# billing.py
#
# Zavo's OWN SaaS subscription billing (Pro/Business plans) — replaces the
# old stripe_handler.py. Zavo itself is the merchant here, charging Zavo's
# own customers, using Zavo's own Razorpay account (plain Basic Auth via
# the SDK's Client(auth=(id, secret)) — no OAuth counterparty involved).
#
# This is a THIRD, unrelated Razorpay integration in this codebase — do not
# confuse with:
#   - razorpay_oauth.py — Shop/Appointment merchants connect THEIR OWN
#     Razorpay account via OAuth, so Zavo can process payments on THEIR
#     behalf for THEIR customers.
#   - shop.py's RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET — legacy manual
#     per-merchant fallback keys, also not Zavo's own account.
# See config.py's RAZORPAY_BILLING_* vars and the plan doc's credential
# table for the full picture.
#
# Razorpay has no all-in-one hosted "Billing Portal" the way Stripe did —
# plan changes and cancellation are API-only, so this file also exposes
# /billing/change-plan and /billing/cancel for the app's own self-built
# Manage Billing UI (see src/app/account/page.js).
import hashlib
import hmac
import json

import razorpay
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from clients import supabase
from auth import verify_token
from config import (
    RAZORPAY_BILLING_KEY_ID, RAZORPAY_BILLING_KEY_SECRET,
    RAZORPAY_BILLING_WEBHOOK_SECRET,
    PLAN_TO_RAZORPAY_PLAN_ID, RAZORPAY_PLAN_TO_PLAN,
)

router = APIRouter()

client = razorpay.Client(auth=(RAZORPAY_BILLING_KEY_ID, RAZORPAY_BILLING_KEY_SECRET))

# Razorpay subscriptions require SOME bound — there's no "bill until
# cancelled" flag. Originally used end_at (a far-future timestamp), which
# consistently caused a live, unexplained 5xx ServerError from Razorpay's
# API on every real subscribe attempt, confirmed NOT to be an account-side
# issue (an identical request made directly via curl, using total_count
# instead of end_at, succeeded immediately with the same plan and key).
# total_count (a cycle count) is Razorpay's own more standard way to bound
# a subscription, and it's what's actually proven to work — 10 years'
# worth of cycles is still effectively indefinite for this purpose (no
# real subscription survives a full decade untouched).
_INDEFINITE_CYCLES = {"monthly": 120, "yearly": 10}  # ~10 years either way


class SubscribeRequest(BaseModel):
    plan: str      # "pro" | "business"
    billing: str   # "monthly" | "yearly"


class CancelRequest(BaseModel):
    at_cycle_end: bool = True


def _resolve_plan_id(plan: str, billing: str) -> str:
    plan_id = PLAN_TO_RAZORPAY_PLAN_ID.get((plan, billing))
    if not plan_id:
        raise HTTPException(status_code=400, detail=f"Unknown plan/billing combination: {plan}/{billing}")
    return plan_id


def _get_profile(user_id: str) -> dict:
    res = supabase.table("profiles").select("*").eq("id", user_id).maybe_single().execute()
    return (res.data if res else None) or {}


@router.post("/billing/subscribe")
def subscribe(body: SubscribeRequest, user=Depends(verify_token)):
    plan_id = _resolve_plan_id(body.plan, body.billing)
    total_count = _INDEFINITE_CYCLES.get(body.billing, 120)

    try:
        subscription = client.subscription.create({
            "plan_id": plan_id,
            "quantity": 1,
            "customer_notify": 1,
            "total_count": total_count,
            "notes": {
                "supabase_user_id": user.id,
                "plan": body.plan,
                "billing": body.billing,
            },
        })
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Billing subscribe error: plan_id={plan_id}, user={user.id}, error={e}")
        raise HTTPException(status_code=502, detail="Could not start the subscription. Please try again.")

    # Persisted immediately — the subscription id exists before payment is
    # authorized (unlike Stripe, where no Subscription object existed until
    # checkout completed). `plan` itself is NOT touched here — it only
    # flips on a confirmed-payment webhook (subscription.activated/charged),
    # same "never trust an unconfirmed checkout" rule Stripe's flow followed.
    supabase.table("profiles").upsert({
        "id": user.id,
        "razorpay_subscription_id": subscription["id"],
    }, on_conflict="id").execute()

    return {"url": subscription["short_url"]}


def _verify_billing_webhook_signature(body_bytes: bytes, signature: str) -> bool:
    if not RAZORPAY_BILLING_WEBHOOK_SECRET or not signature:
        return False
    expected = hmac.new(RAZORPAY_BILLING_WEBHOOK_SECRET.encode(), body_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _find_profile_by_subscription(subscription_id: str, notes: dict) -> Optional[dict]:
    res = supabase.table("profiles").select("*").eq("razorpay_subscription_id", subscription_id).maybe_single().execute()
    profile = res.data if res else None
    if profile:
        return profile
    # Fallback for a real race: webhook arriving before /billing/subscribe's
    # own DB write lands. Stripe's flow needed this as its ONLY lookup path
    # for the first event (no subscription id existed on the profile until
    # checkout completed) — here it's just a rare-race backstop.
    user_id = (notes or {}).get("supabase_user_id")
    if not user_id:
        return None
    res = supabase.table("profiles").select("*").eq("id", user_id).maybe_single().execute()
    return res.data if res else None


@router.post("/webhook/razorpay-billing")
async def billing_webhook(request: Request):
    body_bytes = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not _verify_billing_webhook_signature(body_bytes, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = json.loads(body_bytes)
    event = payload.get("event")
    entity = ((payload.get("payload") or {}).get("subscription") or {}).get("entity") or {}
    subscription_id = entity.get("id")
    plan_id = entity.get("plan_id")
    notes = entity.get("notes") or {}
    # Razorpay's docs don't confirm a customer_id field directly on the
    # Subscription entity — but a payment entity (with a well-documented
    # customer_id) rides along in the same payload once a charge has
    # actually happened, so pull it from there opportunistically. Purely
    # informational (shown on the admin page) — never used for auth/lookup.
    payment_entity = ((payload.get("payload") or {}).get("payment") or {}).get("entity") or {}
    customer_id = payment_entity.get("customer_id")

    if not subscription_id:
        return {"status": "ok"}

    profile = _find_profile_by_subscription(subscription_id, notes)
    if not profile:
        return {"status": "ok"}

    user_id = profile["id"]

    if event in ("subscription.activated", "subscription.charged", "subscription.updated", "subscription.resumed"):
        plan = RAZORPAY_PLAN_TO_PLAN.get(plan_id, "free")
        update = {"plan": plan, "razorpay_subscription_id": subscription_id}
        if customer_id:
            update["razorpay_customer_id"] = customer_id
        supabase.table("profiles").update(update).eq("id", user_id).execute()
        print(f"Billing: {user_id} -> {plan} ({event})")

    elif event == "subscription.authenticated":
        # Mandate set up, no charge confirmed yet — nothing to do until an
        # activated/charged event actually confirms payment.
        pass

    elif event == "subscription.pending":
        # Payment attempt failed, Razorpay is auto-retrying — grace period:
        # leave the plan alone. Only subscription.halted (retries exhausted)
        # or a later subscription.charged (retry succeeded) change anything.
        print(f"Billing: {user_id} payment pending/retrying, no plan change")

    elif event in ("subscription.halted", "subscription.paused", "subscription.completed", "subscription.cancelled"):
        update = {"plan": "free"}
        if event == "subscription.cancelled":
            update["razorpay_subscription_id"] = None
        supabase.table("profiles").update(update).eq("id", user_id).execute()
        print(f"Billing: {user_id} -> free ({event})")

    return {"status": "ok"}


@router.get("/billing/plan")
def get_plan(user=Depends(verify_token)):
    profile = _get_profile(user.id)
    subscription_id = profile.get("razorpay_subscription_id")
    result = {
        "plan": profile.get("plan", "free"),
        "has_subscription": bool(subscription_id),
    }

    if subscription_id:
        try:
            sub = client.subscription.fetch(subscription_id)
            result["status"] = sub.get("status")
            result["charge_at"] = sub.get("charge_at")
            # Razorpay flips status to "cancelled" immediately when a
            # cancel-at-cycle-end is requested, even though access keeps
            # running until current_end — the frontend needs this to show
            # "cancels on X" instead of silently looking unchanged.
            result["current_end"] = sub.get("current_end")
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Billing get_plan fetch error: subscription_id={subscription_id}, user={user.id}, error={e}")

    return result


@router.get("/billing/invoices")
def list_invoices(user=Depends(verify_token)):
    profile = _get_profile(user.id)
    subscription_id = profile.get("razorpay_subscription_id")
    if not subscription_id:
        return {"invoices": []}

    try:
        res = client.invoice.all({"subscription_id": subscription_id})
        return {"invoices": res.get("items", [])}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Billing list_invoices error: subscription_id={subscription_id}, user={user.id}, error={e}")
        return {"invoices": []}


@router.post("/billing/change-plan")
def change_plan(body: SubscribeRequest, user=Depends(verify_token)):
    profile = _get_profile(user.id)
    subscription_id = profile.get("razorpay_subscription_id")
    if not subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription to change.")

    new_plan_id = _resolve_plan_id(body.plan, body.billing)

    try:
        # "now" prorates (Razorpay auto charges/refunds the difference);
        # "cycle_end" applies at the next cycle with no adjustment needed.
        client.subscription.edit(subscription_id, {
            "plan_id": new_plan_id,
            "schedule_change_at": "now",
        })
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Billing change_plan error: subscription_id={subscription_id}, new_plan_id={new_plan_id}, user={user.id}, error={e}")
        # Real, permanent Razorpay platform limitation, not a transient
        # failure — a UPI AutoPay mandate is registered for one specific
        # plan/amount and can't be edited in place the way a card can.
        # "Please try again" would be actively misleading here since
        # retrying can never succeed; tell the customer what to actually do.
        if "payment mode is upi" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail="UPI payments don't support changing plans. Please cancel your subscription and re-subscribe to a different plan — using a card instead of UPI lets you change plans directly next time.",
            )
        raise HTTPException(status_code=502, detail="Could not change your plan. Please try again.")

    # Does NOT optimistically flip profiles.plan — waits for the resulting
    # subscription.updated webhook, same latency model as the old Stripe
    # portal (which also never mutated `plan` client-side).
    return {"status": "pending"}


@router.post("/billing/cancel")
def cancel_subscription(body: CancelRequest, user=Depends(verify_token)):
    profile = _get_profile(user.id)
    subscription_id = profile.get("razorpay_subscription_id")
    if not subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription to cancel.")

    try:
        client.subscription.cancel(subscription_id, {
            "cancel_at_cycle_end": 1 if body.at_cycle_end else 0,
        })
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Billing cancel_subscription error: subscription_id={subscription_id}, user={user.id}, error={e}")
        raise HTTPException(status_code=502, detail="Could not cancel your subscription. Please try again.")

    return {"status": "pending"}
