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
import requests
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from clients import supabase
from webhook_dedup import already_processed
from auth import verify_token
from ratelimit import is_rate_limited
from config import (
    RAZORPAY_BILLING_KEY_ID, RAZORPAY_BILLING_KEY_SECRET,
    RAZORPAY_BILLING_WEBHOOK_SECRET,
    PLAN_TO_RAZORPAY_PLAN_ID, RAZORPAY_PLAN_TO_PLAN,
)

router = APIRouter()

class _TimeoutSession(requests.Session):
    """Applies a default timeout to every Razorpay call.

    get_plan and list_invoices hit Razorpay synchronously while the billing
    page loads, and the SDK creates a plain requests.Session, which waits
    forever by default — so a slow (not even down) Razorpay pinned a worker
    per request. Same wrapper whatsapp.py uses for the Graph API.
    """

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", 20)
        return super().request(*args, **kwargs)


client = razorpay.Client(
    session=_TimeoutSession(),
    auth=(RAZORPAY_BILLING_KEY_ID, RAZORPAY_BILLING_KEY_SECRET),
)

# Razorpay states in which a subscription is still the user's live one.
# "cancelled"/"completed"/"expired" are terminal, so a profile still
# pointing at one of those must not block a fresh subscribe.
_LIVE_SUBSCRIPTION_STATES = {"created", "authenticated", "active", "pending", "halted", "paused"}

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


def _live_subscription_status(subscription_id: str) -> Optional[str]:
    """The Razorpay status of a subscription, if it is still live.

    Returns None when it is terminal, unknown, or Razorpay can't be reached
    — the caller treats that as "nothing blocking a new subscription".
    Failing open here is deliberate: refusing to let someone subscribe
    because Razorpay had a bad minute costs a sale, whereas the duplicate it
    might admit is caught by the unique index on the profile column.
    """
    if not subscription_id:
        return None
    try:
        sub = client.subscription.fetch(subscription_id)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return None
    status = sub.get("status")
    return status if status in _LIVE_SUBSCRIPTION_STATES else None


@router.post("/billing/subscribe")
def subscribe(body: SubscribeRequest, user=Depends(verify_token)):
    plan_id = _resolve_plan_id(body.plan, body.billing)
    total_count = _INDEFINITE_CYCLES.get(body.billing, 120)

    # Each call creates a real Razorpay subscription object. Nothing bounded
    # that before.
    if is_rate_limited(f"billing-subscribe:{user.id}", limit=5, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a few minutes and try again.")

    # Nothing stopped a second subscription. A double-click during checkout
    # created a SECOND Razorpay subscription and overwrote
    # razorpay_subscription_id with it — so the first kept billing the
    # customer with no record of it anywhere, and its eventual
    # subscription.cancelled webhook downgraded a still-paying customer.
    profile = _get_profile(user.id)
    existing_id = profile.get("razorpay_subscription_id")
    if existing_id and _live_subscription_status(existing_id):
        raise HTTPException(
            status_code=409,
            detail="You already have an active subscription. Use Change Plan to switch, or cancel it first.",
        )

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
        raise HTTPException(status_code=502, detail="Could not start the subscription. Please try again.")

    # Persisted immediately — the subscription id exists before payment is
    # authorized (unlike Stripe, where no Subscription object existed until
    # checkout completed). `plan` itself is NOT touched here — it only
    # flips on a confirmed-payment webhook (subscription.activated/charged),
    # same "never trust an unconfirmed checkout" rule Stripe's flow followed.
    supabase.table("profiles").upsert({
        "id": user.id,
        "razorpay_subscription_id": subscription["id"],
        # Reset in case they're re-subscribing after a previous
        # cancel-at-cycle-end — this is a brand new subscription, not
        # scheduled to end.
        "subscription_cancel_scheduled": False,
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
    candidate = res.data if res else None
    if not candidate:
        return None
    # Only accept the fallback while the profile has NO subscription on file.
    # If it already points at a different subscription, this event belongs to
    # a superseded one and must not be allowed to rewrite the live plan.
    if candidate.get("razorpay_subscription_id"):
        return None
    return candidate


@router.post("/webhook/razorpay-billing")
async def billing_webhook(request: Request):
    body_bytes = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not _verify_billing_webhook_signature(body_bytes, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        payload = json.loads(body_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")

    # A valid signature proves Razorpay sent this body — it does NOT prove
    # we haven't already acted on it. Razorpay retries, and without this a
    # replayed subscription.cancelled downgrades a paying customer to free,
    # while a replayed subscription.activated resurrects a cancelled plan.
    # Razorpay normally sends X-Razorpay-Event-Id, but when it doesn't, the
    # old `if event_id` guard silently skipped dedup entirely — on the one
    # path that changes what a customer pays. A hash of the raw body is
    # stable across retries of the same event, so it works as the key.
    event_id = request.headers.get("X-Razorpay-Event-Id") or hashlib.sha256(body_bytes).hexdigest()
    if already_processed("razorpay-billing", event_id):
        return {"status": "duplicate_ignored"}

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
        # Defaulting to "free" meant a renamed or rotated Razorpay plan id
        # silently downgraded a paying customer on the next subscription
        # event. Note config collapses to {"": "business"} when the plan env
        # vars are unset, so an unrecognised id is a real possibility.
        plan = RAZORPAY_PLAN_TO_PLAN.get(plan_id)
        if not plan:
            sentry_sdk.capture_message(
                f"Razorpay billing webhook: unknown plan_id {plan_id!r} for user {user_id} - leaving plan unchanged"
            )
            return {"status": "unknown_plan_ignored"}
        # subscription.resumed specifically means a cancel-at-cycle-end got
        # reversed — not scheduled to end anymore either way here.
        # Scoped so an event from a superseded subscription can't take the
        # profile over: either the profile already points at this
        # subscription, or it points at nothing yet.
        current_sub = profile.get("razorpay_subscription_id")
        if current_sub and current_sub != subscription_id:
            sentry_sdk.capture_message(
                f"Razorpay billing webhook: {event} for superseded subscription - ignored"
            )
            return {"status": "superseded_ignored"}

        update = {"plan": plan, "razorpay_subscription_id": subscription_id, "subscription_cancel_scheduled": False}
        if customer_id:
            update["razorpay_customer_id"] = customer_id
        supabase.table("profiles").update(update).eq("id", user_id).execute()
        print(f"Billing: plan -> {plan} ({event})")

    elif event == "subscription.authenticated":
        # Mandate set up, no charge confirmed yet — nothing to do until an
        # activated/charged event actually confirms payment.
        pass

    elif event == "subscription.pending":
        # Payment attempt failed, Razorpay is auto-retrying — grace period:
        # leave the plan alone. Only subscription.halted (retries exhausted)
        # or a later subscription.charged (retry succeeded) change anything.
        print("Billing: payment pending/retrying, no plan change")

    elif event in ("subscription.halted", "subscription.paused", "subscription.completed", "subscription.cancelled"):
        # THE important guard. This downgraded on ANY subscription's
        # cancellation, so an orphaned earlier subscription ending would
        # drop a currently-paying customer to free. The write now only
        # matches while the profile still points at the subscription this
        # event is about.
        update = {"plan": "free", "subscription_cancel_scheduled": False}
        if event == "subscription.cancelled":
            update["razorpay_subscription_id"] = None
        res = supabase.table("profiles").update(update)             .eq("id", user_id)             .eq("razorpay_subscription_id", subscription_id)             .execute()
        if not res.data:
            sentry_sdk.capture_message(
                f"Razorpay billing webhook: {event} for a subscription not on file - plan left unchanged"
            )
            return {"status": "superseded_ignored"}
        print(f"Billing: plan -> free ({event})")

    return {"status": "ok"}


@router.get("/billing/plan")
def get_plan(user=Depends(verify_token)):
    profile = _get_profile(user.id)
    subscription_id = profile.get("razorpay_subscription_id")
    result = {
        "plan": profile.get("plan", "free"),
        "has_subscription": bool(subscription_id),
        # NOT derived from Razorpay's live status field — confirmed live
        # that Razorpay keeps status "active" the whole time for a
        # cancel-at-cycle-end subscription, right up until it actually
        # ends. Tracked on our own side instead (set in cancel_subscription
        # below) so the frontend can reliably show "cancels on X".
        "cancel_scheduled": bool(profile.get("subscription_cancel_scheduled")),
    }

    if subscription_id:
        try:
            sub = client.subscription.fetch(subscription_id)
            result["status"] = sub.get("status")
            result["charge_at"] = sub.get("charge_at")
            result["current_end"] = sub.get("current_end")
        except Exception as e:
            sentry_sdk.capture_exception(e)

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
        return {"invoices": []}


@router.post("/billing/change-plan")
def change_plan(body: SubscribeRequest, user=Depends(verify_token)):
    # schedule_change_at="now" PRORATES on every call — Razorpay charges or
    # refunds the difference each time — so an unthrottled loop moves real
    # money against a real card repeatedly.
    if is_rate_limited(f"billing-change-plan:{user.id}", limit=5, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Too many plan changes. Please wait a few minutes and try again.")

    profile = _get_profile(user.id)
    subscription_id = profile.get("razorpay_subscription_id")
    if not subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription to change.")

    new_plan_id = _resolve_plan_id(body.plan, body.billing)

    # Switching to the plan already active still issued a prorated edit.
    current_plan_id = PLAN_TO_RAZORPAY_PLAN_ID.get((profile.get("plan"), body.billing))
    if current_plan_id and current_plan_id == new_plan_id:
        return {"status": "unchanged"}

    try:
        # "now" prorates (Razorpay auto charges/refunds the difference);
        # "cycle_end" applies at the next cycle with no adjustment needed.
        client.subscription.edit(subscription_id, {
            "plan_id": new_plan_id,
            "schedule_change_at": "now",
        })
    except Exception as e:
        sentry_sdk.capture_exception(e)
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
    if is_rate_limited(f"billing-cancel:{user.id}", limit=5, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a few minutes and try again.")

    profile = _get_profile(user.id)
    subscription_id = profile.get("razorpay_subscription_id")
    if not subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription to cancel.")

    # Already scheduled to end — calling Razorpay again changes nothing.
    if body.at_cycle_end and profile.get("subscription_cancel_scheduled"):
        return {"status": "already_scheduled"}

    try:
        client.subscription.cancel(subscription_id, {
            "cancel_at_cycle_end": 1 if body.at_cycle_end else 0,
        })
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=502, detail="Could not cancel your subscription. Please try again.")

    # See get_plan's comment — Razorpay's own status field won't reflect
    # this, so this is the only record that a cancel-at-cycle-end is
    # scheduled. Only meaningful for at_cycle_end=True; an immediate cancel
    # gets caught by the subscription.cancelled webhook flipping plan to
    # free almost immediately anyway, but setting it False either way keeps
    # this field always accurate rather than stale from a previous cancel.
    supabase.table("profiles").update({
        "subscription_cancel_scheduled": body.at_cycle_end,
    }).eq("id", user.id).execute()

    return {"status": "pending"}
