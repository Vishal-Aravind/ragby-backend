import json
import stripe
from fastapi import APIRouter, Depends, HTTPException, Request

from clients import supabase
from config import (
    STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, FRONTEND_URL,
    PRICE_TO_PLAN
)
from auth import verify_token

stripe.api_key = STRIPE_SECRET_KEY

router = APIRouter()


@router.post("/stripe/checkout")
def create_checkout(data: dict, user=Depends(verify_token)):
    price_id = data["priceId"]
    user_id = user.id
    user_email = user.email

    profile = supabase.table("profiles") \
        .select("stripe_customer_id") \
        .eq("id", user_id) \
        .single() \
        .execute()

    stripe_customer_id = None
    if profile.data:
        stripe_customer_id = profile.data.get("stripe_customer_id")

    if not stripe_customer_id:
        customer = stripe.Customer.create(
            email=user_email,
            metadata={"supabase_user_id": user_id}
        )
        stripe_customer_id = customer.id
        supabase.table("profiles").upsert({
            "id": user_id,
            "stripe_customer_id": stripe_customer_id,
        }, on_conflict="id").execute()

    session = stripe.checkout.Session.create(
        customer=stripe_customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=f"{FRONTEND_URL}/dashboard?upgraded=true",
        cancel_url=f"{FRONTEND_URL}/pricing?cancelled=true",
        metadata={
            "supabase_user_id": user_id,
            "price_id": price_id,
        },
        subscription_data={
            "metadata": {
                "supabase_user_id": user_id,
                "price_id": price_id,
            }
        }
    )

    return {"url": session.url}


@router.post("/stripe/portal")
def customer_portal(user=Depends(verify_token)):
    user_id = user.id

    profile = supabase.table("profiles") \
        .select("stripe_customer_id") \
        .eq("id", user_id) \
        .single() \
        .execute()

    if not profile.data or not profile.data.get("stripe_customer_id"):
        raise HTTPException(status_code=400, detail="No Stripe customer found.")

    session = stripe.billing_portal.Session.create(
        customer=profile.data["stripe_customer_id"],
        return_url=f"{FRONTEND_URL}/dashboard",
    )

    return {"url": session.url}


@router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    raw = json.loads(payload)
    event_type = raw["type"]
    data = raw["data"]["object"]

    if event_type == "checkout.session.completed":
        metadata = data.get("metadata") or {}
        user_id = metadata.get("supabase_user_id")
        price_id = metadata.get("price_id")
        subscription_id = data.get("subscription")

        if user_id and price_id:
            plan = PRICE_TO_PLAN.get(price_id, "free")
            supabase.table("profiles").upsert({
                "id": user_id,
                "plan": plan,
                "stripe_subscription_id": subscription_id,
            }, on_conflict="id").execute()
            print(f"Plan activated: {user_id} → {plan}")

    elif event_type == "customer.subscription.updated":
        subscription_id = data.get("id")
        price_id = data.get("items", {}).get("data", [{}])[0].get("price", {}).get("id", "")
        status = data.get("status")

        profile = supabase.table("profiles") \
            .select("id") \
            .eq("stripe_subscription_id", subscription_id) \
            .single() \
            .execute()

        if profile.data:
            user_id = profile.data["id"]
            plan = PRICE_TO_PLAN.get(price_id, "free") if status in ("active", "trialing") else "free"
            supabase.table("profiles").update({"plan": plan}).eq("id", user_id).execute()
            print(f"Plan updated: {user_id} → {plan}")

    elif event_type == "customer.subscription.deleted":
        subscription_id = data.get("id")

        profile = supabase.table("profiles") \
            .select("id") \
            .eq("stripe_subscription_id", subscription_id) \
            .single() \
            .execute()

        if profile.data:
            supabase.table("profiles").update({
                "plan": "free",
                "stripe_subscription_id": None,
            }).eq("id", profile.data["id"]).execute()
            print(f"Plan cancelled: {profile.data['id']} → free")

    return {"status": "ok"}


@router.get("/stripe/plan")
def get_plan(user=Depends(verify_token)):
    user_id = user.id

    profile = supabase.table("profiles") \
        .select("plan, stripe_customer_id, stripe_subscription_id") \
        .eq("id", user_id) \
        .single() \
        .execute()

    if not profile.data:
        return {"plan": "free"}

    return {
        "plan": profile.data.get("plan", "free"),
        "has_subscription": bool(profile.data.get("stripe_subscription_id")),
    }