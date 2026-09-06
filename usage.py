import sentry_sdk
from datetime import datetime
from fastapi import APIRouter, Depends

from clients import supabase
from config import PLAN_LIMITS
from auth import verify_token

router = APIRouter()


def get_current_month() -> str:
    return datetime.utcnow().strftime("%Y-%m")


def get_plan_limits(project_id: str) -> dict:
    """Plan limits for whoever OWNS this project (not the caller — a
    teammate's uploads count against the owner's plan). Single source of
    truth: config.PLAN_LIMITS, so these can't drift the way the seat limit
    did when it was copied into three different files."""
    proj = supabase.table("projects")         .select("user_id")         .eq("id", project_id)         .maybe_single()         .execute()
    if not proj or not proj.data:
        return PLAN_LIMITS["free"]

    profile = supabase.table("profiles")         .select("plan")         .eq("id", proj.data["user_id"])         .maybe_single()         .execute()
    plan = (profile.data or {}).get("plan") or "free" if profile else "free"
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


def check_rate_limit(project_id: str) -> dict:
    # FIX: was .single(), which raises PGRST116 ("0 rows") instead of
    # returning empty data — any chat request against a stale/deleted
    # project_id (e.g. an old widget embed still live on a merchant's site
    # after they delete the project) threw an unhandled 500 here, leaving
    # the widget's "..." typing indicator stuck forever with no error shown.
    proj = supabase.table("projects") \
        .select("user_id") \
        .eq("id", project_id) \
        .maybe_single() \
        .execute()

    if not proj or not proj.data:
        return {"allowed": False, "reason": "Project not found"}

    user_id = proj.data["user_id"]

    profile = supabase.table("profiles") \
        .select("plan") \
        .eq("id", user_id) \
        .maybe_single() \
        .execute()

    plan = "free"
    if profile.data and profile.data.get("plan"):
        plan = profile.data["plan"]

    limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    month = get_current_month()

    usage = supabase.table("usage") \
        .select("count") \
        .eq("user_id", user_id) \
        .eq("month", month) \
        .execute()

    current_count = usage.data[0]["count"] if usage.data else 0

    if current_count >= limit["conversations"]:
        return {
            "allowed": False,
            "reason": f"Monthly limit of {limit['conversations']} conversations reached. Please upgrade your plan.",
            "plan": plan,
            "usage": current_count,
            "limit": limit["conversations"],
        }

    return {
        "allowed": True,
        "plan": plan,
        "usage": current_count,
        "limit": limit["conversations"],
    }


def increment_usage(project_id: str):
    try:
        proj = supabase.table("projects") \
            .select("user_id") \
            .eq("id", project_id) \
            .single() \
            .execute()

        if not proj.data:
            return

        user_id = proj.data["user_id"]
        month = get_current_month()

        existing = supabase.table("usage") \
            .select("id, count") \
            .eq("user_id", user_id) \
            .eq("month", month) \
            .execute()

        if existing.data:
            supabase.table("usage") \
                .update({"count": existing.data[0]["count"] + 1}) \
                .eq("id", existing.data[0]["id"]) \
                .execute()
        else:
            supabase.table("usage").insert({
                "user_id": user_id,
                "month": month,
                "count": 1,
            }).execute()

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"increment_usage error: {e}")


@router.get("/usage/status")
def usage_status(user=Depends(verify_token)):
    user_id = user.id
    month = get_current_month()

    profile = supabase.table("profiles") \
        .select("plan") \
        .eq("id", user_id) \
        .single() \
        .execute()

    plan = "free"
    if profile.data and profile.data.get("plan"):
        plan = profile.data["plan"]

    limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

    usage = supabase.table("usage") \
        .select("count") \
        .eq("user_id", user_id) \
        .eq("month", month) \
        .execute()

    count = usage.data[0]["count"] if usage.data else 0

    return {
        "plan": plan,
        "usage": count,
        "limit": limit["conversations"],
        "remaining": max(0, limit["conversations"] - count),
        "month": month,
    }