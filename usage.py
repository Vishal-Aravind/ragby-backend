import sentry_sdk
from datetime import datetime
from fastapi import APIRouter, Depends

from clients import supabase
from config import PLAN_LIMITS
from auth import verify_token, require_project_access

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


@router.get("/projects/{project_id}/limits")
def project_limits(project_id: str, user=Depends(verify_token)):
    """Exposes the caller's effective plan limits to the frontend, scoped to
    THIS project (i.e. its owner's plan, via get_plan_limits — not the
    caller's own /usage/status, which is keyed on the caller's own profile
    and would be wrong for a teammate uploading into someone else's
    project). Currently just the subset the upload flow needs; add more
    keys here rather than inventing a second endpoint."""
    require_project_access(user.id, project_id, tab="documents")
    limits = get_plan_limits(project_id)
    return {
        "items": limits["items"],
        "used": count_knowledge_items(project_id),
        "maxFileMB": limits["maxFileMB"],
    }


def count_knowledge_items(project_id: str, exclude_file_id: str = None) -> int:
    """Everything counted against the plan's single "items" limit: uploaded
    documents and text notes (files) plus connected sources (sheets, Excel,
    websites, databases). Shopify is excluded — it's a store integration
    connected through OAuth, not something uploaded into the knowledge base.
    exclude_file_id: the file being ingested right now, whose row already
    exists (re-saving a note must not count against itself)."""
    # A file whose processing failed stays listed (so the merchant can see
    # the error and delete it) but isn't in the knowledge base — with a
    # limit of 3 it must not eat a slot.
    files = (
        supabase.table("files").select("id", count="exact")
        .eq("project_id", project_id).neq("status", "failed")
    )
    if exclude_file_id:
        files = files.neq("id", exclude_file_id)
    # Same for a source whose indexing failed (private sheet, blocked site):
    # it stays listed with the reason, but holds nothing. Written as an
    # "or": a plain neq would also drop rows with no sync_status at all.
    sources = (
        supabase.table("data_sources").select("id", count="exact")
        .eq("project_id", project_id).neq("type", "shopify")
        .or_("config->>sync_status.is.null,config->>sync_status.neq.failed")
    )
    return (files.execute().count or 0) + (sources.execute().count or 0)


def knowledge_limit_message(limit: int) -> str:
    return (
        f"You've reached your plan's limit of {limit} items in your knowledge base "
        f"(documents, notes, sheets, Excel files, websites and databases combined). "
        f"Delete one, or upgrade your plan, to add more."
    )


def check_rate_limit(project_id: str) -> dict:
    # FIX: was .single(), which raises PGRST116 ("0 rows") instead of
    # returning empty data — any chat request against a stale/deleted
    # project_id (e.g. an old widget embed still live on a merchant's site
    # after they delete the project) threw an unhandled 500 here, leaving
    # the widget's "..." typing indicator stuck forever with no error shown.
    proj = supabase.table("projects") \
        .select("user_id, suspended") \
        .eq("id", project_id) \
        .maybe_single() \
        .execute()

    if not proj or not proj.data:
        return {"allowed": False, "reason": "Project not found"}

    # Suspension was only enforced deep inside run_chat, which runs AFTER
    # public_chat has already inserted the `chats` row — so a suspended
    # project's inbox could still be filled with conversations. Checked here
    # instead, which puts it ahead of every caller's own writes and covers
    # WhatsApp and Telegram at the same time, not just the widget.
    if proj.data.get("suspended"):
        return {"allowed": False, "reason": "Project suspended"}

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
        # maybe_single, not single: a webhook arriving for a project that was
        # just deleted raised PGRST116 here, and the caller swallowed it as a
        # generic failure.
        proj = supabase.table("projects") \
            .select("user_id") \
            .eq("id", project_id) \
            .maybe_single() \
            .execute()

        if not proj or not proj.data:
            return

        user_id = proj.data["user_id"]
        month = get_current_month()

        # One statement, evaluated by Postgres against the locked row.
        # This was a read-then-write: two concurrent messages both read N
        # and both wrote N+1, so we billed for fewer replies than we served.
        # The insert branch was worse — two concurrent first-messages of a
        # month created two usage rows, and check_rate_limit only ever reads
        # the first, so the second accumulated invisibly and the monthly cap
        # was undercounted for that user from then on.
        supabase.rpc("increment_usage_atomic", {
            "p_user_id": user_id,
            "p_month": month,
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
        .maybe_single() \
        .execute()

    plan = "free"
    if profile and profile.data and profile.data.get("plan"):
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