import hashlib
import re
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from clients import supabase
from auth import verify_token, require_project_access
from whatsapp import send_whatsapp_message
from ratelimit import is_rate_limited

router = APIRouter()

MAX_MESSAGE_LENGTH = 4096  # WhatsApp rejects a text message longer than this


def generate_key():
    return f"ak_{secrets.token_urlsafe(32)}"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _valid_phone(raw) -> Optional[str]:
    """Normalise a recipient number, or None if it can't be one.

    Mirrors campaigns.py's _valid_phone. A malformed number isn't merely a
    failed send: Meta counts the failure against the WABA's quality rating.
    """
    if not isinstance(raw, str):
        return None
    digits = re.sub(r"\D", "", raw)
    if not (8 <= len(digits) <= 15):
        return None
    return digits


def resolve_api_key(api_key: str) -> Optional[dict]:
    """Validate an API key and return {project_id, id}, or None.

    The single lookup shared by every API-key-authenticated endpoint. There
    used to be two — this one and send_template_api.py's — and they had
    already drifted: that one honoured is_active while this one did not, so
    a key disabled in the database still worked on /public/send.

    Compares by hash; the raw key is never stored, only ever returned once
    at creation/regeneration time.
    """
    if not api_key or not isinstance(api_key, str):
        return None

    res = supabase.table("api_keys") \
        .select("id, project_id, is_active") \
        .eq("key_hash", hash_key(api_key)) \
        .limit(1) \
        .execute()

    if not res.data:
        return None

    row = res.data[0]
    if row.get("is_active") is False:
        return None
    return {"id": row["id"], "project_id": row["project_id"]}


def touch_api_key(key_id: str):
    """Record that a key was used.

    Deliberately called AFTER the rate-limit and quota checks, not during
    lookup: it used to fire on every authenticated request including ones
    about to be refused, which doubled the database writes on the hot path.
    Never allowed to fail a request that has otherwise succeeded.
    """
    try:
        supabase.table("api_keys").update({"last_used_at": "now()"}).eq("id", key_id).execute()
    except Exception as e:
        import sentry_sdk
        sentry_sdk.capture_exception(e)


def get_project_by_key(api_key: str) -> Optional[str]:
    """Backwards-compatible shim — returns just the project_id."""
    resolved = resolve_api_key(api_key)
    return resolved["project_id"] if resolved else None


# -------------------------------------------------
# MANAGEMENT ENDPOINTS (authenticated)
# -------------------------------------------------
@router.get("/api-keys/{project_id}")
def get_api_key(project_id: str, user=Depends(verify_token)):
    """Read-only. Reports whether a key exists; never mints or reveals one.

    This used to CREATE a key when none existed and return the raw value in
    the response — a read verb performing a write, and handing back a live
    WhatsApp sending credential. Combined with the old require_project_role
    (which passes for any role), an agent locked out of this tab could call
    the route and walk away with a working key. Minting now happens only on
    the explicit POST /regenerate below.
    """
    require_project_access(user.id, project_id, tab="api")

    res = supabase.table("api_keys") \
        .select("id, name, created_at, last_used_at, is_active") \
        .eq("project_id", project_id) \
        .limit(1) \
        .execute()

    if not res.data:
        return {"has_key": False}

    return {**res.data[0], "has_key": True}


@router.post("/api-keys/{project_id}/regenerate")
def regenerate_api_key(project_id: str, user=Depends(verify_token)):
    """Mint a new key, replacing any existing one. The only path that ever
    returns the raw value, and the only path that creates one.

    Requires admin: rotating the key instantly breaks every external system
    the merchant has wired up, which is the most destructive action here.
    """
    require_project_access(user.id, project_id, tab="api", min_role="admin")

    if is_rate_limited(f"api-key-regen:{project_id}", limit=10, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Too many key regenerations. Please wait a few minutes.")

    key = generate_key()
    existing = supabase.table("api_keys") \
        .select("id") \
        .eq("project_id", project_id) \
        .limit(1) \
        .execute()

    if existing.data:
        # Scoped to one row id. This was .eq("project_id", ...) with no id
        # filter, so a project that had somehow ended up with two rows got
        # BOTH rewritten to the same hash — which then violates the unique
        # index on key_hash and makes regeneration fail forever after.
        supabase.table("api_keys") \
            .update({"key_hash": hash_key(key), "is_active": True}) \
            .eq("id", existing.data[0]["id"]) \
            .execute()
    else:
        try:
            supabase.table("api_keys").insert({
                "project_id": project_id,
                "key_hash": hash_key(key),
                "name": "Default",
            }).execute()
        except Exception as e:
            # Lost a race with a concurrent first mint. The unique index on
            # project_id makes that a duplicate-key error rather than a
            # second row; tell the caller to retry rather than returning a
            # key that was never stored.
            message = str(e).lower()
            if "duplicate key" in message or "23505" in message:
                raise HTTPException(
                    status_code=409,
                    detail="A key was just created for this project. Reload and try again.",
                )
            raise

    return {"key": key}


# -------------------------------------------------
# PUBLIC SEND ENDPOINT (API key auth)
# -------------------------------------------------
class PublicSendRequest(BaseModel):
    """Was a raw dict: `body.get("to", "").strip()` raised AttributeError on
    a non-string (a 500), malformed JSON was an unhandled 500, and neither
    field had any length cap on an endpoint that spends the merchant's
    WhatsApp budget."""
    to: str = Field(max_length=32)
    message: str = Field(max_length=MAX_MESSAGE_LENGTH)


@router.post("/public/send")
def public_send(
    body: PublicSendRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Send a free-form WhatsApp message from an external system.

    Note for callers: free-form text only reaches a customer inside Meta's
    24-hour customer service window. Outside it, use /api/send-template.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    resolved = resolve_api_key(x_api_key)
    if not resolved:
        raise HTTPException(status_code=401, detail="Invalid API key")

    project_id = resolved["project_id"]

    # Short-window burst limit on top of the monthly usage cap below — a
    # leaked key could otherwise blast messages at full speed until the
    # monthly ceiling is hit, unlike the other public send-adjacent
    # endpoints in this codebase which all layer one on top of that cap.
    if is_rate_limited(f"api-send:{project_id}", limit=20, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please slow down.")

    phone = _valid_phone(body.to)
    if not phone:
        raise HTTPException(status_code=400, detail="'to' must be a valid phone number")

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="'message' is required")

    # Reads access_token too: this passed the MERCHANT's phone_number_id
    # with no token, so whatsapp.py fell back to the PLATFORM's global
    # WHATSAPP_TOKEN. Same defect fixed in campaigns, flows and appointments.
    wa = supabase.table("whatsapp_integrations") \
        .select("phone_number_id, access_token") \
        .eq("project_id", project_id) \
        .limit(1) \
        .execute()

    if not wa.data or not wa.data[0].get("phone_number_id"):
        raise HTTPException(status_code=400, detail="WhatsApp not connected for this project")

    phone_number_id = wa.data[0]["phone_number_id"]
    token = wa.data[0].get("access_token")

    from usage import check_rate_limit, increment_usage
    rate_check = check_rate_limit(project_id)
    if not rate_check["allowed"]:
        raise HTTPException(status_code=429, detail="Monthly message limit reached")

    res = send_whatsapp_message(f"+{phone}", message, phone_number_id, token)

    if not (res and res.ok):
        raise HTTPException(status_code=502, detail="Failed to send WhatsApp message")

    # Both of these now happen only AFTER a confirmed send. increment_usage
    # used to run before the result was checked, so a failed send still
    # consumed the merchant's monthly allowance.
    increment_usage(project_id)
    touch_api_key(resolved["id"])

    return {"status": "sent", "to": f"+{phone}"}
