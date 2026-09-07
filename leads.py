import re
import time

import sentry_sdk
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from clients import supabase
from auth import verify_token, require_project_role, require_project_access
from config import MAX_LEADS_PER_PROJECT
from ratelimit import is_rate_limited, client_ip

router = APIRouter()

_UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

# Deliberately permissive but structural — the point is to reject "@" and
# "not an email", not to adjudicate RFC 5322. Anything stricter rejects real
# addresses, which on a lead form means losing the lead.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")

# The columns the Leads tab actually renders. Was `select("*")`, which shipped
# every column of every row — including any column added later — to the
# browser on each page load.
_LEAD_COLUMNS = (
    "id, project_id, name, email, phone, source, channel, "
    "whatsapp_number, tags, last_seen_at, created_at"
)

LEADS_PAGE_SIZE = 200
MAX_LEADS_PAGE_SIZE = 500


def _is_duplicate_error(e: Exception) -> bool:
    """Postgres unique-violation, however the client surfaces it.
    Same test used in webhook_dedup.already_processed."""
    text = str(e).lower()
    return "duplicate key" in text or "23505" in text


# -------------------------------------------------
# LEAD CAPTURE CONFIG (read path, shared with chat.py)
# -------------------------------------------------
# chat.py consults this on every public message once the feature is on, so an
# uncached read would add a query per message. Bounded + TTL'd on purpose: an
# unbounded module-level dict keyed by project_id is a slow memory leak, which
# is exactly what the rest of this codebase has been cleaning up.
_CONFIG_TTL_SECONDS = 60
_CONFIG_CACHE_MAX = 500
_config_cache: dict[str, tuple[float, dict]] = {}


def _fetch_lead_config(project_id: str) -> dict:
    res = supabase.table("lead_capture_config") \
        .select("enabled, trigger_after_messages, form_title, form_subtitle") \
        .eq("project_id", project_id) \
        .limit(1) \
        .execute()

    row = (res.data or [None])[0]
    if not row or not row.get("enabled"):
        return {"enabled": False}
    return row


def get_lead_config(project_id: str) -> dict:
    """Cached lead-capture config. Returns {"enabled": False} when the feature
    is off or the project doesn't exist — never raises, so a config lookup can
    never take down the chat path."""
    now = time.time()
    hit = _config_cache.get(project_id)
    if hit and now - hit[0] < _CONFIG_TTL_SECONDS:
        return hit[1]

    try:
        config = _fetch_lead_config(project_id)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {"enabled": False}

    if len(_config_cache) >= _CONFIG_CACHE_MAX:
        # Cheapest bounded eviction: drop whatever is stalest. This cache is a
        # latency optimisation, not a correctness mechanism.
        oldest = min(_config_cache, key=lambda k: _config_cache[k][0])
        _config_cache.pop(oldest, None)

    _config_cache[project_id] = (now, config)
    return config


def invalidate_lead_config(project_id: str):
    _config_cache.pop(project_id, None)


@router.get("/public/lead-config/{project_id}")
def get_lead_config_public(project_id: str, request: Request):
    # Unauthenticated and unthrottled before this: a project-existence oracle
    # anyone could hammer for free. A non-UUID also reached Postgres as a
    # `uuid = text` type error and surfaced as a 500 with driver detail.
    if not _UUID_RE.match(project_id or ""):
        raise HTTPException(status_code=400, detail="Invalid project id")

    if is_rate_limited(f"leadcfg:{client_ip(request)}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please slow down.")

    return get_lead_config(project_id)


# -------------------------------------------------
# PUBLIC LEAD SUBMISSION
# -------------------------------------------------
class LeadSubmitRequest(BaseModel):
    project_id: str
    # The durable per-browser id from widget.js — the dedup key, and the key
    # chat.py's gate looks the lead up by. Not a chat session.
    session_id: str = Field(max_length=64)
    # The REAL chat session, verified against the chats table below. Without
    # it this endpoint was an open write into any merchant's contact list.
    chat_session_id: str
    name: str = Field(max_length=120)
    email: str = Field(max_length=254)
    phone: str = Field(max_length=32)


def _attach_to_existing_contact(project_id: str, session_id: str, name: str, email: str, phone: str) -> bool:
    """Link this browser to an existing contact with the same phone.

    Returns True if a row was claimed. session_id is overwritten rather than
    only filled when empty: it is what the chat gate matches on, so the
    browser currently chatting has to be the one it points at, or that visitor
    stays blocked. Name and email only fill gaps, so a WhatsApp profile name
    already on file isn't clobbered by whatever was typed into the form.
    """
    row = supabase.table("leads") \
        .select("id, name, email") \
        .eq("project_id", project_id) \
        .eq("phone", phone) \
        .limit(1) \
        .execute()

    if not row.data:
        return False

    contact = row.data[0]
    update = {"session_id": session_id, "last_seen_at": "now()"}
    if not (contact.get("name") or "").strip():
        update["name"] = name
    if not (contact.get("email") or "").strip():
        update["email"] = email

    supabase.table("leads").update(update).eq("id", contact["id"]).execute()
    return True


@router.post("/public/leads")
def submit_lead(req: LeadSubmitRequest, request: Request):
    ip = client_ip(request)
    if is_rate_limited(f"leads:{req.project_id}:{ip}", limit=5):
        raise HTTPException(status_code=429, detail="Too many attempts — please wait a moment and try again.")

    if not _UUID_RE.match(req.project_id or ""):
        raise HTTPException(status_code=400, detail="Invalid project id")

    # GATE 1 — the feature must actually be on. Stops injection into every
    # project that never enabled lead capture, which is most of them.
    if not get_lead_config(req.project_id).get("enabled"):
        raise HTTPException(status_code=403, detail="Lead capture is not enabled for this chatbot.")

    # GATE 2 — you must have actually had a conversation. Creating a chat is
    # already quota-capped (usage.check_rate_limit) and IP rate-limited in
    # chat.py, so this puts a real cost on forging leads in bulk. Without it,
    # anyone with the project id from the public embed snippet could stuff the
    # contact list — and campaigns.py:resolve_contacts sends a paid WhatsApp
    # template to every row with a phone number.
    if not _UUID_RE.match(req.chat_session_id or ""):
        raise HTTPException(status_code=400, detail="Invalid session")

    chat = supabase.table("chats") \
        .select("id") \
        .eq("id", req.chat_session_id) \
        .eq("project_id", req.project_id) \
        .in_("channel", ["public", "shopify"]) \
        .limit(1) \
        .execute()
    if not chat.data:
        raise HTTPException(status_code=403, detail="Start a conversation before sharing your details.")

    name = req.name.strip()
    email = req.email.strip()
    phone = req.phone.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Please enter your name.")

    # Was `"@" not in email`, which accepted the single character "@".
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")

    # Was len() on the raw string, so seven spaces and a letter passed. The
    # widget already strips non-digits client-side; the server never did.
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 7:
        raise HTTPException(status_code=400, detail="Please enter a valid phone number.")

    existing = supabase.table("leads") \
        .select("id") \
        .eq("session_id", req.session_id) \
        .eq("project_id", req.project_id) \
        .limit(1) \
        .execute()

    if existing.data:
        return {"status": "already_captured"}

    # This same person may already exist as a WhatsApp contact under the same
    # phone, saved by upsert_contact. Claim that row instead of inserting a
    # second one: the unique index on (project_id, phone) would reject the
    # insert anyway, and — more importantly — chat.py's gate looks a lead up
    # by session_id, so leaving this browser unlinked would lock the visitor
    # out of the chat permanently after they'd just handed over their details.
    if _attach_to_existing_contact(req.project_id, req.session_id, name, email, phone):
        return {"status": "captured"}

    # Hard ceiling on every plan. Nothing bounded this table at all before —
    # get_plan_limits has no leads key — so an unauthenticated endpoint could
    # grow it without limit. Mirrors MAX_CHUNKS_PER_INGEST / MAX_SHEET_ROWS.
    count_res = supabase.table("leads") \
        .select("id", count="exact") \
        .eq("project_id", req.project_id) \
        .limit(1) \
        .execute()
    if (count_res.count or 0) >= MAX_LEADS_PER_PROJECT:
        raise HTTPException(
            status_code=403,
            detail="This chatbot isn't accepting new contacts right now.",
        )

    try:
        supabase.table("leads").insert({
            "project_id": req.project_id,
            "session_id": req.session_id,
            "name": name,
            "email": email,
            "phone": phone,
            "source": "widget",
            "channel": "web",
            "last_seen_at": "now()",
        }).execute()
    except Exception as e:
        # The selects above are a fast path, not a guarantee — two concurrent
        # submits both see nothing and both insert. The unique index added in
        # the leads_unique_constraints migration turns the loser into this.
        if not _is_duplicate_error(e):
            raise
        # Whoever won may have been an upsert_contact for the same phone, so
        # try once more to claim it rather than stranding this visitor.
        if not _attach_to_existing_contact(req.project_id, req.session_id, name, email, phone):
            return {"status": "already_captured"}

    return {"status": "captured"}


class LeadConfigRequest(BaseModel):
    projectId: str
    enabled: bool
    # 0 or negative made widget.js's `userMessageCount >= threshold` fire
    # before the first message — and with the server-side gate in place, that
    # locks a visitor out before a chat session exists to validate against.
    triggerAfterMessages: Optional[int] = Field(default=2, ge=1, le=20)
    formTitle: Optional[str] = Field(default="Before we continue...", max_length=120)
    formSubtitle: Optional[str] = Field(
        default="Please share your details to keep chatting.", max_length=240
    )


@router.put("/lead-config")
def save_lead_config(req: LeadConfigRequest, user=Depends(verify_token)):
    # Was require_project_role, which only checks that SOME role exists — so an
    # agent could curl this even though the toggle lives in the Integrations
    # tab they're blocked from. "integrations" is the key in tabs-config.js.
    require_project_access(user.id, req.projectId, tab="integrations")
    supabase.table("lead_capture_config").upsert({
        "project_id": req.projectId,
        "enabled": req.enabled,
        "trigger_after_messages": req.triggerAfterMessages,
        "form_title": req.formTitle,
        "form_subtitle": req.formSubtitle,
    }, on_conflict="project_id").execute()
    invalidate_lead_config(req.projectId)
    return {"status": "saved"}


@router.get("/leads")
def get_leads(
    project_id: str,
    user=Depends(verify_token),
    offset: int = 0,
    limit: int = LEADS_PAGE_SIZE,
):
    require_project_role(user.id, project_id)

    offset = max(0, offset)
    limit = max(1, min(limit, MAX_LEADS_PAGE_SIZE))

    res = supabase.table("leads") \
        .select(_LEAD_COLUMNS, count="exact") \
        .eq("project_id", project_id) \
        .order("created_at", desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()

    return {
        "leads": res.data or [],
        "total": res.count or 0,
        "offset": offset,
        "limit": limit,
    }


class LeadTagsUpdate(BaseModel):
    # Was an unbounded list of unbounded strings, written into a text[] that
    # LeadsClient re-aggregates across every lead on each render.
    tags: list[str] = Field(default_factory=list, max_length=25)


@router.put("/leads/{lead_id}")
def update_lead_tags(lead_id: str, req: LeadTagsUpdate, user=Depends(verify_token)):
    if not _UUID_RE.match(lead_id or ""):
        raise HTTPException(status_code=400, detail="Invalid lead id")

    existing = supabase.table("leads").select("project_id").eq("id", lead_id).maybe_single().execute()
    lead = existing.data if existing else None
    if not lead:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_role(user.id, lead["project_id"])
    # Normalize: trim, lowercase, drop empties, dedupe — "VIP" and "vip" must
    # collapse into one tag, or campaign tag filtering silently misses people.
    clean_tags = sorted(set(t.strip().lower()[:40] for t in req.tags if t.strip()))
    supabase.table("leads").update({"tags": clean_tags}).eq("id", lead_id).execute()
    res = supabase.table("leads").select(_LEAD_COLUMNS).eq("id", lead_id).single().execute()
    return res.data


# -------------------------------------------------
# AUTO-SAVE CONTACT (called internally)
# -------------------------------------------------
def upsert_contact(project_id: str, phone: str, name: str = None, channel: str = "whatsapp"):
    """Auto-save or update a contact when they message."""
    try:
        existing = supabase.table("leads") \
            .select("id, name") \
            .eq("project_id", project_id) \
            .eq("phone", phone) \
            .limit(1) \
            .execute()

        if existing.data:
            # Update last_seen_at and name if we now have one
            update = {"last_seen_at": "now()"}
            if name and not existing.data[0].get("name"):
                update["name"] = name
            supabase.table("leads").update(update) \
                .eq("id", existing.data[0]["id"]).execute()
            return

        try:
            # Insert new contact
            supabase.table("leads").insert({
                "project_id": project_id,
                "phone": phone,
                "name": name or "",
                "email": "",
                "source": channel,
                "channel": channel,
                "whatsapp_number": phone if channel == "whatsapp" else None,
                "last_seen_at": "now()",
            }).execute()
        except Exception as insert_error:
            # This races in normal operation, not just under attack: Meta
            # retries webhooks, so two deliveries for a new number can both
            # pass the select above. The duplicate that used to result is not
            # cosmetic — chat.py's _get_known_customer_name reads this row with
            # .maybe_single(), which RAISES on multiple rows and takes down the
            # WhatsApp chat path for that customer.
            if not _is_duplicate_error(insert_error):
                raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"upsert_contact error: {e}")
