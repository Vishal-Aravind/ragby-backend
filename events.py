"""
Event Registration system — for webinars, expos, workshops, demos, walk-in promos.
Merchant creates an event, broadcasts a rich card via WhatsApp,
customer registers via a public page, gets WhatsApp confirmation.
"""
import re
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
from clients import supabase
from auth import verify_token, require_project_access
from config import WHATSAPP_TOKEN, FRONTEND_URL
from ratelimit import is_rate_limited, client_ip
from datetime import datetime, timedelta

router = APIRouter()

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

# Statuses the rest of this file actually branches on. Was a free string
# written straight to the row, and because the capacity count excludes only
# 'cancelled', an arbitrary value silently kept consuming a seat.
VALID_REGISTRATION_STATUSES = {"confirmed", "attended", "no_show", "cancelled"}

# Ceiling on confirmation messages per event per hour. Past it registrations
# still save; only the WhatsApp is skipped. This is the last line of defence
# on spend if every other guard fails — the send goes out on the merchant's
# number and their bill.
MAX_CONFIRMATIONS_PER_EVENT_HOUR = 100


def _require_uuid(value: str, label: str = "id") -> str:
    """A non-UUID used to reach Postgres and come back as a 500 carrying
    driver detail. Same helper shape as leads.py."""
    if not value or not _UUID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")
    return value


def _deadline_passed(deadline) -> bool:
    """True if this event's registration deadline is in the past.

    registration_deadline is a timestamptz, so PostgREST hands it back in
    UTC. Stripping the tzinfo makes it a naive UTC value, and it was then
    compared against datetime.now() — the server's LOCAL clock. That is
    only correct because the host happens to run UTC; anywhere else every
    deadline was wrong by the UTC offset, in the direction of accepting
    registrations after they should have closed.

    An unparseable deadline is treated as "not passed": refusing every
    registration because of one malformed field is the worse failure.
    """
    if not deadline:
        return False
    try:
        dl = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
    except Exception:
        return False
    if dl.tzinfo is not None:
        dl = dl.replace(tzinfo=None) - dl.utcoffset()
    return datetime.utcnow() > dl


def _valid_phone(raw) -> Optional[str]:
    """Normalise a registration phone, or None if it can't be one.

    Registration used to accept any non-empty string here and hand it
    straight to send_whatsapp_message, so an unauthenticated caller could
    drive messages to arbitrary numbers on the merchant's account. Same
    E.164 digit test campaigns.py already applies to its recipients.
    """
    if not isinstance(raw, str):
        return None
    digits = re.sub(r"\D", "", raw)
    if not (8 <= len(digits) <= 15):
        return None
    return digits


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class EventCreate(BaseModel):
    title: str
    description: Optional[str] = None
    banner_url: Optional[str] = None
    event_date: Optional[str] = None       # YYYY-MM-DD
    event_time: Optional[str] = None       # free text e.g. "10:00 AM - 4:00 PM"
    location: Optional[str] = None
    capacity: Optional[int] = None
    registration_deadline: Optional[str] = None
    contact_phone: Optional[str] = None
    accent_color: Optional[str] = "#6366f1"
    page_json: Optional[list] = None
    form_schema: Optional[list] = None

class EventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    banner_url: Optional[str] = None
    event_date: Optional[str] = None
    event_time: Optional[str] = None
    location: Optional[str] = None
    capacity: Optional[int] = None
    registration_deadline: Optional[str] = None
    contact_phone: Optional[str] = None
    accent_color: Optional[str] = None
    is_active: Optional[bool] = None
    page_json: Optional[list] = None
    form_schema: Optional[list] = None
    bot_can_register: Optional[bool] = None

class RegistrationCreate(BaseModel):
    event_id: str
    # project_id deliberately absent. The browser used to send one, and it
    # was used as part of the rate-limit key while being ignored everywhere
    # else — so varying it handed the caller a fresh bucket per request and
    # defeated the limit entirely. The real project is derived from the
    # event row in register_for_event_core. Pydantic ignores the extra
    # field, so an older page that still sends it keeps working.
    data: dict   # dynamic — keyed by form field id, e.g. {"name": "John", "phone": "919...", "dietary": "Veg"}


class RegistrationStatusUpdate(BaseModel):
    status: str = Field(..., min_length=1, max_length=32)


# -------------------------------------------------
# MERCHANT — EVENT CRUD
# -------------------------------------------------
# Every endpoint below used require_project_role, which passes for ANY
# role. "events" is a grantable tab and events/page.js gates it correctly,
# so an agent locked out of Registrations could still read and edit
# everything by calling the API directly.
@router.get("/events")
def list_events(project_id: str, user=Depends(verify_token)):
    _require_uuid(project_id, "project")
    require_project_access(user.id, project_id, tab="events")
    res = supabase.table("events") \
        .select(
            "id, project_id, title, description, banner_url, event_date, "
            "event_time, location, capacity, registration_deadline, "
            "contact_phone, accent_color, page_json, form_schema, "
            "is_active, bot_can_register, created_at"
        ) \
        .eq("project_id", project_id) \
        .order("created_at", desc=True) \
        .execute()

    events = res.data or []
    # One count query PER event was an N+1 that grew with the list. A
    # single grouped read covers every event on the page.
    if events:
        counts = {}
        reg_res = supabase.table("event_registrations") \
            .select("event_id") \
            .eq("project_id", project_id) \
            .neq("status", "cancelled") \
            .execute()
        for row in (reg_res.data or []):
            counts[row["event_id"]] = counts.get(row["event_id"], 0) + 1
        for e in events:
            e["registration_count"] = counts.get(e["id"], 0)

    return events


@router.post("/events")
def create_event(project_id: str, body: EventCreate, user=Depends(verify_token)):
    _require_uuid(project_id, "project")
    require_project_access(user.id, project_id, tab="events", min_role="admin")
    res = supabase.table("events").insert({
        "project_id": project_id,
        **body.dict(exclude_none=True),
    }).execute()
    return res.data[0]


def _require_role_for_event(user_id: str, event_id: str, min_role: Optional[str] = None):
    _require_uuid(event_id, "event")
    res = supabase.table("events").select("project_id").eq("id", event_id).maybe_single().execute()
    event = res.data if res else None
    if not event:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user_id, event["project_id"], tab="events", min_role=min_role)
    return event["project_id"]

def _require_role_for_registration(user_id: str, registration_id: str, min_role: Optional[str] = None):
    _require_uuid(registration_id, "registration")
    res = supabase.table("event_registrations").select("project_id").eq("id", registration_id).maybe_single().execute()
    registration = res.data if res else None
    if not registration:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user_id, registration["project_id"], tab="events", min_role=min_role)
    return registration["project_id"]

@router.put("/events/{event_id}")
def update_event(event_id: str, body: EventUpdate, user=Depends(verify_token)):
    _require_role_for_event(user.id, event_id, min_role="admin")
    # exclude_unset, not "drop every None": the page builder sends null to
    # CLEAR a field, and filtering nulls meant an emptied banner or
    # location silently kept its old value on the next reload.
    update = body.dict(exclude_unset=True)
    if not update:
        raise HTTPException(status_code=400, detail="Nothing to update.")
    supabase.table("events").update(update).eq("id", event_id).execute()
    res = supabase.table("events").select("*").eq("id", event_id).maybe_single().execute()
    if not res or not res.data:
        raise HTTPException(status_code=404, detail="Not found")
    return res.data


@router.delete("/events/{event_id}")
def delete_event(event_id: str, user=Depends(verify_token)):
    _require_role_for_event(user.id, event_id, min_role="admin")
    supabase.table("events").delete().eq("id", event_id).execute()
    return {"status": "deleted"}


@router.get("/events/{event_id}/registrations")
def list_registrations(event_id: str, user=Depends(verify_token)):
    _require_role_for_event(user.id, event_id)
    res = supabase.table("event_registrations") \
        .select("id, event_id, name, phone, email, notes, status, created_at") \
        .eq("event_id", event_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data or []


@router.put("/events/registrations/{registration_id}")
def update_registration(registration_id: str, body: RegistrationStatusUpdate, user=Depends(verify_token)):
    _require_role_for_registration(user.id, registration_id, min_role="admin")
    # Was a raw dict with a free-string status written straight to the row.
    # Because the capacity count excludes only 'cancelled', an arbitrary
    # value silently kept consuming a seat.
    status = body.status
    if status not in VALID_REGISTRATION_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(sorted(VALID_REGISTRATION_STATUSES))}.",
        )

    if status == "cancelled":
        try:
            return cancel_registration_core(registration_id, notify_customer=True)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    supabase.table("event_registrations").update({"status": status}).eq("id", registration_id).execute()
    res = supabase.table("event_registrations").select("*").eq("id", registration_id).maybe_single().execute()
    if not res or not res.data:
        raise HTTPException(status_code=404, detail="Not found")
    return res.data


# -------------------------------------------------
# AI-FACING CORE FUNCTIONS — reused by both the public routes below and
# backend/chat.py's in-chat registration tools (bot_can_register, opt-in
# per-event, off by default).
# -------------------------------------------------
def get_bot_registrable_events(project_id: str) -> list:
    """Cheap existence check — does this project have ANY event the
    merchant opted into bot registration for? Used purely to gate whether
    EVENT_TOOLS get added to active_tools at all in chat.py's run_chat."""
    res = supabase.table("events").select("id") \
        .eq("project_id", project_id) \
        .eq("is_active", True) \
        .eq("bot_can_register", True) \
        .limit(1).execute()
    return res.data or []


def get_upcoming_events_for_ai(project_id: str) -> list:
    """Read-only 'what's open for registration right now' — used by the
    in-chat browse tool and find_event_by_title. Recomputes the same
    capacity/deadline logic as public_event_details so chat and the public
    page can never disagree."""
    # Was select("*"), which pulled page_json, form_schema and contact_phone
    # into the chat layer. Only _shape_event_for_ai's whitelist kept them
    # from reaching the model, so any new caller forwarding this raw would
    # have leaked them.
    res = supabase.table("events") \
        .select("id, title, description, event_date, event_time, location, capacity, registration_deadline") \
        .eq("project_id", project_id) \
        .eq("is_active", True) \
        .eq("bot_can_register", True) \
        .execute()
    events = res.data or []

    # event_date is an IST wall-clock date — datetime.now() would read the
    # server's UTC clock, which can lag IST by up to a day near midnight.
    today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
    open_events = []
    for event in events:
        if event.get("event_date"):
            try:
                event_date = datetime.strptime(event["event_date"], "%Y-%m-%d").date()
                if event_date < today:
                    continue
            except Exception:
                pass

        deadline = event.get("registration_deadline")
        if _deadline_passed(deadline):
            continue

        spots_left = None
        if event.get("capacity"):
            count_res = supabase.table("event_registrations") \
                .select("id", count="exact") \
                .eq("event_id", event["id"]) \
                .neq("status", "cancelled") \
                .execute()
            registered = count_res.count or 0
            spots_left = max(0, event["capacity"] - registered)
            if spots_left <= 0:
                continue

        event["spots_left"] = spots_left
        open_events.append(event)

    return open_events


def find_event_by_title(project_id: str, title: str) -> list:
    """Case-insensitive exact-then-substring match against currently
    bot-registrable, open events — used by register_for_event and
    cancel_event_registration to resolve free text like 'the Chennai expo'.
    Always returns the full candidate list (0/1/many); the caller decides
    what to do with ambiguity."""
    events = get_upcoming_events_for_ai(project_id)
    query = (title or "").strip().lower()
    if not query:
        return events

    exact = [e for e in events if e["title"].strip().lower() == query]
    if exact:
        return exact

    return [e for e in events if query in e["title"].strip().lower() or e["title"].strip().lower() in query]


def get_registrations_for_phone(
    project_id: str, phone: str, limit: int = 20, bot_registrable_only: bool = False
) -> list:
    """Used by 'check my registrations' and 'cancel my registration' — the
    customer's own non-cancelled registrations across the project's events
    (a customer can be registered for more than one). Two-query approach
    since no tracked migration proves a real FK exists here.

    bot_registrable_only exists because the chat tools are switched on by a
    PROJECT-level check: one event with bot_can_register turned the tools on
    for every event in the project, so the bot could cancel registrations
    for events the merchant had deliberately left opted out. The chat path
    passes True; the dashboard has no such restriction.
    """
    # Normalised the same way writes are (_valid_phone strips to digits),
    # so a number stored via a different path still matches.
    clean_phone = _valid_phone(phone) or re.sub(r"\D", "", phone or "")
    if not clean_phone:
        return []

    reg_res = supabase.table("event_registrations") \
        .select("id, event_id, status, created_at") \
        .eq("project_id", project_id) \
        .eq("phone", clean_phone) \
        .neq("status", "cancelled") \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    registrations = reg_res.data or []
    if not registrations:
        return []

    # project_id is re-applied here on purpose. The ids came from rows that
    # were already project-scoped, but nothing in the DB enforces that
    # event_registrations.project_id agrees with events.project_id, and this
    # list is what the chat tool is allowed to cancel.
    event_ids = list({r["event_id"] for r in registrations})
    events_query = supabase.table("events") \
        .select("id, title, event_date, event_time, location") \
        .eq("project_id", project_id) \
        .in_("id", event_ids)
    if bot_registrable_only:
        events_query = events_query.eq("bot_can_register", True)
    events_res = events_query.execute()
    events_by_id = {e["id"]: e for e in (events_res.data or [])}

    today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
    result = []
    for r in registrations:
        event = events_by_id.get(r["event_id"])
        if not event:
            continue
        if event.get("event_date"):
            try:
                event_date = datetime.strptime(event["event_date"], "%Y-%m-%d").date()
                if event_date < today:
                    continue
            except Exception:
                pass
        result.append({
            "registration_id": r["id"],
            "event_id": event["id"],
            "event_title": event["title"],
            "event_date": event.get("event_date"),
            "event_time": event.get("event_time"),
            "location": event.get("location"),
            "status": r["status"],
        })
    return result


def register_for_event_core(event_id: str, data: dict) -> dict:
    """Core registration logic — shared by the public route and the
    in-chat tool. Re-validates is_active, registration_deadline, capacity,
    and duplicate-phone here — never trusts a stale snapshot (e.g. from an
    earlier browse call). Raises ValueError with a human-readable message
    on any failure.

    project_id is deliberately NOT a parameter — it's derived from the
    fetched event row below (FIX: a caller-supplied project_id used to be
    trusted directly for both the registration insert and the WhatsApp
    integration lookup, so an event_id from one project combined with an
    unrelated project_id let an attacker trigger a real WhatsApp
    confirmation — containing the victim event's real details — sent
    through a DIFFERENT tenant's connected WhatsApp number to any phone)."""
    # Scoped rather than select("*"): this is the unauthenticated
    # registration path, and these are the only fields it reads.
    event_res = supabase.table("events") \
        .select("id, project_id, title, is_active, capacity, registration_deadline, event_date, event_time, location") \
        .eq("id", event_id) \
        .maybe_single() \
        .execute()
    if not event_res or not event_res.data:
        raise ValueError("Event not found")

    event = event_res.data
    project_id = event["project_id"]
    if not event.get("is_active"):
        raise ValueError("Registration is closed for this event")

    if _deadline_passed(event.get("registration_deadline")):
        raise ValueError("The registration deadline for this event has passed")

    name = data.get("name", "")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Name and phone are required")
    name = name.strip()[:200]

    # Was `str(phone).replace("+","").replace(" ","")` with a non-empty
    # test — every value reached send_whatsapp_message below.
    phone = _valid_phone(data.get("phone"))
    if not phone:
        raise ValueError("Please enter a valid phone number with country code")

    email = data.get("email")
    email = email.strip()[:320] if isinstance(email, str) and email.strip() else None
    notes = data.get("notes")
    notes = notes.strip()[:2000] if isinstance(notes, str) and notes.strip() else None

    # Capacity and duplicate-phone were separate check-then-insert steps
    # with nothing atomic between them, so concurrent submits oversold the
    # event and created duplicate registrations — each duplicate firing its
    # own billable WhatsApp confirmation. The function locks the event row
    # before re-checking, the same shape as book_appointment_slot.
    try:
        rpc_res = supabase.rpc("register_for_event_atomic", {
            "p_event_id": event_id,
            "p_project_id": project_id,
            "p_name": name,
            "p_phone": phone,
            "p_email": email,
            "p_notes": notes,
        }).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"register_for_event_atomic failed: {type(e).__name__}")
        raise ValueError("We couldn't complete that registration. Please try again.")

    result = rpc_res.data if isinstance(rpc_res.data, dict) else (rpc_res.data or [None])[0]
    if not result:
        raise ValueError("We couldn't complete that registration. Please try again.")

    status = result.get("status")
    if status == "full":
        raise ValueError("This event is full")
    if status == "duplicate":
        raise ValueError("You are already registered for this event")
    if status != "ok" or not result.get("registration"):
        raise ValueError("Registration is closed for this event")

    registration = result["registration"]

    try:
        supabase.table("form_submissions").insert({
            "entity_type": "event",
            "entity_id": event_id,
            "project_id": project_id,
            "data": data,
        }).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"form_submissions insert error: {e}")

    # Hard ceiling on confirmations per event per hour. Everything above
    # bounds who can register; this bounds what registering can COST if one
    # of those guards is ever wrong. The registration is already saved and
    # visible in the dashboard — only the message is skipped.
    #
    # Note there is no in-app surface that surfaces this to the merchant
    # today, so the signal goes to Sentry.
    if is_rate_limited(
        f"event-confirm:{event_id}",
        limit=MAX_CONFIRMATIONS_PER_EVENT_HOUR,
        window_seconds=3600,
    ):
        sentry_sdk.capture_message(
            f"Event {event_id} hit the hourly WhatsApp confirmation cap "
            f"({MAX_CONFIRMATIONS_PER_EVENT_HOUR}/hr); registration saved without a message."
        )
        print(f"Event {event_id}: confirmation cap reached, skipping WhatsApp")
        return {
            "status": "confirmed",
            "registration_id": registration["id"],
            "event_title": event["title"],
            "event_date": event.get("event_date"),
            "event_time": event.get("event_time"),
            "location": event.get("location"),
            "confirmation_sent": False,
        }

    try:
        wa_res = supabase.table("whatsapp_integrations") \
            .select("phone_number_id, access_token") \
            .eq("project_id", project_id) \
            .maybe_single() \
            .execute()
        wa_data = (wa_res.data if wa_res else None)
        if wa_data:
            from whatsapp import send_whatsapp_message
            phone_number_id = wa_data["phone_number_id"]
            token = wa_data.get("access_token") or WHATSAPP_TOKEN

            date_str = ""
            if event.get("event_date"):
                try:
                    date_obj = datetime.strptime(event["event_date"], "%Y-%m-%d")
                    date_str = date_obj.strftime("%d %B %Y")
                except Exception:
                    date_str = event["event_date"]

            msg = f"✅ *Registration Confirmed!*\n\n"
            msg += f"📋 {event['title']}\n"
            if date_str:
                msg += f"📅 {date_str}"
                if event.get("event_time"):
                    msg += f", {event['event_time']}"
                msg += "\n"
            if event.get("location"):
                msg += f"📍 {event['location']}\n"
            msg += f"\n👤 {name}\n"
            msg += f"\nBooking ID: #{registration['id'][:8].upper()}\n\nSee you there!"

            send_whatsapp_message(to=phone, text=msg, phone_number_id=phone_number_id, token=token)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Event registration WhatsApp confirmation error: {e}")

    return {
        "status": "confirmed",
        "registration_id": registration["id"],
        "event_title": event["title"],
        "event_date": event.get("event_date"),
        "event_time": event.get("event_time"),
        "location": event.get("location"),
    }


def cancel_registration_core(registration_id: str, notify_customer: bool = True) -> dict:
    """Core cancel logic — used by the dashboard PUT route AND the in-chat
    cancel tool. notify_customer=False when the customer cancels it
    themselves in the same conversation — they don't need a separate
    notification for something they just did.

    CALLER CONTRACT: this performs NO authorization of its own. It takes a
    bare registration_id and acts on it. Both current callers pre-authorize
    — the dashboard route via _require_role_for_registration, the chat tool
    by only ever passing an id drawn from the verified sender's own
    registrations — and any new caller must do the same, or it is an IDOR.
    """
    reg_res = supabase.table("event_registrations").select("*").eq("id", registration_id).maybe_single().execute()
    if not reg_res or not reg_res.data:
        raise ValueError("Registration not found")
    registration = reg_res.data

    # Idempotent. Cancelling twice used to send a second billable WhatsApp
    # for a registration that was already cancelled.
    if registration.get("status") == "cancelled":
        return registration

    if notify_customer:
        try:
            event_res = supabase.table("events").select("title").eq("id", registration["event_id"]).maybe_single().execute()
            event = (event_res.data if event_res else None)
            wa_res = supabase.table("whatsapp_integrations") \
                .select("phone_number_id, access_token") \
                .eq("project_id", registration["project_id"]) \
                .maybe_single() \
                .execute()
            wa_data = (wa_res.data if wa_res else None)
            if event and wa_data:
                from whatsapp import send_whatsapp_message
                send_whatsapp_message(
                    to=registration["phone"],
                    text=f"❌ *Registration Cancelled*\n\nYour registration for {event['title']} has been cancelled.",
                    phone_number_id=wa_data["phone_number_id"],
                    token=wa_data.get("access_token") or WHATSAPP_TOKEN,
                )
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Registration cancel notification error: {e}")

    supabase.table("event_registrations").update({"status": "cancelled"}).eq("id", registration_id).execute()
    res = supabase.table("event_registrations").select("*").eq("id", registration_id).maybe_single().execute()
    if not res or not res.data:
        raise ValueError("Registration not found")
    return res.data


# -------------------------------------------------
# PUBLIC — Registration page APIs
# -------------------------------------------------
@router.get("/public/events/{event_id}")
def public_event_details(event_id: str, request: Request):
    _require_uuid(event_id, "event")

    if is_rate_limited(f"event-view:{event_id}:{client_ip(request)}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again in a moment.")

    # Was select("*"), which handed an anonymous visitor the whole row —
    # project_id, bot_can_register, contact_phone and form_schema included.
    # form_schema IS needed (it renders the form); the rest are not.
    res = supabase.table("events") \
        .select(
            "id, title, description, banner_url, event_date, event_time, "
            "location, capacity, registration_deadline, accent_color, "
            "page_json, form_schema, is_active"
        ) \
        .eq("id", event_id) \
        .maybe_single() \
        .execute()
    if not res or not res.data:
        raise HTTPException(status_code=404, detail="Event not found")

    event = res.data
    if not event.get("is_active"):
        raise HTTPException(status_code=403, detail="Registration closed")

    # Check registration deadline
    event["registration_open"] = not _deadline_passed(event.get("registration_deadline"))

    # Check capacity
    if event.get("capacity"):
        count_res = supabase.table("event_registrations") \
            .select("id", count="exact") \
            .eq("event_id", event_id) \
            .neq("status", "cancelled") \
            .execute()
        registered = count_res.count or 0
        event["registered_count"] = registered
        event["spots_left"] = max(0, event["capacity"] - registered)
        if event["spots_left"] <= 0:
            event["registration_open"] = False
    else:
        event["registered_count"] = None
        event["spots_left"] = None

    return event


@router.post("/public/events/register")
def register_for_event(body: RegistrationCreate, request: Request):
    _require_uuid(body.event_id, "event")

    # Every key here is server-derived. The old key mixed in a
    # browser-supplied project_id that nothing else used, so rotating that
    # one field gave the caller an unlimited supply of fresh buckets — and
    # each accepted request sent a WhatsApp to a phone number of their
    # choosing, billed to the merchant.
    ip = client_ip(request)
    if is_rate_limited(f"register-ip:{ip}", limit=10, window_seconds=300):
        raise HTTPException(status_code=429, detail="Too many attempts — please wait a moment and try again.")

    # An event id can't be forged; it has to name a real, open event.
    if is_rate_limited(f"register-event:{body.event_id}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="This event is receiving a lot of registrations right now. Please try again in a moment.")

    # Stops one number being cycled across every open event in a loop.
    # Checked before the handler so a rejected phone costs nothing.
    phone_key = _valid_phone((body.data or {}).get("phone"))
    if phone_key and is_rate_limited(f"register-phone:{phone_key}", limit=5, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Too many registrations from this number. Please try again later.")

    try:
        return register_for_event_core(body.event_id, body.data or {})
    except ValueError as e:
        msg = str(e)
        if msg == "Event not found":
            raise HTTPException(status_code=404, detail=msg)
        if msg == "Registration is closed for this event":
            raise HTTPException(status_code=403, detail=msg)
        raise HTTPException(status_code=400, detail=msg)