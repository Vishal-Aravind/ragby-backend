"""
Event Registration system — for webinars, expos, workshops, demos, walk-in promos.
Merchant creates an event, broadcasts a rich card via WhatsApp,
customer registers via a public page, gets WhatsApp confirmation.
"""
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from clients import supabase
from auth import verify_token, require_project_role
from config import WHATSAPP_TOKEN, FRONTEND_URL
from ratelimit import is_rate_limited
from datetime import datetime, timedelta

router = APIRouter()


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
    project_id: str
    data: dict   # dynamic — keyed by form field id, e.g. {"name": "John", "phone": "919...", "dietary": "Veg"}


# -------------------------------------------------
# MERCHANT — EVENT CRUD
# -------------------------------------------------
@router.get("/events")
def list_events(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("events") \
        .select("*") \
        .eq("project_id", project_id) \
        .order("created_at", desc=True) \
        .execute()

    events = res.data or []
    # Attach registration count to each event
    for e in events:
        count_res = supabase.table("event_registrations") \
            .select("id", count="exact") \
            .eq("event_id", e["id"]) \
            .neq("status", "cancelled") \
            .execute()
        e["registration_count"] = count_res.count or 0

    return events


@router.post("/events")
def create_event(project_id: str, body: EventCreate, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("events").insert({
        "project_id": project_id,
        **body.dict(exclude_none=True),
    }).execute()
    return res.data[0]


def _require_role_for_event(user_id: str, event_id: str):
    res = supabase.table("events").select("project_id").eq("id", event_id).maybe_single().execute()
    event = res.data if res else None
    if not event:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_role(user_id, event["project_id"])

def _require_role_for_registration(user_id: str, registration_id: str):
    res = supabase.table("event_registrations").select("project_id").eq("id", registration_id).maybe_single().execute()
    registration = res.data if res else None
    if not registration:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_role(user_id, registration["project_id"])

@router.put("/events/{event_id}")
def update_event(event_id: str, body: EventUpdate, user=Depends(verify_token)):
    _require_role_for_event(user.id, event_id)
    update = {k: v for k, v in body.dict().items() if v is not None}
    supabase.table("events").update(update).eq("id", event_id).execute()
    res = supabase.table("events").select("*").eq("id", event_id).single().execute()
    return res.data


@router.delete("/events/{event_id}")
def delete_event(event_id: str, user=Depends(verify_token)):
    _require_role_for_event(user.id, event_id)
    supabase.table("events").delete().eq("id", event_id).execute()
    return {"status": "deleted"}


@router.get("/events/{event_id}/registrations")
def list_registrations(event_id: str, user=Depends(verify_token)):
    _require_role_for_event(user.id, event_id)
    res = supabase.table("event_registrations") \
        .select("*") \
        .eq("event_id", event_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data or []


@router.put("/events/registrations/{registration_id}")
def update_registration(registration_id: str, data: dict, user=Depends(verify_token)):
    _require_role_for_registration(user.id, registration_id)
    status = data.get("status")
    if status == "cancelled":
        try:
            return cancel_registration_core(registration_id, notify_customer=True)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    if status:
        supabase.table("event_registrations").update({"status": status}).eq("id", registration_id).execute()
    res = supabase.table("event_registrations").select("*").eq("id", registration_id).single().execute()
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
    res = supabase.table("events").select("*") \
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
        if deadline:
            try:
                dl = datetime.fromisoformat(deadline.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                dl = None
            if dl and datetime.now() > dl:
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


def get_registrations_for_phone(project_id: str, phone: str, limit: int = 20) -> list:
    """Used by 'check my registrations' and 'cancel my registration' — the
    customer's own non-cancelled registrations across ALL of the project's
    events (a customer can be registered for more than one). Two-query
    approach since no tracked migration proves a real FK exists here."""
    clean_phone = phone.replace("+", "").replace(" ", "")
    reg_res = supabase.table("event_registrations") \
        .select("*") \
        .eq("project_id", project_id) \
        .eq("phone", clean_phone) \
        .neq("status", "cancelled") \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    registrations = reg_res.data or []
    if not registrations:
        return []

    event_ids = list({r["event_id"] for r in registrations})
    events_res = supabase.table("events").select("id, title, event_date, event_time, location").in_("id", event_ids).execute()
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
    event_res = supabase.table("events").select("*").eq("id", event_id).maybe_single().execute()
    if not event_res or not event_res.data:
        raise ValueError("Event not found")

    event = event_res.data
    project_id = event["project_id"]
    if not event.get("is_active"):
        raise ValueError("Registration is closed for this event")

    deadline = event.get("registration_deadline")
    if deadline:
        try:
            dl = datetime.fromisoformat(deadline.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            dl = None
        if dl and datetime.now() > dl:
            raise ValueError("The registration deadline for this event has passed")

    name = data.get("name", "")
    phone = str(data.get("phone", "")).replace("+", "").replace(" ", "")
    if not name or not phone:
        raise ValueError("Name and phone are required")

    if event.get("capacity"):
        count_res = supabase.table("event_registrations") \
            .select("id", count="exact") \
            .eq("event_id", event_id) \
            .neq("status", "cancelled") \
            .execute()
        if (count_res.count or 0) >= event["capacity"]:
            raise ValueError("This event is full")

    existing = supabase.table("event_registrations") \
        .select("id") \
        .eq("event_id", event_id) \
        .eq("phone", phone) \
        .neq("status", "cancelled") \
        .maybe_single() \
        .execute()
    if existing and existing.data:
        raise ValueError("You are already registered for this event")

    reg_res = supabase.table("event_registrations").insert({
        "event_id": event_id,
        "project_id": project_id,
        "name": name,
        "phone": phone,
        "email": data.get("email"),
        "notes": data.get("notes"),
    }).execute()
    registration = reg_res.data[0]

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

    try:
        wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", project_id).maybe_single().execute()
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
    notification for something they just did."""
    reg_res = supabase.table("event_registrations").select("*").eq("id", registration_id).maybe_single().execute()
    if not reg_res or not reg_res.data:
        raise ValueError("Registration not found")
    registration = reg_res.data

    if notify_customer:
        try:
            event_res = supabase.table("events").select("*").eq("id", registration["event_id"]).maybe_single().execute()
            event = (event_res.data if event_res else None)
            wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", registration["project_id"]).maybe_single().execute()
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
    res = supabase.table("event_registrations").select("*").eq("id", registration_id).single().execute()
    return res.data


# -------------------------------------------------
# PUBLIC — Registration page APIs
# -------------------------------------------------
@router.get("/public/events/{event_id}")
def public_event_details(event_id: str):
    res = supabase.table("events").select("*").eq("id", event_id).maybe_single().execute()
    if not res or not res.data:
        raise HTTPException(status_code=404, detail="Event not found")

    event = res.data
    if not event.get("is_active"):
        raise HTTPException(status_code=403, detail="Registration closed")

    # Check registration deadline
    deadline = event.get("registration_deadline")
    if deadline:
        try:
            dl = datetime.fromisoformat(deadline.replace("Z", "+00:00")).replace(tzinfo=None)
            if datetime.now() > dl:
                event["registration_open"] = False
            else:
                event["registration_open"] = True
        except Exception:
            event["registration_open"] = True
    else:
        event["registration_open"] = True

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
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"register:{body.project_id}:{ip}", limit=5):
        raise HTTPException(status_code=429, detail="Too many attempts — please wait a moment and try again.")
    try:
        return register_for_event_core(body.event_id, body.data or {})
    except ValueError as e:
        msg = str(e)
        if msg == "Event not found":
            raise HTTPException(status_code=404, detail=msg)
        if msg == "Registration is closed for this event":
            raise HTTPException(status_code=403, detail=msg)
        raise HTTPException(status_code=400, detail=msg)