"""
Appointments system — Calendly-style booking via WhatsApp bot.
Supports Google Calendar integration for availability.
"""
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
from clients import supabase
from auth import verify_token, require_project_role
from config import WHATSAPP_TOKEN, FRONTEND_URL
from ratelimit import is_rate_limited
import os
import requests
from datetime import datetime, date, timedelta, time
import json

router = APIRouter()

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", f"{os.getenv('BACKEND_URL', 'https://ragby-backend.onrender.com')}/appointments/google/callback")

# How long a 'hold_to_confirm' booking provisionally reserves its slot
# before the background sweep (see release_expired_holds / main.py's
# scheduler) releases it back to availability if unpaid.
HOLD_MINUTES = 15


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class AppointmentSettingsUpdate(BaseModel):
    # service_name/duration_minutes moved to appointment_services (see
    # ServiceCreate/ServiceUpdate below) — no longer editable here.
    buffer_minutes: Optional[int] = None
    slot_capacity: Optional[int] = None
    working_hours: Optional[dict] = None
    advance_booking_days: Optional[int] = None
    reminder_hours: Optional[int] = None
    google_calendar_id: Optional[str] = None
    accent_color: Optional[str] = None
    currency_code: Optional[str] = None
    is_enabled: Optional[bool] = None
    bot_can_book: Optional[bool] = None

class ServiceCreate(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    duration_minutes: int = 30
    is_active: bool = True
    sort_order: int = 0

class ServiceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    duration_minutes: Optional[int] = None
    is_active: Optional[bool] = None
    sort_order: Optional[int] = None

class BookingCreate(BaseModel):
    project_id: str
    # Optional only because a reschedule always inherits the original
    # appointment's service — see create_appointment(), which resolves it
    # authoritatively server-side rather than trusting whatever (if
    # anything) the caller sends when reschedule_id is set. Required for a
    # brand-new booking.
    service_id: Optional[str] = None
    customer_name: str
    customer_phone: str
    appointment_date: str  # YYYY-MM-DD
    start_time: str        # HH:MM
    notes: Optional[str] = None
    reschedule_id: Optional[str] = None  # old appointment ID being rescheduled

class AppointmentStatusUpdate(BaseModel):
    status: str  # confirmed, cancelled, rescheduled, completed


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_google_access_token(refresh_token: str) -> Optional[str]:
    """Exchange refresh token for access token."""
    res = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    })
    if res.ok:
        return res.json().get("access_token")
    print(f"Google token refresh error: {res.text}")
    return None


def get_busy_slots(access_token: str, calendar_id: str, date_str: str) -> List[dict]:
    """Get busy time slots from Google Calendar for a specific date."""
    start = f"{date_str}T00:00:00Z"
    end   = f"{date_str}T23:59:59Z"

    res = requests.post(
        "https://www.googleapis.com/calendar/v3/freeBusy",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "timeMin": start,
            "timeMax": end,
            "items": [{"id": calendar_id}],
        }
    )
    if res.ok:
        calendars = res.json().get("calendars", {})
        busy = calendars.get(calendar_id, {}).get("busy", [])
        return busy
    print(f"Google freeBusy error: {res.text}")
    return []


def create_google_event(access_token: str, calendar_id: str, event: dict) -> Optional[str]:
    """Create a Google Calendar event and return event ID."""
    res = requests.post(
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=event,
    )
    if res.ok:
        return res.json().get("id")
    print(f"Google create event error: {res.text}")
    return None


def delete_google_event(access_token: str, calendar_id: str, event_id: str):
    """Delete a Google Calendar event."""
    requests.delete(
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )


def generate_slots(date_str: str, settings: dict, service: dict, busy_slots: List[dict]) -> List[str]:
    """Generate available time slots for a given date, for a specific
    service (duration comes from `service`, everything else — working
    hours, buffer, capacity — comes from the project-level `settings`).

    Booking-overlap check is interval-based, not exact-start-time-match:
    once different services can have different durations, two bookings can
    overlap in time without ever sharing a start time (e.g. a 60-min
    booking at 10:00 and a 30-min booking at 10:15 overlap 10:15–10:30 but
    never match on start_time). An exact-match count would silently allow
    double-booking across services sharing the same calendar/capacity —
    this checks real interval overlap against every one of the project's
    existing bookings, not just ones for this same service."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_map = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}
    day_key = day_map[dt.weekday()]

    working_hours = settings.get("working_hours", {})
    day_config = working_hours.get(day_key, {})

    if not day_config.get("enabled", False):
        return []

    start_str = day_config.get("start", "09:00")
    end_str   = day_config.get("end", "18:00")
    duration  = service.get("duration_minutes", 30)
    buffer    = settings.get("buffer_minutes", 0)

    # Parse start/end times
    start_h, start_m = map(int, start_str.split(":"))
    end_h, end_m     = map(int, end_str.split(":"))

    start_dt = datetime(dt.year, dt.month, dt.day, start_h, start_m)
    end_dt   = datetime(dt.year, dt.month, dt.day, end_h, end_m)

    # Parse Google Calendar busy slots (genuinely UTC) into datetime ranges
    busy_ranges = []
    for b in busy_slots:
        b_start = datetime.fromisoformat(b["start"].replace("Z", "+00:00")).replace(tzinfo=None)
        b_end   = datetime.fromisoformat(b["end"].replace("Z", "+00:00")).replace(tzinfo=None)
        # Adjust for IST (UTC+5:30) if needed — approximate, since no timezone handling in MVP
        b_start = b_start + timedelta(hours=5, minutes=30)
        b_end   = b_end   + timedelta(hours=5, minutes=30)
        busy_ranges.append((b_start, b_end))

    slot_capacity = settings.get("slot_capacity", 1)

    # Existing bookings for this date, across ALL services of the project
    # (they share the same provider/calendar/capacity) — start_time/end_time
    # here are already IST wall-clock strings, so no UTC shift needed,
    # unlike the Google busy_slots above. A 'pending_payment' hold blocks
    # the slot too, but only while it's still live — an expired-but-not-
    # yet-swept hold (see release_expired_holds) must NOT keep blocking it,
    # hence the extra hold_expires_at condition alongside the status check.
    now_iso = datetime.utcnow().isoformat() + "Z"
    existing_bookings_res = supabase.table("appointments") \
        .select("start_time, end_time") \
        .eq("project_id", settings.get("project_id", "")) \
        .eq("appointment_date", date_str) \
        .or_(f"status.in.(confirmed,rescheduled),and(status.eq.pending_payment,hold_expires_at.gt.{now_iso})") \
        .execute()

    booking_ranges = []
    for b in (existing_bookings_res.data or []):
        b_start_h, b_start_m = map(int, str(b["start_time"])[:5].split(":"))
        b_end_h, b_end_m = map(int, str(b["end_time"])[:5].split(":"))
        booking_ranges.append((
            datetime(dt.year, dt.month, dt.day, b_start_h, b_start_m),
            datetime(dt.year, dt.month, dt.day, b_end_h, b_end_m),
        ))

    # Generate slots
    slots = []
    current = start_dt
    # datetime.now() reads the SERVER's clock, which runs on UTC — 5.5
    # hours behind India time. That silently let already-passed times
    # today (e.g. 5:45 PM asked for at 11 PM IST) look "still upcoming"
    # to this filter. Match the IST conversion already used elsewhere
    # (see chat.py's date-grounding) so "past" is judged correctly.
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)

    while current + timedelta(minutes=duration) <= end_dt:
        slot_end = current + timedelta(minutes=duration)
        slot_str = current.strftime("%H:%M")

        # Skip past slots
        if current <= now:
            current += timedelta(minutes=duration + buffer)
            continue

        # Interval-overlap capacity check — how many existing bookings
        # (any service) overlap this candidate slot at all, not just ones
        # starting at exactly this time.
        overlap_count = sum(
            1 for (b_start, b_end) in booking_ranges
            if not (slot_end <= b_start or current >= b_end)
        )
        if overlap_count >= slot_capacity:
            current += timedelta(minutes=duration + buffer)
            continue

        is_busy = False
        for b_start, b_end in busy_ranges:
            if not (slot_end <= b_start or current >= b_end):
                is_busy = True
                break

        if not is_busy:
            remaining = slot_capacity - overlap_count
            slot_label = slot_str if slot_capacity == 1 else f"{slot_str} ({remaining} left)"
            slots.append(slot_label)

        current += timedelta(minutes=duration + buffer)

    return slots


# -------------------------------------------------
# GOOGLE OAUTH
# -------------------------------------------------
@router.get("/appointments/google/auth/{project_id}")
def google_auth(project_id: str, user=Depends(verify_token)):
    """Start Google OAuth flow for calendar access."""
    require_project_role(user.id, project_id)
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Google OAuth not configured. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to env vars.")

    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        "&scope=https://www.googleapis.com/auth/calendar"
        "&access_type=offline"
        "&prompt=consent"
        f"&state={project_id}"
    )
    return {"auth_url": auth_url}


@router.get("/appointments/google/callback")
def google_callback(code: str, state: str):
    """Handle Google OAuth callback — exchange code for tokens."""
    project_id = state

    res = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    })

    # FIX: FastAPI's default response_class is JSONResponse — a bare string
    # return gets json.dumps()'d (wrapped in quotes, Content-Type
    # application/json), so the <script> below never actually executed in
    # the popup and window.opener.postMessage never fired. The token was
    # still saved correctly (that part never depended on this), but the
    # frontend's "Connecting..." spinner would hang until a manual refresh.
    # Wrapping every return in HTMLResponse() makes the popup really close
    # and notify its opener as intended.
    if not res.ok:
        return HTMLResponse(f"<html><body><script>window.opener.postMessage({{type:'GOOGLE_AUTH',event:'ERROR',error:'{res.text}'}}, '*'); window.close();</script></body></html>")

    token_data = res.json()
    refresh_token = token_data.get("refresh_token")

    if not refresh_token:
        return HTMLResponse("<html><body><script>window.opener.postMessage({type:'GOOGLE_AUTH',event:'ERROR',error:'No refresh token returned. Try disconnecting and reconnecting.'}, '*'); window.close();</script></body></html>")

    # Save refresh token
    existing = supabase.table("appointment_settings").select("id").eq("project_id", project_id).maybe_single().execute()
    if existing and existing.data:
        supabase.table("appointment_settings").update({
            "google_refresh_token": refresh_token,
        }).eq("project_id", project_id).execute()
    else:
        supabase.table("appointment_settings").insert({
            "project_id": project_id,
            "google_refresh_token": refresh_token,
        }).execute()

    return HTMLResponse("<html><body><script>window.opener.postMessage({type:'GOOGLE_AUTH',event:'FINISH'}, '*'); window.close();</script></body></html>")


@router.delete("/appointments/google/disconnect/{project_id}")
def google_disconnect(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    supabase.table("appointment_settings").update({
        "google_refresh_token": None,
        "google_calendar_id": "primary",
    }).eq("project_id", project_id).execute()
    return {"success": True}


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
@router.get("/appointment-settings/{project_id}")
def get_settings(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        return {
            "project_id": project_id,
            "service_name": "Appointment",
            "duration_minutes": 30,
            "buffer_minutes": 0,
            "working_hours": {
                "mon": {"start": "09:00", "end": "18:00", "enabled": True},
                "tue": {"start": "09:00", "end": "18:00", "enabled": True},
                "wed": {"start": "09:00", "end": "18:00", "enabled": True},
                "thu": {"start": "09:00", "end": "18:00", "enabled": True},
                "fri": {"start": "09:00", "end": "18:00", "enabled": True},
                "sat": {"start": "09:00", "end": "14:00", "enabled": False},
                "sun": {"start": "09:00", "end": "14:00", "enabled": False},
            },
            "advance_booking_days": 30,
            "reminder_hours": 24,
            "slot_capacity": 1,
            "google_refresh_token": None,
            "google_calendar_id": "primary",
            "accent_color": "#6366f1",
            "is_enabled": False,
            "google_connected": False,
        }
    data = res.data
    data["google_connected"] = bool(data.get("google_refresh_token"))
    data.pop("google_refresh_token", None)  # never expose token to frontend
    return data


@router.put("/appointment-settings/{project_id}")
def update_settings(project_id: str, body: AppointmentSettingsUpdate, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    update = {k: v for k, v in body.dict().items() if v is not None}
    existing = supabase.table("appointment_settings").select("id").eq("project_id", project_id).maybe_single().execute()
    if existing and existing.data:
        supabase.table("appointment_settings").update(update).eq("project_id", project_id).execute()
    else:
        supabase.table("appointment_settings").insert({"project_id": project_id, **update}).execute()
    res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).single().execute()
    data = res.data
    data["google_connected"] = bool(data.get("google_refresh_token"))
    data.pop("google_refresh_token", None)
    return data


# -------------------------------------------------
# PUBLIC — Booking page APIs
# -------------------------------------------------
@router.get("/public/appointments/{project_id}/settings")
def public_settings(project_id: str, request: Request):
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"appt-settings:{project_id}:{ip}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    # FIX: previously also selected google_refresh_token into memory (then
    # popped it before returning) — this is a public, unauthenticated route,
    # so don't pull a merchant's Calendar token into scope here at all.
    res = supabase.table("appointment_settings").select(
        "working_hours,advance_booking_days,accent_color,currency_code,is_enabled,google_calendar_id"
    ).eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        raise HTTPException(status_code=404, detail="Booking not configured")
    data = res.data
    if not data.get("is_enabled"):
        raise HTTPException(status_code=403, detail="Booking not enabled")
    return data


@router.get("/public/appointments/{project_id}/services")
def public_services(project_id: str, request: Request):
    """Active services a customer can pick from on the public booking page
    — no auth, matches the rest of the /public/* surface."""
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"appt-services:{project_id}:{ip}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    res = supabase.table("appointment_services").select(
        "id,name,description,duration_minutes,price,payment_mode"
    ).eq("project_id", project_id).eq("is_active", True).order("sort_order", desc=False).execute()
    return res.data or []


@router.get("/public/appointments/{project_id}/booking-status/{appointment_id}")
def public_booking_status(project_id: str, appointment_id: str, request: Request):
    """Polled by the public booking page while a 'hold_to_confirm' payment
    is in flight — the customer pays on Razorpay's hosted page in a new
    tab, this is how the original tab finds out payment landed (mirrors
    the Shopify storefront widget's cart-status refocus-poll). Rate-limited
    generously (it's legitimately polled every few seconds by one honest
    client) — just enough to stop it being an open, unbounded polling
    target, matching every other /public/* write/lookup endpoint in this
    file."""
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"booking-status:{project_id}:{ip}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    res = supabase.table("appointments").select("status, payment_status") \
        .eq("id", appointment_id).eq("project_id", project_id).maybe_single().execute()
    data = res.data if res else None
    if not data:
        raise HTTPException(status_code=404, detail="Not found")
    return data


@router.get("/public/appointments/{project_id}/reschedule/{appointment_id}")
def public_reschedule_context(project_id: str, appointment_id: str, request: Request):
    """Used only by the reschedule flow on the public booking page to learn
    which service the original appointment was for, so it can skip service
    selection and fetch slots with the right duration — changing service on
    reschedule isn't supported (see create_appointment's docstring)."""
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"appt-reschedule-ctx:{project_id}:{ip}", limit=30, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    res = supabase.table("appointments").select("service_id, service_name") \
        .eq("id", appointment_id).eq("project_id", project_id).maybe_single().execute()
    appt = res.data if res else None
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {"service_id": appt.get("service_id"), "service_name": appt.get("service_name")}


def get_service(project_id: str, service_id: str) -> Optional[dict]:
    res = supabase.table("appointment_services").select("*").eq("id", service_id).eq("project_id", project_id).maybe_single().execute()
    service = res.data if res else None
    if not service or not service.get("is_active", True):
        return None
    return service


def find_service_by_name(project_id: str, name: str) -> Optional[dict]:
    """Case-insensitive, substring-tolerant match against a project's real
    active services — mirrors shop.py's find_product_by_name. Used by the
    in-chat booking tools, which only ever see a service *name* from the
    model, never an internal id. Returns None if there's no match OR more
    than one equally-good match (ambiguous — caller should ask the customer
    to clarify)."""
    res = supabase.table("appointment_services").select("*").eq("project_id", project_id).eq("is_active", True).execute()
    services = res.data or []
    name_lower = name.strip().lower()

    exact = [s for s in services if s["name"].strip().lower() == name_lower]
    if len(exact) == 1:
        return exact[0]

    partial = [s for s in services if name_lower in s["name"].lower() or s["name"].lower() in name_lower]
    if len(partial) == 1:
        return partial[0]

    return None


def get_available_slots(project_id: str, date: str, service_id: str) -> list:
    """
    Core slot-lookup logic — used by the public booking page AND by the
    in-chat booking tool (backend/chat.py). Single source of truth so both
    paths can never disagree about what's actually free.
    Raises ValueError on a bad date, missing settings, or missing/inactive
    service.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD")

    res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        raise ValueError("Appointment settings not found for this project")
    settings = res.data

    service = get_service(project_id, service_id)
    if not service:
        raise ValueError("Service not found")

    refresh_token = settings.get("google_refresh_token")
    calendar_id   = settings.get("google_calendar_id", "primary")

    # Google Calendar busy times only. FIX: this used to also fold the
    # project's own appointments into this same list as fake UTC timestamps
    # (f"{date}T{start_time}Z"), which then got shifted +5:30 *again* inside
    # generate_slots — double-converting an already-IST wall-clock time as
    # if it were UTC. Harmless before now only because the separate
    # exact-start-time count also (correctly) blocked the real slot anyway;
    # our own bookings are now checked directly and correctly inside
    # generate_slots() via unshifted-IST interval overlap, so they no
    # longer need to go through this UTC-oriented busy_slots path at all.
    busy_slots = []
    if refresh_token:
        access_token = get_google_access_token(refresh_token)
        if access_token:
            busy_slots = get_busy_slots(access_token, calendar_id, date)

    settings["project_id"] = project_id
    return generate_slots(date, settings, service, busy_slots)


def generate_appointment_payment_link(appointment: dict, service: dict, settings: dict) -> Optional[str]:
    """Razorpay Payment Link for a paid appointment. OAuth-only — unlike
    shop.py's generate_razorpay_link, there's no legacy manual-key fallback
    here, since appointments never had a manual-key UI; if the project
    hasn't connected Razorpay yet this just returns None and the
    appointment stays payment_status='unpaid' (booking still succeeds for
    request_after mode; for hold_to_confirm the slot will simply expire
    unpaid via the hold sweep — see the merchant-facing "connect Razorpay"
    prompt on the payment_mode selector for why that's an acceptable
    failure mode rather than something to special-case further here)."""
    import time as time_module
    from razorpay_oauth import _razorpay_api_request

    price = service.get("price") or 0
    if price <= 0:
        return None

    payload = {
        "amount": int(round(price * 100)),
        "currency": settings.get("currency_code") or "INR",
        "description": f"{service.get('name', 'Appointment')} — Booking #{appointment['id'][:8].upper()}",
        "customer": {"contact": f"+{appointment['customer_phone']}"},
        "notify": {"sms": False, "email": False},
        "reminder_enable": False,
        "expire_by": int(time_module.time()) + 5400,
    }

    try:
        res = _razorpay_api_request("POST", "/payment_links", appointment["project_id"], json=payload)
    except ValueError:
        return None  # project hasn't connected Razorpay yet
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return None

    if res.status_code >= 300:
        sentry_sdk.capture_message(f"Razorpay payment_link.create failed ({res.status_code}) for appointment {appointment['id']}: {res.text[:300]}")
        return None

    link = res.json()
    supabase.table("appointments").update({
        "payment_id": link["id"],
        "payment_status": "link_sent",
    }).eq("id", appointment["id"]).execute()
    return link["short_url"]


def create_appointment(
    project_id: str,
    service_id: str,
    customer_name: str,
    customer_phone: str,
    appointment_date: str,
    start_time: str,
    notes: str = None,
    reschedule_id: str = None,
) -> dict:
    """
    Core booking logic — used by the public booking page AND by the in-chat
    booking tool (backend/chat.py). Callers are responsible for confirming
    the slot with the customer BEFORE calling this — this function books
    unconditionally once called. Raises ValueError if settings/service are
    missing or the slot is no longer free (re-checked here, not trusted
    from the caller, since an AI-proposed slot could be stale by the time
    it's used).

    Changing service on reschedule is out of scope — if reschedule_id is
    set, the ORIGINAL appointment's service is used regardless of what
    service_id was passed in (present "change service" as cancel + rebook
    instead). This is resolved authoritatively here, not trusted from the
    caller, same "never trust the caller" philosophy as the slot re-check.

    Payment handling by the resolved service's payment_mode — NEVER
    re-triggered on a reschedule, even for a paid service: a reschedule
    moves an already-decided booking to a new time, it is not a fresh
    purchase decision, and re-running payment collection here would risk
    double-charging a customer who already paid for the original slot.
    Rescheduling a paid appointment instead directly carries over the
    original's payment_status/payment_id and confirms immediately.
      - 'free' (or a paid service with price <= 0, treated the same):
        confirms immediately, Google event created immediately — today's
        behavior, unchanged.
      - 'hold_to_confirm': books as status='pending_payment' with a
        HOLD_MINUTES-minute hold_expires_at, slot is excluded from
        availability by the hold itself (see get_available_slots), no
        Google event yet (deferred to the payment webhook finalizing it —
        avoids creating-then-cleaning-up a tentative event for every
        expired hold). A Razorpay payment link is generated and sent as
        its own WhatsApp message.
      - 'request_after': confirms immediately exactly like 'free' (Google
        event included), then a payment link is generated and sent as a
        second, non-blocking WhatsApp message right after the normal
        confirmation.
    """
    from whatsapp import send_whatsapp_buttons, send_whatsapp_cta_url
    from config import WHATSAPP_TOKEN

    res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        raise ValueError("Appointment settings not found for this project")
    settings = res.data

    old_appt_data = None
    if reschedule_id:
        old_appt_res = supabase.table("appointments").select("*").eq("id", reschedule_id).maybe_single().execute()
        old_appt_data = old_appt_res.data if old_appt_res else None
        if not old_appt_data:
            raise ValueError("Original appointment not found")
        # FIX: previously anyone who learned another customer's
        # appointment_id (visible in WhatsApp button payloads and the
        # public reschedule-context endpoint) could move that appointment
        # to a new time under different customer details, cancelling the
        # real customer's booking out from under them. Require the caller's
        # own phone/project to match the original booking before honoring
        # a reschedule — never trust reschedule_id alone.
        if old_appt_data.get("project_id") != project_id:
            raise ValueError("Original appointment not found")
        if old_appt_data.get("customer_phone") != customer_phone.replace("+", "").replace(" ", ""):
            raise ValueError("This appointment does not belong to that phone number")
        if old_appt_data.get("service_id"):
            service_id = old_appt_data["service_id"]

    if not service_id:
        raise ValueError("A service must be selected")

    service_row = get_service(project_id, service_id)
    if not service_row:
        raise ValueError("Service not found")

    duration = service_row.get("duration_minutes", 30)
    service  = service_row.get("name", "Appointment")
    price    = service_row.get("price") or 0

    # Reschedule of an already-paid booking is never re-charged (see
    # docstring) — otherwise, apply the service's real payment_mode.
    if reschedule_id and old_appt_data and old_appt_data.get("payment_status") == "paid":
        payment_mode = "free"
    elif price <= 0:
        payment_mode = "free"
    else:
        payment_mode = service_row.get("payment_mode") or "free"

    is_fresh_hold = payment_mode == "hold_to_confirm" and not reschedule_id

    # Re-validate the slot is still actually free — never trust a
    # previously-computed slot list as still true at execution time.
    if not reschedule_id:
        available = get_available_slots(project_id, appointment_date, service_id)
        available_times = {s.split(" ")[0] for s in available}  # strip "(N left)" suffix
        if start_time not in available_times:
            raise ValueError(f"{start_time} on {appointment_date} is no longer available")

    start_dt = datetime.strptime(f"{appointment_date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt   = start_dt + timedelta(minutes=duration)
    end_time = end_dt.strftime("%H:%M")

    google_event_id = None
    refresh_token   = settings.get("google_refresh_token")
    calendar_id     = settings.get("google_calendar_id", "primary")

    # Skip creating a Google event for a fresh hold — the slot is already
    # excluded from availability via the appointments row itself (see
    # get_available_slots' pending_payment handling), and this avoids
    # having to clean up a tentative event for every hold that expires
    # unpaid. The event is created when the webhook finalizes payment.
    if refresh_token and not is_fresh_hold:
        access_token = get_google_access_token(refresh_token)
        if access_token:
            event = {
                "summary": f"{service} — {customer_name}",
                "description": f"Customer: {customer_name}\nPhone: {customer_phone}\nNotes: {notes or 'None'}",
                "start": {
                    "dateTime": f"{appointment_date}T{start_time}:00",
                    "timeZone": "Asia/Kolkata",
                },
                "end": {
                    "dateTime": f"{appointment_date}T{end_time}:00",
                    "timeZone": "Asia/Kolkata",
                },
                "reminders": {
                    "useDefault": False,
                    "overrides": [{"method": "popup", "minutes": 30}],
                },
            }
            google_event_id = create_google_event(access_token, calendar_id, event)

    if reschedule_id:
        try:
            if old_appt_data:
                supabase.table("appointments").update({"status": "rescheduled"}).eq("id", reschedule_id).execute()
                if settings.get("google_refresh_token") and old_appt_data.get("google_event_id"):
                    old_token = get_google_access_token(settings["google_refresh_token"])
                    if old_token:
                        delete_google_event(old_token, calendar_id, old_appt_data["google_event_id"])
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Reschedule old appointment error: {e}")

    insert_row = {
        "project_id": project_id,
        "service_id": service_id,
        "customer_name": customer_name,
        "customer_phone": customer_phone.replace("+", ""),
        "service_name": service,
        "appointment_date": appointment_date,
        "start_time": start_time,
        "end_time": end_time,
        "status": "pending_payment" if is_fresh_hold else "confirmed",
        "google_event_id": google_event_id,
        "notes": notes,
        "payment_status": "not_required" if payment_mode == "free" else "unpaid",
    }
    if reschedule_id and old_appt_data and old_appt_data.get("payment_status") == "paid":
        insert_row["payment_status"] = "paid"
        insert_row["payment_id"] = old_appt_data.get("payment_id")
    if is_fresh_hold:
        insert_row["hold_expires_at"] = (datetime.utcnow() + timedelta(minutes=HOLD_MINUTES)).isoformat()

    appt_res = supabase.table("appointments").insert(insert_row).execute()
    appointment = appt_res.data[0]

    date_obj = datetime.strptime(appointment_date, "%Y-%m-%d")
    date_formatted = date_obj.strftime("%A, %d %B %Y")

    payment_url = None
    if payment_mode in ("hold_to_confirm", "request_after") and not (reschedule_id and old_appt_data and old_appt_data.get("payment_status") == "paid"):
        payment_url = generate_appointment_payment_link(appointment, service_row, settings)

    try:
        wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", project_id).maybe_single().execute()
        wa_data = (wa_res.data if wa_res else None)

        if wa_data:
            phone_number_id = wa_data["phone_number_id"]
            token = wa_data.get("access_token") or WHATSAPP_TOKEN
            phone = customer_phone.replace("+", "").replace(" ", "")

            if is_fresh_hold:
                # Not confirmed yet — no Reschedule/Cancel buttons (nothing
                # to reschedule/cancel), just the summary + a "Pay Now"
                # CTA-URL if a link was generated, or a plain heads-up if
                # Razorpay isn't connected for this project yet.
                msg = f"⏳ *Slot Reserved — Payment Needed*\n\n"
                msg += f"📋 Service: {service}\n"
                msg += f"📅 Date: {date_formatted}\n"
                msg += f"⏰ Time: {start_time}\n\n"
                if payment_url:
                    msg += f"Pay within {HOLD_MINUTES} minutes to confirm your booking."
                    send_whatsapp_cta_url(phone, msg, "Pay Now", payment_url, phone_number_id, token)
                else:
                    msg += "We'll be in touch shortly to arrange payment and confirm your booking."
                    from whatsapp import send_whatsapp_message
                    send_whatsapp_message(to=phone, text=msg, phone_number_id=phone_number_id, token=token)
            else:
                action = "Rescheduled" if reschedule_id else "Confirmed"
                msg = f"✅ *Booking {action}!*\n\n"
                msg += f"📋 Service: {service}\n"
                msg += f"📅 Date: {date_formatted}\n"
                msg += f"⏰ Time: {start_time}\n"
                msg += f"👤 Name: {customer_name}\n\n"
                msg += f"Booking ID: #{appointment['id'][:8].upper()}"

                send_whatsapp_buttons(
                    to=phone,
                    body=msg,
                    buttons=[
                        {"id": f"reschedule_{appointment['id']}", "title": "Reschedule 🔄"},
                        {"id": f"cancel_appt_{appointment['id']}", "title": "Cancel ❌"},
                    ],
                    phone_number_id=phone_number_id,
                    token=token,
                )

                # Non-blocking payment request, sent as its own message —
                # WhatsApp interactive messages can't mix quick-reply
                # buttons and a CTA-URL in one message.
                if payment_url:
                    send_whatsapp_cta_url(
                        phone,
                        f"💳 Optional: pay {service} in advance.",
                        "Pay Now",
                        payment_url,
                        phone_number_id, token,
                    )

            supabase.table("whatsapp_sessions").upsert({
                "project_id": project_id,
                "phone_number": phone,
                "mode": "appointment_confirmed",
                "metadata": {"appointment_id": appointment["id"]},
            }, on_conflict="project_id,phone_number").execute()

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"WhatsApp confirmation error: {e}")

    return {
        "status": appointment["status"],
        "appointment_id": appointment["id"],
        "date": date_formatted,
        "time": start_time,
        "service": service,
        "payment_required": payment_mode != "free",
        "payment_url": payment_url,
        "hold_minutes": HOLD_MINUTES if is_fresh_hold else None,
    }


@router.get("/public/appointments/{project_id}/slots")
def public_slots(project_id: str, date: str, service_id: str, request: Request):
    """Get available slots for a specific date + service."""
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"appt-slots:{project_id}:{ip}", limit=30, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    try:
        slots = get_available_slots(project_id, date, service_id)
    except ValueError as e:
        status = 400 if "format" in str(e) else 404
        raise HTTPException(status_code=status, detail=str(e))
    return {"date": date, "slots": slots}


@router.post("/public/appointments/book")
def book_appointment(body: BookingCreate, request: Request):
    """Create a new appointment."""
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"book:{body.project_id}:{ip}", limit=5):
        raise HTTPException(status_code=429, detail="Too many booking attempts — please wait a moment and try again.")
    if not body.service_id and not body.reschedule_id:
        raise HTTPException(status_code=400, detail="A service must be selected")
    try:
        return create_appointment(
            project_id=body.project_id,
            service_id=body.service_id,
            customer_name=body.customer_name,
            customer_phone=body.customer_phone,
            appointment_date=body.appointment_date,
            start_time=body.start_time,
            notes=body.notes,
            reschedule_id=body.reschedule_id,
        )
    except ValueError as e:
        status = 404 if "not found" in str(e) else 409
        raise HTTPException(status_code=status, detail=str(e))


# -------------------------------------------------
# APPOINTMENT SERVICES CRUD (dashboard) — Calendly-style "event types"
# -------------------------------------------------
@router.get("/appointment-services")
def list_services(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("appointment_services").select("*").eq("project_id", project_id).order("sort_order", desc=False).execute()
    return res.data or []


@router.post("/appointment-services")
def create_service(body: ServiceCreate, user=Depends(verify_token)):
    require_project_role(user.id, body.project_id)
    supabase.table("appointment_services").insert({
        "project_id": body.project_id,
        "name": body.name,
        "description": body.description,
        "duration_minutes": body.duration_minutes,
        "is_active": body.is_active,
        "sort_order": body.sort_order,
    }).execute()
    res = supabase.table("appointment_services").select("*").eq("project_id", body.project_id).order("created_at", desc=True).limit(1).execute()
    return res.data[0]


def _require_role_for_service(user_id: str, service_id: str):
    res = supabase.table("appointment_services").select("project_id").eq("id", service_id).maybe_single().execute()
    service = res.data if res else None
    if not service:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_role(user_id, service["project_id"])
    return service["project_id"]


@router.put("/appointment-services/{service_id}")
def update_service(service_id: str, body: ServiceUpdate, user=Depends(verify_token)):
    _require_role_for_service(user.id, service_id)
    update = {k: v for k, v in body.dict().items() if v is not None}
    supabase.table("appointment_services").update(update).eq("id", service_id).execute()
    res = supabase.table("appointment_services").select("*").eq("id", service_id).single().execute()
    return res.data


@router.delete("/appointment-services/{service_id}")
def delete_service(service_id: str, user=Depends(verify_token)):
    _require_role_for_service(user.id, service_id)
    referenced = supabase.table("appointments").select("id").eq("service_id", service_id).limit(1).execute()
    if referenced.data:
        # Has historical bookings — deleting would orphan their service_id
        # reference (appointments.service_id is ON DELETE SET NULL, so it
        # wouldn't break, but the merchant almost certainly means "stop
        # offering this," not "erase which service past customers booked").
        # Deactivate instead; hard delete is only for services nobody's
        # ever booked.
        raise HTTPException(status_code=409, detail="This service has existing bookings — deactivate it instead of deleting.")
    supabase.table("appointment_services").delete().eq("id", service_id).execute()
    return {"status": "deleted"}


# -------------------------------------------------
# APPOINTMENTS CRUD (dashboard)
# -------------------------------------------------
@router.get("/appointments")
def list_appointments(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("appointments") \
        .select("*") \
        .eq("project_id", project_id) \
        .order("appointment_date", desc=False) \
        .order("start_time", desc=False) \
        .execute()
    return res.data or []


def get_upcoming_appointments(project_id: str, phone: str, limit: int = 5) -> list:
    """Used by the in-chat 'show my bookings' tool, and by
    get_latest_upcoming_appointment below."""
    clean_phone = phone.replace("+", "").replace(" ", "")
    today = date.today().isoformat()
    res = supabase.table("appointments") \
        .select("*") \
        .eq("project_id", project_id) \
        .eq("customer_phone", clean_phone) \
        .in_("status", ["confirmed", "rescheduled"]) \
        .gte("appointment_date", today) \
        .order("appointment_date", desc=False) \
        .order("start_time", desc=False) \
        .limit(limit) \
        .execute()
    return res.data or []


def get_latest_upcoming_appointment(project_id: str, phone: str) -> Optional[dict]:
    """Used by the in-chat 'cancel my appointment' tool — finds the one
    appointment a customer would mean by 'my appointment' without needing
    them to specify an ID."""
    results = get_upcoming_appointments(project_id, phone, limit=1)
    return results[0] if results else None


def cancel_appointment(appointment_id: str, notify_customer: bool = True) -> dict:
    """Core cancel logic — used by the dashboard PUT route AND the in-chat
    cancel tool. notify_customer=False when the customer is the one
    cancelling it themselves in the same conversation — they don't need a
    separate notification for something they just did."""
    appt_res = supabase.table("appointments").select("*").eq("id", appointment_id).maybe_single().execute()
    if not appt_res or not appt_res.data:
        raise ValueError("Appointment not found")
    appt = appt_res.data

    settings_res = supabase.table("appointment_settings").select("*").eq("project_id", appt["project_id"]).maybe_single().execute()
    settings = (settings_res.data if settings_res else None) or {}
    refresh_token = settings.get("google_refresh_token")
    if refresh_token and appt.get("google_event_id"):
        access_token = get_google_access_token(refresh_token)
        if access_token:
            delete_google_event(access_token, settings.get("google_calendar_id", "primary"), appt["google_event_id"])

    if notify_customer:
        try:
            wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", appt["project_id"]).maybe_single().execute()
            wa_data = (wa_res.data if wa_res else None)
            if wa_data:
                from whatsapp import send_whatsapp_message
                send_whatsapp_message(
                    to=appt["customer_phone"],
                    text=f"❌ *Appointment Cancelled*\n\nYour {appt['service_name']} on {appt['appointment_date']} at {appt['start_time']} has been cancelled.\n\nReply *book* to schedule a new appointment.",
                    phone_number_id=wa_data["phone_number_id"],
                    token=wa_data.get("access_token") or WHATSAPP_TOKEN,
                )
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Cancel notification error: {e}")

    supabase.table("appointments").update({"status": "cancelled"}).eq("id", appointment_id).execute()
    res = supabase.table("appointments").select("*").eq("id", appointment_id).single().execute()
    return res.data


@router.put("/appointments/{appointment_id}")
def update_appointment(appointment_id: str, body: AppointmentStatusUpdate, user=Depends(verify_token)):
    appt_check = supabase.table("appointments").select("project_id").eq("id", appointment_id).maybe_single().execute()
    appt_data = appt_check.data if appt_check else None
    if not appt_data:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_role(user.id, appt_data["project_id"])
    if body.status == "cancelled":
        try:
            return cancel_appointment(appointment_id, notify_customer=True)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    supabase.table("appointments").update({"status": body.status}).eq("id", appointment_id).execute()
    res = supabase.table("appointments").select("*").eq("id", appointment_id).single().execute()
    return res.data


# -------------------------------------------------
# REMINDER SENDER — actual reminders are sent by send_reminders_job() below,
# called directly by the in-process scheduler in main.py. There used to
# also be an HTTP POST /appointments/send-reminders route here that ANY
# logged-in user (not project-scoped at all) could hit to trigger a
# reminder blast across every tenant's customers on demand — it had no
# frontend caller (confirmed via repo-wide search), duplicated
# send_reminders_job's logic, and served no legitimate purpose. Removed
# rather than patched, per the security audit.
# -------------------------------------------------
def _send_reminder_template(to: str, customer_name: str, date: str, time: str, phone_number_id: str, token: str) -> bool:
    """
    Send appointment_reminder template message.
    Template body: Hi {{1}}! Your appointment is confirmed for {{2}} at {{3}}.
                   Reply CONFIRM to confirm or CANCEL to cancel.
    Returns True if sent successfully, False if template not approved or failed.
    """
    try:
        res = requests.post(
            f"https://graph.facebook.com/v19.0/{phone_number_id}/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "template",
                "template": {
                    "name": "appointment_reminder",
                    "language": {"code": "en_US"},
                    "components": [
                        {
                            "type": "body",
                            "parameters": [
                                {"type": "text", "text": customer_name},
                                {"type": "text", "text": date},
                                {"type": "text", "text": time},
                            ]
                        }
                    ]
                }
            }
        )
        if res.ok:
            print(f"Reminder template sent to {to}")
            return True
        else:
            print(f"Reminder template failed for {to}: {res.text}")
            return False
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Reminder template error: {e}")
        return False


# -------------------------------------------------
# STANDALONE JOB — called by background scheduler in main.py
# -------------------------------------------------
def send_reminders_job():
    """
    Called directly by APScheduler every hour.
    Same logic as the /appointments/send-reminders endpoint
    but runs internally without needing an HTTP request.
    """
    from datetime import datetime, timedelta
    # Same IST-vs-server-UTC fix as generate_slots/send_reminders above.
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    sent = 0
    failed = 0

    try:
        appts_res = supabase.table("appointments") \
            .select("*") \
            .eq("status", "confirmed") \
            .eq("reminder_sent", False) \
            .execute()

        for appt in (appts_res.data or []):
            try:
                settings_res = supabase.table("appointment_settings") \
                    .select("reminder_hours") \
                    .eq("project_id", appt["project_id"]) \
                    .maybe_single() \
                    .execute()
                settings = (settings_res.data if settings_res else None) or {}
                reminder_hours = settings.get("reminder_hours", 24)

                start_time_str = str(appt["start_time"])[:5]
                appt_dt = datetime.strptime(f"{appt['appointment_date']} {start_time_str}", "%Y-%m-%d %H:%M")
                hours_until = (appt_dt - now).total_seconds() / 3600

                if 0 < hours_until <= reminder_hours:
                    wa_res = supabase.table("whatsapp_integrations") \
                        .select("*") \
                        .eq("project_id", appt["project_id"]) \
                        .maybe_single() \
                        .execute()
                    wa_data = (wa_res.data if wa_res else None)

                    if wa_data:
                        phone_number_id = wa_data["phone_number_id"]
                        token = wa_data.get("access_token") or WHATSAPP_TOKEN
                        date_obj = datetime.strptime(appt["appointment_date"], "%Y-%m-%d")
                        date_formatted = date_obj.strftime("%d %B %Y")

                        template_sent = _send_reminder_template(
                            to=appt["customer_phone"],
                            customer_name=appt["customer_name"],
                            date=date_formatted,
                            time=start_time_str,
                            phone_number_id=phone_number_id,
                            token=token,
                        )

                        if not template_sent:
                            from whatsapp import send_whatsapp_message
                            msg = (
                                f"\u23f0 *Appointment Reminder*\n\n"
                                f"Hi {appt['customer_name']}! Your {appt['service_name']} is coming up.\n\n"
                                f"\U0001f4c5 {date_formatted}\n"
                                f"\u23f0 {start_time_str}\n\n"
                                f"Reply *CANCEL* if you need to cancel."
                            )
                            send_whatsapp_message(
                                to=appt["customer_phone"],
                                text=msg,
                                phone_number_id=phone_number_id,
                                token=token,
                            )

                        supabase.table("appointments").update({"reminder_sent": True}).eq("id", appt["id"]).execute()
                        sent += 1

            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"Reminder job error for appointment {appt.get('id')}: {e}")
                failed += 1

        print(f"Reminder job done — sent: {sent}, failed: {failed}")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Reminder job fatal error: {e}")


# -------------------------------------------------
# PAYMENT FINALIZATION — called from shop.py's unified Razorpay webhook
# -------------------------------------------------
def is_appointment_slot_still_available(appointment: dict, settings: dict) -> bool:
    """Re-validates that finalizing THIS appointment wouldn't put its slot
    over capacity — i.e. counts overlapping bookings excluding the
    appointment itself. Deliberately not implemented via
    get_available_slots(), which would always report this exact slot as
    unavailable (it's already counting this very appointment as a busy
    booking)."""
    date_str = appointment["appointment_date"]
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_h, start_m = map(int, str(appointment["start_time"])[:5].split(":"))
    end_h, end_m = map(int, str(appointment["end_time"])[:5].split(":"))
    my_start = datetime(dt.year, dt.month, dt.day, start_h, start_m)
    my_end   = datetime(dt.year, dt.month, dt.day, end_h, end_m)

    now_iso = datetime.utcnow().isoformat() + "Z"
    others_res = supabase.table("appointments") \
        .select("start_time, end_time") \
        .eq("project_id", appointment["project_id"]) \
        .eq("appointment_date", date_str) \
        .neq("id", appointment["id"]) \
        .or_(f"status.in.(confirmed,rescheduled),and(status.eq.pending_payment,hold_expires_at.gt.{now_iso})") \
        .execute()

    overlap_count = 0
    for b in (others_res.data or []):
        b_start_h, b_start_m = map(int, str(b["start_time"])[:5].split(":"))
        b_end_h, b_end_m = map(int, str(b["end_time"])[:5].split(":"))
        b_start = datetime(dt.year, dt.month, dt.day, b_start_h, b_start_m)
        b_end   = datetime(dt.year, dt.month, dt.day, b_end_h, b_end_m)
        if not (my_end <= b_start or my_start >= b_end):
            overlap_count += 1

    slot_capacity = settings.get("slot_capacity", 1)
    return overlap_count < slot_capacity


def handle_appointment_payment_paid(appointment: dict):
    """Finalizes a Razorpay-paid appointment — called from shop.py's
    unified /webhook/razorpay handler once it's identified the paid
    payment_link belongs to an appointment, not an order. Idempotent: a
    retried webhook for an already-paid appointment is a safe no-op.
    """
    from whatsapp import send_whatsapp_buttons
    from config import WHATSAPP_TOKEN

    if appointment.get("payment_status") == "paid":
        return

    project_id = appointment["project_id"]

    if appointment["status"] != "pending_payment":
        # 'request_after' path — already confirmed at booking time (Google
        # event included), this just records that payment came through.
        supabase.table("appointments").update({"payment_status": "paid"}).eq("id", appointment["id"]).execute()
        return

    # 'hold_to_confirm' path — finalize now. Re-validate the slot wasn't
    # taken by someone else in the interim before confirming — never trust
    # a hold as still valid at payment time, same "never trust a
    # previously-computed slot" philosophy create_appointment already uses.
    settings_res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    settings = (settings_res.data if settings_res else None) or {}

    if not is_appointment_slot_still_available(appointment, settings):
        # Genuine conflict — money was collected but the slot's gone (e.g.
        # payment landed right as the hold was swept to 'expired' and
        # someone else's booking took it first). Flagged for manual ops
        # resolution rather than silently dropping a paid booking or
        # building automatic refund integration — refunds are a
        # deliberate non-goal here.
        supabase.table("appointments").update({
            "payment_status": "paid",
            "status": "payment_conflict",
        }).eq("id", appointment["id"]).execute()
        sentry_sdk.capture_message(f"Appointment {appointment['id']} paid but its slot was taken in the interim — needs manual resolution")
        return

    google_event_id = None
    refresh_token = settings.get("google_refresh_token")
    calendar_id   = settings.get("google_calendar_id", "primary")
    if refresh_token:
        access_token = get_google_access_token(refresh_token)
        if access_token:
            start_hm = str(appointment["start_time"])[:5]
            end_hm   = str(appointment["end_time"])[:5]
            event = {
                "summary": f"{appointment['service_name']} — {appointment['customer_name']}",
                "description": f"Customer: {appointment['customer_name']}\nPhone: {appointment['customer_phone']}\nNotes: {appointment.get('notes') or 'None'}",
                "start": {"dateTime": f"{appointment['appointment_date']}T{start_hm}:00", "timeZone": "Asia/Kolkata"},
                "end": {"dateTime": f"{appointment['appointment_date']}T{end_hm}:00", "timeZone": "Asia/Kolkata"},
                "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 30}]},
            }
            # A failed Google event here is logged (inside create_google_event)
            # but never blocks finalizing a *paid* confirmation.
            google_event_id = create_google_event(access_token, calendar_id, event)

    supabase.table("appointments").update({
        "payment_status": "paid",
        "status": "confirmed",
        "google_event_id": google_event_id,
        "hold_expires_at": None,
    }).eq("id", appointment["id"]).execute()

    try:
        wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", project_id).maybe_single().execute()
        wa_data = (wa_res.data if wa_res else None)
        if wa_data:
            phone_number_id = wa_data["phone_number_id"]
            token = wa_data.get("access_token") or WHATSAPP_TOKEN
            date_obj = datetime.strptime(appointment["appointment_date"], "%Y-%m-%d")
            date_formatted = date_obj.strftime("%A, %d %B %Y")
            msg = f"✅ *Payment Confirmed — Booking Confirmed!*\n\n"
            msg += f"📋 Service: {appointment['service_name']}\n"
            msg += f"📅 Date: {date_formatted}\n"
            msg += f"⏰ Time: {str(appointment['start_time'])[:5]}\n\n"
            msg += f"Booking ID: #{appointment['id'][:8].upper()}"
            send_whatsapp_buttons(
                to=appointment["customer_phone"],
                body=msg,
                buttons=[
                    {"id": f"reschedule_{appointment['id']}", "title": "Reschedule 🔄"},
                    {"id": f"cancel_appt_{appointment['id']}", "title": "Cancel ❌"},
                ],
                phone_number_id=phone_number_id,
                token=token,
            )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Appointment payment confirmation WhatsApp error: {e}")


# -------------------------------------------------
# HOLD-EXPIRY SWEEP — called by background scheduler in main.py
# -------------------------------------------------
def release_expired_holds():
    """Runs periodically (every 2 minutes — see main.py's scheduler) to
    flip unpaid 'hold_to_confirm' bookings whose HOLD_MINUTES window has
    passed to status='expired', freeing their slot back up (see
    generate_slots — a 'pending_payment' row only blocks availability
    while hold_expires_at is still in the future). Kept distinct from
    customer-initiated 'cancelled' so dashboard/reporting can tell a
    genuine cancellation apart from an abandoned, unpaid checkout.

    Never deletes the row — if a Razorpay webhook for this exact
    appointment arrives just after the sweep fires, the payment webhook
    handler (shop.py's razorpay_webhook) can still find it by payment_id
    and re-confirm it if the slot is still free (see that handler's
    'already-expired-but-paid' reconciliation path)."""
    now_iso = datetime.utcnow().isoformat() + "Z"
    released = 0
    failed = 0
    try:
        expired_res = supabase.table("appointments") \
            .select("id") \
            .eq("status", "pending_payment") \
            .lt("hold_expires_at", now_iso) \
            .execute()

        for row in (expired_res.data or []):
            try:
                supabase.table("appointments").update({"status": "expired"}).eq("id", row["id"]).eq("status", "pending_payment").execute()
                released += 1
            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"Hold release error for appointment {row.get('id')}: {e}")
                failed += 1

        print(f"Hold-expiry sweep done — released: {released}, failed: {failed}")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Hold-expiry sweep fatal error: {e}")