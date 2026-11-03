"""
Appointments system — Calendly-style booking via WhatsApp bot.
Supports Google Calendar integration for availability.
"""
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from clients import supabase
from auth import verify_token, require_project_access
from config import WHATSAPP_TOKEN, FRONTEND_URL
from ratelimit import is_rate_limited, client_ip
from oauth_state import issue_state, consume_state
# whatsapp.py's _TimeoutSession. Every Google call in this file used bare
# `requests` with no timeout, and get_busy_slots is reached from the PUBLIC
# /public/appointments/{id}/slots endpoint — so a slow Google pinned a
# worker thread per request, triggerable by anyone on the internet.
from whatsapp import http
import os
import re
from urllib.parse import quote
from zoneinfo import ZoneInfo
from datetime import datetime, date, timedelta
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
# Google's IANA zone for the merchant's calendar. Previously "Asia/Kolkata"
# was a literal in three event payloads and a +5:30 shift in three more
# places, so any merchant outside India got calendar events at the wrong
# time with no way to correct it. Default keeps today's behaviour exactly.
DEFAULT_TIMEZONE = "Asia/Kolkata"

# Statuses the rest of this file actually branches on. `status` used to be a
# free string written straight to the row, so any value at all could be
# stored and then silently fail to match anywhere.
VALID_APPOINTMENT_STATUSES = {"confirmed", "cancelled", "rescheduled", "completed", "no_show", "declined"}

# Google calendar ids are "primary" or an email-shaped address.
_CALENDAR_ID_RE = re.compile(r"^(primary|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})$")


class AppointmentSettingsUpdate(BaseModel):
    # service_name/duration_minutes moved to appointment_services (see
    # ServiceCreate/ServiceUpdate below) — no longer editable here.
    #
    # Every int here was unbounded. slot_capacity is the one that matters
    # most: it drives the `overlap_count >= slot_capacity` gate in
    # generate_slots, so a huge value means unlimited bookings per slot and
    # a negative one means none at all.
    buffer_minutes: Optional[int] = Field(default=None, ge=0, le=240)
    slot_capacity: Optional[int] = Field(default=None, ge=1, le=100)
    working_hours: Optional[dict] = None
    date_overrides: Optional[list] = None
    buffer_before_minutes: Optional[int] = Field(default=None, ge=0, le=240)
    # null = back to back (duration + buffers); set = a start time every N minutes
    slot_interval_minutes: Optional[int] = Field(default=None, ge=5, le=720)
    min_notice_minutes: Optional[int] = Field(default=None, ge=0, le=43200)
    max_per_day: Optional[int] = Field(default=None, ge=1, le=1000)
    advance_booking_days: Optional[int] = Field(default=None, ge=1, le=365)
    reminder_hours: Optional[int] = Field(default=None, ge=1, le=168)
    reminders_enabled: Optional[bool] = None
    # Short link name: /book/<booking_slug> (empty clears it)
    booking_slug: Optional[str] = Field(default=None, max_length=60)
    # Optional WhatsApp message some hours after the appointment
    followup_enabled: Optional[bool] = None
    followup_hours: Optional[int] = Field(default=None, ge=1, le=72)
    followup_message: Optional[str] = Field(default=None, max_length=200)
    # Customers can't cancel or reschedule closer than this many hours before
    # the appointment (empty/0 = no limit)
    cancel_cutoff_hours: Optional[int] = Field(default=None, ge=0, le=720)
    # Hours before the appointment, one reminder each (largest first), up to 4
    reminder_offsets: Optional[List[int]] = Field(default=None, max_length=4)
    google_calendar_id: Optional[str] = Field(default=None, max_length=254)
    accent_color: Optional[str] = Field(default=None, max_length=32)
    currency_code: Optional[str] = Field(default=None, max_length=8)
    timezone: Optional[str] = Field(default=None, max_length=64)
    is_enabled: Optional[bool] = None
    bot_can_book: Optional[bool] = None
    # WhatsApp alert to the owner when someone books. The phone arrives as a
    # country code and a number; an empty number clears it.
    notify_owner_booking: Optional[bool] = None
    owner_phone_code: Optional[str] = Field(default=None, max_length=6)
    owner_phone_number: Optional[str] = Field(default=None, max_length=20)

class ServiceCreate(BaseModel):
    project_id: str
    name: str = Field(max_length=120)
    description: Optional[str] = Field(default=None, max_length=1000)
    duration_minutes: int = Field(default=30, ge=5, le=1440)
    is_active: bool = True
    sort_order: int = Field(default=0, ge=0, le=10000)
    price: float = Field(default=0, ge=0, le=10000000)
    payment_mode: str = Field(default="free", pattern="^(free|hold_to_confirm|request_after)$")
    # Optional per-service overrides; empty = use the project setting.
    slot_capacity: Optional[int] = Field(default=None, ge=1, le=100)
    slot_interval_minutes: Optional[int] = Field(default=None, ge=5, le=720)
    buffer_before_minutes: Optional[int] = Field(default=None, ge=0, le=240)
    buffer_after_minutes: Optional[int] = Field(default=None, ge=0, le=240)
    min_notice_minutes: Optional[int] = Field(default=None, ge=0, le=43200)
    fixed_times: Optional[List[str]] = Field(default=None, max_length=48)
    schedule_id: Optional[str] = Field(default=None, max_length=64)   # empty = the default schedule
    # rolling = next N days, range = between two dates, indefinite = no end; empty = the project setting
    booking_window_type: Optional[str] = Field(default=None, pattern="^(rolling|range|indefinite)$")
    booking_window_days: Optional[int] = Field(default=None, ge=1, le=730)
    booking_window_start: Optional[str] = Field(default=None, max_length=10)
    booking_window_end: Optional[str] = Field(default=None, max_length=10)
    max_per_day: Optional[int] = Field(default=None, ge=1, le=1000)
    is_hidden: bool = False
    questions: Optional[list] = None
    requires_approval: Optional[bool] = None
    location: Optional[str] = Field(default=None, max_length=300)
    confirmation_note: Optional[str] = Field(default=None, max_length=500)
    cancel_cutoff_hours: Optional[int] = Field(default=None, ge=0, le=720)
    duration_options: Optional[List[int]] = Field(default=None, max_length=6)

class ServiceUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)
    description: Optional[str] = Field(default=None, max_length=1000)
    duration_minutes: Optional[int] = Field(default=None, ge=5, le=1440)
    is_active: Optional[bool] = None
    sort_order: Optional[int] = Field(default=None, ge=0, le=10000)
    price: Optional[float] = Field(default=None, ge=0, le=10000000)
    payment_mode: Optional[str] = Field(default=None, pattern="^(free|hold_to_confirm|request_after)$")
    slot_capacity: Optional[int] = Field(default=None, ge=1, le=100)
    slot_interval_minutes: Optional[int] = Field(default=None, ge=5, le=720)
    buffer_before_minutes: Optional[int] = Field(default=None, ge=0, le=240)
    buffer_after_minutes: Optional[int] = Field(default=None, ge=0, le=240)
    min_notice_minutes: Optional[int] = Field(default=None, ge=0, le=43200)
    fixed_times: Optional[List[str]] = Field(default=None, max_length=48)
    schedule_id: Optional[str] = Field(default=None, max_length=64)
    booking_window_type: Optional[str] = Field(default=None, pattern="^(rolling|range|indefinite)$")
    booking_window_days: Optional[int] = Field(default=None, ge=1, le=730)
    booking_window_start: Optional[str] = Field(default=None, max_length=10)
    booking_window_end: Optional[str] = Field(default=None, max_length=10)
    max_per_day: Optional[int] = Field(default=None, ge=1, le=1000)
    is_hidden: Optional[bool] = None
    questions: Optional[list] = None
    requires_approval: Optional[bool] = None
    location: Optional[str] = Field(default=None, max_length=300)
    confirmation_note: Optional[str] = Field(default=None, max_length=500)
    cancel_cutoff_hours: Optional[int] = Field(default=None, ge=0, le=720)
    duration_options: Optional[List[int]] = Field(default=None, max_length=6)

class ScheduleCreate(BaseModel):
    project_id: str = Field(max_length=64)
    name: str = Field(min_length=1, max_length=80)
    is_default: bool = False
    working_hours: Optional[dict] = None
    date_overrides: Optional[list] = None

class ScheduleUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    is_default: Optional[bool] = None
    working_hours: Optional[dict] = None
    date_overrides: Optional[list] = None

class BookingCreate(BaseModel):
    """Reached from /public/appointments/book, which is unauthenticated —
    so these bounds are the only thing standing between a stranger and an
    unbounded write. notes in particular ends up in the merchant's Google
    Calendar description and in a WhatsApp message."""
    project_id: str = Field(max_length=64)
    # Optional only because a reschedule always inherits the original
    # appointment's service — see create_appointment(), which resolves it
    # authoritatively server-side rather than trusting whatever (if
    # anything) the caller sends when reschedule_id is set. Required for a
    # brand-new booking.
    service_id: Optional[str] = Field(default=None, max_length=64)
    customer_name: str = Field(max_length=120)
    customer_phone: str = Field(max_length=32)
    appointment_date: str = Field(max_length=10)  # YYYY-MM-DD
    start_time: str = Field(max_length=5)         # HH:MM
    notes: Optional[str] = Field(default=None, max_length=1000)
    reschedule_id: Optional[str] = Field(default=None, max_length=64)  # old appointment ID being rescheduled
    answers: Optional[dict] = None  # booking questions: {question id: answer}
    duration_minutes: Optional[int] = Field(default=None, ge=5, le=1440)  # one of the service's lengths


class ManualBooking(BaseModel):
    """A booking the owner adds from the dashboard (phone call, walk-in)."""
    project_id: str = Field(max_length=64)
    service_id: str = Field(max_length=64)
    customer_name: str = Field(min_length=1, max_length=120)
    customer_phone: str = Field(max_length=32)
    appointment_date: str = Field(max_length=10)
    start_time: str = Field(max_length=5)
    duration_minutes: Optional[int] = Field(default=None, ge=5, le=1440)
    notes: Optional[str] = Field(default=None, max_length=1000)

class AppointmentStatusUpdate(BaseModel):
    status: str = Field(max_length=32)  # confirmed, cancelled, rescheduled, completed


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_google_access_token(refresh_token: str) -> Optional[str]:
    """Exchange refresh token for access token."""
    res = http.post("https://oauth2.googleapis.com/token", data={
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    })
    if res.ok:
        return res.json().get("access_token")
    # Body deliberately not logged — Google's error payloads echo token
    # material and internal ids.
    print(f"Google token refresh failed: HTTP {res.status_code}")
    return None


def get_busy_slots(access_token: str, calendar_id: str, date_str: str) -> List[dict]:
    """Get busy time slots from Google Calendar for a specific date."""
    start = f"{date_str}T00:00:00Z"
    end   = f"{date_str}T23:59:59Z"

    res = http.post(
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
    print(f"Google freeBusy failed: HTTP {res.status_code}")
    return []


def create_google_event(access_token: str, calendar_id: str, event: dict) -> Optional[str]:
    """Create a Google Calendar event and return event ID."""
    res = http.post(
        f"https://www.googleapis.com/calendar/v3/calendars/{quote(calendar_id, safe='@.')}/events",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=event,
    )
    if res.ok:
        return res.json().get("id")
    print(f"Google create event failed: HTTP {res.status_code}")
    return None


def delete_google_event(access_token: str, calendar_id: str, event_id: str):
    """Delete a Google Calendar event."""
    # calendar_id and event_id are percent-encoded: calendar_id comes from
    # merchant-editable settings and was interpolated into the API path raw.
    http.delete(
        f"https://www.googleapis.com/calendar/v3/calendars/{quote(calendar_id, safe='@.')}"
        f"/events/{quote(event_id, safe='')}",
        headers={"Authorization": f"Bearer {access_token}"},
    )


DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _to_min(t: str) -> int:
    h, m = str(t)[:5].split(":")
    return int(h) * 60 + int(m)


def _from_min(n: int) -> str:
    return f"{n // 60:02d}:{n % 60:02d}"


def day_windows(day_cfg: Optional[dict]) -> List[tuple]:
    """Open (start, end) ranges of one weekday, in minutes since midnight.
    Reads both the old {start, end, enabled} shape and the new one with a
    "windows" list (several ranges a day, e.g. a lunch break)."""
    if not day_cfg or not day_cfg.get("enabled", False):
        return []
    raw = day_cfg.get("windows")
    if not raw:
        raw = [{"start": day_cfg.get("start", "09:00"), "end": day_cfg.get("end", "18:00")}]
    out = []
    for w in raw:
        try:
            s, e = _to_min(w["start"]), _to_min(w["end"])
        except Exception:
            continue
        if e > s:
            out.append((s, e))
    return sorted(out)


def windows_for_date(settings: dict, date_str: str) -> List[tuple]:
    """Open ranges on one date: a date-specific override wins over the
    weekly hours (closed, or its own ranges)."""
    for o in settings.get("date_overrides") or []:
        if isinstance(o, dict) and o.get("date") == date_str:
            if o.get("closed") or not o.get("windows"):
                return []
            return day_windows({"enabled": True, "windows": o["windows"]})
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return day_windows((settings.get("working_hours") or {}).get(DAY_KEYS[dt.weekday()], {}))


DEFAULT_WORKING_HOURS = {
    **{d: {"enabled": True, "start": "09:00", "end": "18:00", "windows": [{"start": "09:00", "end": "18:00"}]}
       for d in ("mon", "tue", "wed", "thu", "fri")},
    **{d: {"enabled": False, "start": "09:00", "end": "14:00", "windows": [{"start": "09:00", "end": "14:00"}]}
       for d in ("sat", "sun")},
}


def _load_schedules(project_id: str) -> list:
    try:
        res = supabase.table("appointment_schedules").select("id, is_default, working_hours, date_overrides") \
            .eq("project_id", project_id).execute()
        return res.data or []
    except Exception as e:
        # Before the schedules migration the table doesn't exist: fall back
        # to the hours stored on appointment_settings.
        print(f"appointment_schedules unavailable: {e}")
        return []


def pick_schedule(schedules: list, service: dict, settings: dict) -> tuple:
    """(working_hours, date_overrides) for a service: its own schedule, else
    the project's default schedule, else the old hours on settings."""
    by_id = {x["id"]: x for x in schedules}
    chosen = by_id.get(service.get("schedule_id")) or next((x for x in schedules if x.get("is_default")), None)
    if chosen:
        return chosen.get("working_hours") or {}, chosen.get("date_overrides") or []
    return settings.get("working_hours") or {}, settings.get("date_overrides") or []


def _first_set(*values, default=None):
    for v in values:
        if v is not None:
            return v
    return default


def generate_slots(date_str: str, settings: dict, service: dict, busy_slots: List[dict]) -> List[str]:
    """Available start times for one date and one service.

    Everything a merchant can tune: several time ranges a day, date-specific
    hours, a start-time interval (default: back to back), buffers before and
    after, minimum notice, how far ahead people can book, bookings per day,
    capacity per slot, and hand-picked start times. Service settings (when
    set) win over the project's.

    Bookings of every service of the project count against the same
    calendar: a candidate slot (padded by its buffers) is compared to every
    existing booking by real interval overlap, not by equal start time."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    windows = windows_for_date(settings, date_str)
    if not windows:
        return []

    # "Now" in the merchant's own timezone: appointment dates and times are
    # stored as that wall clock.
    now = _reminder_now(settings.get("timezone"))

    # How far ahead: the service's own booking window wins over the project's
    window = service.get("booking_window_type")
    if window == "range":
        first, last = service.get("booking_window_start"), service.get("booking_window_end")
        if (first and date_str < str(first)[:10]) or (last and date_str > str(last)[:10]):
            return []
    elif window != "indefinite":
        advance = (service.get("booking_window_days") if window == "rolling" else None) or settings.get("advance_booking_days")
        if advance and dt.date() > (now + timedelta(days=advance)).date():
            return []

    duration = service.get("duration_minutes", 30)
    before = _first_set(service.get("buffer_before_minutes"), settings.get("buffer_before_minutes"), default=0)
    after = _first_set(service.get("buffer_after_minutes"), settings.get("buffer_minutes"), default=0)
    interval = _first_set(service.get("slot_interval_minutes"), settings.get("slot_interval_minutes"))
    notice = _first_set(service.get("min_notice_minutes"), settings.get("min_notice_minutes"), default=0)
    slot_capacity = _first_set(service.get("slot_capacity"), settings.get("slot_capacity"), default=1)
    max_per_day = settings.get("max_per_day")

    # Existing bookings for this date, across ALL services of the project
    # (they share the same provider/calendar). A 'pending_payment' hold
    # blocks the slot only while it is still live; an expired-but-not-yet-
    # swept hold must NOT keep blocking it.
    now_iso = datetime.utcnow().isoformat() + "Z"
    existing_bookings_res = supabase.table("appointments") \
        .select("start_time, end_time, service_id") \
        .eq("project_id", settings.get("project_id", "")) \
        .eq("appointment_date", date_str) \
        .or_(f"status.in.(confirmed,pending_approval),and(status.eq.pending_payment,hold_expires_at.gt.{now_iso})") \
        .execute()
    existing = existing_bookings_res.data or []

    if max_per_day and len(existing) >= max_per_day:
        return []
    service_max = service.get("max_per_day")
    if service_max and sum(1 for b in existing if b.get("service_id") == service.get("id")) >= service_max:
        return []

    booking_ranges = []
    for b in existing:
        booking_ranges.append((
            dt + timedelta(minutes=_to_min(b["start_time"])),
            dt + timedelta(minutes=_to_min(b["end_time"])),
        ))

    # Google Calendar busy times are genuinely UTC.
    busy_ranges = []
    for b in busy_slots:
        b_start = datetime.fromisoformat(b["start"].replace("Z", "+00:00")).replace(tzinfo=None)
        b_end   = datetime.fromisoformat(b["end"].replace("Z", "+00:00")).replace(tzinfo=None)
        # Approximate: shifted to IST, like before.
        busy_ranges.append((b_start + timedelta(hours=5, minutes=30), b_end + timedelta(hours=5, minutes=30)))

    # Candidate start times (minutes since midnight)
    fixed = [t for t in (service.get("fixed_times") or []) if isinstance(t, str) and _HHMM_RE.match(t)]
    starts = set()
    for (w_start, w_end) in windows:
        if fixed:
            for t in fixed:
                m = _to_min(t)
                if w_start <= m and m + duration <= w_end:
                    starts.add(m)
        else:
            step = interval or (before + duration + after)
            m = w_start
            while m + duration <= w_end:
                starts.add(m)
                m += step

    earliest = now + timedelta(minutes=notice)
    slots = []
    for m in sorted(starts):
        current = dt + timedelta(minutes=m)
        slot_end = current + timedelta(minutes=duration)

        if current <= earliest:
            continue

        # Padded by the buffers: the real appointment plus its breathing room.
        pad_start = current - timedelta(minutes=before)
        pad_end = slot_end + timedelta(minutes=after)

        overlap_count = sum(
            1 for (b_start, b_end) in booking_ranges
            if not (pad_end <= b_start or pad_start >= b_end)
        )
        if overlap_count >= slot_capacity:
            continue

        if any(not (pad_end <= b_start or pad_start >= b_end) for b_start, b_end in busy_ranges):
            continue

        remaining = slot_capacity - overlap_count
        slot_str = current.strftime("%H:%M")
        slots.append(slot_str if slot_capacity == 1 else f"{slot_str} ({remaining} left)")

    return slots


# -------------------------------------------------
# GOOGLE OAUTH
# -------------------------------------------------
@router.get("/appointments/google/auth/{project_id}")
def google_auth(project_id: str, user=Depends(verify_token)):
    """Start Google OAuth flow for calendar access."""
    require_project_access(user.id, project_id, tab="appointments")
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Google OAuth not configured. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to env vars.")

    # state was the bare project_id, and the callback authenticated nothing —
    # so anyone could run Google's consent screen themselves and then hand
    # the callback SOMEONE ELSE'S project id, writing their own refresh
    # token into that merchant's settings. From then on every booking landed
    # on the attacker's calendar (customer name, phone and notes included)
    # and the attacker's calendar decided which slots looked free.
    #
    # Google was the only provider missed when durable state was introduced;
    # shopify_oauth.py and razorpay_oauth.py have both used these two
    # helpers since. The nonce records who minted it and is single-use.
    state = issue_state("google", project_id, user.id)

    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        "&scope=https://www.googleapis.com/auth/calendar"
        "&access_type=offline"
        "&prompt=consent"
        f"&state={quote(state, safe='')}"
    )
    return {"auth_url": auth_url}


def _popup_html(event: str, error: str = None) -> HTMLResponse:
    # Matches razorpay_oauth.py/shopify_oauth.py's own helper — json.dumps
    # so a raw error string (which may contain quotes) can't break the
    # <script> tag's syntax, unlike the old inline f-string interpolation.
    payload = {"type": "GOOGLE_AUTH", "event": event}
    if error:
        payload["error"] = error
    # Targeted at FRONTEND_URL rather than '*'. The wildcard handed the auth
    # result to whatever origin happened to own the opener.
    return HTMLResponse(
        f"<html><body><script>"
        f"window.opener.postMessage({json.dumps(payload)}, {json.dumps(FRONTEND_URL)}); window.close();"
        f"</script></body></html>"
    )


@router.get("/appointments/google/callback")
def google_callback(code: str, state: str, request: Request):
    """Handle Google OAuth callback — exchange code for tokens."""
    # Unauthenticated by necessity (Google redirects the browser here), so
    # it needs its own ceiling — it posts to Google on every call.
    if is_rate_limited(f"google-oauth-cb:{client_ip(request)}", limit=20, window_seconds=300):
        return _popup_html("ERROR", "Too many attempts — please wait a moment and try again.")

    # project_id comes from the single-use state row, NOT from the query
    # string. It used to be `project_id = state`, which let a caller name
    # any project they liked and attach their own Google account to it.
    state_row = consume_state("google", state)
    if not state_row:
        return _popup_html("ERROR", "This connection link expired or was already used. Please try connecting again.")

    project_id = state_row["project_id"]

    res = http.post("https://oauth2.googleapis.com/token", data={
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
        print(f"Google Calendar OAuth token exchange failed: {res.text}")
        return _popup_html("ERROR", "Could not connect Google Calendar. Please try again.")

    token_data = res.json()
    refresh_token = token_data.get("refresh_token")

    if not refresh_token:
        return _popup_html("ERROR", "No refresh token returned. Try disconnecting and reconnecting.")

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

    return _popup_html("FINISH")


@router.delete("/appointments/google/disconnect/{project_id}")
def google_disconnect(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="appointments")
    supabase.table("appointment_settings").update({
        "google_refresh_token": None,
        "google_calendar_id": "primary",
    }).eq("project_id", project_id).execute()
    return {"success": True}


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
def _clean_windows(raw, label: str) -> list:
    """Validated, sorted, non-overlapping [{start, end}] ranges."""
    if not isinstance(raw, list) or not raw or len(raw) > 8:
        raise HTTPException(status_code=400, detail=f"{label}: use between 1 and 8 time ranges")
    out = []
    for w in raw:
        if not isinstance(w, dict) or not _HHMM_RE.match(str(w.get("start", ""))) or not _HHMM_RE.match(str(w.get("end", ""))):
            raise HTTPException(status_code=400, detail=f"{label}: times must look like 09:00")
        s_m, e_m = _to_min(w["start"]), _to_min(w["end"])
        if e_m <= s_m:
            raise HTTPException(status_code=400, detail=f"{label}: a time range must end after it starts")
        out.append((s_m, e_m))
    out.sort()
    for a, b in zip(out, out[1:]):
        if b[0] < a[1]:
            raise HTTPException(status_code=400, detail=f"{label}: time ranges overlap")
    return [{"start": _from_min(s_m), "end": _from_min(e_m)} for s_m, e_m in out]


def _clean_working_hours(raw) -> dict:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="Invalid working hours")
    out = {}
    for key in DAY_KEYS:
        cfg = raw.get(key)
        if cfg is None:
            continue
        if not isinstance(cfg, dict):
            raise HTTPException(status_code=400, detail="Invalid working hours")
        enabled = bool(cfg.get("enabled"))
        windows = cfg.get("windows")
        if windows is None:
            windows = [{"start": cfg.get("start", "09:00"), "end": cfg.get("end", "18:00")}]
        label = key.capitalize()
        if windows:
            cleaned = _clean_windows(windows, label)
        elif enabled:
            raise HTTPException(status_code=400, detail=f"{label}: add a time range or switch the day off")
        else:
            cleaned = [{"start": "09:00", "end": "18:00"}]
        # start/end keep describing the whole day for anything reading the old shape
        out[key] = {"enabled": enabled, "start": cleaned[0]["start"], "end": cleaned[-1]["end"], "windows": cleaned}
    return out


def _clean_date_overrides(raw) -> list:
    if not isinstance(raw, list) or len(raw) > 400:
        raise HTTPException(status_code=400, detail="Too many date-specific entries")
    by_date = {}
    for o in raw:
        if not isinstance(o, dict) or not _DATE_RE.match(str(o.get("date", ""))):
            raise HTTPException(status_code=400, detail="Each date-specific entry needs a date")
        try:
            datetime.strptime(o["date"], "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{o['date']} is not a real date")
        entry = {"date": o["date"], "closed": bool(o.get("closed"))}
        note = str(o.get("note") or "").strip()[:80]
        if note:
            entry["note"] = note
        if not entry["closed"]:
            entry["windows"] = _clean_windows(o.get("windows") or [], o["date"])
        by_date[o["date"]] = entry
    return [by_date[d] for d in sorted(by_date)]


def _clean_fixed_times(raw) -> Optional[list]:
    if raw is None:
        return None
    times = sorted({str(t).strip() for t in raw})
    for t in times:
        if not _HHMM_RE.match(t):
            raise HTTPException(status_code=400, detail=f"{t} is not a time like 09:00")
    return times or None


def _clean_questions(raw) -> Optional[list]:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) > 10:
        raise HTTPException(status_code=400, detail="Up to 10 booking questions")
    out, seen = [], set()
    for i, q in enumerate(raw):
        if not isinstance(q, dict):
            raise HTTPException(status_code=400, detail="Invalid booking question")
        label = str(q.get("label") or "").strip()[:120]
        if not label:
            raise HTTPException(status_code=400, detail="Each booking question needs a label")
        qtype = q.get("type") or "text"
        if qtype not in ("text", "textarea", "select"):
            raise HTTPException(status_code=400, detail="Invalid booking question type")
        qid = re.sub(r"[^a-zA-Z0-9_-]", "", str(q.get("id") or ""))[:40] or f"q{i + 1}"
        if qid in seen:
            qid = f"{qid}_{i + 1}"
        seen.add(qid)
        options = []
        if qtype == "select":
            options = [str(o).strip()[:80] for o in (q.get("options") or []) if str(o).strip()][:20]
            if not options:
                raise HTTPException(status_code=400, detail=f'"{label}" needs at least one option')
        out.append({"id": qid, "label": label, "type": qtype, "options": options, "required": bool(q.get("required"))})
    return out


def _check_booking_window(kind, days, first, last):
    if kind == "rolling" and not days:
        raise HTTPException(status_code=400, detail="Set how many days ahead people can book this service")
    if kind == "range":
        if not first or not last:
            raise HTTPException(status_code=400, detail="Pick a start and an end date for this service")
        try:
            a, b = datetime.strptime(str(first)[:10], "%Y-%m-%d"), datetime.strptime(str(last)[:10], "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Booking dates must be real dates")
        if b < a:
            raise HTTPException(status_code=400, detail="The end date must be after the start date")


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,39}$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# Words that are other pages, or other parts of the booking API
_RESERVED_SLUGS = {"api", "admin", "new", "dashboard", "login", "signup", "book", "confirm", "event", "shop",
                   "pricing", "privacy", "account", "verification", "settings", "services", "slots",
                   "booking-status", "reschedule", "resolve", "reset-password", "forgot-password"}


def clean_slug(raw) -> Optional[str]:
    """A valid booking link name, or None for empty. Raises HTTPException."""
    slug = str(raw or "").strip().lower()
    if not slug:
        return None
    if not _SLUG_RE.match(slug) or _UUID_RE.match(slug) or slug in _RESERVED_SLUGS:
        raise HTTPException(status_code=400, detail="Use 3 to 40 letters, numbers or dashes, like clinic or dr-asha. That name isn't allowed.")
    return slug


def _clean_duration_options(raw) -> Optional[list]:
    if raw is None:
        return None
    opts = sorted({int(x) for x in raw})
    if any(x < 5 or x > 1440 for x in opts):
        raise HTTPException(status_code=400, detail="Lengths must be between 5 and 1440 minutes")
    return opts or None


def allowed_durations(service: dict) -> list:
    return sorted({service.get("duration_minutes", 30), *(service.get("duration_options") or [])})


def _check_approval(requires_approval, payment_mode):
    if requires_approval and payment_mode not in (None, "free"):
        raise HTTPException(status_code=400, detail="Approval is only available for free services. Switch the service's payment to Free.")


def _check_schedule(project_id: str, schedule_id: Optional[str]):
    if not schedule_id:
        return
    res = supabase.table("appointment_schedules").select("id").eq("id", schedule_id).eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        raise HTTPException(status_code=400, detail="That schedule doesn't exist")


class ChangeNotAllowed(ValueError):
    """A customer tried to cancel/reschedule inside the cut-off."""


def change_cutoff_problem(appt: dict, settings: dict, service: Optional[dict]) -> Optional[str]:
    """A message for the customer when it's too late for them to cancel or
    reschedule this appointment, else None. The dashboard isn't limited by
    this: only the customer's own changes."""
    hours = _first_set((service or {}).get("cancel_cutoff_hours"), settings.get("cancel_cutoff_hours"))
    if not hours:
        return None
    now = _reminder_now(settings.get("timezone"))
    start = datetime.strptime(f"{appt['appointment_date']} {str(appt['start_time'])[:5]}", "%Y-%m-%d %H:%M")
    if (start - now).total_seconds() < hours * 3600:
        unit = "hour" if hours == 1 else "hours"
        return f"Cancelling or rescheduling is only possible up to {hours} {unit} before the appointment. Please contact us directly."
    return None


def appointment_change_problem(appointment_id: str, project_id: str, phone: str) -> Optional[str]:
    """Cut-off check for the WhatsApp Reschedule button. None = allowed (also
    when the booking can't be found: the booking page deals with that)."""
    appt_res = supabase.table("appointments").select("*").eq("id", appointment_id).eq("project_id", project_id).maybe_single().execute()
    appt = appt_res.data if appt_res else None
    if not appt or appt.get("customer_phone") != str(phone).replace("+", "").replace(" ", ""):
        return None
    st_res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    svc_res = supabase.table("appointment_services").select("cancel_cutoff_hours").eq("id", appt.get("service_id") or "").maybe_single().execute()
    return change_cutoff_problem(appt, (st_res.data if st_res else None) or {}, (svc_res.data if svc_res else None) or {})


def booking_details(service_row: dict) -> str:
    """One line shown in the confirmation: where, and the owner's note."""
    from shop import one_line
    parts = []
    if service_row.get("location"):
        parts.append(f"Where: {one_line(service_row['location'])}.")
    if service_row.get("confirmation_note"):
        parts.append(one_line(service_row["confirmation_note"]))
    return " ".join(parts)[:400]


def send_booking_confirmed_messages(appointment: dict, details: str, phone_number_id: str, token: str,
                                    action: str = "Confirmed") -> None:
    """The customer's confirmation: the details template (when there are
    details), else the plain confirmation template, else buttons as free
    text, which only arrives inside WhatsApp's 24-hour window."""
    from whatsapp import send_whatsapp_buttons
    phone = str(appointment["customer_phone"]).replace("+", "").replace(" ", "")
    name = appointment.get("customer_name") or "there"
    service = appointment.get("service_name") or "Appointment"
    date_formatted = datetime.strptime(appointment["appointment_date"], "%Y-%m-%d").strftime("%A, %d %B %Y")
    start = str(appointment["start_time"])[:5]
    short_id = appointment["id"][:8].upper()
    payloads = [f"reschedule_{appointment['id']}", f"cancel_appt_{appointment['id']}"]

    msg = f"✅ *Booking {action}!*\n\n📋 Service: {service}\n📅 Date: {date_formatted}\n⏰ Time: {start}\n👤 Name: {name}\n"
    if details:
        msg += f"ℹ️ {details}\n"
    msg += f"\nBooking ID: #{short_id}"

    # Templates first (they reach website bookers too); their Reschedule /
    # Cancel buttons carry the same ids as the buttons below.
    if details and _send_appt_template("appointment_confirmed_details", phone,
                                       [name, service, date_formatted, start, details, short_id],
                                       phone_number_id, token, button_payloads=payloads):
        return
    if _send_appt_template("appointment_confirmed", phone, [name, service, date_formatted, start, short_id],
                           phone_number_id, token, button_payloads=payloads):
        return
    send_whatsapp_buttons(
        to=phone, body=msg,
        buttons=[{"id": payloads[0], "title": "Reschedule 🔄"}, {"id": payloads[1], "title": "Cancel ❌"}],
        phone_number_id=phone_number_id, token=token,
    )


class AnswerError(ValueError):
    """A required booking question is missing or an answer isn't allowed."""


def clean_answers(questions: list, answers) -> list:
    """[{question, answer}] stored on the booking. The question text is kept
    with the answer, so later edits to the questions don't change history."""
    answers = answers if isinstance(answers, dict) else {}
    out = []
    for q in questions or []:
        val = str(answers.get(q.get("id")) or "").strip()[:1000]
        if not val:
            if q.get("required"):
                raise AnswerError(f"Please answer: {q.get('label')}")
            continue
        if q.get("type") == "select" and val not in (q.get("options") or []):
            raise AnswerError(f"Please pick one of the options for: {q.get('label')}")
        out.append({"question": q.get("label"), "answer": val})
    return out


def _fields_set(body) -> set:
    return set(getattr(body, "model_fields_set", None) or getattr(body, "__fields_set__", set()))


# Empty here means "back to the default", so an explicit null is saved, not dropped.
_SETTINGS_NULLABLE = ("slot_interval_minutes", "max_per_day", "cancel_cutoff_hours")
_SERVICE_NULLABLE = ("slot_capacity", "slot_interval_minutes", "buffer_before_minutes",
                     "buffer_after_minutes", "min_notice_minutes", "fixed_times", "schedule_id",
                     "booking_window_type", "booking_window_days", "booking_window_start",
                     "booking_window_end", "max_per_day", "cancel_cutoff_hours", "location", "confirmation_note",
                     "duration_options")


@router.get("/appointment-settings/{project_id}")
def get_settings(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="appointments")
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
            "timezone": DEFAULT_TIMEZONE,
            "is_enabled": False,
            "reminders_enabled": True,
            "reminder_offsets": [],
            "cancel_cutoff_hours": None,
            "notify_owner_booking": True,
            "owner_phone": None,
            "owner_phone_code": None,
            "google_connected": False,
        }
    data = res.data
    data["google_connected"] = bool(data.get("google_refresh_token"))
    data.pop("google_refresh_token", None)  # never expose token to frontend
    return data


@router.put("/appointment-settings/{project_id}")
def update_settings(project_id: str, body: AppointmentSettingsUpdate, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="appointments")
    update = {k: v for k, v in body.dict().items() if v is not None}
    sent = _fields_set(body)
    for k in _SETTINGS_NULLABLE:
        if k in sent and getattr(body, k) is None:
            update[k] = None
    if "reminder_offsets" in update:
        if any(o < 1 or o > 168 for o in update["reminder_offsets"]):
            raise HTTPException(status_code=400, detail="Reminders go out 1 to 168 hours before")
        update["reminder_offsets"] = sorted(set(update["reminder_offsets"]), reverse=True)
    update.pop("booking_slug", None)
    if "booking_slug" in sent:
        slug = clean_slug(body.booking_slug)
        cur_res = supabase.table("appointment_settings").select("booking_slug, old_slugs").eq("project_id", project_id).maybe_single().execute()
        cur = (cur_res.data if cur_res else None) or {}
        current = cur.get("booking_slug")
        if slug != current:
            if slug:
                taken = supabase.table("appointment_settings").select("project_id").eq("booking_slug", slug).neq("project_id", project_id).limit(1).execute().data
                taken_old = supabase.table("appointment_settings").select("project_id").contains("old_slugs", [slug]).neq("project_id", project_id).limit(1).execute().data
                if taken or taken_old:
                    raise HTTPException(status_code=409, detail="That link name is already taken. Try another.")
            # The name being replaced keeps working, so shared links don't break
            old = [x for x in (cur.get("old_slugs") or []) if x not in (current, slug)]
            if current:
                old.append(current)
            update["booking_slug"] = slug
            update["old_slugs"] = old[-10:]
    update.pop("owner_phone_code", None)
    update.pop("owner_phone_number", None)
    if "owner_phone_number" in sent:
        code = re.sub(r"\D", "", body.owner_phone_code or "")
        number = re.sub(r"\D", "", body.owner_phone_number or "")
        if not number:
            update["owner_phone"] = None
            update["owner_phone_code"] = None
        else:
            from phones import normalize, default_region_for_project
            digits, reason = normalize(f"+{code}{number}" if code else number, default_region_for_project(project_id))
            if not digits:
                raise HTTPException(status_code=400, detail=f"Your WhatsApp number: {reason}")
            update["owner_phone"] = digits
            update["owner_phone_code"] = code or None
    if "working_hours" in update:
        update["working_hours"] = _clean_working_hours(update["working_hours"])
    if "date_overrides" in update:
        update["date_overrides"] = _clean_date_overrides(update["date_overrides"])

    # timezone reaches Google as the event's timeZone and drives slot
    # generation, so it has to be a real IANA zone, not any 64-char string.
    if "timezone" in update:
        try:
            ZoneInfo(update["timezone"])
        except Exception:
            raise HTTPException(status_code=400, detail="Unknown timezone")

    # calendar_id is interpolated into the Google API path (it is
    # percent-encoded there now, but a nonsense value just means every
    # calendar call silently fails, so reject it here instead).
    if "google_calendar_id" in update:
        cal = update["google_calendar_id"].strip()
        if not cal or not _CALENDAR_ID_RE.match(cal):
            raise HTTPException(status_code=400, detail="Invalid calendar id")
        update["google_calendar_id"] = cal

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
    ip = client_ip(request)
    if is_rate_limited(f"appt-settings:{project_id}:{ip}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    # FIX: previously also selected google_refresh_token into memory (then
    # popped it before returning) — this is a public, unauthenticated route,
    # so don't pull a merchant's Calendar token into scope here at all.
    res = supabase.table("appointment_settings").select(
        "working_hours,date_overrides,advance_booking_days,accent_color,currency_code,is_enabled,google_calendar_id,timezone"
    ).eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        raise HTTPException(status_code=404, detail="Booking not configured")
    data = res.data
    if not data.get("is_enabled"):
        raise HTTPException(status_code=403, detail="Booking not enabled")
    return data


@router.get("/public/appointments/resolve/{slug}")
def resolve_booking_slug(slug: str, request: Request):
    """/book/<slug> -> the project id behind it. Earlier names of a project
    resolve too."""
    ip = client_ip(request)
    if is_rate_limited(f"appt-resolve:{ip}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    slug = slug.strip().lower()[:60]
    res = supabase.table("appointment_settings").select("project_id").eq("booking_slug", slug).limit(1).execute()
    rows = res.data or supabase.table("appointment_settings").select("project_id").contains("old_slugs", [slug]).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Booking page not found")
    return {"project_id": rows[0]["project_id"]}


@router.get("/public/appointments/{project_id}/services")
def public_services(project_id: str, request: Request, service: Optional[str] = None):
    """Active services a customer can pick from on the public booking page
    — no auth, matches the rest of the /public/* surface. Hidden services
    are left out unless the page was opened with that service's own link
    (?service=<id>). Each carries the hours and booking window it uses, so
    the page can grey out the right dates."""
    ip = client_ip(request)
    if is_rate_limited(f"appt-services:{project_id}:{ip}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    res = supabase.table("appointment_services").select(
        "id,name,description,duration_minutes,price,payment_mode,is_hidden,schedule_id,"
        "booking_window_type,booking_window_days,booking_window_start,booking_window_end,questions,requires_approval,"
        "duration_options,location,confirmation_note"
    ).eq("project_id", project_id).eq("is_active", True).order("sort_order", desc=False).execute()
    rows = [r for r in (res.data or []) if not r.get("is_hidden") or r["id"] == service]
    if not rows:
        return []
    st = supabase.table("appointment_settings").select("working_hours,date_overrides") \
        .eq("project_id", project_id).maybe_single().execute()
    settings = (st.data if st else None) or {}
    schedules = _load_schedules(project_id)
    for r in rows:
        r["working_hours"], r["date_overrides"] = pick_schedule(schedules, r, settings)
        r.pop("schedule_id", None)
    return rows


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
    ip = client_ip(request)
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
    ip = client_ip(request)
    if is_rate_limited(f"appt-reschedule-ctx:{project_id}:{ip}", limit=30, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    res = supabase.table("appointments").select("service_id, service_name, start_time, end_time") \
        .eq("id", appointment_id).eq("project_id", project_id).maybe_single().execute()
    appt = res.data if res else None
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {"service_id": appt.get("service_id"), "service_name": appt.get("service_name"),
            "duration_minutes": _to_min(appt["end_time"]) - _to_min(appt["start_time"])}


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
    res = supabase.table("appointment_services").select("*").eq("project_id", project_id).eq("is_active", True) \
        .eq("is_hidden", False).execute()
    services = res.data or []
    name_lower = name.strip().lower()

    exact = [s for s in services if s["name"].strip().lower() == name_lower]
    if len(exact) == 1:
        return exact[0]

    partial = [s for s in services if name_lower in s["name"].lower() or s["name"].lower() in name_lower]
    if len(partial) == 1:
        return partial[0]

    return None


def get_available_slots(project_id: str, date: str, service_id: str, duration: Optional[int] = None) -> list:
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
    if duration is not None and duration != service.get("duration_minutes"):
        if duration not in (service.get("duration_options") or []):
            raise ValueError("That length isn't offered for this service")
        service = {**service, "duration_minutes": duration}

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
    settings["working_hours"], settings["date_overrides"] = pick_schedule(_load_schedules(project_id), service, settings)
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
    answers: Optional[dict] = None,
    duration_minutes: Optional[int] = None,
    by_owner: bool = False,
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

    # Booking questions: checked only for the booking page (answers given).
    # The AI chat books without them; a reschedule keeps the old answers.
    answer_list = None
    if reschedule_id and old_appt_data:
        answer_list = old_appt_data.get("answers")
    elif answers is not None:
        answer_list = clean_answers(service_row.get("questions") or [], answers) or None

    if reschedule_id and old_appt_data:
        problem = change_cutoff_problem(old_appt_data, settings, service_row)
        if problem:
            raise ChangeNotAllowed(problem)

    # A new booking for an approval service waits for the owner; moving an
    # already-approved booking doesn't ask again.
    needs_approval = bool(service_row.get("requires_approval")) and not reschedule_id and not by_owner
    details = booking_details(service_row)

    duration = service_row.get("duration_minutes", 30)
    if reschedule_id and old_appt_data:
        # A reschedule keeps the length that was booked
        duration = (_to_min(old_appt_data["end_time"]) - _to_min(old_appt_data["start_time"])) or duration
    elif duration_minutes and duration_minutes != duration:
        if duration_minutes not in (service_row.get("duration_options") or []):
            raise ValueError("That length isn't offered for this service")
        duration = duration_minutes
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

    if by_owner:
        payment_mode = "free"  # the owner takes any payment outside Zavo
    is_fresh_hold = payment_mode == "hold_to_confirm" and not reschedule_id

    # Re-validate the slot is still actually free — never trust a
    # previously-computed slot list as still true at execution time.
    if not reschedule_id:
        available = get_available_slots(project_id, appointment_date, service_id, duration)
        available_times = {s.split(" ")[0] for s in available}  # strip "(N left)" suffix
        if start_time not in available_times:
            raise ValueError(f"{start_time} on {appointment_date} is no longer available")

    start_dt = datetime.strptime(f"{appointment_date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt   = start_dt + timedelta(minutes=duration)
    end_time = end_dt.strftime("%H:%M")

    refresh_token   = settings.get("google_refresh_token")
    calendar_id     = settings.get("google_calendar_id", "primary")
    tz              = settings.get("timezone") or DEFAULT_TIMEZONE

    insert_row = {
        "project_id": project_id,
        "service_id": service_id,
        "customer_name": customer_name,
        "customer_phone": customer_phone.replace("+", ""),
        "service_name": service,
        "appointment_date": appointment_date,
        "start_time": start_time,
        "end_time": end_time,
        "status": "pending_approval" if needs_approval else ("pending_payment" if is_fresh_hold else "confirmed"),
        "notes": notes,
        "payment_status": "not_required" if payment_mode == "free" else "unpaid",
    }
    if reschedule_id and old_appt_data and old_appt_data.get("payment_status") == "paid":
        insert_row["payment_status"] = "paid"
        insert_row["payment_id"] = old_appt_data.get("payment_id")
    if reschedule_id:
        insert_row["rescheduled_from"] = reschedule_id
    if answer_list:
        insert_row["answers"] = answer_list
    if is_fresh_hold:
        insert_row["hold_expires_at"] = (datetime.utcnow() + timedelta(minutes=HOLD_MINUTES)).isoformat()

    # The row goes in FIRST, inside a transaction that locks the project's
    # settings and recounts overlapping bookings under that lock. Previously
    # this was a plain insert after an unlocked availability check, so two
    # customers taking the last slot both succeeded — and the Google event
    # was created BEFORE the insert, so a failed insert left an orphan event
    # on the merchant's calendar carrying the customer's name and phone.
    #
    # Releasing the original on a reschedule happens in this same
    # transaction, so a failure can no longer leave the customer with
    # neither the old booking nor a new one.
    try:
        booked = supabase.rpc("book_appointment_slot", {
            "p_project_id": project_id,
            "p_row": insert_row,
            "p_start_time": start_time,
            "p_end_time": end_time,
            "p_date": appointment_date,
            "p_reschedule_id": reschedule_id,
            "p_capacity": service_row.get("slot_capacity"),
        }).execute()
    except Exception as e:
        message = str(e)
        if "slot_taken" in message:
            raise ValueError(f"{start_time} on {appointment_date} is no longer available")
        if "settings_not_found" in message:
            raise ValueError("Appointment settings not found for this project")
        # Neither expected case — an unrecognized failure from the core
        # atomic-booking RPC. Both callers (the public booking route and
        # the in-chat booking tool) only catch ValueError, so this
        # propagated as an unhandled 500 with nothing telling Sentry.
        sentry_sdk.capture_exception(e)
        raise

    appointment = booked.data[0] if isinstance(booked.data, list) else booked.data
    if not appointment:
        raise ValueError(f"{start_time} on {appointment_date} is no longer available")

    # Calendar work happens only after the booking is durably stored, so a
    # failure here costs a missing event, never a lost or duplicated booking.
    #
    # Skip creating a Google event for a fresh hold — the slot is already
    # excluded from availability via the appointments row itself (see
    # get_available_slots' pending_payment handling), and this avoids
    # having to clean up a tentative event for every hold that expires
    # unpaid. The event is created when the webhook finalizes payment.
    if refresh_token and not is_fresh_hold and not needs_approval:
        try:
            access_token = get_google_access_token(refresh_token)
            if access_token:
                event = {
                    "summary": f"{service} — {customer_name}",
                    "description": f"Customer: {customer_name}\nPhone: {customer_phone}\nNotes: {notes or 'None'}"
                                   + "".join(f"\n{a['question']}: {a['answer']}" for a in (answer_list or [])),
                    "start": {"dateTime": f"{appointment_date}T{start_time}:00", "timeZone": tz},
                    "end": {"dateTime": f"{appointment_date}T{end_time}:00", "timeZone": tz},
                    "reminders": {
                        "useDefault": False,
                        "overrides": [{"method": "popup", "minutes": 30}],
                    },
                }
                google_event_id = create_google_event(access_token, calendar_id, event)
                if google_event_id:
                    supabase.table("appointments").update(
                        {"google_event_id": google_event_id}
                    ).eq("id", appointment["id"]).execute()
                    appointment["google_event_id"] = google_event_id

                # The replaced booking's event is removed only now that the
                # new one exists.
                if reschedule_id and old_appt_data and old_appt_data.get("google_event_id"):
                    delete_google_event(access_token, calendar_id, old_appt_data["google_event_id"])
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Google Calendar sync failed for appointment {appointment.get('id')}: {e}")

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

            if needs_approval:
                # Not confirmed yet: the owner decides. The template reaches
                # website bookers too; free text only inside the 24-hour window.
                if not _send_appt_template("appointment_request_received", phone,
                                           [customer_name, service, date_formatted, start_time],
                                           phone_number_id, token):
                    from whatsapp import send_whatsapp_message
                    send_whatsapp_message(
                        to=phone,
                        text=(f"🕒 *Request received*\n\n📋 {service}\n📅 {date_formatted}\n⏰ {start_time}\n\n"
                              "We'll confirm it shortly."),
                        phone_number_id=phone_number_id, token=token,
                    )
            elif is_fresh_hold:
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
                send_booking_confirmed_messages(
                    {**appointment, "customer_name": customer_name, "service_name": service},
                    details, phone_number_id, token,
                    action="Rescheduled" if reschedule_id else "Confirmed",
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

    if by_owner:
        pass  # the owner just made this booking themselves
    elif needs_approval:
        notify_owner_booking(appointment, "request", answer_list)
    elif not is_fresh_hold:
        notify_owner_booking(appointment, "rescheduled" if reschedule_id else "new", answer_list)

    return {
        "status": appointment["status"],
        "appointment_id": appointment["id"],
        "date": date_formatted,
        "time": start_time,
        "service": service,
        "details": details,
        **({"message": "Request sent: the business has to approve it. Tell the customer it is requested and they will be "
                       "messaged on WhatsApp once it is confirmed. Do NOT say it is confirmed."} if needs_approval else {}),
        "payment_required": payment_mode != "free",
        "payment_url": payment_url,
        "hold_minutes": HOLD_MINUTES if is_fresh_hold else None,
    }


@router.get("/public/appointments/{project_id}/slots")
def public_slots(project_id: str, date: str, service_id: str, request: Request, duration: Optional[int] = None):
    """Get available slots for a specific date + service."""
    ip = client_ip(request)
    if is_rate_limited(f"appt-slots:{project_id}:{ip}", limit=30, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    try:
        slots = get_available_slots(project_id, date, service_id, duration)
    except ValueError as e:
        status = 400 if ("format" in str(e) or "length" in str(e)) else 404
        raise HTTPException(status_code=status, detail=str(e))
    return {"date": date, "slots": slots}


@router.post("/public/appointments/book")
def book_appointment(body: BookingCreate, request: Request):
    """Create a new appointment."""
    ip = client_ip(request)
    if is_rate_limited(f"book:{body.project_id}:{ip}", limit=5):
        raise HTTPException(status_code=429, detail="Too many booking attempts — please wait a moment and try again.")
    if not body.service_id and not body.reschedule_id:
        raise HTTPException(status_code=400, detail="A service must be selected")
    customer_phone = body.customer_phone
    if not body.reschedule_id:
        from phones import normalize, default_region_for_project
        customer_phone, reason = normalize(customer_phone, default_region_for_project(body.project_id))
        if not customer_phone:
            raise HTTPException(status_code=400, detail=reason or "Please enter a valid phone number.")
    try:
        return create_appointment(
            project_id=body.project_id,
            service_id=body.service_id,
            customer_name=body.customer_name,
            customer_phone=customer_phone,
            appointment_date=body.appointment_date,
            start_time=body.start_time,
            notes=body.notes,
            reschedule_id=body.reschedule_id,
            answers=None if body.reschedule_id else (body.answers or {}),
            duration_minutes=body.duration_minutes,
        )
    except AnswerError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        status = 404 if "not found" in str(e) else 409
        raise HTTPException(status_code=status, detail=str(e))


# -------------------------------------------------
# APPOINTMENT SERVICES CRUD (dashboard) — Calendly-style "event types"
# -------------------------------------------------
@router.get("/appointment-services")
def list_services(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="appointments")
    res = supabase.table("appointment_services").select("*").eq("project_id", project_id).order("sort_order", desc=False).execute()
    return res.data or []


@router.post("/appointment-services")
def create_service(body: ServiceCreate, user=Depends(verify_token)):
    require_project_access(user.id, body.project_id, tab="appointments")
    price = body.price if body.payment_mode != "free" else 0
    if body.payment_mode != "free" and price <= 0:
        raise HTTPException(status_code=400, detail="A paid service needs a price")
    _check_approval(body.requires_approval, body.payment_mode)
    if body.duration_options and body.payment_mode != "free":
        raise HTTPException(status_code=400, detail="Several lengths are only available for free services (the price is the same for every length).")
    _check_schedule(body.project_id, body.schedule_id)
    _check_booking_window(body.booking_window_type, body.booking_window_days,
                          body.booking_window_start, body.booking_window_end)
    supabase.table("appointment_services").insert({
        "project_id": body.project_id,
        "name": body.name,
        "description": body.description,
        "duration_minutes": body.duration_minutes,
        "is_active": body.is_active,
        "sort_order": body.sort_order,
        "price": price,
        "payment_mode": body.payment_mode,
        "slot_capacity": body.slot_capacity,
        "slot_interval_minutes": body.slot_interval_minutes,
        "buffer_before_minutes": body.buffer_before_minutes,
        "buffer_after_minutes": body.buffer_after_minutes,
        "min_notice_minutes": body.min_notice_minutes,
        "fixed_times": _clean_fixed_times(body.fixed_times),
        "schedule_id": body.schedule_id or None,
        "booking_window_type": body.booking_window_type,
        "booking_window_days": body.booking_window_days,
        "booking_window_start": body.booking_window_start or None,
        "booking_window_end": body.booking_window_end or None,
        "max_per_day": body.max_per_day,
        "is_hidden": body.is_hidden,
        "questions": _clean_questions(body.questions) or [],
        "requires_approval": bool(body.requires_approval),
        "location": (body.location or "").strip() or None,
        "confirmation_note": (body.confirmation_note or "").strip() or None,
        "cancel_cutoff_hours": body.cancel_cutoff_hours,
        "duration_options": _clean_duration_options(body.duration_options),
    }).execute()
    res = supabase.table("appointment_services").select("*").eq("project_id", body.project_id).order("created_at", desc=True).limit(1).execute()
    return res.data[0]


def _require_role_for_service(user_id: str, service_id: str):
    res = supabase.table("appointment_services").select("project_id").eq("id", service_id).maybe_single().execute()
    service = res.data if res else None
    if not service:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user_id, service["project_id"], tab="appointments")
    return service["project_id"]


@router.put("/appointment-services/{service_id}")
def update_service(service_id: str, body: ServiceUpdate, user=Depends(verify_token)):
    _require_role_for_service(user.id, service_id)
    update = {k: v for k, v in body.dict().items() if v is not None}
    sent = _fields_set(body)
    for k in _SERVICE_NULLABLE:
        if k in sent and getattr(body, k) is None:
            update[k] = None
    if "fixed_times" in update:
        update["fixed_times"] = _clean_fixed_times(update["fixed_times"])
    if update.get("payment_mode") == "free":
        update["price"] = 0
    if "questions" in update:
        update["questions"] = _clean_questions(update["questions"]) or []
    if "duration_options" in update:
        update["duration_options"] = _clean_duration_options(update["duration_options"])
    for k in ("schedule_id", "booking_window_start", "booking_window_end", "location", "confirmation_note"):
        if k in update and not str(update[k] or "").strip():
            update[k] = None
    current = supabase.table("appointment_services").select("*").eq("id", service_id).single().execute().data
    merged = {**current, **update}
    _check_approval(merged.get("requires_approval"), merged.get("payment_mode"))
    if merged.get("duration_options") and merged.get("payment_mode") not in (None, "free"):
        raise HTTPException(status_code=400, detail="Several lengths are only available for free services (the price is the same for every length).")
    _check_schedule(current["project_id"], merged.get("schedule_id"))
    _check_booking_window(merged.get("booking_window_type"), merged.get("booking_window_days"),
                          merged.get("booking_window_start"), merged.get("booking_window_end"))
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
# SCHEDULES CRUD (dashboard) — named weekly hours + date-specific hours
# -------------------------------------------------
def _schedule_project(user_id: str, schedule_id: str) -> dict:
    res = supabase.table("appointment_schedules").select("*").eq("id", schedule_id).maybe_single().execute()
    row = res.data if res else None
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user_id, row["project_id"], tab="appointments")
    return row


def _unset_other_defaults(project_id: str, keep_id: Optional[str] = None):
    q = supabase.table("appointment_schedules").update({"is_default": False}).eq("project_id", project_id).eq("is_default", True)
    if keep_id:
        q = q.neq("id", keep_id)
    q.execute()


@router.get("/appointment-schedules")
def list_schedules(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="appointments")
    q = lambda: supabase.table("appointment_schedules").select("*").eq("project_id", project_id) \
        .order("created_at", desc=False).execute().data or []
    rows = q()
    if not rows:
        # Projects set up after the migration: start from their current hours.
        st = supabase.table("appointment_settings").select("working_hours,date_overrides") \
            .eq("project_id", project_id).maybe_single().execute()
        st = (st.data if st else None) or {}
        supabase.table("appointment_schedules").insert({
            "project_id": project_id, "name": "Working hours", "is_default": True,
            "working_hours": st.get("working_hours") or DEFAULT_WORKING_HOURS,
            "date_overrides": st.get("date_overrides") or [],
        }).execute()
        rows = q()
    return rows


@router.post("/appointment-schedules")
def create_schedule(body: ScheduleCreate, user=Depends(verify_token)):
    require_project_access(user.id, body.project_id, tab="appointments")
    row = {
        "project_id": body.project_id,
        "name": body.name.strip(),
        "is_default": body.is_default,
        "working_hours": _clean_working_hours(body.working_hours or DEFAULT_WORKING_HOURS),
        "date_overrides": _clean_date_overrides(body.date_overrides or []),
    }
    if body.is_default:
        _unset_other_defaults(body.project_id)
    res = supabase.table("appointment_schedules").insert(row).execute()
    return res.data[0]


@router.put("/appointment-schedules/{schedule_id}")
def update_schedule(schedule_id: str, body: ScheduleUpdate, user=Depends(verify_token)):
    current = _schedule_project(user.id, schedule_id)
    update = {}
    if body.name is not None:
        update["name"] = body.name.strip()
    if body.working_hours is not None:
        update["working_hours"] = _clean_working_hours(body.working_hours)
    if body.date_overrides is not None:
        update["date_overrides"] = _clean_date_overrides(body.date_overrides)
    if body.is_default is False and current.get("is_default"):
        raise HTTPException(status_code=400, detail="Make another schedule the default instead")
    if body.is_default and not current.get("is_default"):
        _unset_other_defaults(current["project_id"], keep_id=schedule_id)
        update["is_default"] = True
    if update:
        supabase.table("appointment_schedules").update(update).eq("id", schedule_id).execute()
    return supabase.table("appointment_schedules").select("*").eq("id", schedule_id).single().execute().data


@router.delete("/appointment-schedules/{schedule_id}")
def delete_schedule(schedule_id: str, user=Depends(verify_token)):
    current = _schedule_project(user.id, schedule_id)
    if current.get("is_default"):
        raise HTTPException(status_code=409, detail="The default schedule can't be deleted. Make another one the default first.")
    # Services using it fall back to the default schedule (schedule_id on delete set null).
    supabase.table("appointment_schedules").delete().eq("id", schedule_id).execute()
    return {"status": "deleted"}


# -------------------------------------------------
# APPOINTMENTS CRUD (dashboard)
# -------------------------------------------------
@router.post("/appointments/manual")
def manual_booking(body: ManualBooking, user=Depends(verify_token)):
    """The owner adds a booking (phone call, walk-in). Confirmed straight away
    and the customer gets the usual WhatsApp confirmation; the owner isn't
    alerted about their own booking and no payment is collected here."""
    require_project_access(user.id, body.project_id, tab="appointments")
    from phones import normalize, default_region_for_project
    phone, reason = normalize(body.customer_phone, default_region_for_project(body.project_id))
    if not phone:
        raise HTTPException(status_code=400, detail=f"Customer number: {reason}")
    try:
        return create_appointment(
            project_id=body.project_id, service_id=body.service_id, customer_name=body.customer_name.strip(),
            customer_phone=phone, appointment_date=body.appointment_date, start_time=body.start_time,
            notes=body.notes, duration_minutes=body.duration_minutes, by_owner=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e) else 409, detail=str(e))


@router.get("/appointments")
def list_appointments(project_id: str, user=Depends(verify_token), limit: int = 200, offset: int = 0):
    require_project_access(user.id, project_id, tab="appointments")
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    # Explicit columns and a page. This returned every appointment ever —
    # every customer name, phone and note — in one unbounded response.
    res = supabase.table("appointments") \
        .select("id, project_id, service_id, service_name, customer_name, "
                "customer_phone, appointment_date, start_time, end_time, "
                "status, payment_status, notes, google_event_id, created_at, rescheduled_from, answers") \
        .eq("project_id", project_id) \
        .order("appointment_date", desc=False) \
        .order("start_time", desc=False) \
        .range(offset, offset + limit - 1) \
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
        .eq("status", "confirmed") \
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
    if not notify_customer:
        # The customer is cancelling themselves: the cut-off applies. The
        # owner cancelling from the dashboard isn't limited by it.
        svc_res = supabase.table("appointment_services").select("cancel_cutoff_hours").eq("id", appt.get("service_id") or "").maybe_single().execute()
        problem = change_cutoff_problem(appt, settings, (svc_res.data if svc_res else None) or {})
        if problem:
            raise ChangeNotAllowed(problem)
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
                pid = wa_data["phone_number_id"]
                tok = wa_data.get("access_token") or WHATSAPP_TOKEN
                if not _send_appt_template(
                    "appointment_cancelled", appt["customer_phone"],
                    [appt.get("customer_name") or "there", appt["service_name"], appt["appointment_date"], appt["start_time"]],
                    pid, tok,
                ):
                    send_whatsapp_message(
                        to=appt["customer_phone"],
                        text=f"❌ *Appointment Cancelled*\n\nYour {appt['service_name']} on {appt['appointment_date']} at {appt['start_time']} has been cancelled.\n\nReply *book* to schedule a new appointment.",
                        phone_number_id=pid,
                        token=tok,
                    )
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Cancel notification error: {e}")

    supabase.table("appointments").update({"status": "cancelled"}).eq("id", appointment_id).execute()
    res = supabase.table("appointments").select("*").eq("id", appointment_id).single().execute()
    if not notify_customer:
        # The customer cancelled it themselves (WhatsApp button or AI chat):
        # tell the owner. A cancel from the dashboard is the owner's own doing.
        notify_owner_booking(res.data, "cancelled", res.data.get("answers"))
    return res.data


def _wa_for(project_id: str):
    res = supabase.table("whatsapp_integrations").select("phone_number_id, access_token").eq("project_id", project_id).maybe_single().execute()
    return res.data if res else None


def approve_appointment(appt: dict) -> dict:
    """Owner approved a pending request: it becomes a normal confirmed booking."""
    from config import WHATSAPP_TOKEN
    project_id = appt["project_id"]
    st_res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    settings = (st_res.data if st_res else None) or {}
    svc_res = supabase.table("appointment_services").select("location, confirmation_note").eq("id", appt.get("service_id") or "").maybe_single().execute()
    details = booking_details((svc_res.data if svc_res else None) or {})

    google_event_id = None
    refresh_token = settings.get("google_refresh_token")
    if refresh_token:
        try:
            access_token = get_google_access_token(refresh_token)
            if access_token:
                tz = settings.get("timezone") or DEFAULT_TIMEZONE
                start, end = str(appt["start_time"])[:5], str(appt["end_time"])[:5]
                answers = "".join(f"\n{a['question']}: {a['answer']}" for a in (appt.get("answers") or []))
                google_event_id = create_google_event(access_token, settings.get("google_calendar_id", "primary"), {
                    "summary": f"{appt['service_name']} — {appt['customer_name']}",
                    "description": f"Customer: {appt['customer_name']}\nPhone: {appt['customer_phone']}\nNotes: {appt.get('notes') or 'None'}{answers}",
                    "start": {"dateTime": f"{appt['appointment_date']}T{start}:00", "timeZone": tz},
                    "end": {"dateTime": f"{appt['appointment_date']}T{end}:00", "timeZone": tz},
                    "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 30}]},
                })
        except Exception as e:
            sentry_sdk.capture_exception(e)

    supabase.table("appointments").update({"status": "confirmed", "google_event_id": google_event_id}) \
        .eq("id", appt["id"]).eq("status", "pending_approval").execute()
    row = supabase.table("appointments").select("*").eq("id", appt["id"]).single().execute().data
    try:
        wa = _wa_for(project_id)
        if wa:
            send_booking_confirmed_messages(row, details, wa["phone_number_id"], wa.get("access_token") or WHATSAPP_TOKEN)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Approval confirmation WhatsApp error: {e}")
    return row


def decline_appointment(appt: dict) -> dict:
    """Owner declined a pending request: the slot is free again."""
    from config import WHATSAPP_TOKEN
    supabase.table("appointments").update({"status": "declined"}).eq("id", appt["id"]).eq("status", "pending_approval").execute()
    row = supabase.table("appointments").select("*").eq("id", appt["id"]).single().execute().data
    try:
        wa = _wa_for(appt["project_id"])
        if wa:
            pid, tok = wa["phone_number_id"], wa.get("access_token") or WHATSAPP_TOKEN
            date_formatted = datetime.strptime(appt["appointment_date"], "%Y-%m-%d").strftime("%A, %d %B %Y")
            start = str(appt["start_time"])[:5]
            if not _send_appt_template("appointment_cancelled", appt["customer_phone"],
                                       [appt.get("customer_name") or "there", appt["service_name"], date_formatted, start], pid, tok):
                from whatsapp import send_whatsapp_message
                send_whatsapp_message(
                    to=appt["customer_phone"],
                    text=f"Sorry, we couldn't confirm your {appt['service_name']} on {date_formatted} at {start}. Please pick another time.",
                    phone_number_id=pid, token=tok,
                )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Decline WhatsApp error: {e}")
    return row


@router.put("/appointments/{appointment_id}")
def update_appointment(appointment_id: str, body: AppointmentStatusUpdate, user=Depends(verify_token)):
    appt_check = supabase.table("appointments").select("project_id").eq("id", appointment_id).maybe_single().execute()
    appt_data = appt_check.data if appt_check else None
    if not appt_data:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user.id, appt_data["project_id"], tab="appointments")
    if body.status not in VALID_APPOINTMENT_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    if body.status in ("confirmed", "declined"):
        cur = supabase.table("appointments").select("*").eq("id", appointment_id).single().execute().data
        if cur and cur.get("status") == "pending_approval":
            return approve_appointment(cur) if body.status == "confirmed" else decline_appointment(cur)
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
def _send_appt_template(name: str, to: str, params: list, phone_number_id: str, token: str,
                        button_payloads: list = None) -> bool:
    """Send an appointment template (Templates > library). False when it
    isn't in the account or WhatsApp refuses it - the caller then sends the
    old free-form message, which only arrives inside WhatsApp's 24-hour
    window (i.e. for people who messaged the business recently). Website
    bookers have no such window, so without the template they got nothing."""
    components = [{"type": "body", "parameters": [{"type": "text", "text": str(p or "-")[:200]} for p in params]}]
    for i, payload in enumerate(button_payloads or []):
        components.append({"type": "button", "sub_type": "quick_reply", "index": str(i),
                           "parameters": [{"type": "payload", "payload": payload}]})
    try:
        res = http.post(
            f"https://graph.facebook.com/v25.0/{phone_number_id}/messages",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"messaging_product": "whatsapp", "to": to, "type": "template",
                  "template": {"name": name, "language": {"code": "en_US"}, "components": components}},
        )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return False
    if not res.ok:
        print(f"Appointment template {name} not sent ({res.status_code}); falling back")
    return res.ok


def _send_reminder_template(to: str, customer_name: str, date: str, time: str, phone_number_id: str, token: str) -> bool:
    """
    Send appointment_reminder template message.
    Template body: Hi {{1}}! Your appointment is confirmed for {{2}} at {{3}}.
                   Reply CONFIRM to confirm or CANCEL to cancel.
    Returns True if sent successfully, False if template not approved or failed.

    A request-level exception (network error, timeout) is NOT caught here —
    it propagates to send_reminders_job's own per-appointment try/except,
    which batches and reports once per run. This used to catch and capture
    here too, called from inside that same per-appointment loop — up to
    MAX_REMINDERS_PER_RUN times an hour, and worst-case ALL of them at once
    if the cause is systemic (an expired WhatsApp token fails identically
    for every appointment in the run).
    """
    res = http.post(
        f"https://graph.facebook.com/v25.0/{phone_number_id}/messages",
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
    print(f"Reminder template failed for {to}: {res.text}")
    return False


# -------------------------------------------------
# STANDALONE JOB — called by background scheduler in main.py
# -------------------------------------------------
# How many un-reminded appointments one hourly pass will look at. Without a
# ceiling this query grew without bound: a past appointment never satisfies
# the "0 < hours_until" window below, so it was never marked and stayed in
# the result set for ever. The date floor in the query is the real fix for
# that; this is the backstop.
MAX_REMINDERS_PER_RUN = 500


def _reminder_now(tz_name: str, on_invalid=None) -> datetime:
    """Local wall-clock time for a project, for deciding whether an
    appointment has entered its reminder window.

    appointment_date and start_time are stored as the merchant's local wall
    clock, so the comparison has to happen in the merchant's zone. This was
    utcnow() + 5:30 — hardcoded IST — even after the booking path started
    honouring appointment_settings.timezone (see create_appointment), so a
    merchant outside India got their reminders at the wrong hour.

    on_invalid, if given, is called with the bad tz_name before falling
    back. This is called once per appointment in the reminder job's loop —
    up to MAX_REMINDERS_PER_RUN times — so the caller uses on_invalid to
    dedupe by project and report once per run instead of capturing an
    exception here on every single appointment a broken project has.
    """
    try:
        return datetime.now(ZoneInfo(tz_name or DEFAULT_TIMEZONE)).replace(tzinfo=None)
    except Exception:
        if on_invalid:
            on_invalid(tz_name)
        return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).replace(tzinfo=None)


FOLLOWUP_TEMPLATE = "appointment_followup"


def send_followups_job():
    """Hourly: the optional thank-you / review-request message, a set number of
    hours after the appointment ended (not for cancelled or no-show bookings).
    Claimed before sending, so overlapping runs can't double-send."""
    sent = failed = 0
    try:
        from config import WHATSAPP_TOKEN
        projects = supabase.table("appointment_settings") \
            .select("project_id, timezone, followup_hours, followup_message") \
            .eq("followup_enabled", True).execute().data or []
        for st in projects:
            now = _reminder_now(st.get("timezone"))
            hours = st.get("followup_hours") or 2
            wa = _wa_for(st["project_id"])
            if not wa:
                continue
            pid, tok = wa["phone_number_id"], wa.get("access_token") or WHATSAPP_TOKEN
            rows = supabase.table("appointments") \
                .select("id, appointment_date, end_time, customer_name, customer_phone, service_name") \
                .eq("project_id", st["project_id"]).in_("status", ["confirmed", "completed"]) \
                .eq("followup_sent", False) \
                .gte("appointment_date", (now.date() - timedelta(days=2)).isoformat()) \
                .lte("appointment_date", now.date().isoformat()).limit(200).execute().data or []
            for a in rows:
                try:
                    end = datetime.strptime(f"{a['appointment_date']} {str(a['end_time'])[:5]}", "%Y-%m-%d %H:%M")
                    if not (timedelta(hours=hours) <= now - end <= timedelta(hours=48)):
                        continue
                    claim = supabase.table("appointments").update({"followup_sent": True}) \
                        .eq("id", a["id"]).eq("followup_sent", False).execute()
                    if not claim.data:
                        continue
                    note = (st.get("followup_message") or "").strip() or "We would love to hear how it went."
                    name = a.get("customer_name") or "there"
                    if not _send_appt_template(FOLLOWUP_TEMPLATE, a["customer_phone"], [name, a["service_name"], note], pid, tok):
                        from whatsapp import send_whatsapp_message
                        send_whatsapp_message(to=a["customer_phone"], text=f"Hi {name}! Thank you for your {a['service_name']} appointment. {note}",
                                              phone_number_id=pid, token=tok)
                    sent += 1
                except Exception as e:
                    print(f"Follow-up error for appointment {a.get('id')}: {e}")
                    failed += 1
        if failed:
            sentry_sdk.capture_message(f"send_followups_job: {failed} of {sent + failed} failed in one run", level="warning")
        print(f"Follow-up job done — sent: {sent}, failed: {failed}")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Follow-up job fatal error: {e}")


def reminder_offsets_for(settings: dict) -> list:
    """Hours-before values to remind at, largest first. No list set = the
    single reminder_hours value, as before."""
    offs = sorted({int(o) for o in (settings.get("reminder_offsets") or [])
                   if isinstance(o, (int, float)) and 1 <= o <= 168}, reverse=True)[:4]
    return offs or [settings.get("reminder_hours") or 24]


def reminder_stage_now(offsets: list, hours_until: float, stage: int) -> int:
    """How many reminders (largest first) should have gone out by now. A
    reminder is due once the appointment is within its offset; when several
    fall due together (booked late), one message covers them all. Returns
    `stage` unchanged when nothing new is due."""
    if hours_until <= 0:
        return stage
    due = sum(1 for o in offsets if hours_until <= o)
    return due if due > stage else stage


def send_reminders_job():
    """
    Called directly by APScheduler every hour.
    Same logic as the /appointments/send-reminders endpoint
    but runs internally without needing an HTTP request.
    """
    sent = 0
    failed = 0
    # Deduped by project, not appended per-appointment — a project with a
    # broken timezone setting has every one of its appointments hit this on
    # every run, and a set means that only ever costs one entry regardless
    # of how many appointments it has.
    bad_timezones = set()

    try:
        # From yesterday onwards only. A past appointment can never enter the
        # reminder window, so scanning them was waste that grew for ever. One
        # day of slack absorbs any timezone difference between the floor
        # (computed in the default zone) and the merchant's own.
        floor = (datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).date() - timedelta(days=1)).isoformat()

        appts_res = supabase.table("appointments") \
            .select("id, project_id, appointment_date, start_time, customer_name, customer_phone, service_name, reminder_stage") \
            .eq("status", "confirmed") \
            .eq("reminder_sent", False) \
            .gte("appointment_date", floor) \
            .order("appointment_date") \
            .limit(MAX_REMINDERS_PER_RUN) \
            .execute()

        for appt in (appts_res.data or []):
            try:
                settings_res = supabase.table("appointment_settings") \
                    .select("reminder_hours, timezone, reminders_enabled, reminder_offsets") \
                    .eq("project_id", appt["project_id"]) \
                    .maybe_single() \
                    .execute()
                settings = (settings_res.data if settings_res else None) or {}
                if settings.get("reminders_enabled") is False:
                    continue
                offsets = reminder_offsets_for(settings)
                stage = appt.get("reminder_stage") or 0
                now = _reminder_now(settings.get("timezone"), on_invalid=bad_timezones.add)

                start_time_str = str(appt["start_time"])[:5]
                appt_dt = datetime.strptime(f"{appt['appointment_date']} {start_time_str}", "%Y-%m-%d %H:%M")
                hours_until = (appt_dt - now).total_seconds() / 3600

                new_stage = reminder_stage_now(offsets, hours_until, stage)
                if new_stage > stage:
                    wa_res = supabase.table("whatsapp_integrations") \
                        .select("phone_number_id, access_token") \
                        .eq("project_id", appt["project_id"]) \
                        .maybe_single() \
                        .execute()
                    wa_data = (wa_res.data if wa_res else None)

                    if wa_data:
                        phone_number_id = wa_data["phone_number_id"]
                        token = wa_data.get("access_token") or WHATSAPP_TOKEN

                        # CLAIM BEFORE SENDING. This was select, then send,
                        # then mark — so two overlapping runs both selected
                        # the row, both sent a PAID WhatsApp template, and
                        # both marked it. The customer got two reminders and
                        # the merchant was billed twice. Render's
                        # zero-downtime deploys overlap two instances, so
                        # that window opens on every deploy, not only if a
                        # worker flag is added later.
                        #
                        # Same conditional-update claim that
                        # dispatch_scheduled_campaigns uses. Marking first
                        # means a crash before the send loses a reminder
                        # rather than duplicating a charge, which is the
                        # right way round for something spending money.
                        claim = supabase.table("appointments") \
                            .update({"reminder_stage": new_stage, "reminder_sent": new_stage >= len(offsets)}) \
                            .eq("id", appt["id"]) \
                            .eq("reminder_sent", False) \
                            .eq("reminder_stage", stage) \
                            .execute()
                        if not claim.data:
                            continue

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

                        sent += 1

            except Exception as e:
                # NOT captured per appointment — this loop runs up to
                # MAX_REMINDERS_PER_RUN times an hour, and a systemic cause
                # (e.g. a dead WhatsApp integration) fails identically for
                # every appointment in the run. One capture_message below,
                # after the loop, reports the count instead of spending one
                # Sentry event per failure.
                print(f"Reminder job error for appointment {appt.get('id')}: {e}")
                failed += 1

        if failed:
            sentry_sdk.capture_message(
                f"send_reminders_job: {failed} of {sent + failed} appointment(s) failed in one run",
                level="warning",
            )
        if bad_timezones:
            sentry_sdk.capture_message(
                f"send_reminders_job: invalid timezone(s) fell back to default: {sorted(bad_timezones)[:10]}",
                level="warning",
            )

        print(f"Reminder job done — sent: {sent}, failed: {failed}")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Reminder job fatal error: {e}")


# -------------------------------------------------
# OWNER ALERT — "someone booked"
# -------------------------------------------------
OWNER_BOOKING_TEMPLATE = "appointment_new_booking_alert"
_OWNER_EVENT_TITLES = {
    "new": ("📅 *New booking!*", "New booking"),
    "rescheduled": ("🔄 *Booking rescheduled*", "Rescheduled booking"),
    "paid": ("💳 *New booking (paid)*", "New paid booking"),
    "cancelled": ("❌ *Booking cancelled by the customer*", "Cancelled booking"),
    "request": ("🕒 *Booking request — needs your approval*", "Booking request"),
}


def notify_owner_booking(appointment: dict, event: str = "new", answers: Optional[list] = None) -> None:
    """Tell the business owner about a booking on WhatsApp. Best-effort, never
    raises. Off when the switch is off or no number is set. Inside WhatsApp's
    24-hour window the full text goes out (free); outside it the approved
    appointment_new_booking_alert template does, and if that isn't added yet
    the text is still tried."""
    try:
        project_id = appointment["project_id"]
        st_res = supabase.table("appointment_settings").select("notify_owner_booking, owner_phone") \
            .eq("project_id", project_id).maybe_single().execute()
        st = (st_res.data if st_res else None) or {}
        phone = st.get("owner_phone")
        # A missing key (column not added yet) counts as on, but with no phone nothing is sent.
        if st.get("notify_owner_booking") is False or not phone:
            return
        wa_res = supabase.table("whatsapp_integrations").select("phone_number_id, access_token") \
            .eq("project_id", project_id).maybe_single().execute()
        wa = wa_res.data if wa_res else None
        if not wa:
            return
        from config import WHATSAPP_TOKEN
        from shop import _owner_window_open, one_line
        from whatsapp import send_whatsapp_message
        phone_number_id = wa["phone_number_id"]
        token = wa.get("access_token") or WHATSAPP_TOKEN

        title, label = _OWNER_EVENT_TITLES.get(event, _OWNER_EVENT_TITLES["new"])
        date_formatted = datetime.strptime(appointment["appointment_date"], "%Y-%m-%d").strftime("%A, %d %B %Y")
        start = str(appointment["start_time"])[:5]
        customer = one_line(appointment.get("customer_name") or "") or f"+{appointment['customer_phone']}"
        who = f"+{appointment['customer_phone']}" + (f" ({one_line(appointment['customer_name'])})" if appointment.get("customer_name") else "")

        msg = f"{title}\n\n📋 {appointment.get('service_name') or 'Appointment'}\n📅 {date_formatted}\n⏰ {start}\n👤 {who}\n"
        for a in answers or []:
            msg += f"❓ {one_line(a['question'])}: {one_line(a['answer'])}\n"
        if appointment.get("notes"):
            msg += f"📝 {one_line(appointment['notes'])}\n"
        msg += f"Booking ID: #{appointment['id'][:8].upper()}"
        if event == "request":
            msg += "\n\nOpen Appointments in your Zavo dashboard to approve or decline it."

        if _owner_window_open(project_id, phone):
            send_whatsapp_message(to=phone, text=msg, phone_number_id=phone_number_id, token=token)
            return
        if _send_appt_template(OWNER_BOOKING_TEMPLATE, phone,
                               [label, appointment.get("service_name") or "Appointment", customer, date_formatted, start],
                               phone_number_id, token):
            return
        send_whatsapp_message(to=phone, text=msg, phone_number_id=phone_number_id, token=token)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Owner booking alert failed: {e}")


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
        .or_(f"status.in.(confirmed,pending_approval),and(status.eq.pending_payment,hold_expires_at.gt.{now_iso})") \
        .execute()

    overlap_count = 0
    for b in (others_res.data or []):
        b_start_h, b_start_m = map(int, str(b["start_time"])[:5].split(":"))
        b_end_h, b_end_m = map(int, str(b["end_time"])[:5].split(":"))
        b_start = datetime(dt.year, dt.month, dt.day, b_start_h, b_start_m)
        b_end   = datetime(dt.year, dt.month, dt.day, b_end_h, b_end_m)
        if not (my_end <= b_start or my_start >= b_end):
            overlap_count += 1

    svc_capacity = None
    if appointment.get("service_id"):
        svc_res = supabase.table("appointment_services").select("slot_capacity").eq("id", appointment["service_id"]).maybe_single().execute()
        svc_capacity = ((svc_res.data if svc_res else None) or {}).get("slot_capacity")
    slot_capacity = _first_set(svc_capacity, settings.get("slot_capacity"), default=1)
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
    _tz           = settings.get("timezone") or DEFAULT_TIMEZONE
    if refresh_token:
        access_token = get_google_access_token(refresh_token)
        if access_token:
            start_hm = str(appointment["start_time"])[:5]
            end_hm   = str(appointment["end_time"])[:5]
            event = {
                "summary": f"{appointment['service_name']} — {appointment['customer_name']}",
                "description": f"Customer: {appointment['customer_name']}\nPhone: {appointment['customer_phone']}\nNotes: {appointment.get('notes') or 'None'}",
                "start": {"dateTime": f"{appointment['appointment_date']}T{start_hm}:00", "timeZone": _tz},
                "end": {"dateTime": f"{appointment['appointment_date']}T{end_hm}:00", "timeZone": _tz},
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

    notify_owner_booking({**appointment, "status": "confirmed"}, "paid", appointment.get("answers"))


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
        # Bounded like the reminder sweep. This one runs every two minutes,
        # so a backlog drains quickly; the cap only stops a single pass
        # loading an unbounded result set.
        expired_res = supabase.table("appointments") \
            .select("id") \
            .eq("status", "pending_payment") \
            .lt("hold_expires_at", now_iso) \
            .order("hold_expires_at") \
            .limit(MAX_REMINDERS_PER_RUN) \
            .execute()

        for row in (expired_res.data or []):
            try:
                supabase.table("appointments").update({"status": "expired"}).eq("id", row["id"]).eq("status", "pending_payment").execute()
                released += 1
            except Exception as e:
                # NOT captured per row — this runs every 2 minutes (the
                # tightest interval of any job here), so a systemic cause
                # (e.g. the update RPC itself misbehaving) could spend up to
                # MAX_REMINDERS_PER_RUN Sentry events every 2 minutes. One
                # capture_message below reports the count instead.
                print(f"Hold release error for appointment {row.get('id')}: {e}")
                failed += 1

        if failed:
            sentry_sdk.capture_message(
                f"release_expired_holds: {failed} of {released + failed} hold(s) failed to release in one run",
                level="warning",
            )

        print(f"Hold-expiry sweep done — released: {released}, failed: {failed}")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Hold-expiry sweep fatal error: {e}")