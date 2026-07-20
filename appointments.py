"""
Appointments system — Calendly-style booking via WhatsApp bot.
Supports Google Calendar integration for availability.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from clients import supabase
from auth import verify_token
from config import WHATSAPP_TOKEN, FRONTEND_URL
import os
import requests
from datetime import datetime, date, timedelta, time
import json

router = APIRouter()

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", f"{os.getenv('BACKEND_URL', 'https://ragby-backend.onrender.com')}/appointments/google/callback")


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class AppointmentSettingsUpdate(BaseModel):
    service_name: Optional[str] = None
    duration_minutes: Optional[int] = None
    buffer_minutes: Optional[int] = None
    slot_capacity: Optional[int] = None
    working_hours: Optional[dict] = None
    advance_booking_days: Optional[int] = None
    reminder_hours: Optional[int] = None
    google_calendar_id: Optional[str] = None
    accent_color: Optional[str] = None
    is_enabled: Optional[bool] = None
    bot_can_book: Optional[bool] = None

class BookingCreate(BaseModel):
    project_id: str
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


def generate_slots(date_str: str, settings: dict, busy_slots: List[dict]) -> List[str]:
    """Generate available time slots for a given date."""
    import re

    # Get day of week
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_map = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}
    day_key = day_map[dt.weekday()]

    working_hours = settings.get("working_hours", {})
    day_config = working_hours.get(day_key, {})

    if not day_config.get("enabled", False):
        return []

    start_str = day_config.get("start", "09:00")
    end_str   = day_config.get("end", "18:00")
    duration  = settings.get("duration_minutes", 30)
    buffer    = settings.get("buffer_minutes", 0)

    # Parse start/end times
    start_h, start_m = map(int, start_str.split(":"))
    end_h, end_m     = map(int, end_str.split(":"))

    start_dt = datetime(dt.year, dt.month, dt.day, start_h, start_m)
    end_dt   = datetime(dt.year, dt.month, dt.day, end_h, end_m)

    # Parse busy slots into datetime ranges
    busy_ranges = []
    for b in busy_slots:
        b_start = datetime.fromisoformat(b["start"].replace("Z", "+00:00")).replace(tzinfo=None)
        b_end   = datetime.fromisoformat(b["end"].replace("Z", "+00:00")).replace(tzinfo=None)
        # Adjust for IST (UTC+5:30) if needed — approximate, since no timezone handling in MVP
        b_start = b_start + timedelta(hours=5, minutes=30)
        b_end   = b_end   + timedelta(hours=5, minutes=30)
        busy_ranges.append((b_start, b_end))

    slot_capacity = settings.get("slot_capacity", 1)

    # Count existing confirmed bookings per slot from our own DB
    existing_bookings_res = supabase.table("appointments")         .select("start_time")         .eq("project_id", settings.get("project_id", ""))         .eq("appointment_date", date_str)         .in_("status", ["confirmed", "rescheduled"])         .execute()

    # Build a count map: {start_time_str: count}
    booking_counts = {}
    for b in (existing_bookings_res.data or []):
        t = str(b["start_time"])[:5]  # "HH:MM"
        booking_counts[t] = booking_counts.get(t, 0) + 1

    # Generate slots
    slots = []
    current = start_dt
    now = datetime.now()

    while current + timedelta(minutes=duration) <= end_dt:
        slot_end = current + timedelta(minutes=duration)
        slot_str = current.strftime("%H:%M")

        # Skip past slots
        if current <= now:
            current += timedelta(minutes=duration + buffer)
            continue

        # Check capacity — if existing bookings >= capacity, slot is full
        existing_count = booking_counts.get(slot_str, 0)
        if existing_count >= slot_capacity:
            current += timedelta(minutes=duration + buffer)
            continue

        # Check Google Calendar busy only if capacity is 1 (exclusive slots)
        # For capacity > 1, Google Calendar is used as a personal block-out only
        is_busy = False
        if slot_capacity == 1:
            for b_start, b_end in busy_ranges:
                if not (slot_end <= b_start or current >= b_end):
                    is_busy = True
                    break
        else:
            # For capacity > 1, only block if entire capacity would be exceeded
            # Google Calendar events still block the slot completely (owner blocked)
            for b_start, b_end in busy_ranges:
                if not (slot_end <= b_start or current >= b_end):
                    is_busy = True
                    break

        if not is_busy:
            remaining = slot_capacity - existing_count
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

    if not res.ok:
        return f"<html><body><script>window.opener.postMessage({{type:'GOOGLE_AUTH',event:'ERROR',error:'{res.text}'}}, '*'); window.close();</script></body></html>"

    token_data = res.json()
    refresh_token = token_data.get("refresh_token")

    if not refresh_token:
        return f"<html><body><script>window.opener.postMessage({{type:'GOOGLE_AUTH',event:'ERROR',error:'No refresh token returned. Try disconnecting and reconnecting.'}}, '*'); window.close();</script></body></html>"

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

    return f"<html><body><script>window.opener.postMessage({{type:'GOOGLE_AUTH',event:'FINISH'}}, '*'); window.close();</script></body></html>"


@router.delete("/appointments/google/disconnect/{project_id}")
def google_disconnect(project_id: str, user=Depends(verify_token)):
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
def public_settings(project_id: str):
    res = supabase.table("appointment_settings").select(
        "service_name,duration_minutes,working_hours,advance_booking_days,accent_color,is_enabled,google_refresh_token,google_calendar_id"
    ).eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        raise HTTPException(status_code=404, detail="Booking not configured")
    data = res.data
    if not data.get("is_enabled"):
        raise HTTPException(status_code=403, detail="Booking not enabled")
    data.pop("google_refresh_token", None)
    return data


def get_available_slots(project_id: str, date: str) -> list:
    """
    Core slot-lookup logic — used by the public booking page AND by the
    in-chat booking tool (backend/chat.py). Single source of truth so both
    paths can never disagree about what's actually free.
    Raises ValueError on a bad date or missing settings.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD")

    res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        raise ValueError("Appointment settings not found for this project")

    settings = res.data
    refresh_token = settings.get("google_refresh_token")
    calendar_id   = settings.get("google_calendar_id", "primary")

    busy_slots = []
    if refresh_token:
        access_token = get_google_access_token(refresh_token)
        if access_token:
            busy_slots = get_busy_slots(access_token, calendar_id, date)

    our_appointments = supabase.table("appointments") \
        .select("start_time, end_time") \
        .eq("project_id", project_id) \
        .eq("appointment_date", date) \
        .neq("status", "cancelled") \
        .execute()

    for appt in (our_appointments.data or []):
        busy_slots.append({
            "start": f"{date}T{appt['start_time']}Z",
            "end":   f"{date}T{appt['end_time']}Z",
        })

    settings["project_id"] = project_id
    return generate_slots(date, settings, busy_slots)


def create_appointment(
    project_id: str,
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
    unconditionally once called. Raises ValueError if settings are missing
    or the slot is no longer free (re-checked here, not trusted from the
    caller, since an AI-proposed slot could be stale by the time it's used).
    """
    from whatsapp import send_whatsapp_buttons
    from config import WHATSAPP_TOKEN

    res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        raise ValueError("Appointment settings not found for this project")

    settings = res.data
    duration  = settings.get("duration_minutes", 30)
    service   = settings.get("service_name", "Appointment")

    # Re-validate the slot is still actually free — never trust a
    # previously-computed slot list as still true at execution time.
    if not reschedule_id:
        available = get_available_slots(project_id, appointment_date)
        available_times = {s.split(" ")[0] for s in available}  # strip "(N left)" suffix
        if start_time not in available_times:
            raise ValueError(f"{start_time} on {appointment_date} is no longer available")

    start_dt = datetime.strptime(f"{appointment_date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt   = start_dt + timedelta(minutes=duration)
    end_time = end_dt.strftime("%H:%M")

    google_event_id = None
    refresh_token   = settings.get("google_refresh_token")
    calendar_id     = settings.get("google_calendar_id", "primary")

    if refresh_token:
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
            old_appt = supabase.table("appointments").select("*").eq("id", reschedule_id).maybe_single().execute()
            if old_appt and old_appt.data:
                supabase.table("appointments").update({"status": "rescheduled"}).eq("id", reschedule_id).execute()
                old_refresh = (supabase.table("appointment_settings").select("google_refresh_token,google_calendar_id").eq("project_id", project_id).maybe_single().execute())
                old_settings = (old_refresh.data if old_refresh else None) or {}
                if old_settings.get("google_refresh_token") and old_appt.data.get("google_event_id"):
                    old_token = get_google_access_token(old_settings["google_refresh_token"])
                    if old_token:
                        delete_google_event(old_token, old_settings.get("google_calendar_id", "primary"), old_appt.data["google_event_id"])
        except Exception as e:
            print(f"Reschedule old appointment error: {e}")

    appt_res = supabase.table("appointments").insert({
        "project_id": project_id,
        "customer_name": customer_name,
        "customer_phone": customer_phone.replace("+", ""),
        "service_name": service,
        "appointment_date": appointment_date,
        "start_time": start_time,
        "end_time": end_time,
        "status": "confirmed",
        "google_event_id": google_event_id,
        "notes": notes,
    }).execute()

    appointment = appt_res.data[0]

    date_obj = datetime.strptime(appointment_date, "%Y-%m-%d")
    date_formatted = date_obj.strftime("%A, %d %B %Y")

    try:
        wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", project_id).maybe_single().execute()
        wa_data = (wa_res.data if wa_res else None)

        if wa_data:
            phone_number_id = wa_data["phone_number_id"]
            token = wa_data.get("access_token") or WHATSAPP_TOKEN
            phone = customer_phone.replace("+", "").replace(" ", "")

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

            supabase.table("whatsapp_sessions").upsert({
                "project_id": project_id,
                "phone_number": phone,
                "mode": "appointment_confirmed",
                "metadata": {"appointment_id": appointment["id"]},
            }, on_conflict="project_id,phone_number").execute()

    except Exception as e:
        print(f"WhatsApp confirmation error: {e}")

    return {
        "status": "confirmed",
        "appointment_id": appointment["id"],
        "date": date_formatted,
        "time": start_time,
        "service": service,
    }


@router.get("/public/appointments/{project_id}/slots")
def public_slots(project_id: str, date: str):
    """Get available slots for a specific date."""
    try:
        slots = get_available_slots(project_id, date)
    except ValueError as e:
        status = 400 if "format" in str(e) else 404
        raise HTTPException(status_code=status, detail=str(e))
    return {"date": date, "slots": slots}


@router.post("/public/appointments/book")
def book_appointment(body: BookingCreate):
    """Create a new appointment."""
    try:
        return create_appointment(
            project_id=body.project_id,
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
# APPOINTMENTS CRUD (dashboard)
# -------------------------------------------------
@router.get("/appointments")
def list_appointments(project_id: str, user=Depends(verify_token)):
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
            print(f"Cancel notification error: {e}")

    supabase.table("appointments").update({"status": "cancelled"}).eq("id", appointment_id).execute()
    res = supabase.table("appointments").select("*").eq("id", appointment_id).single().execute()
    return res.data


@router.put("/appointments/{appointment_id}")
def update_appointment(appointment_id: str, body: AppointmentStatusUpdate, user=Depends(verify_token)):
    if body.status == "cancelled":
        try:
            return cancel_appointment(appointment_id, notify_customer=True)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    supabase.table("appointments").update({"status": body.status}).eq("id", appointment_id).execute()
    res = supabase.table("appointments").select("*").eq("id", appointment_id).single().execute()
    return res.data


# -------------------------------------------------
# REMINDER SENDER (called by a cron job or manually)
# -------------------------------------------------
@router.post("/appointments/send-reminders")
def send_reminders(user=Depends(verify_token)):
    """Send reminders for upcoming appointments. Call this every hour via a cron job."""
    now = datetime.now()
    sent = 0
    failed = 0

    # Get all confirmed, unreminded appointments
    appts_res = supabase.table("appointments") \
        .select("*") \
        .eq("status", "confirmed") \
        .eq("reminder_sent", False) \
        .execute()

    for appt in (appts_res.data or []):
        try:
            settings_res = supabase.table("appointment_settings").select("reminder_hours").eq("project_id", appt["project_id"]).maybe_single().execute()
            settings = (settings_res.data if settings_res else None) or {}
            reminder_hours = settings.get("reminder_hours", 24)

            # Parse start_time — handle both "HH:MM:SS" and "HH:MM" formats
            start_time_str = str(appt["start_time"])[:5]
            appt_dt = datetime.strptime(f"{appt['appointment_date']} {start_time_str}", "%Y-%m-%d %H:%M")
            hours_until = (appt_dt - now).total_seconds() / 3600

            if 0 < hours_until <= reminder_hours:
                wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", appt["project_id"]).maybe_single().execute()
                wa_data = (wa_res.data if wa_res else None)

                if wa_data:
                    phone_number_id = wa_data["phone_number_id"]
                    token = wa_data.get("access_token") or WHATSAPP_TOKEN
                    date_obj = datetime.strptime(appt["appointment_date"], "%Y-%m-%d")
                    date_formatted = date_obj.strftime("%d %B %Y")

                    # Try approved template first (works outside 24hr window)
                    template_sent = _send_reminder_template(
                        to=appt["customer_phone"],
                        customer_name=appt["customer_name"],
                        date=date_formatted,
                        time=start_time_str,
                        phone_number_id=phone_number_id,
                        token=token,
                    )

                    if not template_sent:
                        # Fallback to plain text (only works within 24hr window)
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
            print(f"Reminder error for {appt['id']}: {e}")
            failed += 1

    return {"status": "done", "reminders_sent": sent, "failed": failed}


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
    from datetime import datetime
    now = datetime.now()
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
                print(f"Reminder job error for appointment {appt.get('id')}: {e}")
                failed += 1

        print(f"Reminder job done — sent: {sent}, failed: {failed}")

    except Exception as e:
        print(f"Reminder job fatal error: {e}")