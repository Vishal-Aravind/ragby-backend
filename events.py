"""
Event Registration system — for webinars, expos, workshops, demos, walk-in promos.
Merchant creates an event, broadcasts a rich card via WhatsApp,
customer registers via a public page, gets WhatsApp confirmation.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from clients import supabase
from auth import verify_token
from config import WHATSAPP_TOKEN, FRONTEND_URL
from datetime import datetime

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

class RegistrationCreate(BaseModel):
    event_id: str
    project_id: str
    data: dict   # dynamic — keyed by form field id, e.g. {"name": "John", "phone": "919...", "dietary": "Veg"}


# -------------------------------------------------
# MERCHANT — EVENT CRUD
# -------------------------------------------------
@router.get("/events")
def list_events(project_id: str, user=Depends(verify_token)):
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
    res = supabase.table("events").insert({
        "project_id": project_id,
        **body.dict(exclude_none=True),
    }).execute()
    return res.data[0]


@router.put("/events/{event_id}")
def update_event(event_id: str, body: EventUpdate, user=Depends(verify_token)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    supabase.table("events").update(update).eq("id", event_id).execute()
    res = supabase.table("events").select("*").eq("id", event_id).single().execute()
    return res.data


@router.delete("/events/{event_id}")
def delete_event(event_id: str, user=Depends(verify_token)):
    supabase.table("events").delete().eq("id", event_id).execute()
    return {"status": "deleted"}


@router.get("/events/{event_id}/registrations")
def list_registrations(event_id: str, user=Depends(verify_token)):
    res = supabase.table("event_registrations") \
        .select("*") \
        .eq("event_id", event_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data or []


@router.put("/events/registrations/{registration_id}")
def update_registration(registration_id: str, data: dict, user=Depends(verify_token)):
    status = data.get("status")
    if status:
        supabase.table("event_registrations").update({"status": status}).eq("id", registration_id).execute()
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
def register_for_event(body: RegistrationCreate):
    # Verify event exists and is open
    event_res = supabase.table("events").select("*").eq("id", body.event_id).maybe_single().execute()
    if not event_res or not event_res.data:
        raise HTTPException(status_code=404, detail="Event not found")

    event = event_res.data
    if not event.get("is_active"):
        raise HTTPException(status_code=403, detail="Registration closed")

    data = body.data or {}
    name = data.get("name", "")
    phone = str(data.get("phone", "")).replace("+", "").replace(" ", "")

    if not name or not phone:
        raise HTTPException(status_code=400, detail="Name and phone are required")

    # Check capacity
    if event.get("capacity"):
        count_res = supabase.table("event_registrations") \
            .select("id", count="exact") \
            .eq("event_id", body.event_id) \
            .neq("status", "cancelled") \
            .execute()
        if (count_res.count or 0) >= event["capacity"]:
            raise HTTPException(status_code=400, detail="Event is full")

    # Prevent duplicate registration by same phone
    existing = supabase.table("event_registrations") \
        .select("id") \
        .eq("event_id", body.event_id) \
        .eq("phone", phone) \
        .neq("status", "cancelled") \
        .maybe_single() \
        .execute()
    if existing and existing.data:
        raise HTTPException(status_code=400, detail="You are already registered for this event")

    # Insert into event_registrations (legacy columns, kept for compatibility)
    reg_res = supabase.table("event_registrations").insert({
        "event_id": body.event_id,
        "project_id": body.project_id,
        "name": name,
        "phone": phone,
        "email": data.get("email"),
        "notes": data.get("notes"),
    }).execute()

    registration = reg_res.data[0]

    # Also insert full dynamic data into form_submissions for custom fields
    try:
        supabase.table("form_submissions").insert({
            "entity_type": "event",
            "entity_id": body.event_id,
            "project_id": body.project_id,
            "data": data,
        }).execute()
    except Exception as e:
        print(f"form_submissions insert error: {e}")

    # Send WhatsApp confirmation
    try:
        wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", body.project_id).maybe_single().execute()
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

            send_whatsapp_message(
                to=phone,
                text=msg,
                phone_number_id=phone_number_id,
                token=token,
            )
    except Exception as e:
        print(f"Event registration WhatsApp confirmation error: {e}")

    return {
        "status": "confirmed",
        "registration_id": registration["id"],
        "event_title": event["title"],
    }