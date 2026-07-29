import asyncio
import calendar
import sentry_sdk
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from clients import supabase
from auth import verify_token
from config import WHATSAPP_TOKEN

router = APIRouter()


# -------------------------------------------------
# FETCH TEMPLATES FROM META
# -------------------------------------------------
@router.get("/campaigns/templates")
def get_templates(project_id: str, user=Depends(verify_token)):
    """Fetch approved templates from Meta for this project's WABA."""
    import requests

    wa = supabase.table("whatsapp_integrations") \
        .select("waba_id") \
        .eq("project_id", project_id) \
        .execute()

    if not wa.data or not wa.data[0].get("waba_id"):
        raise HTTPException(status_code=400, detail="WhatsApp not connected")

    waba_id = wa.data[0]["waba_id"]

    res = requests.get(
        f"https://graph.facebook.com/v19.0/{waba_id}/message_templates",
        params={
            "fields": "name,status,components,language",
            "limit": 50,
            "access_token": WHATSAPP_TOKEN,
        }
    )

    if not res.ok:
        raise HTTPException(status_code=400, detail="Failed to fetch templates from Meta")

    data = res.json()
    templates = data.get("data", [])

    # Filter only approved templates
    approved = [t for t in templates if t.get("status") == "APPROVED"]
    return approved


# -------------------------------------------------
# RECIPIENT RESOLUTION — shared by immediate + scheduled + recurring sends
# -------------------------------------------------
def resolve_contacts(project_id: str, recipient_filter: str, tag_filter, csv_contacts):
    if csv_contacts is not None:
        contacts = [{"phone": c["phone"], "name": c.get("name", "")} for c in csv_contacts if c.get("phone")]

        # Save to leads table (upsert) — happens immediately at creation
        # time even for a scheduled campaign, so they show up in Leads
        # right away rather than only once the campaign fires.
        from leads import upsert_contact
        for c in contacts:
            upsert_contact(project_id, c["phone"], c.get("name") or None, channel="whatsapp")

        return [c for c in contacts if c.get("phone", "").strip()]

    query = supabase.table("leads") \
        .select("phone, name") \
        .eq("project_id", project_id) \
        .neq("phone", "")

    if recipient_filter == "whatsapp":
        query = query.eq("channel", "whatsapp")
    elif recipient_filter == "web":
        query = query.eq("channel", "web")
    elif recipient_filter == "tag" and tag_filter:
        query = query.contains("tags", [tag_filter])

    contacts_res = query.execute()
    contacts = contacts_res.data or []
    return [c for c in contacts if c.get("phone", "").strip()]


VALID_RECURRENCES = {"daily", "weekly", "monthly"}


def _next_occurrence(dt: datetime, recurrence: str) -> datetime:
    if recurrence == "daily":
        return dt + timedelta(days=1)
    if recurrence == "weekly":
        return dt + timedelta(weeks=1)
    if recurrence == "monthly":
        month = dt.month + 1
        year = dt.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return dt.replace(year=year, month=month, day=day)
    raise ValueError(f"Unknown recurrence: {recurrence}")


def _parse_and_validate_schedule(scheduled_at, recurrence, recipient_filter):
    """Shared by create + edit. Returns (scheduled_dt_iso, is_future, recurrence)."""
    if recurrence and recurrence not in VALID_RECURRENCES:
        raise HTTPException(status_code=400, detail="Invalid recurrence")
    if recurrence and recipient_filter == "csv":
        raise HTTPException(status_code=400, detail="Recurring campaigns can't use a one-time CSV upload — pick a live filter instead")

    if not scheduled_at:
        return None, False, None

    try:
        scheduled_dt = datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid scheduled time")

    if scheduled_dt <= datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Scheduled time must be in the future")

    return scheduled_dt.isoformat(), True, (recurrence if recurrence else None)


# -------------------------------------------------
# CREATE + SEND (or SCHEDULE) CAMPAIGN
# -------------------------------------------------
@router.post("/campaigns")
async def create_campaign(data: dict, background_tasks: BackgroundTasks, user=Depends(verify_token)):
    project_id     = data["project_id"]
    name           = data["name"]
    template_name  = data["template_name"]
    template_lang  = data.get("template_language", "en_US")
    variables      = data.get("variables", [])  # list of values for {{1}}, {{2}} etc
    recipient_filter = data.get("recipient_filter", "all")  # all | whatsapp | web | tag | csv
    tag_filter     = data.get("tag_filter", None)
    csv_contacts   = data.get("csv_contacts", None)
    scheduled_at   = data.get("scheduled_at", None)  # ISO string, optional
    recurrence     = data.get("recurrence", None)    # None | daily | weekly | monthly

    # Recipients are resolved once, up front — the list a one-time
    # scheduled campaign sends to is locked in at creation time. Recurring
    # campaigns re-resolve fresh on every occurrence instead (see
    # dispatch_scheduled_campaigns) so new matching leads get included.
    contacts = resolve_contacts(project_id, recipient_filter, tag_filter, csv_contacts)
    if not contacts:
        raise HTTPException(status_code=400, detail="No contacts found for this filter")

    wa = supabase.table("whatsapp_integrations") \
        .select("phone_number_id") \
        .eq("project_id", project_id) \
        .execute()

    if not wa.data:
        raise HTTPException(status_code=400, detail="WhatsApp not connected")

    phone_number_id = wa.data[0]["phone_number_id"]

    scheduled_dt_iso, is_future, recurrence = _parse_and_validate_schedule(scheduled_at, recurrence, recipient_filter)

    campaign_res = supabase.table("campaigns").insert({
        "project_id": project_id,
        "name": name,
        "template_name": template_name,
        "template_variables": {"variables": variables, "language": template_lang},
        "status": "scheduled" if is_future else "sending",
        "recipient_filter": recipient_filter,
        "tag_filter": tag_filter,
        "scheduled_at": scheduled_dt_iso,
        "recurrence": recurrence,
        "resolved_contacts": contacts if is_future else None,
        "phone_number_id": phone_number_id if is_future else None,
        "total_count": len(contacts),
        "sent_count": 0,
        "failed_count": 0,
    }).execute()

    campaign_id = campaign_res.data[0]["id"]

    if is_future:
        return {"id": campaign_id, "total": len(contacts), "status": "scheduled", "scheduled_at": scheduled_dt_iso}

    background_tasks.add_task(
        send_campaign_messages,
        campaign_id, contacts, template_name, template_lang,
        variables, phone_number_id
    )

    return {"id": campaign_id, "total": len(contacts), "status": "sending"}


# -------------------------------------------------
# EDIT A SCHEDULED CAMPAIGN (before it fires)
# -------------------------------------------------
@router.put("/campaigns/{campaign_id}")
def update_campaign(campaign_id: str, data: dict, user=Depends(verify_token)):
    existing = supabase.table("campaigns").select("status, project_id").eq("id", campaign_id).maybe_single().execute()
    if not existing or not existing.data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if existing.data["status"] != "scheduled":
        raise HTTPException(status_code=400, detail="Only scheduled campaigns can be edited")

    project_id = existing.data["project_id"]
    name           = data["name"]
    template_name  = data["template_name"]
    template_lang  = data.get("template_language", "en_US")
    variables      = data.get("variables", [])
    recipient_filter = data.get("recipient_filter", "all")
    tag_filter     = data.get("tag_filter", None)
    csv_contacts   = data.get("csv_contacts", None)
    scheduled_at   = data.get("scheduled_at", None)
    recurrence     = data.get("recurrence", None)

    scheduled_dt_iso, is_future, recurrence = _parse_and_validate_schedule(scheduled_at, recurrence, recipient_filter)
    if not is_future:
        raise HTTPException(status_code=400, detail="Scheduled time is required when editing a scheduled campaign")

    # Re-resolve recipients — leads/tags may have changed since it was first created.
    contacts = resolve_contacts(project_id, recipient_filter, tag_filter, csv_contacts)
    if not contacts:
        raise HTTPException(status_code=400, detail="No contacts found for this filter")

    supabase.table("campaigns").update({
        "name": name,
        "template_name": template_name,
        "template_variables": {"variables": variables, "language": template_lang},
        "recipient_filter": recipient_filter,
        "tag_filter": tag_filter,
        "scheduled_at": scheduled_dt_iso,
        "recurrence": recurrence,
        "resolved_contacts": contacts,
        "total_count": len(contacts),
    }).eq("id", campaign_id).execute()

    return {"id": campaign_id, "total": len(contacts), "status": "scheduled", "scheduled_at": scheduled_dt_iso}


async def send_campaign_messages(
    campaign_id, contacts, template_name, template_lang,
    variables, phone_number_id
):
    """Send template messages to all contacts in background."""
    import requests
    import time

    sent = 0
    failed = 0

    for contact in contacts:
        phone = contact["phone"].strip().replace(" ", "").replace("-", "")
        if not phone.startswith("+"):
            phone = f"+{phone}"

        # Build components with variables
        components = []
        if variables:
            params = [{"type": "text", "text": str(v)} for v in variables]
            components.append({"type": "body", "parameters": params})

        try:
            res = requests.post(
                f"https://graph.facebook.com/v19.0/{phone_number_id}/messages",
                headers={
                    "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "messaging_product": "whatsapp",
                    "to": phone,
                    "type": "template",
                    "template": {
                        "name": template_name,
                        "language": {"code": template_lang},
                        "components": components,
                    },
                }
            )

            if res.ok:
                sent += 1
            else:
                failed += 1
                print(f"Campaign send error to {phone}: {res.text}")

        except Exception as e:
            sentry_sdk.capture_exception(e)
            failed += 1
            print(f"Campaign send exception to {phone}: {e}")

        # Rate limiting — WhatsApp allows ~80 messages/sec on low tier
        time.sleep(0.05)

    # Update campaign status
    supabase.table("campaigns").update({
        "status": "sent",
        "sent_count": sent,
        "failed_count": failed,
        "sent_at": "now()",
    }).eq("id", campaign_id).execute()


# -------------------------------------------------
# SCHEDULED CAMPAIGN DISPATCH — called by APScheduler in main.py
# -------------------------------------------------
def dispatch_scheduled_campaigns():
    """
    Runs every 30 seconds. Finds campaigns whose scheduled_at has arrived
    and sends them. Reuses the same background scheduler already running
    for appointment reminders, instead of introducing a second scheduling
    mechanism. If a campaign has a recurrence set, it's rescheduled to its
    next occurrence (with freshly re-resolved contacts) instead of being
    left in a final "sent" state.
    """
    try:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        due = supabase.table("campaigns") \
            .select("*") \
            .eq("status", "scheduled") \
            .lte("scheduled_at", now_iso) \
            .execute()

        for camp in (due.data or []):
            try:
                # Flip status first so a slow send (or a crash mid-send)
                # can't cause the same campaign to be picked up twice by
                # the next tick.
                supabase.table("campaigns").update({"status": "sending"}).eq("id", camp["id"]).execute()

                contacts = camp.get("resolved_contacts") or []
                tv = camp.get("template_variables") or {}
                variables = tv.get("variables", [])
                template_lang = tv.get("language", "en_US")
                phone_number_id = camp.get("phone_number_id")

                if not contacts or not phone_number_id:
                    print(f"Scheduled campaign {camp['id']} missing contacts or phone_number_id — skipping")
                    continue

                asyncio.run(send_campaign_messages(
                    camp["id"], contacts, camp["template_name"], template_lang,
                    variables, phone_number_id
                ))

                recurrence = camp.get("recurrence")
                if recurrence:
                    try:
                        current_dt = datetime.fromisoformat(str(camp["scheduled_at"]).replace("Z", "+00:00"))
                        next_dt = _next_occurrence(current_dt, recurrence)
                        fresh_contacts = resolve_contacts(
                            camp["project_id"], camp.get("recipient_filter", "all"), camp.get("tag_filter"), None
                        )
                        supabase.table("campaigns").update({
                            "status": "scheduled",
                            "scheduled_at": next_dt.isoformat(),
                            "resolved_contacts": fresh_contacts,
                            "total_count": len(fresh_contacts),
                        }).eq("id", camp["id"]).execute()
                    except Exception as e:
                        sentry_sdk.capture_exception(e)
                        print(f"Failed to reschedule recurring campaign {camp['id']}: {e}")

            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"dispatch_scheduled_campaigns error for {camp.get('id')}: {e}")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"dispatch_scheduled_campaigns fatal error: {e}")


# -------------------------------------------------
# CANCEL A SCHEDULED CAMPAIGN (also stops a recurring series)
# -------------------------------------------------
@router.post("/campaigns/{campaign_id}/cancel")
def cancel_campaign(campaign_id: str, user=Depends(verify_token)):
    camp = supabase.table("campaigns").select("status").eq("id", campaign_id).maybe_single().execute()
    if not camp or not camp.data or camp.data["status"] != "scheduled":
        raise HTTPException(status_code=400, detail="Only scheduled campaigns can be cancelled")

    supabase.table("campaigns").update({"status": "cancelled"}).eq("id", campaign_id).execute()
    return {"status": "cancelled"}


# -------------------------------------------------
# LIST CAMPAIGNS
# -------------------------------------------------
@router.get("/campaigns")
def list_campaigns(project_id: str, user=Depends(verify_token)):
    res = supabase.table("campaigns") \
        .select("*") \
        .eq("project_id", project_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data
