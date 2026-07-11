import asyncio
from datetime import datetime, timezone
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
# RECIPIENT RESOLUTION — shared by immediate + scheduled sends
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
    recipient_filter = data.get("recipient_filter", "all")  # all | whatsapp | web | tag
    tag_filter     = data.get("tag_filter", None)
    csv_contacts   = data.get("csv_contacts", None)
    scheduled_at   = data.get("scheduled_at", None)  # ISO string, optional

    # Recipients are resolved once, up front — the list a scheduled
    # campaign sends to is locked in at creation time, not re-computed
    # when it fires. Predictable: what you see when you schedule it is
    # exactly who gets it later.
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

    scheduled_dt_iso = None
    is_future = False
    if scheduled_at:
        try:
            scheduled_dt = datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid scheduled time")
        if scheduled_dt > datetime.now(timezone.utc):
            is_future = True
            scheduled_dt_iso = scheduled_dt.isoformat()

    campaign_res = supabase.table("campaigns").insert({
        "project_id": project_id,
        "name": name,
        "template_name": template_name,
        "template_variables": {"variables": variables, "language": template_lang},
        "status": "scheduled" if is_future else "sending",
        "recipient_filter": recipient_filter,
        "tag_filter": tag_filter,
        "scheduled_at": scheduled_dt_iso,
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
    Runs every couple of minutes. Finds campaigns whose scheduled_at has
    arrived and actually sends them. Reuses the same background scheduler
    already running for appointment reminders, instead of introducing a
    second scheduling mechanism.
    """
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
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
            except Exception as e:
                print(f"dispatch_scheduled_campaigns error for {camp.get('id')}: {e}")

    except Exception as e:
        print(f"dispatch_scheduled_campaigns fatal error: {e}")


# -------------------------------------------------
# CANCEL A SCHEDULED CAMPAIGN
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
