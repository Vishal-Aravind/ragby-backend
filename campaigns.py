import asyncio
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
# CREATE + SEND CAMPAIGN
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

    # Get contacts
    csv_contacts = data.get("csv_contacts", None)

    if csv_contacts is not None:
        # Use uploaded CSV contacts
        contacts = [{"phone": c["phone"], "name": c.get("name", "")} for c in csv_contacts if c.get("phone")]

        # Save to leads table (upsert)
        from leads import upsert_contact
        for c in contacts:
            upsert_contact(project_id, c["phone"], c.get("name") or None, channel="whatsapp")
    else:
        # Get from leads table
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

    # Filter out empty phones
    contacts = [c for c in contacts if c.get("phone", "").strip()]

    if not contacts:
        raise HTTPException(status_code=400, detail="No contacts found for this filter")

    # Get WhatsApp integration
    wa = supabase.table("whatsapp_integrations") \
        .select("phone_number_id") \
        .eq("project_id", project_id) \
        .execute()

    if not wa.data:
        raise HTTPException(status_code=400, detail="WhatsApp not connected")

    phone_number_id = wa.data[0]["phone_number_id"]

    # Create campaign record
    campaign_res = supabase.table("campaigns").insert({
        "project_id": project_id,
        "name": name,
        "template_name": template_name,
        "template_variables": {"variables": variables, "language": template_lang},
        "status": "sending",
        "recipient_filter": recipient_filter,
        "total_count": len(contacts),
        "sent_count": 0,
        "failed_count": 0,
    }).execute()

    campaign_id = campaign_res.data[0]["id"]

    # Send in background
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