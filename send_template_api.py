"""
Send Template API — allows external systems to trigger WhatsApp template messages.
Merchants can call this from their own booking software, CRM, inventory systems etc.
Authentication is via the project's API key (stored in api_keys table).
"""
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from clients import supabase
from config import WHATSAPP_TOKEN
import requests

router = APIRouter()


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class SendTemplateRequest(BaseModel):
    to: str                          # recipient phone number e.g. "919150530543"
    template: str                    # template name e.g. "appointment_reminder"
    variables: Optional[List[str]] = []  # values for {{1}}, {{2}}, {{3}} etc
    language: Optional[str] = "en_US"


class SendTemplateResponse(BaseModel):
    status: str
    message_id: Optional[str] = None
    message: str


# -------------------------------------------------
# AUTH — API KEY VALIDATION
# -------------------------------------------------
def get_project_from_api_key(api_key: str) -> dict:
    """Validate API key and return project + WhatsApp integration details."""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required. Pass it as X-API-Key header.")

    # Look up API key
    key_res = supabase.table("api_keys") \
        .select("project_id, is_active") \
        .eq("key", api_key) \
        .maybe_single() \
        .execute()

    key_data = (key_res.data if key_res else None)
    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    if not key_data.get("is_active", True):
        raise HTTPException(status_code=401, detail="API key is inactive.")

    project_id = key_data["project_id"]

    # Get WhatsApp integration
    wa_res = supabase.table("whatsapp_integrations") \
        .select("phone_number_id, waba_id, access_token") \
        .eq("project_id", project_id) \
        .maybe_single() \
        .execute()

    wa_data = (wa_res.data if wa_res else None)
    if not wa_data or not wa_data.get("phone_number_id"):
        raise HTTPException(status_code=400, detail="WhatsApp not connected for this project.")

    return {
        "project_id": project_id,
        "phone_number_id": wa_data["phone_number_id"],
        "token": wa_data.get("access_token") or WHATSAPP_TOKEN,
    }


# -------------------------------------------------
# SEND TEMPLATE ENDPOINT
# -------------------------------------------------
@router.post("/api/send-template", response_model=SendTemplateResponse)
def send_template(
    body: SendTemplateRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    Send a WhatsApp template message to a single recipient.

    Example usage from external system:
    POST /api/send-template
    Headers: X-API-Key: your_api_key_here
    Body:
    {
        "to": "919150530543",
        "template": "appointment_reminder",
        "variables": ["John", "25 Dec 2026", "3:00 PM"]
    }
    """
    project = get_project_from_api_key(x_api_key)

    # Normalize phone number
    phone = body.to.replace("+", "").replace(" ", "").replace("-", "")
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number.")

    # Build template components
    components = []
    if body.variables:
        params = [{"type": "text", "text": str(v)} for v in body.variables]
        components.append({"type": "body", "parameters": params})

    # Send via WhatsApp API
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "template",
        "template": {
            "name": body.template,
            "language": {"code": body.language},
            "components": components,
        },
    }

    res = requests.post(
        f"https://graph.facebook.com/v19.0/{project['phone_number_id']}/messages",
        headers={
            "Authorization": f"Bearer {project['token']}",
            "Content-Type": "application/json",
        },
        json=payload,
    )

    resp = res.json()

    if res.ok:
        message_id = resp.get("messages", [{}])[0].get("id")

        # Log to a notifications table for tracking
        try:
            supabase.table("template_notifications").insert({
                "project_id": project["project_id"],
                "to_phone": phone,
                "template_name": body.template,
                "variables": body.variables,
                "message_id": message_id,
                "status": "sent",
            }).execute()
        except Exception as e:
            print(f"Notification log error (non-fatal): {e}")

        return SendTemplateResponse(
            status="sent",
            message_id=message_id,
            message=f"Template '{body.template}' sent successfully to +{phone}.",
        )
    else:
        error_msg = resp.get("error", {}).get("message", "Failed to send template.")
        print(f"Send template error: {resp}")
        raise HTTPException(status_code=400, detail=error_msg)


# -------------------------------------------------
# LIST AVAILABLE TEMPLATES ENDPOINT
# -------------------------------------------------
@router.get("/api/templates")
def list_available_templates(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    List all approved templates available for this project's WhatsApp account.
    Use these template names in the send-template endpoint.
    """
    project = get_project_from_api_key(x_api_key)

    # Get WABA ID
    wa_res = supabase.table("whatsapp_integrations") \
        .select("waba_id") \
        .eq("project_id", project["project_id"]) \
        .maybe_single() \
        .execute()

    wa_data = (wa_res.data if wa_res else None)
    waba_id = wa_data.get("waba_id") if wa_data else None

    if not waba_id:
        raise HTTPException(status_code=400, detail="WABA ID not found.")

    res = requests.get(
        f"https://graph.facebook.com/v19.0/{waba_id}/message_templates",
        params={
            "fields": "name,status,components,language",
            "limit": 50,
            "access_token": project["token"],
        }
    )

    if not res.ok:
        raise HTTPException(status_code=400, detail="Failed to fetch templates.")

    templates = res.json().get("data", [])
    approved = [
        {
            "name": t["name"],
            "status": t["status"],
            "language": t.get("language", "en_US"),
            "variables": len([
                p for c in t.get("components", [])
                if c["type"] == "BODY"
                for p in __import__("re").findall(r"\{\{\d+\}\}", c.get("text", ""))
            ]),
        }
        for t in templates
        if t.get("status") == "APPROVED"
    ]

    return {"templates": approved}


# -------------------------------------------------
# SEND BULK TEMPLATE (for scheduled notifications)
# -------------------------------------------------
@router.post("/api/send-template/bulk")
def send_template_bulk(
    body: dict,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    Send the same template to multiple recipients with different variables.

    Body:
    {
        "template": "appointment_reminder",
        "recipients": [
            {"to": "919150530543", "variables": ["John", "25 Dec", "3pm"]},
            {"to": "919884313063", "variables": ["Kumar", "26 Dec", "4pm"]}
        ]
    }
    """
    project = get_project_from_api_key(x_api_key)

    template = body.get("template")
    recipients = body.get("recipients", [])

    if not template:
        raise HTTPException(status_code=400, detail="template is required.")
    if not recipients:
        raise HTTPException(status_code=400, detail="recipients list is required.")
    if len(recipients) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 recipients per bulk call.")

    results = []
    for r in recipients:
        phone = r.get("to", "").replace("+", "").replace(" ", "").replace("-", "")
        variables = r.get("variables", [])

        components = []
        if variables:
            params = [{"type": "text", "text": str(v)} for v in variables]
            components.append({"type": "body", "parameters": params})

        res = requests.post(
            f"https://graph.facebook.com/v19.0/{project['phone_number_id']}/messages",
            headers={
                "Authorization": f"Bearer {project['token']}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "template",
                "template": {
                    "name": template,
                    "language": {"code": body.get("language", "en_US")},
                    "components": components,
                },
            }
        )

        resp = res.json()
        if res.ok:
            results.append({"to": phone, "status": "sent", "message_id": resp.get("messages", [{}])[0].get("id")})
        else:
            results.append({"to": phone, "status": "failed", "error": resp.get("error", {}).get("message", "Unknown error")})

        # Rate limiting
        import time
        time.sleep(0.05)

    sent = sum(1 for r in results if r["status"] == "sent")
    failed = sum(1 for r in results if r["status"] == "failed")

    return {
        "status": "completed",
        "total": len(recipients),
        "sent": sent,
        "failed": failed,
        "results": results,
    }