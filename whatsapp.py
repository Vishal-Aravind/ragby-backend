import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import PlainTextResponse

from clients import supabase
from config import WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN, META_APP_ID, META_APP_SECRET
from auth import verify_token

router = APIRouter()


# -------------------------------------------------
# SEND HELPERS
# -------------------------------------------------
def send_whatsapp_message(to: str, text: str, phone_number_id: str = None, token: str = None):
    pid = phone_number_id or WHATSAPP_PHONE_NUMBER_ID
    tok = token or WHATSAPP_TOKEN
    url = f"https://graph.facebook.com/v19.0/{pid}/messages"
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    res = requests.post(url, headers=headers, json=payload)
    if not res.ok:
        print(f"WhatsApp send error: {res.text}")
    return res


def send_whatsapp_buttons(to: str, body: str, buttons: list, phone_number_id: str, token: str):
    url = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": btn["id"], "title": btn["title"]}}
                    for btn in buttons[:3]
                ]
            }
        }
    }
    res = requests.post(url, headers=headers, json=payload)
    if not res.ok:
        print(f"WhatsApp button send error: {res.text}")
    return res


def send_whatsapp_list(to: str, body: str, button_text: str, sections: list, phone_number_id: str, token: str):
    url = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": body},
            "action": {"button": button_text, "sections": sections}
        }
    }
    res = requests.post(url, headers=headers, json=payload)
    if not res.ok:
        print(f"WhatsApp list send error: {res.text}")
    return res


def send_whatsapp_cta_url(to: str, body: str, button_text: str, url_link: str, phone_number_id: str, token: str):
    url = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "cta_url",
            "body": {"text": body},
            "action": {
                "name": "cta_url",
                "parameters": {
                    "display_text": button_text,
                    "url": url_link.replace("{{phone_number}}", to)
                }
            }
        }
    }
    res = requests.post(url, headers=headers, json=payload)
    if not res.ok:
        print(f"WhatsApp CTA send error: {res.text}")
    return res


# -------------------------------------------------
# WEBHOOK VERIFY
# -------------------------------------------------
@router.get("/webhook/whatsapp")
async def whatsapp_verify(request: Request):
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        print("WhatsApp webhook verified")
        return PlainTextResponse(challenge)

    raise HTTPException(status_code=403, detail="Verification failed")


# -------------------------------------------------
# WEBHOOK HANDLER
# -------------------------------------------------
@router.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    body = await request.json()

    try:
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        if "statuses" in value:
            return {"status": "ignored"}

        messages = value.get("messages", [])
        if not messages:
            return {"status": "ignored"}

        message = messages[0]
        msg_type = message.get("type")
        from_number = message["from"]
        waba_id = value.get("metadata", {}).get("phone_number_id")

        # Look up project
        res = supabase.table("whatsapp_integrations") \
            .select("project_id, phone_number_id") \
            .eq("phone_number_id", waba_id or WHATSAPP_PHONE_NUMBER_ID) \
            .execute()

        if not res.data:
            print(f"No project found for phone_number_id: {waba_id}")
            return {"status": "ignored"}

        project_id = res.data[0]["project_id"]
        phone_number_id = res.data[0]["phone_number_id"]
        token = WHATSAPP_TOKEN

        # Get or create chat record
        chat = supabase.table("chats") \
            .select("id") \
            .eq("project_id", project_id) \
            .eq("external_id", from_number) \
            .eq("channel", "whatsapp") \
            .limit(1) \
            .execute()

        if chat.data:
            chat_id = chat.data[0]["id"]
        else:
            new_chat = supabase.table("chats").insert({
                "project_id": project_id,
                "external_id": from_number,
                "channel": "whatsapp",
                "title": f"WhatsApp {from_number}",
            }).execute()
            chat_id = new_chat.data[0]["id"]

        from flows import get_session, handle_interactive, handle_text

        # ── Interactive (button/list click) ──────────────
        if msg_type == "interactive":
            interactive = message.get("interactive", {})
            if interactive.get("type") == "button_reply":
                trigger = interactive["button_reply"]["id"]
            elif interactive.get("type") == "list_reply":
                trigger = interactive["list_reply"]["id"]
            else:
                return {"status": "ignored"}

            session = get_session(project_id, from_number)
            if session:
                handle_interactive(session, trigger, from_number, phone_number_id, token, project_id)
            return {"status": "ok"}

        # ── Text message ──────────────────────────────────
        if msg_type == "text":
            text = message["text"]["body"].strip()
            session = get_session(project_id, from_number)
            handle_text(session, text, project_id, chat_id, from_number, phone_number_id, token)
            return {"status": "ok"}

        return {"status": "ignored"}

    except Exception as e:
        print(f"WHATSAPP WEBHOOK ERROR: {e}")
        return {"status": "error"}


# -------------------------------------------------
# MANAGEMENT ENDPOINTS
# -------------------------------------------------
@router.get("/whatsapp/status/{project_id}")
def whatsapp_status(project_id: str, user=Depends(verify_token)):
    res = supabase.table("whatsapp_integrations") \
        .select("phone_number_id, display_phone_number, waba_id") \
        .eq("project_id", project_id) \
        .execute()
    if res.data:
        return {"connected": True, **res.data[0]}
    return {"connected": False}


@router.post("/whatsapp/connect")
def whatsapp_connect(data: dict, user=Depends(verify_token)):
    supabase.table("whatsapp_integrations").upsert({
        "project_id": data["projectId"],
        "phone_number_id": data["phone_number_id"],
        "waba_id": data.get("waba_id", ""),
        "display_phone_number": data.get("display_phone_number", ""),
    }, on_conflict="project_id").execute()
    return {"success": True}


@router.delete("/whatsapp/disconnect/{project_id}")
def whatsapp_disconnect(project_id: str, user=Depends(verify_token)):
    supabase.table("whatsapp_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}


@router.post("/whatsapp/onboard")
def whatsapp_onboard(data: dict, user=Depends(verify_token)):
    code = data["code"]
    project_id = data["projectId"]

    token_res = requests.get(
        "https://graph.facebook.com/v19.0/oauth/access_token",
        params={"client_id": META_APP_ID, "client_secret": META_APP_SECRET, "code": code}
    )
    token_data = token_res.json()
    print(f"Token exchange: {token_data}")

    if "access_token" not in token_data:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {token_data}")

    access_token = token_data["access_token"]

    waba_res = requests.get(
        "https://graph.facebook.com/v19.0/me/whatsapp_business_accounts",
        params={"access_token": access_token}
    )
    waba_id = waba_res.json().get("data", [{}])[0].get("id", "")

    phone_res = requests.get(
        f"https://graph.facebook.com/v19.0/{waba_id}/phone_numbers",
        params={"access_token": access_token}
    )
    phone_data = phone_res.json().get("data", [{}])[0]
    phone_number_id = phone_data.get("id", "")
    display_phone = phone_data.get("display_phone_number", "")

    supabase.table("whatsapp_integrations").upsert({
        "project_id": project_id,
        "phone_number_id": phone_number_id,
        "waba_id": waba_id,
        "display_phone_number": display_phone,
    }, on_conflict="project_id").execute()

    return {
        "success": True,
        "phone_number_id": phone_number_id,
        "display_phone_number": display_phone,
        "waba_id": waba_id,
    }