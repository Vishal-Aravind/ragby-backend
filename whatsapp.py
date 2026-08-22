import hmac
import hashlib
import sentry_sdk
import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import PlainTextResponse

from clients import supabase
from config import WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN, META_APP_ID, META_APP_SECRET
from auth import verify_token, require_project_role

router = APIRouter()


def verify_meta_signature(raw_body: bytes, signature_header: str) -> bool:
    """Meta signs every real webhook POST with X-Hub-Signature-256
    (sha256=<hex>, HMAC-SHA256 of the raw body using the app secret).
    Without this check, anyone who learns a project's phone_number_id could
    POST a fully spoofed message that still triggers a real OpenAI-costing
    chat reply — unlike Slack's webhook (see slack.py's
    verify_slack_signature) and Stripe's, this one had no verification at
    all. Fails closed: a missing/misconfigured secret rejects the request
    rather than silently accepting everything."""
    if not signature_header or not META_APP_SECRET:
        return False
    expected = "sha256=" + hmac.new(META_APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def _get_or_create_chat(project_id: str, external_id: str) -> str:
    """Shared by the live-message path and the coexistence history/echo
    ingestion paths — same (project_id, external_id, channel) key used
    everywhere else in this codebase for WhatsApp chats."""
    chat = supabase.table("chats") \
        .select("id") \
        .eq("project_id", project_id) \
        .eq("external_id", external_id) \
        .eq("channel", "whatsapp") \
        .limit(1) \
        .execute()
    if chat.data:
        return chat.data[0]["id"]
    new_chat = supabase.table("chats").insert({
        "project_id": project_id,
        "external_id": external_id,
        "channel": "whatsapp",
        "title": f"WhatsApp {external_id}",
    }).execute()
    return new_chat.data[0]["id"]


# -------------------------------------------------
# COEXISTENCE — history/contacts sync + live echoes
# -------------------------------------------------
def _extract_message_content(msg_type: str, msg: dict) -> str:
    if msg_type == "text":
        return msg.get("text", {}).get("body", "")
    if msg_type == "media_placeholder":
        # Meta only retains retrievable media for 14 days after a number
        # connects via coexistence — anything older arrives as a stub with
        # no asset. Say so honestly rather than storing an empty message.
        return "[Media message — not available. Meta only syncs media sent within 14 days of connecting.]"
    if msg_type in ("image", "video", "document", "audio", "sticker"):
        caption = msg.get(msg_type, {}).get("caption")
        return f"[{msg_type}]" + (f" {caption}" if caption else "")
    if msg_type == "revoke":
        return "[Message deleted]"
    if msg_type == "edit":
        return msg.get("edit", {}).get("text", {}).get("body") or "[Message edited]"
    return f"[{msg_type} message]"


def _save_synced_message(chat_id: str, wa_message_id, from_number: str, business_phone_number: str,
                          msg_type: str, content: str, created_at: str = None, prefix: str = ""):
    """Shared by history-sync ingestion and live message-echo ingestion.
    Upserts on (chat_id, wa_message_id) — Meta redelivers webhooks at-least-
    once, and a message could plausibly arrive via both history sync and a
    live webhook if timing overlaps, so this must be safe to call twice
    with the same message. One bad message must not abort an entire
    history-sync batch, so errors are caught and reported, not raised."""
    row = {
        "chat_id": chat_id,
        "role": "assistant" if from_number == business_phone_number else "user",
        "content": f"{prefix}{content}",
        "message_type": msg_type,
    }
    if wa_message_id:
        row["wa_message_id"] = wa_message_id
    if created_at:
        row["created_at"] = created_at
    try:
        if wa_message_id:
            supabase.table("chat_messages").upsert(row, on_conflict="chat_id,wa_message_id").execute()
        else:
            supabase.table("chat_messages").insert(row).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"WhatsApp synced-message save error: {e}")


def _handle_history_sync(project_id: str, business_phone_number: str, history_items: list):
    """One-time backfill of a client's pre-existing chat history, delivered
    in phases/chunks after initiate_coexistence_sync() requests it.
    Declining to share history is a normal, expected outcome — it's the
    business's own choice on their phone — handled as its own status, not
    folded into 'failed'."""
    from datetime import datetime, timezone

    for item in history_items:
        errors = item.get("errors")
        if errors:
            title = errors[0].get("title") or ""
            status = "declined" if "turned off" in title.lower() or "declined" in title.lower() else "failed"
            update = {"history_sync_status": status}
            if status == "failed":
                update["last_sync_error"] = title or "Unknown error"
            supabase.table("whatsapp_integrations").update(update).eq("project_id", project_id).execute()
            continue

        supabase.table("whatsapp_integrations") \
            .update({"history_sync_status": "in_progress"}) \
            .eq("project_id", project_id).eq("history_sync_status", "pending").execute()

        for thread in item.get("threads", []):
            contact_id = thread.get("id")
            if not contact_id:
                continue
            chat_id = _get_or_create_chat(project_id, contact_id)
            for msg in thread.get("messages", []):
                msg_type = msg.get("type", "text")
                ts = msg.get("timestamp")
                try:
                    created_at = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat() if ts else None
                except (TypeError, ValueError):
                    created_at = None
                _save_synced_message(
                    chat_id=chat_id,
                    wa_message_id=msg.get("id"),
                    from_number=msg.get("from"),
                    business_phone_number=business_phone_number,
                    msg_type=msg_type,
                    content=_extract_message_content(msg_type, msg),
                    created_at=created_at,
                )

        if item.get("metadata", {}).get("progress") == 100:
            supabase.table("whatsapp_integrations").update({
                "history_sync_status": "completed",
                "history_sync_completed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("project_id", project_id).execute()


def _handle_state_sync(project_id: str, state_sync_items: list):
    """The client's phone contact list, synced once on connect then
    incrementally as they add/edit contacts. 'remove' is intentionally a
    no-op: there's no safe, obviously-correct meaning for 'delete this
    lead' just because a phone contact was removed, so existing lead data
    is never touched on remove — only 'add' feeds upsert_contact."""
    from leads import upsert_contact
    for item in state_sync_items:
        if item.get("type") != "contact" or item.get("action") != "add":
            continue
        contact = item.get("contact", {})
        phone = contact.get("phone_number")
        if not phone:
            continue
        name = contact.get("full_name") or contact.get("first_name")
        upsert_contact(project_id, phone, name=name, channel="whatsapp")


def _handle_message_echoes(project_id: str, echoes: list, business_phone_number: str):
    """Messages the business owner sent/received directly from their own
    phone after connecting — keeps Zavo's own transcript accurate, and
    marks the chat as recently human-handled (see run_chat()'s best-effort
    bot-suppression check in chat.py)."""
    from datetime import datetime, timezone
    for echo in echoes:
        from_number = echo.get("from")
        contact_number = echo.get("to") if from_number == business_phone_number else from_number
        if not contact_number:
            continue
        chat_id = _get_or_create_chat(project_id, contact_number)
        msg_type = echo.get("type", "text")
        ts = echo.get("timestamp")
        try:
            created_at = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat() if ts else None
        except (TypeError, ValueError):
            created_at = None
        _save_synced_message(
            chat_id=chat_id,
            wa_message_id=echo.get("id"),
            from_number=from_number,
            business_phone_number=business_phone_number,
            msg_type=msg_type,
            content=_extract_message_content(msg_type, echo),
            created_at=created_at,
            prefix="[Business App] ",
        )
        if from_number == business_phone_number:
            supabase.table("chats").update({
                "last_human_reply_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", chat_id).execute()


def initiate_coexistence_sync(project_id: str, phone_number_id: str, access_token: str):
    """Called synchronously from whatsapp_onboard right after a coexistence
    completion — Meta gives a hard 24-hour window to request both syncs or
    the client must be fully offboarded and redo Embedded Signup (with up
    to 48h for deregistration to clear before they can retry), so this
    cannot ride on a queued background job with loose timing."""
    from datetime import datetime, timezone
    url = f"https://graph.facebook.com/v19.0/{phone_number_id}/smb_app_data"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    update = {
        "coexistence_enabled": True,
        "history_sync_status": "pending",
        "history_sync_requested_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        contacts_res = requests.post(url, headers=headers, json={
            "messaging_product": "whatsapp", "sync_type": "smb_app_state_sync",
        })
        if contacts_res.ok:
            update["contacts_sync_request_id"] = contacts_res.json().get("request_id")
        else:
            print(f"WhatsApp coexistence contacts-sync request failed: {contacts_res.text}")

        history_res = requests.post(url, headers=headers, json={
            "messaging_product": "whatsapp", "sync_type": "history",
        })
        if history_res.ok:
            update["history_sync_request_id"] = history_res.json().get("request_id")
        else:
            update["history_sync_status"] = "failed"
            update["last_sync_error"] = history_res.text[:500]
            print(f"WhatsApp coexistence history-sync request failed: {history_res.text}")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        update["history_sync_status"] = "failed"
        update["last_sync_error"] = str(e)[:500]

    supabase.table("whatsapp_integrations").update(update).eq("project_id", project_id).execute()


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
    # WhatsApp rejects the ENTIRE message if the body exceeds 1024 chars —
    # a last-resort safety net so a too-long body (e.g. a merchant-authored
    # flow node) fails as a truncated message, not a completely silent one.
    if len(body) > 1024:
        body = body[:1021] + "..."
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
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_meta_signature(raw_body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    body = await request.json()

    try:
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        if "statuses" in value:
            return {"status": "ignored"}

        # Coexistence fields (history/smb_app_state_sync/smb_message_echoes)
        # and normal inbound messages all need the same project lookup —
        # resolved once here, before branching.
        webhook_phone_number_id = value.get("metadata", {}).get("phone_number_id")
        res = supabase.table("whatsapp_integrations") \
            .select("project_id, phone_number_id") \
            .eq("phone_number_id", webhook_phone_number_id or WHATSAPP_PHONE_NUMBER_ID) \
            .execute()

        if not res.data:
            print(f"No project found for phone_number_id: {webhook_phone_number_id}")
            return {"status": "ignored"}

        project_id = res.data[0]["project_id"]
        phone_number_id = res.data[0]["phone_number_id"]
        token = WHATSAPP_TOKEN
        business_phone_number = value.get("metadata", {}).get("display_phone_number")

        if "history" in value:
            _handle_history_sync(project_id, business_phone_number, value["history"])
            return {"status": "ok"}

        if "state_sync" in value:
            _handle_state_sync(project_id, value["state_sync"])
            return {"status": "ok"}

        if "message_echoes" in value:
            _handle_message_echoes(project_id, value["message_echoes"], business_phone_number)
            return {"status": "ok"}

        messages = value.get("messages", [])
        if not messages:
            return {"status": "ignored"}

        message = messages[0]
        msg_type = message.get("type")
        from_number = message["from"]

        # WhatsApp includes the sender's real profile name on every message —
        # previously never captured anywhere, so bookings/orders had no real
        # name to fall back on and leads showed no name either.
        contacts = value.get("contacts", [])
        profile_name = contacts[0].get("profile", {}).get("name") if contacts else None

        # Get or create chat record
        chat_id = _get_or_create_chat(project_id, from_number)

        # Best-effort suppression for WhatsApp Coexistence — if the business
        # owner just replied to this conversation manually from their own
        # phone (signaled by a smb_message_echoes webhook, handled above),
        # skip the automatic bot reply so it doesn't talk over them. Not a
        # hard guarantee: the echo webhook arrives after the fact, not
        # synchronously with the owner's reply, so a narrow race is possible.
        chat_row = supabase.table("chats").select("last_human_reply_at").eq("id", chat_id).maybe_single().execute()
        last_human_reply_at = (chat_row.data or {}).get("last_human_reply_at") if chat_row else None
        if last_human_reply_at:
            from datetime import datetime, timezone, timedelta
            reply_time = datetime.fromisoformat(last_human_reply_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - reply_time < timedelta(minutes=5):
                return {"status": "suppressed_human_active"}

        from flows import get_session, handle_interactive, handle_text
        from leads import upsert_contact

        # Auto-save contact
        upsert_contact(project_id, from_number, name=profile_name, channel="whatsapp")

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
                handle_interactive(session, trigger, from_number, phone_number_id, token, project_id, chat_id)
            return {"status": "ok"}

        # ── Text message ──────────────────────────────────
        if msg_type == "text":
            text = message["text"]["body"].strip()
            session = get_session(project_id, from_number)
            handle_text(session, text, project_id, chat_id, from_number, phone_number_id, token)
            return {"status": "ok"}

        return {"status": "ignored"}

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"WHATSAPP WEBHOOK ERROR: {e}")
        return {"status": "error"}


# -------------------------------------------------
# MANAGEMENT ENDPOINTS
# -------------------------------------------------
@router.get("/whatsapp/status/{project_id}")
def whatsapp_status(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("whatsapp_integrations") \
        .select("phone_number_id, display_phone_number, waba_id") \
        .eq("project_id", project_id) \
        .execute()
    if res.data:
        return {"connected": True, **res.data[0]}
    return {"connected": False}


@router.post("/whatsapp/connect")
def whatsapp_connect(data: dict, user=Depends(verify_token)):
    require_project_role(user.id, data["projectId"])
    supabase.table("whatsapp_integrations").upsert({
        "project_id": data["projectId"],
        "phone_number_id": data["phone_number_id"],
        "waba_id": data.get("waba_id", ""),
        "display_phone_number": data.get("display_phone_number", ""),
    }, on_conflict="project_id").execute()
    return {"success": True}


@router.delete("/whatsapp/disconnect/{project_id}")
def whatsapp_disconnect(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)

    # Meta's normal Deregister API does not work on a coexistence-enabled
    # number, and its docs don't fully specify the alternative — rather
    # than ship an unconfirmed disconnect path for something explicitly
    # meant to protect client data, refuse and point them at the one
    # mechanism we know is correct: their own phone.
    existing = supabase.table("whatsapp_integrations").select("coexistence_enabled").eq("project_id", project_id).maybe_single().execute()
    if existing and (existing.data or {}).get("coexistence_enabled"):
        raise HTTPException(
            status_code=400,
            detail="This number is connected via WhatsApp Coexistence. Disconnect it from the WhatsApp Business App on your phone instead — Zavo can't safely disconnect a coexistence number without risking your chat history.",
        )

    supabase.table("whatsapp_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}


@router.get("/whatsapp/coexistence-status/{project_id}")
def whatsapp_coexistence_status(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("whatsapp_integrations") \
        .select("coexistence_enabled, history_sync_status, history_sync_requested_at, history_sync_completed_at, last_sync_error, phone_number_id") \
        .eq("project_id", project_id).maybe_single().execute()
    row = res.data if res else None
    if not row or not row.get("coexistence_enabled"):
        return {"coexistence_enabled": False}

    is_on_biz_app = None
    try:
        check_res = requests.get(
            f"https://graph.facebook.com/v19.0/{row['phone_number_id']}",
            params={"fields": "is_on_biz_app,platform_type", "access_token": WHATSAPP_TOKEN},
        )
        if check_res.ok:
            is_on_biz_app = check_res.json().get("is_on_biz_app")
    except Exception as e:
        sentry_sdk.capture_exception(e)

    return {
        "coexistence_enabled": True,
        "history_sync_status": row.get("history_sync_status"),
        "history_sync_requested_at": row.get("history_sync_requested_at"),
        "history_sync_completed_at": row.get("history_sync_completed_at"),
        "last_sync_error": row.get("last_sync_error"),
        "is_on_biz_app": is_on_biz_app,
    }


@router.post("/whatsapp/onboard")
def whatsapp_onboard(data: dict, user=Depends(verify_token)):
    code = data["code"]
    project_id = data["projectId"]
    is_coexistence = bool(data.get("isCoexistence"))
    require_project_role(user.id, project_id)

    token_res = requests.get(
        "https://graph.facebook.com/v19.0/oauth/access_token",
        params={"client_id": META_APP_ID, "client_secret": META_APP_SECRET, "code": code}
    )
    token_data = token_res.json()
    # FIX: was logging the full response, including the live access_token,
    # to stdout on every onboarding — never log the raw token response.
    if "access_token" not in token_data:
        print(f"WhatsApp token exchange failed: {token_data.get('error', token_data)}")
        raise HTTPException(status_code=400, detail="Token exchange failed")

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

    # Coexistence completion — request both syncs now, synchronously, not
    # via a queued job. Meta's 24-hour window to do this starts the moment
    # onboarding finishes, so this can't wait on background scheduling.
    if is_coexistence and phone_number_id:
        initiate_coexistence_sync(project_id, phone_number_id, access_token)

    return {
        "success": True,
        "phone_number_id": phone_number_id,
        "display_phone_number": display_phone,
        "waba_id": waba_id,
        "is_coexistence": is_coexistence,
    }


# -------------------------------------------------
# HUMAN REPLY ENDPOINT
# -------------------------------------------------
@router.post("/whatsapp/reply")
async def whatsapp_reply(data: dict, user=Depends(verify_token)):
    """Send a manual reply from the dashboard to a WhatsApp user."""
    project_id  = data["project_id"]
    phone_number = data["phone_number"]
    message     = data["message"]
    require_project_role(user.id, project_id)

    # Get WhatsApp integration for this project
    res = supabase.table("whatsapp_integrations") \
        .select("phone_number_id") \
        .eq("project_id", project_id) \
        .execute()

    if not res.data:
        raise HTTPException(status_code=404, detail="WhatsApp not connected")

    phone_number_id = res.data[0]["phone_number_id"]

    # Send message
    send_whatsapp_message(phone_number, message, phone_number_id, WHATSAPP_TOKEN)

    # Save to chat_messages so it appears in conversation
    chat = supabase.table("chats") \
        .select("id") \
        .eq("project_id", project_id) \
        .eq("external_id", phone_number) \
        .eq("channel", "whatsapp") \
        .limit(1) \
        .execute()

    if chat.data:
        supabase.table("chat_messages").insert({
            "chat_id": chat.data[0]["id"],
            "role": "assistant",
            "content": f"[Human] {message}",
        }).execute()

    return {"status": "sent"}