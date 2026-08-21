import hmac
import os
import sentry_sdk
import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from clients import supabase
from auth import verify_token, require_project_role
from config import TELEGRAM_WEBHOOK_SECRET
from usage import check_rate_limit, increment_usage
from chat import run_chat, get_history

router = APIRouter()


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def send_telegram_message(bot_token: str, chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def set_telegram_webhook(bot_token: str, webhook_url: str):
    url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    payload = {"url": webhook_url}
    # Telegram echoes this back as a header on every real webhook call, so
    # we can verify a request genuinely came from Telegram — previously
    # nothing checked this at all, unlike WhatsApp's webhook (now fixed) and
    # Slack's (already had it).
    if TELEGRAM_WEBHOOK_SECRET:
        payload["secret_token"] = TELEGRAM_WEBHOOK_SECRET
    res = requests.post(url, json=payload)
    return res.json()

def get_bot_info(bot_token: str):
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    res = requests.get(url)
    return res.json()


# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------
@router.post("/telegram/connect")
def telegram_connect(data: dict, user=Depends(verify_token)):
    bot_token = data["bot_token"]
    project_id = data["projectId"]
    require_project_role(user.id, project_id)

    bot_info = get_bot_info(bot_token)
    if not bot_info.get("ok"):
        raise HTTPException(status_code=400, detail="Invalid bot token.")

    bot_username = bot_info["result"]["username"]

    supabase.table("telegram_integrations").upsert({
        "project_id": project_id,
        "bot_token": bot_token,
        "bot_username": bot_username,
    }, on_conflict="project_id").execute()

    webhook_url = f"{os.getenv('BACKEND_PUBLIC_URL')}/webhook/telegram/{project_id}"
    result = set_telegram_webhook(bot_token, webhook_url)

    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=f"Could not set webhook: {result}")

    return {"success": True, "bot_username": bot_username}


@router.delete("/telegram/disconnect/{project_id}")
def telegram_disconnect(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("telegram_integrations") \
        .select("bot_token") \
        .eq("project_id", project_id) \
        .single() \
        .execute()

    if res.data:
        requests.post(f"https://api.telegram.org/bot{res.data['bot_token']}/deleteWebhook")

    supabase.table("telegram_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}


@router.get("/telegram/status/{project_id}")
def telegram_status(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("telegram_integrations") \
        .select("bot_username, created_at") \
        .eq("project_id", project_id) \
        .execute()

    if res.data:
        return {"connected": True, "bot_username": res.data[0]["bot_username"]}
    return {"connected": False}


@router.post("/webhook/telegram/{project_id}")
async def telegram_webhook(project_id: str, req: Request):
    # Fails closed if the secret isn't configured — same posture as the
    # WhatsApp webhook fix. Note: any Telegram bot connected BEFORE this
    # secret was set up needs to be disconnected and reconnected (via the
    # dashboard) so Telegram actually registers the secret_token — until
    # then its existing webhook registration has no secret to send, and
    # this check would reject it.
    header_token = req.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not TELEGRAM_WEBHOOK_SECRET or not header_token or not hmac.compare_digest(header_token, TELEGRAM_WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Invalid secret token")

    body = await req.json()

    try:
        message = body.get("message") or body.get("edited_message")
        if not message or "text" not in message:
            return {"status": "ignored"}

        text = message["text"]
        chat_id = message["chat"]["id"]
        chat_type = message["chat"]["type"]
        telegram_user_id = str(message["from"]["id"])
        username = message["from"].get("username") or message["from"].get("first_name", "User")

        res = supabase.table("telegram_integrations") \
            .select("bot_token, bot_username") \
            .eq("project_id", project_id) \
            .single() \
            .execute()

        if not res.data:
            return {"error": "integration not found"}

        bot_token = res.data["bot_token"]
        bot_username = res.data["bot_username"]

        if chat_type in ("group", "supergroup"):
            mention = f"@{bot_username}"
            if mention.lower() not in text.lower():
                return {"status": "ignored"}
            text = text.replace(mention, "").replace(mention.lower(), "").strip()
            if not text:
                send_telegram_message(bot_token, chat_id, "👋 Yes? Ask me anything!")
                return {"status": "ok"}

        if text.startswith("/start"):
            send_telegram_message(bot_token, chat_id, f"👋 Hi @{username}! I'm ready to help. Ask me anything!")
            return {"status": "ok"}

        if text.startswith("/"):
            return {"status": "ignored"}

        chat = supabase.table("chats") \
            .select("id") \
            .eq("project_id", project_id) \
            .eq("external_id", telegram_user_id) \
            .eq("channel", "telegram") \
            .limit(1) \
            .execute()

        if chat.data:
            chat_id_db = chat.data[0]["id"]
        else:
            new_chat = supabase.table("chats").insert({
                "project_id": project_id,
                "external_id": telegram_user_id,
                "channel": "telegram",
                "title": f"Telegram @{username}",
            }).execute()
            chat_id_db = new_chat.data[0]["id"]

        rate_check = check_rate_limit(project_id)
        if not rate_check["allowed"]:
            send_telegram_message(bot_token, chat_id, "⚠️ Monthly message limit reached. Please try again next month.")
            return {"status": "rate_limited"}

        history = get_history(chat_id_db, limit=5)
        result = run_chat(project_id, chat_id_db, text, history)
        send_telegram_message(bot_token, chat_id, result["answer"])
        increment_usage(project_id)
        return {"status": "ok"}

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"TELEGRAM WEBHOOK ERROR: {e}")
        return {"status": "error"}