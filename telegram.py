import hmac
import os
import sentry_sdk
import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from clients import supabase
from ratelimit import is_rate_limited
from webhook_dedup import already_processed
from auth import verify_token, require_project_role
from config import TELEGRAM_WEBHOOK_SECRET
from usage import check_rate_limit, increment_usage
from chat import run_chat, get_history

router = APIRouter()


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def send_telegram_message(bot_token: str, chat_id: int, text: str) -> bool:
    """Returns True if Telegram accepted the message.

    The response used to be discarded, so two common failures were silent:
    a revoked token (401) or a blocked bot (403), and an LLM answer
    containing unbalanced Markdown (*, _, `, [) which Telegram rejects with
    a 400. In every case the customer got nothing while usage was still
    charged. On a Markdown failure we retry once as plain text.

    Exceptions are scrubbed before logging: the bot token is in the URL, so
    a raw requests error string would put it in Sentry and stdout.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    for parse_mode in ("Markdown", None):
        payload = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            res = requests.post(url, json=payload, timeout=15)
            if res.ok:
                return True
            # 400 usually means malformed Markdown; anything else won't be
            # fixed by dropping formatting, so stop.
            if res.status_code != 400:
                print(f"Telegram sendMessage failed ({res.status_code}): {res.text[:200]}")
                return False
        except Exception as e:
            sentry_sdk.capture_message(f"Telegram sendMessage error: {type(e).__name__}")
            print(f"Telegram sendMessage error: {type(e).__name__}")
            return False

    print("Telegram sendMessage failed even without Markdown")
    return False

def set_telegram_webhook(bot_token: str, webhook_url: str):
    url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    payload = {"url": webhook_url}
    # Telegram echoes this back as a header on every real webhook call, so
    # we can verify a request genuinely came from Telegram — previously
    # nothing checked this at all, unlike WhatsApp's webhook (now fixed) and
    # Slack's (already had it).
    if TELEGRAM_WEBHOOK_SECRET:
        payload["secret_token"] = TELEGRAM_WEBHOOK_SECRET
    res = requests.post(url, json=payload, timeout=15)
    return res.json()

def get_bot_info(bot_token: str):
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    res = requests.get(url, timeout=15)
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
        print(f"Telegram setWebhook failed: {result}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not connect to Telegram — {result.get('description', 'please check your bot token and try again.')}"
        )

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
        requests.post(f"https://api.telegram.org/bot{res.data['bot_token']}/deleteWebhook", timeout=15)

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

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")

    # Telegram redelivers an update until it is acknowledged, and run_chat
    # is a slow synchronous LLM call — so a redelivery meant a duplicate
    # reply and duplicate OpenAI spend.
    update_id = body.get("update_id")
    if update_id and already_processed("telegram", f"{project_id}:{update_id}"):
        return {"status": "duplicate_ignored"}

    # No per-minute cap existed here, only the monthly quota, so anyone who
    # found the bot could drain a project's whole allowance in seconds.
    if is_rate_limited(f"telegram-webhook:{project_id}", limit=20, window_seconds=60):
        return {"status": "rate_limited"}

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
        delivered = send_telegram_message(bot_token, chat_id, result["answer"])
        # Only bill for a message the customer actually received.
        if delivered:
            increment_usage(project_id)
        return {"status": "ok" if delivered else "send_failed"}

    except Exception as e:
        sentry_sdk.capture_exception(e)
        # str(e) on a requests failure contains the full API URL, which has
        # the bot token embedded in the path.
        print(f"TELEGRAM WEBHOOK ERROR: {type(e).__name__}")
        return {"status": "error"}