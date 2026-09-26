import hmac
import os
import re
import secrets
import sentry_sdk
import requests

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from clients import supabase
from ratelimit import is_rate_limited
from webhook_dedup import already_processed
from auth import verify_token, require_project_access
from config import TELEGRAM_WEBHOOK_SECRET
from usage import check_rate_limit, increment_usage
from chat import run_chat, get_history
from text_split import split_message

router = APIRouter()

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

# Telegram bot tokens are "<bot_id>:<35 char auth string>". Bounding this
# matters because the token is interpolated straight into the outbound URL:
# an unbounded string became an unbounded request to api.telegram.org.
_BOT_TOKEN_RE = re.compile(r"^\d{5,20}:[A-Za-z0-9_-]{30,50}$")


class TelegramConnectRequest(BaseModel):
    bot_token: str = Field(..., min_length=1, max_length=100)
    projectId: str = Field(..., min_length=1, max_length=64)


def _require_uuid(project_id: str) -> str:
    """A non-UUID used to reach Postgres and come back as a 500 carrying
    driver detail. Same helper shape as leads.py."""
    if not project_id or not _UUID_RE.match(project_id):
        raise HTTPException(status_code=400, detail="Invalid project id.")
    return project_id


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def send_telegram_message(bot_token: str, chat_id: int, text: str) -> bool:
    """Returns True if Telegram accepted every part. Text over Telegram's
    4096-char cap (e.g. a full list from a spreadsheet) is sent as several
    messages, split between lines."""
    for part in split_message(text, 4096):
        if not _send_telegram_part(bot_token, chat_id, part):
            return False
    return True


def _send_telegram_part(bot_token: str, chat_id: int, text: str) -> bool:
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

def set_telegram_webhook(bot_token: str, webhook_url: str, secret: str):
    """Registers the webhook and the secret Telegram will echo back.

    The secret is now per project rather than one global value shared by
    every bot. A single shared secret meant anyone who learned it could
    forge updates into ANY project's webhook URL and burn that project's AI
    quota; WhatsApp never had this problem because it HMACs the body.
    """
    url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    payload = {"url": webhook_url}
    if secret:
        payload["secret_token"] = secret
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
def telegram_connect(body: TelegramConnectRequest, user=Depends(verify_token)):
    # Was `data: dict` with raw data["bot_token"] / data["projectId"]
    # subscripts, so a body missing either key was an uncaught KeyError and
    # a 500, and the token had no length or charset bound at all.
    bot_token = body.bot_token.strip()
    project_id = _require_uuid(body.projectId)
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")

    if not _BOT_TOKEN_RE.match(bot_token):
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a Telegram bot token. Copy the whole token BotFather gave you.",
        )

    # Each connect makes two outbound calls to api.telegram.org, so without
    # a cap an authenticated admin could drive unlimited third-party traffic
    # (and probe tokens) through us.
    if is_rate_limited(f"telegram-connect:{project_id}", limit=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a minute.")

    bot_info = get_bot_info(bot_token)
    if not bot_info.get("ok"):
        raise HTTPException(status_code=400, detail="Invalid bot token.")

    bot_username = bot_info["result"]["username"]

    # Nothing stopped two projects registering the same bot. Telegram allows
    # only ONE webhook per bot, so the second connect silently repointed it:
    # the first project's bot began answering from the second's knowledge
    # base and billing them, with no error shown to either. Same collision
    # WhatsApp already guards against on phone_number_id.
    conflict = (
        supabase.table("telegram_integrations")
        .select("project_id")
        .eq("bot_token", bot_token)
        .neq("project_id", project_id)
        .execute()
    )
    if conflict.data:
        raise HTTPException(
            status_code=409,
            detail="This bot is already connected to a different Zavo project. Disconnect it there first.",
        )

    # A fresh secret per connect, so a leaked secret can be rotated by
    # reconnecting and can never be replayed against another project.
    webhook_secret = secrets.token_urlsafe(32)

    supabase.table("telegram_integrations").upsert({
        "project_id": project_id,
        "bot_token": bot_token,
        "bot_username": bot_username,
        "webhook_secret": webhook_secret,
    }, on_conflict="project_id").execute()

    webhook_url = f"{os.getenv('BACKEND_PUBLIC_URL')}/webhook/telegram/{project_id}"
    result = set_telegram_webhook(bot_token, webhook_url, webhook_secret)

    if not result.get("ok"):
        print(f"Telegram setWebhook failed: {result}")
        raise HTTPException(
            status_code=400,
            detail="Could not register the webhook with Telegram. Please check your bot token and try again.",
        )

    return {"success": True, "bot_username": bot_username}


@router.delete("/telegram/disconnect/{project_id}")
def telegram_disconnect(project_id: str, user=Depends(verify_token)):
    _require_uuid(project_id)
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")

    if is_rate_limited(f"telegram-disconnect:{project_id}", limit=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a minute.")

    res = supabase.table("telegram_integrations") \
        .select("bot_token") \
        .eq("project_id", project_id) \
        .execute()

    # .single() raised on zero rows rather than returning empty, so
    # disconnecting an already-disconnected project was a 500.
    if res.data:
        try:
            requests.post(
                f"https://api.telegram.org/bot{res.data[0]['bot_token']}/deleteWebhook",
                timeout=15,
            )
        except Exception as e:
            # The local row still gets deleted below; a failure to reach
            # Telegram shouldn't block the merchant from disconnecting.
            sentry_sdk.capture_message(f"Telegram deleteWebhook failed: {type(e).__name__}")

    supabase.table("telegram_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}


@router.get("/telegram/status/{project_id}")
def telegram_status(project_id: str, user=Depends(verify_token)):
    _require_uuid(project_id)
    require_project_access(user.id, project_id, tab="integrations")
    # bot_token and webhook_secret are deliberately absent — the frontend
    # never needs either, and the token is the full bot credential.
    res = supabase.table("telegram_integrations") \
        .select("bot_username") \
        .eq("project_id", project_id) \
        .execute()

    if res.data:
        return {"connected": True, "bot_username": res.data[0]["bot_username"]}
    return {"connected": False}


def _secret_matches(header_token: str, row: dict) -> bool:
    """Constant-time check of Telegram's echoed secret.

    The per-project secret is authoritative once a project has one. The
    global TELEGRAM_WEBHOOK_SECRET is accepted ONLY for rows that predate
    the per-project column, so bots connected before this change keep
    working and silently upgrade the next time they are reconnected. Once a
    project has its own secret the global one no longer opens its webhook,
    which is the whole point: a single shared secret let anyone who learned
    it forge updates into every project at once.
    """
    if not header_token:
        return False

    project_secret = (row or {}).get("webhook_secret")
    if project_secret:
        return hmac.compare_digest(header_token, project_secret)

    if TELEGRAM_WEBHOOK_SECRET:
        return hmac.compare_digest(header_token, TELEGRAM_WEBHOOK_SECRET)

    # No secret anywhere: fail closed rather than accepting anything.
    return False


def _process_telegram_update(project_id: str, body: dict, bot_token: str, bot_username: str) -> dict:
    """The slow half: DB writes, the LLM call, and the outbound send.

    Runs in a worker thread. This used to sit directly inside the async
    handler, so every Telegram message froze the single event loop for the
    duration of an LLM round trip — the dashboard, WhatsApp and Razorpay
    webhooks all queued behind one chat message. whatsapp.py already splits
    its webhook this way for exactly this reason.
    """
    try:
        message = body.get("message") or body.get("edited_message")
        if not message or "text" not in message:
            return {"status": "ignored"}

        text = message["text"]
        chat_id = message["chat"]["id"]
        chat_type = message["chat"]["type"]
        telegram_user_id = str(message["from"]["id"])
        username = message["from"].get("username") or message["from"].get("first_name", "User")

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

        # Quota check BEFORE the chats insert. It used to run after, so a
        # project already over its monthly limit still created a new chats
        # row for every fresh sender — unbounded table growth for messages
        # that were never going to be answered.
        rate_check = check_rate_limit(project_id)
        if not rate_check["allowed"]:
            send_telegram_message(bot_token, chat_id, "⚠️ Monthly message limit reached. Please try again next month.")
            return {"status": "rate_limited"}

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


@router.post("/webhook/telegram/{project_id}")
async def telegram_webhook(project_id: str, req: Request):
    if not _UUID_RE.match(project_id or ""):
        raise HTTPException(status_code=403, detail="Invalid secret token")

    # Coarse cap before the integration lookup, so an unauthenticated
    # caller can't turn this endpoint into an unbounded database read.
    if is_rate_limited(f"telegram-hook-raw:{project_id}", limit=300, window_seconds=60):
        return {"status": "rate_limited"}

    res = supabase.table("telegram_integrations") \
        .select("bot_token, bot_username, webhook_secret") \
        .eq("project_id", project_id) \
        .execute()

    row = res.data[0] if res.data else None

    # Same 403 whether the secret is wrong or the project has no
    # integration, so this can't be used to enumerate connected projects.
    header_token = req.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not row or not _secret_matches(header_token, row):
        raise HTTPException(status_code=403, detail="Invalid secret token")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    message = body.get("message") or body.get("edited_message") or {}
    sender_id = str((message.get("from") or {}).get("id") or "")

    # Two buckets, matching whatsapp.py. Previously there was only a
    # per-project cap, so a single abusive sender consumed the whole
    # project's budget and locked out every other user of that bot.
    if sender_id and is_rate_limited(
        f"telegram-in:{project_id}:{sender_id}", limit=15, window_seconds=60
    ):
        return {"status": "rate_limited"}

    if is_rate_limited(f"telegram-webhook:{project_id}", limit=120, window_seconds=60):
        return {"status": "rate_limited"}

    # Dedup runs AFTER the rate limit on purpose. It used to run before,
    # which meant a throttled update was still recorded as processed — so
    # Telegram's redelivery of it was discarded as a duplicate and the
    # message was lost permanently rather than merely delayed.
    update_id = body.get("update_id")
    if update_id and already_processed("telegram", f"{project_id}:{update_id}"):
        return {"status": "duplicate_ignored"}

    return await run_in_threadpool(
        _process_telegram_update,
        project_id,
        body,
        row["bot_token"],
        row["bot_username"],
    )