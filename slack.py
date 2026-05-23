import hmac
import hashlib
import time
import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from clients import supabase
from config import SLACK_CLIENT_ID, SLACK_CLIENT_SECRET, SLACK_SIGNING_SECRET, FRONTEND_URL
from auth import verify_token
from usage import check_rate_limit, increment_usage
from chat import run_chat, get_history

router = APIRouter()


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    if abs(time.time() - int(timestamp)) > 300:
        return False
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    my_sig = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(my_sig, signature)

def send_slack_message(access_token: str, channel: str, text: str):
    requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"channel": channel, "text": text}
    )


# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------
@router.get("/slack/auth-url")
def slack_auth_url(project_id: str, user=Depends(verify_token)):
    redirect_uri = f"{FRONTEND_URL}/api/slack/callback"
    scopes = "app_mentions:read,chat:write,channels:history,im:history,im:write"
    url = (
        f"https://slack.com/oauth/v2/authorize"
        f"?client_id={SLACK_CLIENT_ID}"
        f"&scope={scopes}"
        f"&redirect_uri={redirect_uri}"
        f"&state={project_id}"
    )
    return {"url": url}


@router.post("/slack/callback")
def slack_callback(data: dict):
    code = data["code"]
    project_id = data["project_id"]
    redirect_uri = f"{FRONTEND_URL}/api/slack/callback"

    res = requests.post("https://slack.com/api/oauth.v2.access", data={
        "client_id": SLACK_CLIENT_ID,
        "client_secret": SLACK_CLIENT_SECRET,
        "code": code,
        "redirect_uri": redirect_uri,
    })
    token_data = res.json()

    if not token_data.get("ok"):
        raise HTTPException(status_code=400, detail=f"Slack OAuth failed: {token_data.get('error')}")

    supabase.table("slack_integrations").upsert({
        "project_id": project_id,
        "access_token": token_data["access_token"],
        "team_id": token_data["team"]["id"],
        "team_name": token_data["team"]["name"],
        "bot_user_id": token_data["bot_user_id"],
    }, on_conflict="project_id").execute()

    return {"success": True, "team_name": token_data["team"]["name"]}


@router.get("/slack/status/{project_id}")
def slack_status(project_id: str, user=Depends(verify_token)):
    res = supabase.table("slack_integrations") \
        .select("team_name, team_id") \
        .eq("project_id", project_id) \
        .execute()
    if res.data:
        return {"connected": True, "team_name": res.data[0]["team_name"]}
    return {"connected": False}


@router.delete("/slack/disconnect/{project_id}")
def slack_disconnect(project_id: str, user=Depends(verify_token)):
    supabase.table("slack_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}


@router.post("/webhook/slack")
async def slack_webhook(req: Request):
    body_bytes = await req.body()
    body = await req.json()

    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}

    timestamp = req.headers.get("X-Slack-Request-Timestamp", "")
    signature = req.headers.get("X-Slack-Signature", "")
    if not verify_slack_signature(body_bytes, timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    event = body.get("event", {})
    event_type = event.get("type")

    if event_type not in ("app_mention", "message"):
        return {"status": "ignored"}

    if event.get("bot_id") or event.get("subtype"):
        return {"status": "ignored"}

    text = event.get("text", "").strip()
    channel = event.get("channel")
    user_id = event.get("user")
    team_id = body.get("team_id")

    if not text or not channel or not user_id:
        return {"status": "ignored"}

    res = supabase.table("slack_integrations") \
        .select("project_id, access_token, bot_user_id") \
        .eq("team_id", team_id) \
        .single() \
        .execute()

    if not res.data:
        return {"error": "integration not found"}

    project_id = res.data["project_id"]
    access_token = res.data["access_token"]
    bot_user_id = res.data["bot_user_id"]

    text = text.replace(f"<@{bot_user_id}>", "").strip()
    if not text:
        send_slack_message(access_token, channel, "👋 Yes? Ask me anything!")
        return {"status": "ok"}

    chat = supabase.table("chats") \
        .select("id") \
        .eq("project_id", project_id) \
        .eq("external_id", user_id) \
        .eq("channel", "slack") \
        .limit(1) \
        .execute()

    if chat.data:
        chat_id = chat.data[0]["id"]
    else:
        new_chat = supabase.table("chats").insert({
            "project_id": project_id,
            "external_id": user_id,
            "channel": "slack",
            "title": f"Slack {user_id}",
        }).execute()
        chat_id = new_chat.data[0]["id"]

    rate_check = check_rate_limit(project_id)
    if not rate_check["allowed"]:
        send_slack_message(access_token, channel, "⚠️ Monthly message limit reached. Please try again next month.")
        return {"status": "rate_limited"}

    history = get_history(chat_id, limit=5)
    result = run_chat(project_id, chat_id, text, history)
    send_slack_message(access_token, channel, result["answer"])
    increment_usage(project_id)
    return {"status": "ok"}