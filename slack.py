import sentry_sdk
import hmac
import hashlib
import secrets
import time
import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from clients import supabase
from oauth_state import issue_state, consume_state
from ratelimit import is_rate_limited
from webhook_dedup import already_processed
from config import SLACK_CLIENT_ID, SLACK_CLIENT_SECRET, SLACK_SIGNING_SECRET, FRONTEND_URL
from auth import verify_token, require_project_access
from usage import check_rate_limit, increment_usage
from chat import run_chat, get_history

router = APIRouter()

# Short-lived, single-use CSRF nonce for the OAuth handshake — same pattern
# as shopify_oauth.py/razorpay_oauth.py. Previously this used a bare
# state=project_id with nothing binding the callback to the request that
# issued it, and the callback itself had no auth at all — together that
# meant anyone who completed their own real Slack OAuth against this app
# could attach their workspace to any project_id just by POSTing here.






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

def send_slack_message(access_token: str, channel: str, text: str) -> bool:
    """Returns True if Slack actually accepted the message.

    Slack answers HTTP 200 with {"ok": false, "error": ...} for
    not_in_channel, invalid_auth, token_revoked and friends. The response
    used to be discarded entirely, so a revoked token produced a silent
    void: the customer got no reply and usage was still charged.
    """
    try:
        res = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"channel": channel, "text": text},
            timeout=10,
        )
        data = res.json()
        if not data.get("ok"):
            print(f"Slack chat.postMessage failed: {data.get('error')}")
            return False
        return True
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Slack chat.postMessage error: {e}")
        return False


# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------
@router.get("/slack/auth-url")
def slack_auth_url(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")
    redirect_uri = f"{FRONTEND_URL}/api/slack/callback"
    scopes = "app_mentions:read,chat:write,channels:history,im:history,im:write"
    state = issue_state("slack", project_id, user.id)
    url = (
        f"https://slack.com/oauth/v2/authorize"
        f"?client_id={SLACK_CLIENT_ID}"
        f"&scope={scopes}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
    )
    return {"url": url}


@router.post("/slack/callback")
def slack_callback(data: dict, user=Depends(verify_token)):
    code = data["code"]
    state_row = consume_state("slack", data.get("state"))
    if not state_row:
        raise HTTPException(status_code=400, detail="This connection link expired or was already used — please try connecting again.")
    # The nonce recorded only the project before, so any member could redeem
    # one another member had minted.
    if state_row["user_id"] != str(user.id):
        raise HTTPException(status_code=403, detail="This connection link was started by someone else.")
    project_id = state_row["project_id"]
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")
    redirect_uri = f"{FRONTEND_URL}/api/slack/callback"

    res = requests.post("https://slack.com/api/oauth.v2.access", timeout=15, data={
        "client_id": SLACK_CLIENT_ID,
        "client_secret": SLACK_CLIENT_SECRET,
        "code": code,
        "redirect_uri": redirect_uri,
    })
    token_data = res.json()

    if not token_data.get("ok"):
        # Codes like invalid_client_id / bad_redirect_uri describe OUR app's
        # misconfiguration and mean nothing to the merchant.
        print(f"Slack OAuth failed: {token_data.get('error')}")
        raise HTTPException(
            status_code=400,
            detail="Couldn't connect to Slack. Please try again.",
        )

    # Nothing stopped two projects installing into the same workspace; the
    # webhook then resolves by team_id and would pick one arbitrarily.
    team_id_new = (token_data.get("team") or {}).get("id")
    if team_id_new:
        conflict = (
            supabase.table("slack_integrations")
            .select("project_id")
            .eq("team_id", team_id_new)
            .neq("project_id", project_id)
            .execute()
        )
        if conflict.data:
            raise HTTPException(
                status_code=409,
                detail="This Slack workspace is already connected to a different Zavo project. Disconnect it there first.",
            )

    supabase.table("slack_integrations").upsert({
        "project_id": project_id,
        "access_token": token_data["access_token"],
        "team_id": token_data["team"]["id"],
        "team_name": token_data["team"]["name"],
        "bot_user_id": token_data["bot_user_id"],
    }, on_conflict="project_id").execute()

    return {"success": True, "team_name": token_data["team"]["name"], "project_id": project_id}


@router.get("/slack/status/{project_id}")
def slack_status(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="integrations")
    res = supabase.table("slack_integrations") \
        .select("team_name, team_id") \
        .eq("project_id", project_id) \
        .execute()
    if res.data:
        return {"connected": True, "team_name": res.data[0]["team_name"]}
    return {"connected": False}


@router.delete("/slack/disconnect/{project_id}")
def slack_disconnect(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")
    supabase.table("slack_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}


@router.post("/webhook/slack")
async def slack_webhook(req: Request):
    body_bytes = await req.body()

    # Signature first. url_verification used to be answered before this,
    # which made the endpoint an open reflector for unauthenticated callers,
    # and body["challenge"] was an unguarded index (missing key = 500).
    timestamp = req.headers.get("X-Slack-Request-Timestamp", "")
    signature = req.headers.get("X-Slack-Signature", "")
    if not verify_slack_signature(body_bytes, timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")

    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge", "")}

    # Slack retries when it doesn't get a 200 within 3 seconds, and run_chat
    # routinely takes longer — so without this the NORMAL path was up to four
    # identical replies and four times the OpenAI spend for one question.
    if req.headers.get("X-Slack-Retry-Num"):
        return {"status": "retry_ignored"}

    event = body.get("event", {})
    event_type = event.get("type")

    # Slack delivers BOTH app_mention and message.channels for a single
    # mention, so handling both double-charged every question. Only
    # app_mention is handled now, which also means the bot no longer replies
    # to every unrelated message in a channel it happens to be in.
    # A workspace that uninstalls the app used to leave a live row with a
    # dead token forever: the dashboard kept saying "Connected to {team}"
    # while every reply silently failed.
    if event_type in ("app_uninstalled", "tokens_revoked"):
        team_id = body.get("team_id")
        if team_id:
            supabase.table("slack_integrations").delete().eq("team_id", team_id).execute()
            print(f"Slack {event_type}: removed integration for team {team_id}")
        return {"status": "ok"}

    if event_type != "app_mention":
        return {"status": "ignored"}

    if event.get("bot_id") or event.get("subtype"):
        return {"status": "ignored"}

    # Belt and braces against redelivery that arrives without a retry header.
    event_id = body.get("event_id")
    if event_id and already_processed("slack", event_id):
        return {"status": "duplicate_ignored"}

    text = event.get("text", "").strip()
    channel = event.get("channel")
    user_id = event.get("user")
    team_id = body.get("team_id")

    if not text or not channel or not user_id:
        return {"status": "ignored"}

    res = supabase.table("slack_integrations") \
        .select("project_id, access_token, bot_user_id") \
        .eq("team_id", team_id) \
        .execute()

    # Was .single(), which RAISES on zero rows (workspace uninstalled, or
    # the project deleted mid-flight) and on two rows (same workspace
    # connected to two projects) — so this guard was dead code and either
    # case became a 500, which makes Slack retry the delivery.
    if not res.data:
        return {"status": "integration_not_found"}

    row = res.data[0]
    project_id = row["project_id"]
    access_token = row["access_token"]
    bot_user_id = row["bot_user_id"]

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

    # This endpoint had no per-minute cap at all — only the monthly quota —
    # so anyone in the workspace could drain a project's whole allowance.
    if is_rate_limited(f"slack-webhook:{project_id}", limit=20, window_seconds=60):
        return {"status": "rate_limited"}

    rate_check = check_rate_limit(project_id)
    if not rate_check["allowed"]:
        send_slack_message(access_token, channel, "⚠️ Monthly message limit reached. Please try again next month.")
        return {"status": "rate_limited"}

    history = get_history(chat_id, limit=5)
    result = run_chat(project_id, chat_id, text, history)
    delivered = send_slack_message(access_token, channel, result["answer"])
    # Only bill for a message the customer actually received.
    if delivered:
        increment_usage(project_id)
    return {"status": "ok" if delivered else "send_failed"}