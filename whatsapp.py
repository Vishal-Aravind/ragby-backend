import hmac
import hashlib
import random
import re
import sentry_sdk
import requests
import json
from typing import Optional

from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import PlainTextResponse

from clients import supabase
from ratelimit import is_rate_limited
from config import WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN, META_APP_ID, META_APP_SECRET
from auth import verify_token, require_project_access
from text_split import split_message

router = APIRouter()

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

# Digits only, 8-15, matching campaigns.py. WhatsApp stores numbers without
# a leading '+', and every write in this module normalises to that shape.
_PHONE_RE = re.compile(r"^\d{8,15}$")

# Meta's limit on a single text body. Also the bound we apply to anything
# a merchant types, so an oversized body fails here rather than at Meta.
MAX_TEXT_LEN = 4096

# The coexistence history sync replays a merchant's existing WhatsApp
# conversations. It is signed, so it is genuine Meta traffic, but it is
# also unbounded: one payload can carry an arbitrary number of threads and
# messages, each becoming a chat_messages row in a synchronous request.
MAX_SYNC_THREADS = 200
MAX_SYNC_MESSAGES_PER_THREAD = 500
MAX_SYNC_CONTENT_LEN = 8000
MAX_SYNC_ERROR_LEN = 500


def _require_uuid(project_id: str) -> str:
    """A non-UUID used to reach Postgres and come back as a 500 carrying
    driver detail. Same helper shape as leads.py."""
    if not project_id or not _UUID_RE.match(project_id):
        raise HTTPException(status_code=400, detail="Invalid project id.")
    return project_id


def _valid_phone(raw) -> Optional[str]:
    """Normalised digits, or None if this isn't a usable WhatsApp number.

    Every send path took `to` as a bare string and handed it straight to
    Meta, so an unvalidated value became an outbound message to whatever
    number the caller named.
    """
    if not raw or not isinstance(raw, str):
        return None
    cleaned = re.sub(r"[\s\-()+]", "", raw)
    return cleaned if _PHONE_RE.match(cleaned) else None


def _utc_now_iso() -> str:
    """Current time as an ISO string PostgREST will accept.

    Was the literal string "now()", which Postgres cannot cast to a
    timestamp — 'now' is valid input, 'now()' is not. Both call sites sit
    inside try/except blocks, so the write would have failed silently and
    the delivery failure would never have been recorded at all.
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _is_duplicate_key(e: Exception) -> bool:
    """True for a unique-violation.

    Was a bare `"duplicate key" in str(e).lower()`. That phrasing comes
    from the driver, not from us — a driver upgrade or a non-English
    locale would silently turn every redelivery back into a duplicate
    customer reply. Check the SQLSTATE too, which is stable.
    """
    text = str(e).lower()
    return "duplicate key" in text or "23505" in text


def _maybe_prune_wa_dedup() -> None:
    """Bound the dedup table.

    It holds one row per WhatsApp message ever received and nothing has
    ever deleted from it. Pruned opportunistically on a small fraction of
    inserts, the same approach webhook_dedup.py uses.
    """
    if random.random() >= 0.01:
        return
    try:
        supabase.rpc("prune_whatsapp_webhook_dedup", {}).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"whatsapp dedup prune failed: {type(e).__name__}")


class _TimeoutSession(requests.Session):
    """Applies a default timeout to every Graph API call.

    None of the outbound requests in this module set one, and Python's
    requests waits forever by default. A slow (not even down) Meta meant a
    pinned worker thread per call; on the webhook path that wedged the
    event loop and stalled the whole backend, dashboard included.
    """

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", 20)
        return super().request(*args, **kwargs)


http = _TimeoutSession()


def verify_meta_signature(raw_body: bytes, signature_header: str) -> bool:
    """Meta signs every real webhook POST with X-Hub-Signature-256
    (sha256=<hex>, HMAC-SHA256 of the raw body using the app secret).
    Without this check, anyone who learns a project's phone_number_id could
    POST a fully spoofed message that still triggers a real OpenAI-costing
    chat reply — unlike Stripe's webhook, this one had no verification at
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


def _sync_timestamp(ts) -> Optional[str]:
    """A synced message's created_at, or None to let the DB default.

    The timestamp comes from the webhook body and was only type-checked,
    so a synced row could be dated to 1970 or to the year 5000 — enough to
    sit permanently at the top or bottom of every conversation view. Clamp
    it to a sane window instead of trusting it.
    """
    from datetime import datetime, timezone
    if ts is None:
        return None
    try:
        parsed = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    now = datetime.now(timezone.utc)
    # WhatsApp launched in 2009; nothing legitimate predates it, and
    # nothing synced can be from the future.
    if parsed.year < 2009 or parsed > now:
        return None
    return parsed.isoformat()


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
                # Bounded. The other two writers of this column truncate to
                # 500; this one took webhook-supplied text at any length.
                update["last_sync_error"] = (title or "Unknown error")[:MAX_SYNC_ERROR_LEN]
            supabase.table("whatsapp_integrations").update(update).eq("project_id", project_id).execute()
            continue

        supabase.table("whatsapp_integrations") \
            .update({"history_sync_status": "in_progress"}) \
            .eq("project_id", project_id).eq("history_sync_status", "pending").execute()

        # Capped. This loop inserts one chat_messages row per message in a
        # synchronous request, with no ceiling on threads or messages and
        # every error swallowed by _save_synced_message — so a single
        # signed payload could write an unbounded number of rows with no
        # backpressure. The caps are generous enough for a real backfill.
        threads = item.get("threads") or []
        if len(threads) > MAX_SYNC_THREADS:
            print(f"History sync for {project_id}: {len(threads)} threads, capping at {MAX_SYNC_THREADS}")
        for thread in threads[:MAX_SYNC_THREADS]:
            contact_id = _valid_phone(thread.get("id"))
            if not contact_id:
                continue
            chat_id = _get_or_create_chat(project_id, contact_id)
            for msg in (thread.get("messages") or [])[:MAX_SYNC_MESSAGES_PER_THREAD]:
                if not isinstance(msg, dict):
                    continue
                msg_type = msg.get("type", "text")
                created_at = _sync_timestamp(msg.get("timestamp"))
                _save_synced_message(
                    chat_id=chat_id,
                    wa_message_id=msg.get("id"),
                    from_number=msg.get("from"),
                    business_phone_number=business_phone_number,
                    msg_type=msg_type,
                    content=_extract_message_content(msg_type, msg)[:MAX_SYNC_CONTENT_LEN],
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
    is never touched on remove — only 'add' feeds upsert_contact.

    This is an uncapped, full phone contact-list sync — potentially
    thousands of contacts in one call, unlike the campaign upload path's
    1,000-recipient ceiling. upsert_contact's own default behaviour is to
    capture a failure directly, which is right for a single inbound
    message but not for a loop this size: a systemic cause fails every
    contact identically. on_error routes failures here instead, counted
    and reported once."""
    from leads import upsert_contact
    contact_save_failures = 0
    for item in state_sync_items:
        if item.get("type") != "contact" or item.get("action") != "add":
            continue
        contact = item.get("contact", {})
        phone = contact.get("phone_number")
        if not phone:
            continue
        name = contact.get("full_name") or contact.get("first_name")

        def _count_failure(e):
            nonlocal contact_save_failures
            contact_save_failures += 1

        upsert_contact(project_id, phone, name=name, channel="whatsapp", on_error=_count_failure)

    if contact_save_failures:
        sentry_sdk.capture_message(
            f"_handle_state_sync: {contact_save_failures} contact(s) failed to save for project {project_id}",
            level="warning",
        )


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


def _handle_statuses(project_id: str, statuses: list) -> None:
    """Record delivery FAILURES against the conversation.

    sent/delivered/read are still ignored — tracking those would mean
    threading Meta's message id through every outbound path in campaigns,
    flows, appointments and events. Failures are the ones that matter,
    because nothing else in the product ever learns about them: the send
    helpers only print the error and their callers discard the response,
    so a message Meta refused showed in the dashboard as delivered.

    The common causes are all invisible today: the number isn't on
    WhatsApp, the template was rejected, the 24-hour customer service
    window has closed, or Meta has flagged the number for spam.
    """
    # NOT captured per status below — one webhook call can carry up to 50 of
    # these, and a systemic cause (the chats table unreachable) fails every
    # one of them identically. Counted instead, with a single capture_message
    # after the loop if any failed.
    write_failures = 0

    for status in statuses[:50]:
        if not isinstance(status, dict) or status.get("status") != "failed":
            continue

        recipient = _valid_phone(status.get("recipient_id"))
        if not recipient:
            continue

        errors = status.get("errors") or []
        first = errors[0] if errors and isinstance(errors[0], dict) else {}
        # Meta's own wording, bounded. It is shown to the merchant, so it
        # must not be able to grow without limit from a webhook body.
        reason = str(first.get("title") or first.get("message") or "Delivery failed")[:MAX_SYNC_ERROR_LEN]
        code = first.get("code")
        if code:
            reason = f"{reason} (Meta code {code})"

        try:
            supabase.table("chats") \
                .update({"last_send_error": reason, "last_send_error_at": _utc_now_iso()}) \
                .eq("project_id", project_id) \
                .eq("external_id", recipient) \
                .eq("channel", "whatsapp") \
                .execute()
            print(f"WhatsApp send failed to {recipient} on {project_id}: {reason}")
        except Exception as e:
            print(f"Could not record WhatsApp failure: {type(e).__name__}")
            write_failures += 1

    if write_failures:
        sentry_sdk.capture_message(
            f"_handle_statuses: failed to record {write_failures} delivery failure(s) for project {project_id}",
            level="warning",
        )


def initiate_coexistence_sync(project_id: str, phone_number_id: str, access_token: str):
    """Called synchronously from whatsapp_onboard right after a coexistence
    completion — Meta gives a hard 24-hour window to request both syncs or
    the client must be fully offboarded and redo Embedded Signup (with up
    to 48h for deregistration to clear before they can retry), so this
    cannot ride on a queued background job with loose timing."""
    from datetime import datetime, timezone
    url = f"https://graph.facebook.com/v25.0/{phone_number_id}/smb_app_data"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    update = {
        "coexistence_enabled": True,
        "history_sync_status": "pending",
        "history_sync_requested_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        contacts_res = http.post(url, headers=headers, json={
            "messaging_product": "whatsapp", "sync_type": "smb_app_state_sync",
        })
        if contacts_res.ok:
            update["contacts_sync_request_id"] = contacts_res.json().get("request_id")
        else:
            print(f"WhatsApp coexistence contacts-sync request failed: {contacts_res.text}")

        history_res = http.post(url, headers=headers, json={
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
def _integration_for_project(project_id: str) -> Optional[dict]:
    """The project's own phone_number_id and access token.

    access_token is preferred over the global WHATSAPP_TOKEN wherever it is
    present. Onboarding always obtained a real per-merchant token and then
    threw it away, so every send for every tenant went out on one shared
    system token: a single leaked env var meant send-as-any-merchant, and a
    merchant revoking our access had no effect at all.

    Rows written before the access_token column exists fall back to the
    global token, so a connection made earlier keeps working untouched and
    upgrades the next time it is reconnected.
    """
    res = supabase.table("whatsapp_integrations") \
        .select("phone_number_id, access_token") \
        .eq("project_id", project_id) \
        .limit(1) \
        .execute()
    return res.data[0] if res.data else None


def _token_for(row: Optional[dict]) -> str:
    return ((row or {}).get("access_token")) or WHATSAPP_TOKEN


def send_whatsapp_message(to: str, text: str, phone_number_id: str = None, token: str = None):
    pid = phone_number_id or WHATSAPP_PHONE_NUMBER_ID
    tok = token or WHATSAPP_TOKEN
    url = f"https://graph.facebook.com/v25.0/{pid}/messages"
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    # WhatsApp rejects a text message over 4096 chars outright, so an
    # over-long merchant-authored node body (or a long AI answer) would send
    # NOTHING rather than something. Long text (e.g. a full list from a
    # spreadsheet) is sent as several messages, split between lines, rather
    # than cut off. Returns the last response; stops at the first failure.
    res = None
    for part in split_message(text, MAX_TEXT_LEN):
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": part},
        }
        res = http.post(url, headers=headers, json=payload)
        if not res.ok:
            print(f"WhatsApp send error: {res.text}")
            break
    return res


def send_whatsapp_buttons(to: str, body: str, buttons: list, phone_number_id: str, token: str):
    url = f"https://graph.facebook.com/v25.0/{phone_number_id}/messages"
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
    res = http.post(url, headers=headers, json=payload)
    if not res.ok:
        print(f"WhatsApp button send error: {res.text}")
    return res


def send_whatsapp_media(to: str, media_type: str, media: dict, phone_number_id: str, token: str):
    """Send an image/video/document/audio/location/contacts message.

    flows.py used to build seven of these inline with a bare
    `import requests as req; req.post(...)`. That bypassed _TimeoutSession
    (so a slow Meta pinned a worker thread forever — the exact failure that
    wrapper exists to prevent) AND hardcoded the global WHATSAPP_TOKEN,
    ignoring the per-project token every other send path is given. A
    merchant on their own WABA had their media silently sent from ours.
    """
    url = f"https://graph.facebook.com/v25.0/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": media_type,
        media_type: media,
    }
    res = http.post(url, headers=headers, json=payload)
    if not res.ok:
        print(f"WhatsApp {media_type} send error: {res.text}")
    return res


def send_whatsapp_list(to: str, body: str, button_text: str, sections: list, phone_number_id: str, token: str):
    url = f"https://graph.facebook.com/v25.0/{phone_number_id}/messages"
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
    res = http.post(url, headers=headers, json=payload)
    if not res.ok:
        print(f"WhatsApp list send error: {res.text}")
    return res


def send_whatsapp_cta_url(to: str, body: str, button_text: str, url_link: str, phone_number_id: str, token: str):
    url = f"https://graph.facebook.com/v25.0/{phone_number_id}/messages"
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
    res = http.post(url, headers=headers, json=payload)
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

    # Failed OPEN before: WHATSAPP_VERIFY_TOKEN is None when unset, and a
    # request that also omits hub.verify_token gave None == None, so on a
    # misconfigured deploy an attacker could register this URL as their own
    # app's webhook. Requires both sides and compares in constant time.
    if (
        mode == "subscribe"
        and WHATSAPP_VERIFY_TOKEN
        and token
        and hmac.compare_digest(token, WHATSAPP_VERIFY_TOKEN)
    ):
        print("WhatsApp webhook verified")
        return PlainTextResponse(challenge or "")

    raise HTTPException(status_code=403, detail="Verification failed")


def _process_single_message(value, message, project_id, phone_number_id, token):
    """Handle ONE inbound message.

    Meta batches messages into a single webhook, but only messages[0] was
    ever read. The rest were dropped with no dedup row and no retry, and
    we returned 200 so Meta never resent them — a customer sending three
    quick messages got exactly one answer.

    Returns a per-message status; raises to let the caller release the
    dedup row and ask Meta to redeliver.
    """
    msg_type = message.get("type")
    # Was message["from"] — an unguarded subscript on attacker-shaped (but
    # correctly signed) data. It raised, the caller caught it, RELEASED the
    # dedup row and returned 500, so Meta redelivered the same malformed
    # message forever. One bad payload became a permanent retry storm.
    from_number = message.get("from")
    if not from_number:
        print("WhatsApp webhook: message with no 'from', ignoring")
        return {"status": "ignored"}

    # Dedup stops REPEATS of one message; it does nothing about a flood
    # of distinct ones. Signature verification means this needs real
    # WhatsApp traffic, but a single hostile sender could still burn a
    # project's whole monthly quota (and the matching OpenAI + Meta send
    # cost) in minutes. Per-sender first, then a project-wide ceiling.
    #
    # These run BEFORE the dedup insert on purpose. They used to run after,
    # and because a rate-limited message returns normally rather than
    # raising, the dedup row survived and Meta got a 200 — so the message
    # was never answered, never redelivered, and never even written to
    # chat_messages. A burst did not delay messages, it deleted them.
    # Same ordering bug telegram.py had.
    if is_rate_limited(f"wa-in:{project_id}:{from_number}", limit=15, window_seconds=60):
        print(f"WhatsApp inbound rate limited: {project_id} / {from_number}")
        return {"status": "rate_limited"}
    if is_rate_limited(f"wa-in:{project_id}", limit=120, window_seconds=60):
        print(f"WhatsApp inbound rate limited (project-wide): {project_id}")
        return {"status": "rate_limited"}

    # Meta redelivers a webhook at-least-once if we don't ack fast enough
    # or error transiently — without this, a redelivery re-triggers the
    # whole flow below and sends a duplicate reply. Confirmed live: the
    # same bot reply fired 6 extra times over ~2.5 hours with no new
    # customer message in between. First writer wins; a duplicate-key
    # violation here means we've already processed this exact message.
    wa_message_id = message.get("id")
    if wa_message_id:
        try:
            supabase.table("whatsapp_webhook_dedup").insert({"wa_message_id": wa_message_id}).execute()
            _maybe_prune_wa_dedup()
        except Exception as e:
            if _is_duplicate_key(e):
                return {"status": "duplicate_ignored"}
            raise

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
            # Record it before bailing out. The inbound message is otherwise
            # only saved inside handle_text, further down — so suppressing
            # the bot ALSO meant the customer's message never appeared in
            # the inbox, and the owner it was deferring to never saw the
            # thing they were supposed to answer.
            try:
                from chat import save_message
                if msg_type == "text":
                    save_message(chat_id, "user", message["text"]["body"].strip())
            except Exception as e:
                sentry_sdk.capture_exception(e)
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


# -------------------------------------------------
# WEBHOOK HANDLER
# -------------------------------------------------
@router.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_meta_signature(raw_body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        body = json.loads(raw_body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")

    # Everything below is blocking: Supabase queries, an OpenAI completion,
    # and outbound Graph calls. Running that directly in an async handler
    # occupied the event loop for the whole duration, so ONE inbound message
    # froze every other request the backend was serving, dashboard included
    # — and Meta's ack timeout then triggered redeliveries. run_in_threadpool
    # puts it on a worker thread where blocking is fine.
    return await run_in_threadpool(_process_webhook, body)


def _process_webhook(body: dict):
    wa_message_id = None
    try:
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        # Coexistence fields (history/smb_app_state_sync/smb_message_echoes)
        # and normal inbound messages all need the same project lookup —
        # resolved once here, before branching.
        webhook_phone_number_id = value.get("metadata", {}).get("phone_number_id")
        # Previously fell back to WHATSAPP_PHONE_NUMBER_ID when metadata was
        # absent, which routed the message to whichever project holds the
        # global/dev number - an unrelated tenant. Ignoring it is correct.
        if not webhook_phone_number_id:
            print("WhatsApp webhook: no phone_number_id in metadata, ignoring")
            return {"status": "ignored"}
        # Explicitly ordered and limited. This lookup decides which tenant
        # a customer conversation belongs to, and it used to take an
        # arbitrary res.data[0] from an unordered query — so if two rows
        # ever shared a phone_number_id (see the onboarding race), a share
        # of one merchant's real conversations would land in another's.
        # The migration adds the unique index that makes this unambiguous;
        # the ordering makes the behaviour deterministic regardless.
        res = supabase.table("whatsapp_integrations") \
            .select("project_id, phone_number_id, access_token") \
            .eq("phone_number_id", webhook_phone_number_id) \
            .order("created_at", desc=False) \
            .limit(1) \
            .execute()

        if not res.data:
            print(f"No project found for phone_number_id: {webhook_phone_number_id}")
            return {"status": "ignored"}

        project_id = res.data[0]["project_id"]
        phone_number_id = res.data[0]["phone_number_id"]
        token = _token_for(res.data[0])
        business_phone_number = value.get("metadata", {}).get("display_phone_number")

        # Delivery statuses used to return before this point, so a failed
        # send — wrong number, template rejected, 24h window violation,
        # spam block — was discarded before we even knew whose it was.
        if "statuses" in value:
            _handle_statuses(project_id, value.get("statuses") or [])
            return {"status": "ok"}

        # The three coexistence branches below each write rows and none of
        # them passed through dedup or any rate limit — they bypass
        # _process_single_message entirely. A generous per-project ceiling,
        # since a real backfill legitimately arrives in many chunks.
        if "history" in value or "state_sync" in value or "message_echoes" in value:
            if is_rate_limited(f"wa-coex:{project_id}", limit=60, window_seconds=60):
                print(f"WhatsApp coexistence sync rate limited: {project_id}")
                return {"status": "rate_limited"}

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

        # Process EVERY message in the batch, not just the first.
        results = []
        for message in messages:
            wa_message_id = message.get("id")
            results.append(
                _process_single_message(value, message, project_id, phone_number_id, token)
            )
        return {"status": "ok", "processed": len(results)}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"WHATSAPP WEBHOOK ERROR: {e}")
        # The dedup row was written before processing, so leaving it in place
        # meant a message that failed midway was marked "seen" forever: no
        # reply was sent, and returning 200 told Meta not to redeliver, so
        # the customer was silently ignored. Releasing it lets the retry
        # through; the 500 is what prompts Meta to send one.
        if wa_message_id:
            try:
                supabase.table("whatsapp_webhook_dedup").delete().eq("wa_message_id", wa_message_id).execute()
            except Exception as cleanup_error:
                # A DISTINCT failure from the one just captured above — if
                # this delete itself fails, the dedup row stays stuck, so
                # Meta's retry is wrongly treated as a duplicate and the
                # message is dropped for good this time, with nothing
                # telling anyone why.
                sentry_sdk.capture_exception(cleanup_error)
        raise HTTPException(status_code=500, detail="Processing failed")


# -------------------------------------------------
# MANAGEMENT ENDPOINTS
# -------------------------------------------------
@router.get("/whatsapp/status/{project_id}")
def whatsapp_status(project_id: str, user=Depends(verify_token)):
    _require_uuid(project_id)
    require_project_access(user.id, project_id, tab="integrations")
    res = supabase.table("whatsapp_integrations") \
        .select("phone_number_id, display_phone_number, waba_id") \
        .eq("project_id", project_id) \
        .execute()
    if res.data:
        return {"connected": True, **res.data[0]}
    return {"connected": False}


# REMOVED: /whatsapp/connect.
# It upserted a caller-supplied phone_number_id with no ownership proof and
# no conflict check (unlike whatsapp_onboard, which rejects a number already
# bound to another project). Since the webhook resolves the project by
# phone_number_id and takes the first row of an unordered query, an attacker
# could bind a victim's number and receive a share of that merchant's real
# customer conversations, answered from the attacker's knowledge base.
# The frontend deliberately stopped calling it (see the note in
# IntegrationsTab.js), so there was no behaviour left to preserve.


@router.delete("/whatsapp/disconnect/{project_id}")
def whatsapp_disconnect(project_id: str, user=Depends(verify_token)):
    _require_uuid(project_id)
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")

    # Meta's normal Deregister API does not work on a coexistence-enabled
    # number, and its docs don't fully specify the alternative — rather
    # than ship an unconfirmed disconnect path for something explicitly
    # meant to protect client data, refuse and point them at the one
    # mechanism we know is correct: their own phone.
    #
    # Checked two ways, not one: coexistence_enabled directly, AND
    # history_sync_status as a redundant signal (anything other than the
    # not-a-coexistence-connection defaults) — belt-and-suspenders after a
    # real incident where a plain coexistence_enabled check alone let a
    # disconnect through it should have blocked, root cause not yet fully
    # confirmed. Logged explicitly either way so a repeat is diagnosable
    # from Render logs instead of another guessing round.
    existing = supabase.table("whatsapp_integrations") \
        .select("coexistence_enabled, history_sync_status, phone_number_id, waba_id") \
        .eq("project_id", project_id).maybe_single().execute()
    row = (existing.data if existing else None) or {}
    print(f"WhatsApp disconnect check: project={project_id}, coexistence_enabled={row.get('coexistence_enabled')}, history_sync_status={row.get('history_sync_status')}")

    is_coexistence_connection = bool(row.get("coexistence_enabled")) or \
        row.get("history_sync_status") not in (None, "not_applicable", "declined")
    if is_coexistence_connection:
        # coexistence_enabled is set once at connect time and nothing ever
        # clears it — disconnecting from the phone happens entirely inside
        # WhatsApp Business App, with no webhook telling us it happened, so
        # a customer who correctly unlinked their phone was stuck here
        # forever. Check live with Meta (same call coexistence-status
        # already uses) before blocking — only lift the block on an
        # explicit False; any error or missing field fails safe and keeps
        # blocking, same as before.
        confirmed_unlinked = False
        phone_number_id = row.get("phone_number_id")
        if phone_number_id:
            try:
                check_res = http.get(
                    f"https://graph.facebook.com/v25.0/{phone_number_id}",
                    params={"fields": "is_on_biz_app,platform_type", "access_token": WHATSAPP_TOKEN},
                )
                print(f"WhatsApp disconnect live check: project={project_id}, phone_number_id={phone_number_id}, status={check_res.status_code}, body={check_res.text}")
                if check_res.ok:
                    # Confirmed live: is_on_biz_app is NOT the signal — once
                    # a number actually disconnects from the phone side, it
                    # leaves our WABA entirely and this lookup starts
                    # failing instead of returning is_on_biz_app: false.
                    confirmed_unlinked = check_res.json().get("is_on_biz_app") is False
                else:
                    err = (check_res.json() or {}).get("error", {})
                    # Meta's specific "this object no longer exists /
                    # isn't accessible" error (GraphMethodException,
                    # code 100, subcode 33) — confirmed live as exactly
                    # what a real phone-side disconnect produces. Checked
                    # narrowly on all three fields, not "any 400", since a
                    # broken/expired access token would also 400 here but
                    # via a different error type (OAuthException) and
                    # must NOT be treated as "safe to disconnect."
                    confirmed_unlinked = (
                        err.get("type") == "GraphMethodException"
                        and err.get("code") == 100
                        and err.get("error_subcode") == 33
                    )
            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"WhatsApp disconnect live check FAILED: project={project_id}, error={e}")

        if not confirmed_unlinked:
            print(f"WhatsApp disconnect BLOCKED: project={project_id} is a coexistence connection (confirmed_unlinked=False)")
            raise HTTPException(
                status_code=400,
                detail="This number is connected via WhatsApp Coexistence. Disconnect it from the WhatsApp Business App on your phone instead — Zavo can't safely disconnect a coexistence number without risking your chat history.",
            )
        print(f"WhatsApp disconnect ALLOWED: project={project_id} confirmed no longer on Business App")

    # Unsubscribe our app from the merchant's WABA. Without this, "disconnect"
    # only removed OUR row: Meta carried on pushing that merchant's customer
    # messages (and coexistence history) to this app indefinitely. They were
    # dropped on arrival, but they were still being transmitted to us after
    # the merchant believed they had disconnected.
    waba_id = row.get("waba_id")
    if waba_id:
        try:
            unsub = http.delete(
                f"https://graph.facebook.com/v25.0/{waba_id}/subscribed_apps",
                params={"access_token": WHATSAPP_TOKEN},
            )
            if not unsub.ok:
                print(f"WhatsApp unsubscribe failed for waba {waba_id}: {unsub.status_code} {unsub.text[:200]}")
        except Exception as e:
            # Never block the merchant's disconnect on Meta being reachable.
            sentry_sdk.capture_exception(e)
            print(f"WhatsApp unsubscribe error for waba {waba_id}: {e}")

    supabase.table("whatsapp_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}


@router.post("/whatsapp/resubscribe/{project_id}")
def whatsapp_resubscribe(project_id: str, user=Depends(verify_token)):
    """One-off recovery for connections made before the POST
    /{waba_id}/subscribed_apps call existed in whatsapp_onboard — without
    it, Meta never routes ANY webhook (messages, history, echoes) to this
    app for that WABA, regardless of the app-level webhook field toggles.
    New connections don't need this; it's for repairing ones made before
    the fix landed."""
    _require_uuid(project_id)
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")
    if is_rate_limited(f"wa-resubscribe:{project_id}", limit=5, window_seconds=300):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a few minutes.")
    res = supabase.table("whatsapp_integrations").select("waba_id").eq("project_id", project_id).maybe_single().execute()
    waba_id = (res.data or {}).get("waba_id") if res else None
    if not waba_id:
        raise HTTPException(status_code=404, detail="No WhatsApp Business Account on file for this project")

    subscribe_res = http.post(
        f"https://graph.facebook.com/v25.0/{waba_id}/subscribed_apps",
        params={"access_token": WHATSAPP_TOKEN}
    )
    if not subscribe_res.ok:
        print(f"WhatsApp resubscribe failed: project={project_id}, waba_id={waba_id}. Response: {subscribe_res.text}")
        raise HTTPException(status_code=502, detail="Meta rejected the subscription request. Please try again shortly.")

    return {"success": True, "waba_id": waba_id}


@router.post("/whatsapp/resync/{project_id}")
def whatsapp_resync(project_id: str, user=Depends(verify_token)):
    """Deliberately SEPARATE from whatsapp_resubscribe — Meta enforces a
    hard, undocumented-until-you-hit-it rate limit on the sync API per
    phone number ("(#4) Application request limit reached" /
    "Synchronisation request limit exceeded"), confirmed live during
    testing. Bundling an automatic resync into every resubscribe call
    burned through that quota after a few repair attempts. This must stay
    a deliberate, standalone action — never auto-triggered — so it's not
    accidentally called more than genuinely needed."""
    _require_uuid(project_id)
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")

    # The docstring above describes a hard Meta quota per phone number that
    # a few repeat attempts exhaust permanently, and yet nothing stopped an
    # admin from looping this endpoint. Deliberately strict.
    if is_rate_limited(f"wa-resync:{project_id}", limit=3, window_seconds=3600):
        raise HTTPException(
            status_code=429,
            detail="Meta limits how often a number can be resynced. Please wait an hour before trying again.",
        )

    res = supabase.table("whatsapp_integrations").select("phone_number_id, coexistence_enabled, access_token").eq("project_id", project_id).maybe_single().execute()
    row = res.data if res else None
    if not row or not row.get("coexistence_enabled") or not row.get("phone_number_id"):
        raise HTTPException(status_code=400, detail="This project has no active WhatsApp Coexistence connection to resync")

    initiate_coexistence_sync(project_id, row["phone_number_id"], _token_for(row))

    # FIX: initiate_coexistence_sync() catches Meta API failures internally
    # (writes them to last_sync_error, doesn't raise) so it can always
    # finish writing the row — but that meant this endpoint always
    # reported {"success": true} even when Meta rejected the request
    # outright (e.g. the sync rate limit hit live during testing). Read
    # the outcome back so the caller gets the truth, not just "it ran."
    result = supabase.table("whatsapp_integrations").select("history_sync_status, last_sync_error").eq("project_id", project_id).maybe_single().execute()
    row = result.data if result else {}
    if (row or {}).get("history_sync_status") == "failed":
        # last_sync_error holds raw Graph text (fbtrace_id, WABA ids,
        # internal messages). It is deliberately surfaced in the UI behind a
        # "Technical details" toggle, read from the stored column — it does
        # not belong in a plain error detail that gets toasted verbatim.
        print(f"WhatsApp resync failed for {project_id}: {(row or {}).get('last_sync_error')}")
        raise HTTPException(
            status_code=502,
            detail="WhatsApp couldn't start the sync. Please try again in a few minutes.",
        )

    return {"success": True}


@router.get("/whatsapp/coexistence-status/{project_id}")
def whatsapp_coexistence_status(project_id: str, user=Depends(verify_token)):
    _require_uuid(project_id)
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")
    res = supabase.table("whatsapp_integrations") \
        .select("coexistence_enabled, history_sync_status, history_sync_requested_at, history_sync_completed_at, last_sync_error, phone_number_id") \
        .eq("project_id", project_id).maybe_single().execute()
    row = res.data if res else None
    if not row or not row.get("coexistence_enabled"):
        return {"coexistence_enabled": False}

    is_on_biz_app = None
    try:
        check_res = http.get(
            f"https://graph.facebook.com/v25.0/{row['phone_number_id']}",
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
    code = data.get("code")
    project_id = data.get("projectId")
    if not code or not project_id:
        raise HTTPException(status_code=400, detail="Missing code or projectId")
    is_coexistence = bool(data.get("isCoexistence"))
    # For a coexistence completion, Meta's own FINISH_WHATSAPP_BUSINESS_APP_
    # ONBOARDING session event hands us the waba_id directly in its payload
    # — more reliable than /me/whatsapp_business_accounts, which came back
    # empty for a real coexistence connection during testing (business-app-
    # linked WABAs aren't guaranteed to show up in that generic listing, or
    # there's a propagation delay after the phone's "tap Confirm" step).
    waba_id_hint = data.get("wabaIdHint")
    project_id = _require_uuid(project_id)
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")

    # Each onboard drives several outbound Meta calls and a retry loop.
    if is_rate_limited(f"wa-onboard:{project_id}", limit=5, window_seconds=300):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a few minutes.")

    token_res = http.get(
        "https://graph.facebook.com/v25.0/oauth/access_token",
        params={"client_id": META_APP_ID, "client_secret": META_APP_SECRET, "code": code}
    )
    token_data = token_res.json()
    # FIX: was logging the full response, including the live access_token,
    # to stdout on every onboarding — never log the raw token response.
    if "access_token" not in token_data:
        print(f"WhatsApp token exchange failed: {token_data.get('error', token_data)}")
        raise HTTPException(status_code=400, detail="Token exchange failed")

    access_token = token_data["access_token"]

    waba_id = waba_id_hint or ""
    if not waba_id:
        waba_res = http.get(
            "https://graph.facebook.com/v25.0/me/whatsapp_business_accounts",
            params={"access_token": access_token}
        )
        waba_json = waba_res.json()
        waba_id = waba_json.get("data", [{}])[0].get("id", "")
        if not waba_id:
            print(f"WhatsApp onboard: no WABA found for project {project_id}, coexistence={is_coexistence}. Response: {waba_json}")

    # REQUIRED, separate from the App Dashboard's webhook field toggles —
    # those configure which fields your app CAN receive, but Meta won't
    # route any webhook (messages, history, smb_app_state_sync, etc.) for
    # this specific WABA to your app until the WABA is explicitly
    # subscribed. Missing this call was the actual reason nothing arrived
    # during testing, on live messages and history sync alike — not a
    # Render/config issue.
    if waba_id:
        subscribe_res = http.post(
            f"https://graph.facebook.com/v25.0/{waba_id}/subscribed_apps",
            params={"access_token": access_token}
        )
        if not subscribe_res.ok:
            print(f"WhatsApp onboard: failed to subscribe app to waba_id={waba_id}, project={project_id}. Response: {subscribe_res.text}")

    # A newly (or freshly re-)confirmed WABA/number can take a moment to
    # show up via the phone_numbers endpoint — retry a few times rather
    # than failing the whole connect on the first empty response.
    phone_data = {}
    phone_json = {}
    for attempt in range(3):
        phone_res = http.get(
            f"https://graph.facebook.com/v25.0/{waba_id}/phone_numbers",
            params={"access_token": access_token}
        )
        phone_json = phone_res.json()
        candidates = phone_json.get("data", [])
        if candidates:
            phone_data = candidates[0]
            break
        if attempt < 2:
            import time
            time.sleep(2)

    if not phone_data:
        print(f"WhatsApp onboard: no phone number found for project {project_id}, waba_id={waba_id}, coexistence={is_coexistence} after 3 attempts. Last response: {phone_json}")

    phone_number_id = phone_data.get("id", "")
    display_phone = phone_data.get("display_phone_number", "")

    if phone_number_id:
        conflict_res = supabase.table("whatsapp_integrations") \
            .select("project_id") \
            .eq("phone_number_id", phone_number_id) \
            .neq("project_id", project_id) \
            .execute()
        if conflict_res.data:
            raise HTTPException(
                status_code=409,
                detail="This WhatsApp number is already connected to a different Zavo project. Disconnect it there first before connecting it here."
            )

    # access_token is persisted now. It was obtained above, used for the
    # onboarding calls, and then discarded — so every runtime send for
    # every tenant went out on the single global WHATSAPP_TOKEN. One
    # leaked env var meant send-as-any-merchant, and a merchant revoking
    # our access changed nothing.
    #
    # The conflict check above is a read-then-write and two concurrent
    # onboards of one number both pass it. The unique index on
    # phone_number_id added by the migration is what actually stops it;
    # this catch turns that violation into the same 409 the check gives.
    try:
        supabase.table("whatsapp_integrations").upsert({
            "project_id": project_id,
            "phone_number_id": phone_number_id,
            "waba_id": waba_id,
            "display_phone_number": display_phone,
            "access_token": access_token,
        }, on_conflict="project_id").execute()
    except Exception as e:
        if _is_duplicate_key(e):
            raise HTTPException(
                status_code=409,
                detail="This WhatsApp number is already connected to a different Zavo project. Disconnect it there first before connecting it here."
            )
        # Anything else here is unrecognized, and this route has no outer
        # handler — it propagated as an unhandled 500 on the connect flow
        # with nothing telling Sentry.
        sentry_sdk.capture_exception(e)
        raise

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
class WhatsAppReplyRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=64)
    phone_number: str = Field(..., min_length=1, max_length=32)
    message: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)


def _send_manual_reply(project_id: str, phone_number: str, message: str) -> dict:
    """The blocking half of a manual reply.

    Runs in a worker thread. All of this used to sit inside an async
    handler — two Supabase round trips and a 20-second Meta POST directly
    on the event loop — which is the exact failure the webhook was split
    to avoid, never applied here.
    """
    row = _integration_for_project(project_id)
    if not row or not row.get("phone_number_id"):
        raise HTTPException(status_code=404, detail="WhatsApp not connected")

    # The destination must already be a conversation on THIS project.
    # Without it, any member who could reach this endpoint could send a
    # WhatsApp message to any number in the world on the business's
    # account — the chat was looked up only afterwards, and only to decide
    # whether to file a transcript line.
    chat = supabase.table("chats") \
        .select("id, last_human_reply_at") \
        .eq("project_id", project_id) \
        .eq("external_id", phone_number) \
        .eq("channel", "whatsapp") \
        .limit(1) \
        .execute()

    if not chat.data:
        raise HTTPException(
            status_code=404,
            detail="No WhatsApp conversation with that number on this project.",
        )

    chat_id = chat.data[0]["id"]

    res = send_whatsapp_message(
        phone_number, message, row["phone_number_id"], _token_for(row)
    )

    # The result used to be discarded and the transcript written
    # regardless, so a message Meta refused — most often because the
    # 24-hour customer service window had closed — appeared in the
    # dashboard as delivered, with no way for the operator to find out.
    if res is None or not getattr(res, "ok", False):
        detail = "WhatsApp refused the message."
        try:
            body = res.json() if res is not None else {}
            meta_msg = ((body.get("error") or {}).get("message") or "")[:200]
            if meta_msg:
                detail = f"WhatsApp refused the message: {meta_msg}"
        except Exception:
            pass
        try:
            supabase.table("chats") \
                .update({"last_send_error": detail[:MAX_SYNC_ERROR_LEN],
                         "last_send_error_at": _utc_now_iso()}) \
                .eq("id", chat_id) \
                .execute()
        except Exception as e:
            sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=502, detail=detail)

    supabase.table("chat_messages").insert({
        "chat_id": chat_id,
        "role": "assistant",
        "content": f"[Human] {message}",
    }).execute()

    return {"status": "sent"}


@router.post("/whatsapp/reply")
async def whatsapp_reply(body: WhatsAppReplyRequest, user=Depends(verify_token)):
    """Send a manual reply from the dashboard to a WhatsApp user."""
    # Was `data: dict` with raw data["project_id"] / ["phone_number"] /
    # ["message"] subscripts read BEFORE the auth check, so a body missing
    # any key was a 500 reached without authorization.
    project_id = _require_uuid(body.project_id)

    # min_role="admin", matching every sibling endpoint in this file.
    # Without it an agent who merely has the Integrations tab could send
    # WhatsApp messages as the business.
    require_project_access(user.id, project_id, tab="integrations", min_role="admin")

    phone_number = _valid_phone(body.phone_number)
    if not phone_number:
        raise HTTPException(status_code=400, detail="That is not a valid WhatsApp number.")

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Keyed per user as well as per project, so one member cannot consume
    # the whole project's reply budget and lock out their colleagues.
    if is_rate_limited(f"wa-reply:{project_id}:{user.id}", limit=30, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="You're sending messages too quickly. Please wait a moment.",
        )
    if is_rate_limited(f"wa-reply:{project_id}", limit=120, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="This project is sending too many messages. Please wait a moment.",
        )

    return await run_in_threadpool(_send_manual_reply, project_id, phone_number, message)