import calendar
import re
import sentry_sdk
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from clients import supabase
from auth import verify_token, require_project_access
from ratelimit import is_rate_limited
from config import (
    WHATSAPP_TOKEN,
    MAX_CAMPAIGN_RECIPIENTS,
    MAX_CAMPAIGNS_PER_HOUR,
)
from whatsapp import http

router = APIRouter()

# Columns safe to return from the list endpoint. Deliberately NOT "*":
# resolved_contacts holds every recipient's phone number for every campaign,
# and select("*") shipped the whole lot to the browser on each page load.
_CAMPAIGN_COLUMNS = (
    "id, project_id, name, template_name, status, recipient_filter, "
    "tag_filter, scheduled_at, recurrence, total_count, sent_count, "
    "failed_count, sent_at, created_at"
)


def _wa_integration(project_id: str) -> dict:
    """Phone number id, WABA id and the project's OWN access token.

    campaigns.py and template_library.py both paired the merchant's
    phone_number_id/waba_id with the PLATFORM's WHATSAPP_TOKEN, so for any
    merchant on their own WABA the call either failed outright or billed the
    wrong account. send_template_api.py:73 already had this right; this is
    the same lookup, shared.
    """
    wa = supabase.table("whatsapp_integrations") \
        .select("phone_number_id, waba_id, access_token") \
        .eq("project_id", project_id) \
        .limit(1) \
        .execute()

    if not wa.data:
        raise HTTPException(status_code=400, detail="WhatsApp not connected")

    row = wa.data[0]
    return {
        "phone_number_id": row.get("phone_number_id"),
        "waba_id": row.get("waba_id"),
        "token": row.get("access_token") or WHATSAPP_TOKEN,
    }


def _fetch_approved_templates(waba_id: str, token: str) -> list:
    res = http.get(
        f"https://graph.facebook.com/v19.0/{waba_id}/message_templates",
        params={
            "fields": "name,status,components,language",
            "limit": 100,
            "access_token": token,
        },
    )
    if not res.ok:
        # Meta's own error text is not shown to the caller — it can name
        # internal ids and token state.
        print(f"Meta template fetch failed: {res.text}")
        raise HTTPException(status_code=400, detail="Failed to fetch templates from Meta")

    return [t for t in res.json().get("data", []) if t.get("status") == "APPROVED"]


# -------------------------------------------------
# FETCH TEMPLATES FROM META
# -------------------------------------------------
@router.get("/campaigns/templates")
def get_templates(project_id: str, user=Depends(verify_token)):
    """Fetch approved templates from Meta for this project's WABA."""
    require_project_access(user.id, project_id, tab="campaigns")
    wa = _wa_integration(project_id)
    if not wa["waba_id"]:
        raise HTTPException(status_code=400, detail="WhatsApp not connected")
    return _fetch_approved_templates(wa["waba_id"], wa["token"])


# -------------------------------------------------
# RECIPIENT RESOLUTION — shared by immediate + scheduled + recurring sends
# -------------------------------------------------
def _valid_phone(raw) -> Optional[str]:
    """Normalise a recipient phone, or None if it can't be one.

    Every recipient is a paid message, so a malformed number is not just a
    failed send — it is a failure that counts against the WABA's quality
    rating. Rejecting it up front is cheaper than letting Meta reject it.
    """
    if not isinstance(raw, str):
        return None
    digits = re.sub(r"\D", "", raw)
    # E.164 allows 8-15 digits; anything outside that can't be a real number.
    if not (8 <= len(digits) <= 15):
        return None
    return digits


def resolve_contacts(project_id: str, recipient_filter: str, tag_filter, csv_contacts):
    if csv_contacts is not None:
        if not isinstance(csv_contacts, list):
            raise HTTPException(status_code=400, detail="Invalid contact list")
        # Was unbounded, and wrote one row per contact in a sequential loop.
        if len(csv_contacts) > MAX_CAMPAIGN_RECIPIENTS:
            raise HTTPException(
                status_code=400,
                detail=f"That file has {len(csv_contacts)} contacts. A campaign can reach at most {MAX_CAMPAIGN_RECIPIENTS} people.",
            )

        contacts = []
        seen = set()
        for c in csv_contacts:
            if not isinstance(c, dict):
                continue
            phone = _valid_phone(c.get("phone"))
            if not phone or phone in seen:
                continue
            seen.add(phone)
            name = c.get("name")
            contacts.append({"phone": phone, "name": (name if isinstance(name, str) else "")[:120]})

        # Save to leads table (upsert) — happens immediately at creation
        # time even for a scheduled campaign, so they show up in Leads
        # right away rather than only once the campaign fires.
        from leads import upsert_contact
        for c in contacts:
            try:
                upsert_contact(project_id, c["phone"], c.get("name") or None, channel="whatsapp")
            except Exception as e:
                # One bad contact must not abort the whole campaign setup.
                sentry_sdk.capture_exception(e)

        return contacts

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

    # One over the cap, so a project sitting exactly at the limit still
    # resolves cleanly while anything larger is detectably over.
    contacts_res = query.limit(MAX_CAMPAIGN_RECIPIENTS + 1).execute()

    contacts = []
    seen = set()
    for c in (contacts_res.data or []):
        phone = _valid_phone(c.get("phone"))
        if not phone or phone in seen:
            continue
        seen.add(phone)
        contacts.append({"phone": phone, "name": c.get("name") or ""})
    return contacts


def _enforce_recipient_cap(contacts):
    if not contacts:
        raise HTTPException(status_code=400, detail="No contacts found for this filter")
    if len(contacts) > MAX_CAMPAIGN_RECIPIENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This would message {len(contacts)} people. A single campaign can reach at most "
                f"{MAX_CAMPAIGN_RECIPIENTS} — narrow the filter or split it up."
            ),
        )


VALID_RECURRENCES = {"daily", "weekly", "monthly"}


def _next_occurrence(dt: datetime, recurrence: str) -> datetime:
    if recurrence == "daily":
        return dt + timedelta(days=1)
    if recurrence == "weekly":
        return dt + timedelta(weeks=1)
    if recurrence == "monthly":
        month = dt.month + 1
        year = dt.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return dt.replace(year=year, month=month, day=day)
    raise ValueError(f"Unknown recurrence: {recurrence}")


def _parse_and_validate_schedule(scheduled_at, recurrence, recipient_filter):
    """Shared by create + edit. Returns (scheduled_dt_iso, is_future, recurrence)."""
    if recurrence and recurrence not in VALID_RECURRENCES:
        raise HTTPException(status_code=400, detail="Invalid recurrence")
    if recurrence and recipient_filter == "csv":
        raise HTTPException(status_code=400, detail="Recurring campaigns can't use a one-time CSV upload — pick a live filter instead")

    if not scheduled_at:
        return None, False, None

    try:
        scheduled_dt = datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid scheduled time")

    if scheduled_dt <= datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Scheduled time must be in the future")

    return scheduled_dt.isoformat(), True, (recurrence if recurrence else None)


# -------------------------------------------------
# CREATE + SEND (or SCHEDULE) CAMPAIGN
# -------------------------------------------------
@router.get("/campaigns/recipient-count")
def recipient_count(
    project_id: str,
    recipient_filter: str = "all",
    tag_filter: Optional[str] = None,
    user=Depends(verify_token),
):
    """How many people a filter would actually reach.

    Exists so the UI can say "this will message 847 people" and mean it.
    Confirming a send without a real number is theatre, and the browser has
    no way to resolve a server-side lead filter itself.
    """
    require_project_access(user.id, project_id, tab="campaigns")
    if recipient_filter not in VALID_RECIPIENT_FILTERS:
        raise HTTPException(status_code=400, detail="Invalid recipient filter")
    contacts = resolve_contacts(project_id, recipient_filter, tag_filter, None)
    return {
        "count": len(contacts),
        "max": MAX_CAMPAIGN_RECIPIENTS,
        "over_limit": len(contacts) > MAX_CAMPAIGN_RECIPIENTS,
    }


class CampaignRequest(BaseModel):
    """Was a raw dict indexed directly, so a missing key was a 500 and
    nothing bounded the name, the variables, or the CSV."""
    project_id: str
    name: str = Field(max_length=120)
    template_name: str = Field(max_length=512)
    template_language: str = Field(default="en_US", max_length=16)
    variables: List[str] = Field(default_factory=list, max_length=10)
    recipient_filter: str = Field(default="all", max_length=20)
    tag_filter: Optional[str] = Field(default=None, max_length=40)
    csv_contacts: Optional[list] = None
    scheduled_at: Optional[str] = Field(default=None, max_length=64)
    recurrence: Optional[str] = Field(default=None, max_length=16)
    # Idempotency key, minted once per form session by the browser. A
    # double-click or a retry after a timeout previously created a second
    # campaign and sent to everyone twice.
    client_token: Optional[str] = Field(default=None, max_length=64)


VALID_RECIPIENT_FILTERS = {"all", "whatsapp", "web", "tag", "csv"}
_MAX_VARIABLE_LEN = 512


def _validate_common(req: CampaignRequest):
    if req.recipient_filter not in VALID_RECIPIENT_FILTERS:
        raise HTTPException(status_code=400, detail="Invalid recipient filter")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Campaign name is required")
    for v in req.variables:
        if len(v) > _MAX_VARIABLE_LEN:
            raise HTTPException(status_code=400, detail="One of the template values is too long.")


def _assert_template_approved(waba_id: str, token: str, template_name: str, language: str):
    """get_templates carefully filters to APPROVED for display, then create
    accepted any string. A stale or misspelled name means every single
    message fails — and bulk failures are exactly what damages a WABA's
    quality rating, which is the outcome the merchant most needs protecting
    from. Checked once here instead of discovered a thousand times."""
    approved = _fetch_approved_templates(waba_id, token)
    names = {t.get("name") for t in approved}
    if template_name not in names:
        raise HTTPException(
            status_code=400,
            detail="That template isn't approved on your WhatsApp account. Pick one from the list.",
        )


def _is_duplicate_error(e: Exception) -> bool:
    text = str(e).lower()
    return "duplicate key" in text or "23505" in text


@router.post("/campaigns")
def create_campaign(req: CampaignRequest, background_tasks: BackgroundTasks, user=Depends(verify_token)):
    # Starting a paid bulk send is the most expensive action in the app, so
    # it needs the campaigns tab permission AND admin — not merely "has some
    # role on the project", which is what require_project_role allowed.
    require_project_access(user.id, req.project_id, tab="campaigns", min_role="admin")
    _validate_common(req)

    # Each unit of work here is up to MAX_CAMPAIGN_RECIPIENTS paid messages,
    # so this is per-hour rather than the usual per-minute burst limit.
    if is_rate_limited(f"campaign-create:{req.project_id}", limit=MAX_CAMPAIGNS_PER_HOUR, window_seconds=3600):
        raise HTTPException(
            status_code=429,
            detail="Too many campaigns started recently. Please wait a little before sending another.",
        )

    # Recipients are resolved once, up front — the list a one-time
    # scheduled campaign sends to is locked in at creation time. Recurring
    # campaigns re-resolve fresh on every occurrence instead (see
    # dispatch_scheduled_campaigns) so new matching leads get included.
    contacts = resolve_contacts(req.project_id, req.recipient_filter, req.tag_filter, req.csv_contacts)
    _enforce_recipient_cap(contacts)

    wa = _wa_integration(req.project_id)
    if not wa["phone_number_id"]:
        raise HTTPException(status_code=400, detail="WhatsApp not connected")
    if wa["waba_id"]:
        _assert_template_approved(wa["waba_id"], wa["token"], req.template_name, req.template_language)

    scheduled_dt_iso, is_future, recurrence = _parse_and_validate_schedule(
        req.scheduled_at, req.recurrence, req.recipient_filter
    )

    row = {
        "project_id": req.project_id,
        "name": req.name.strip(),
        "template_name": req.template_name,
        "template_variables": {"variables": req.variables, "language": req.template_language},
        "status": "scheduled" if is_future else "sending",
        "recipient_filter": req.recipient_filter,
        "tag_filter": req.tag_filter,
        "scheduled_at": scheduled_dt_iso,
        "recurrence": recurrence,
        # phone_number_id is stored for BOTH paths now. It was only saved for
        # scheduled campaigns, so an immediate campaign that needed retrying
        # had no record of which number it went out from.
        "resolved_contacts": contacts,
        "phone_number_id": wa["phone_number_id"],
        "total_count": len(contacts),
        "sent_count": 0,
        "failed_count": 0,
    }
    if req.client_token:
        row["client_token"] = req.client_token

    try:
        campaign_res = supabase.table("campaigns").insert(row).execute()
    except Exception as e:
        if _is_duplicate_error(e):
            # Same client_token — this is a double submit, not a new campaign.
            existing = supabase.table("campaigns") \
                .select("id, total_count, status, scheduled_at") \
                .eq("project_id", req.project_id) \
                .eq("client_token", req.client_token) \
                .limit(1).execute()
            if existing.data:
                prior = existing.data[0]
                return {
                    "id": prior["id"], "total": prior.get("total_count", 0),
                    "status": prior.get("status"), "duplicate": True,
                }
        raise

    campaign_id = campaign_res.data[0]["id"]

    if is_future:
        return {"id": campaign_id, "total": len(contacts), "status": "scheduled", "scheduled_at": scheduled_dt_iso}

    background_tasks.add_task(
        send_campaign_messages,
        campaign_id, contacts, req.template_name, req.template_language,
        req.variables, wa["phone_number_id"], wa["token"],
    )

    return {"id": campaign_id, "total": len(contacts), "status": "sending"}


# -------------------------------------------------
# EDIT A SCHEDULED CAMPAIGN (before it fires)
# -------------------------------------------------
@router.put("/campaigns/{campaign_id}")
def update_campaign(campaign_id: str, req: CampaignRequest, user=Depends(verify_token)):
    existing = supabase.table("campaigns").select("status, project_id").eq("id", campaign_id).maybe_single().execute()
    if not existing or not existing.data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if existing.data["status"] != "scheduled":
        raise HTTPException(status_code=400, detail="Only scheduled campaigns can be edited")

    project_id = existing.data["project_id"]
    require_project_access(user.id, project_id, tab="campaigns", min_role="admin")
    _validate_common(req)

    scheduled_dt_iso, is_future, recurrence = _parse_and_validate_schedule(
        req.scheduled_at, req.recurrence, req.recipient_filter
    )
    if not is_future:
        raise HTTPException(status_code=400, detail="Scheduled time is required when editing a scheduled campaign")

    # Re-resolve recipients — leads/tags may have changed since it was first created.
    contacts = resolve_contacts(project_id, req.recipient_filter, req.tag_filter, req.csv_contacts)
    _enforce_recipient_cap(contacts)

    wa = _wa_integration(project_id)
    if wa["waba_id"]:
        _assert_template_approved(wa["waba_id"], wa["token"], req.template_name, req.template_language)

    # Conditional on the status STILL being scheduled. Read-then-write was a
    # race with dispatch_scheduled_campaigns: it could flip the campaign to
    # "sending" in between, and this update then rewrote resolved_contacts
    # underneath a send already in flight.
    res = supabase.table("campaigns").update({
        "name": req.name.strip(),
        "template_name": req.template_name,
        "template_variables": {"variables": req.variables, "language": req.template_language},
        "recipient_filter": req.recipient_filter,
        "tag_filter": req.tag_filter,
        "scheduled_at": scheduled_dt_iso,
        "recurrence": recurrence,
        "resolved_contacts": contacts,
        "phone_number_id": wa["phone_number_id"],
        "total_count": len(contacts),
    }).eq("id", campaign_id).eq("status", "scheduled").execute()

    if not res.data:
        raise HTTPException(status_code=409, detail="This campaign already started sending and can no longer be edited.")

    return {"id": campaign_id, "total": len(contacts), "status": "scheduled", "scheduled_at": scheduled_dt_iso}


# How often the sender writes progress and re-checks for cancellation.
# Small enough that cancelling feels immediate and a crash loses little;
# large enough not to add a database round-trip per message.
_PROGRESS_EVERY = 25


def _campaign_still_sending(campaign_id) -> bool:
    """Cancelling used to set status='cancelled' while the send loop, which
    never re-read status, carried on to the last recipient. Every one of
    those messages was billed after the merchant hit cancel."""
    try:
        res = supabase.table("campaigns").select("status").eq("id", campaign_id).maybe_single().execute()
        return bool(res and res.data and res.data.get("status") == "sending")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        # Can't tell — keep going rather than silently abandoning a send.
        return True


def send_campaign_messages(
    campaign_id, contacts, template_name, template_lang,
    variables, phone_number_id, token=None
):
    """Send template messages to all contacts.

    Deliberately a plain `def`, not `async def`. FastAPI runs an async
    BackgroundTask ON THE EVENT LOOP, and this function is blocking
    throughout (network calls plus a sleep between each message) — so as a
    coroutine it froze the entire backend for the duration of the campaign:
    webhooks, dashboard, chat, everything. As a sync function FastAPI runs
    it in the threadpool instead. Same reason dispatch_scheduled_campaigns
    below can now call it directly rather than via asyncio.run().
    """
    import time

    sent = 0
    failed = 0
    token = token or WHATSAPP_TOKEN

    # Built once — it's identical for every recipient.
    components = []
    if variables:
        components.append({
            "type": "body",
            "parameters": [{"type": "text", "text": str(v)} for v in variables],
        })

    def write_progress(status=None):
        update = {"sent_count": sent, "failed_count": failed}
        if status:
            update["status"] = status
            update["sent_at"] = "now()"
        try:
            supabase.table("campaigns").update(update).eq("id", campaign_id).execute()
        except Exception as e:
            sentry_sdk.capture_exception(e)

    stopped = False
    for i, contact in enumerate(contacts):
        # Re-read status periodically so cancel actually stops the send.
        if i and i % _PROGRESS_EVERY == 0:
            write_progress()
            if not _campaign_still_sending(campaign_id):
                stopped = True
                print(f"Campaign {campaign_id} stopped after {sent} sends")
                break

        phone = str(contact.get("phone") or "").strip().replace(" ", "").replace("-", "")
        if not phone:
            failed += 1
            continue
        if not phone.startswith("+"):
            phone = f"+{phone}"

        try:
            # `http` is whatsapp.py's _TimeoutSession — the bare requests.post
            # this used had NO timeout, so a slow Meta pinned the worker
            # indefinitely.
            res = http.post(
                f"https://graph.facebook.com/v19.0/{phone_number_id}/messages",
                headers={
                    "Authorization": f"Bearer {token}",
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
            sentry_sdk.capture_exception(e)
            failed += 1
            print(f"Campaign send exception to {phone}: {e}")

        # Rate limiting — WhatsApp allows ~80 messages/sec on low tier
        time.sleep(0.05)

    write_progress(status="cancelled" if stopped else "sent")
    # Tells dispatch_scheduled_campaigns whether a recurring series should
    # roll forward — a cancelled send must not reschedule itself.
    return not stopped


# -------------------------------------------------
# SCHEDULED CAMPAIGN DISPATCH — called by APScheduler in main.py
# -------------------------------------------------
def dispatch_scheduled_campaigns():
    """
    Runs every 30 seconds. Finds campaigns whose scheduled_at has arrived
    and sends them. Reuses the same background scheduler already running
    for appointment reminders, instead of introducing a second scheduling
    mechanism. If a campaign has a recurrence set, it's rescheduled to its
    next occurrence (with freshly re-resolved contacts) instead of being
    left in a final "sent" state.
    """
    try:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # Recover campaigns abandoned mid-send. Render's free tier sleeps the
        # process, so an interrupted campaign sat in "sending" forever with
        # no way to tell it apart from one still running. Marked failed
        # rather than retried: the sends already paid for cannot be undone,
        # and re-running would bill the merchant a second time.
        stuck_before = (now - timedelta(hours=2)).isoformat()
        try:
            supabase.table("campaigns") \
                .update({"status": "failed"}) \
                .eq("status", "sending") \
                .lt("created_at", stuck_before) \
                .execute()
        except Exception as e:
            sentry_sdk.capture_exception(e)

        due = supabase.table("campaigns") \
            .select("*") \
            .eq("status", "scheduled") \
            .lte("scheduled_at", now_iso) \
            .limit(20) \
            .execute()

        for camp in (due.data or []):
            try:
                # Flip status first so a slow send (or a crash mid-send)
                # can't cause the same campaign to be picked up twice by
                # the next tick.
                #
                # The .eq("status", "scheduled") predicate is what makes this
                # a claim rather than a hope: without it, two overlapping
                # ticks (or two instances) both flipped and both sent, and
                # every recipient was billed twice. If the update matches no
                # row, someone else already claimed this campaign.
                claim = supabase.table("campaigns") \
                    .update({"status": "sending"}) \
                    .eq("id", camp["id"]) \
                    .eq("status", "scheduled") \
                    .execute()
                if not claim.data:
                    continue

                contacts = camp.get("resolved_contacts") or []
                tv = camp.get("template_variables") or {}
                variables = tv.get("variables", [])
                template_lang = tv.get("language", "en_US")
                phone_number_id = camp.get("phone_number_id")

                if not contacts or not phone_number_id:
                    print(f"Scheduled campaign {camp['id']} missing contacts or phone_number_id — skipping")
                    supabase.table("campaigns").update({"status": "failed"}).eq("id", camp["id"]).execute()
                    continue

                wa_token = None
                try:
                    wa_token = _wa_integration(camp["project_id"])["token"]
                except Exception:
                    wa_token = WHATSAPP_TOKEN

                # Direct call, not asyncio.run(). This runs on the
                # APScheduler thread, which is shared with appointment
                # reminders — spinning up an event loop here blocked those
                # for the whole campaign.
                completed = send_campaign_messages(
                    camp["id"], contacts, camp["template_name"], template_lang,
                    variables, phone_number_id, wa_token,
                )

                recurrence = camp.get("recurrence")
                # A cancelled send must not roll the series forward —
                # otherwise cancelling a recurring campaign just delayed it.
                if recurrence and completed:
                    try:
                        current_dt = datetime.fromisoformat(str(camp["scheduled_at"]).replace("Z", "+00:00"))
                        next_dt = _next_occurrence(current_dt, recurrence)
                        fresh_contacts = resolve_contacts(
                            camp["project_id"], camp.get("recipient_filter", "all"), camp.get("tag_filter"), None
                        )
                        supabase.table("campaigns").update({
                            "status": "scheduled",
                            "scheduled_at": next_dt.isoformat(),
                            "resolved_contacts": fresh_contacts,
                            "total_count": len(fresh_contacts),
                        }).eq("id", camp["id"]).execute()
                    except Exception as e:
                        sentry_sdk.capture_exception(e)
                        print(f"Failed to reschedule recurring campaign {camp['id']}: {e}")

            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"dispatch_scheduled_campaigns error for {camp.get('id')}: {e}")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"dispatch_scheduled_campaigns fatal error: {e}")


# -------------------------------------------------
# CANCEL A SCHEDULED CAMPAIGN (also stops a recurring series)
# -------------------------------------------------
@router.post("/campaigns/{campaign_id}/cancel")
def cancel_campaign(campaign_id: str, user=Depends(verify_token)):
    camp = supabase.table("campaigns").select("status, project_id").eq("id", campaign_id).maybe_single().execute()
    if not camp or not camp.data:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user.id, camp.data["project_id"], tab="campaigns")

    # A campaign already sending can now be cancelled too — the send loop
    # re-reads its status and stops. Previously "cancelled" was written
    # while the loop carried on to the last recipient, every message billed.
    if camp.data["status"] not in ("scheduled", "sending"):
        raise HTTPException(status_code=400, detail="This campaign has already finished.")

    res = supabase.table("campaigns") \
        .update({"status": "cancelled"}) \
        .eq("id", campaign_id) \
        .in_("status", ["scheduled", "sending"]) \
        .execute()

    if not res.data:
        raise HTTPException(status_code=409, detail="This campaign has already finished.")

    return {"status": "cancelled"}


# -------------------------------------------------
# LIST CAMPAIGNS
# -------------------------------------------------
@router.get("/campaigns")
def list_campaigns(project_id: str, user=Depends(verify_token), limit: int = 50, offset: int = 0):
    require_project_access(user.id, project_id, tab="campaigns")
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    # Explicit columns, not "*": resolved_contacts holds every recipient's
    # phone number for every campaign, and it was being shipped to the
    # browser on each page load.
    res = supabase.table("campaigns") \
        .select(_CAMPAIGN_COLUMNS) \
        .eq("project_id", project_id) \
        .order("created_at", desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()
    return res.data
