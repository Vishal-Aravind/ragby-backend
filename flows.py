"""
Interactive Message Flows for WhatsApp
- Everything driven by button IDs, no global keywords
- "ask_a_question" button ID → RAG mode + [Back to Menu] after every answer
- "back_to_menu" button ID → restart flow
- "handoff" button ID → human handoff
- Free questions toggle → if ON, text on buttons node → RAG + resend buttons
"""
import sentry_sdk
import threading
from datetime import datetime, timezone
from typing import Optional
from clients import supabase
from config import FRONTEND_URL, MAX_FLOW_DELAY_THREADS
from whatsapp import (
    send_whatsapp_message,
    send_whatsapp_buttons,
    send_whatsapp_list,
    send_whatsapp_cta_url,
    send_whatsapp_media,
)

RESERVED_ASK_AI   = "ask_a_question"
RESERVED_BACK     = "back_to_menu"
RESERVED_HANDOFF  = "talk_to_human"


# -------------------------------------------------
# SESSION MANAGEMENT
# -------------------------------------------------
def get_session(project_id: str, phone_number: str) -> Optional[dict]:
    try:
        res = supabase.table("whatsapp_sessions") \
            .select("*") \
            .eq("project_id", project_id) \
            .eq("phone_number", phone_number) \
            .limit(1) \
            .execute()

        if not res.data:
            return None

        session = res.data[0]
        expires_at = session.get("expires_at")
        if expires_at:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp < datetime.now(timezone.utc):
                supabase.table("whatsapp_sessions") \
                    .delete() \
                    .eq("id", session["id"]) \
                    .execute()
                return None
        return session

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"get_session error: {e}")
        return None


def upsert_session(project_id: str, phone_number: str, data: dict):
    supabase.table("whatsapp_sessions").upsert({
        "project_id": project_id,
        "phone_number": phone_number,
        **data,
    }, on_conflict="project_id,phone_number").execute()


def delete_session(project_id: str, phone_number: str):
    supabase.table("whatsapp_sessions") \
        .delete() \
        .eq("project_id", project_id) \
        .eq("phone_number", phone_number) \
        .execute()


# -------------------------------------------------
# FLOW + NODE LOOKUP
# -------------------------------------------------
def get_active_flow(project_id: str) -> Optional[dict]:
    try:
        res = supabase.table("flows") \
            .select("*") \
            .eq("project_id", project_id) \
            .eq("is_active", True) \
            .limit(1) \
            .execute()
        return res.data[0] if res.data else None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"get_active_flow error: {e}")
        return None


def get_start_node(flow_id: str) -> Optional[dict]:
    res = supabase.table("flow_nodes") \
        .select("*") \
        .eq("flow_id", flow_id) \
        .eq("is_start", True) \
        .limit(1) \
        .execute()
    return res.data[0] if res.data else None


def get_node(node_id: str, flow_id: str) -> Optional[dict]:
    """Fetch a node, always scoped to the flow it must belong to.

    CROSS-TENANT FIX: this used to select by id alone. Nothing validated
    that an edge's to_node_id was a node in the same flow, so an edge
    pointing at ANOTHER project's node id would fetch that node here and
    send_node would deliver its content to the caller's own WhatsApp
    number. The edge routes now reject such an edge on write; this is the
    read-side backstop that closes any future variant of the same mistake,
    including edges already stored from before the fix.

    flow_id is REQUIRED rather than optional on purpose — an optional scope
    is one forgotten argument away from reopening the same hole.
    """
    if not node_id or not flow_id:
        return None
    res = supabase.table("flow_nodes") \
        .select("*") \
        .eq("id", node_id) \
        .eq("flow_id", flow_id) \
        .limit(1) \
        .execute()
    return res.data[0] if res.data else None


def get_next_node(flow_id: str, from_node_id: str, trigger: str) -> Optional[dict]:
    edge = supabase.table("flow_edges") \
        .select("to_node_id") \
        .eq("flow_id", flow_id) \
        .eq("from_node_id", from_node_id) \
        .eq("trigger", trigger) \
        .limit(1) \
        .execute()
    if not edge.data:
        return None
    return get_node(edge.data[0]["to_node_id"], flow_id=flow_id)


# -------------------------------------------------
# NODE SENDER
# -------------------------------------------------
def send_node(node: dict, to: str, phone_number_id: str, token: str, project_id: str = None):
    """Send the right WhatsApp message type for a node."""
    t = node.get("type") or ""
    # `content` is a free-form jsonb column, so nothing guarantees any key is
    # present. This used to index c["body"] directly in several branches: a
    # node saved without a body raised KeyError INSIDE the webhook handler,
    # which killed that customer's conversation mid-flow with no way back.
    c = node.get("content") or {}
    body = c.get("body") or ""

    if t in ("text", "message"):
        if body:
            send_whatsapp_message(to, body, phone_number_id, token)

    elif t in ("buttons", "message_buttons"):
        btns = []
        for btn in c.get("buttons", []) or []:
            label = btn.get("title") or btn.get("label", "")
            btn_id = btn.get("id") or label.strip().lower().replace(" ", "_")
            if label:
                btns.append({"id": btn_id, "title": label})
        # WhatsApp rejects an interactive message with no buttons and one
        # with an empty body, so fall back to plain text rather than
        # sending a request we know will fail.
        if btns and body:
            send_whatsapp_buttons(to, body, btns, phone_number_id, token)
        elif body:
            send_whatsapp_message(to, body, phone_number_id, token)

    elif t in ("list", "message_list"):
        sections = []
        for section in c.get("sections", []) or []:
            rows = []
            for row in section.get("rows", []) or []:
                label = row.get("title") or row.get("label", "")
                row_id = row.get("id") or label.strip().lower().replace(" ", "_")
                if label:
                    rows.append({"id": row_id, "title": label})
            if rows:
                sections.append({"title": section.get("title", ""), "rows": rows})
        if sections and body:
            send_whatsapp_list(
                to, body, c.get("button_text", "View Options"),
                sections, phone_number_id, token
            )
        elif body:
            send_whatsapp_message(to, body, phone_number_id, token)

    elif t == "message_media":
        if c.get("media_url"):
            send_whatsapp_media(
                to, "image",
                {"link": c["media_url"], "caption": body},
                phone_number_id, token,
            )
        elif body:
            send_whatsapp_message(to, body, phone_number_id, token)

    elif t == "message_video":
        if c.get("video_url"):
            send_whatsapp_media(
                to, "video",
                {"link": c["video_url"], "caption": body},
                phone_number_id, token,
            )
        elif body:
            send_whatsapp_message(to, body, phone_number_id, token)

    elif t == "message_document":
        if c.get("document_url"):
            send_whatsapp_media(
                to, "document",
                {
                    "link": c["document_url"],
                    "caption": body,
                    "filename": c.get("filename") or "document",
                },
                phone_number_id, token,
            )
        elif body:
            send_whatsapp_message(to, body, phone_number_id, token)

    elif t == "cta_url":
        if body:
            send_whatsapp_cta_url(
                to, body, c.get("button_text", "Click Here"),
                c.get("url", ""), phone_number_id, token
            )

    elif t == "message_audio":
        if c.get("audio_url"):
            send_whatsapp_media(
                to, "audio", {"link": c["audio_url"]}, phone_number_id, token,
            )

    elif t == "message_location":
        if c.get("latitude") and c.get("longitude"):
            send_whatsapp_media(
                to, "location",
                {
                    "latitude": c["latitude"],
                    "longitude": c["longitude"],
                    "name": c.get("name", ""),
                    "address": c.get("address", ""),
                },
                phone_number_id, token,
            )
        if body:
            send_whatsapp_message(to, body, phone_number_id, token)

    elif t == "message_contact":
        if c.get("contact_name") and c.get("contact_phone"):
            send_whatsapp_media(
                to, "contacts",
                [{
                    "name": {"formatted_name": c["contact_name"], "first_name": c["contact_name"]},
                    "phones": [{"phone": c["contact_phone"], "type": "CELL"}],
                }],
                phone_number_id, token,
            )

    elif t == "ask_a_question":
        if project_id:
            upsert_session(project_id, to, {"mode": "rag_question"})
        send_whatsapp_buttons(
            to,
            c.get("body", "You can now ask me anything!"),
            [{"id": RESERVED_BACK, "title": "↩ Back to Menu"}],
            phone_number_id, token
        )

    elif t == "back_to_menu":
        pass

    elif t in ("handoff", "talk_to_human"):
        send_whatsapp_message(to, c.get("body", "Connecting you to our team. Please wait..."), phone_number_id, token)

    elif t == "call_us":
        phone = c.get("phone", "").replace(" ", "")
        if phone:
            send_whatsapp_cta_url(
                to,
                c.get("body", "Need help? Call us directly!"),
                "📞 Call Us",
                f"tel:{phone}",
                phone_number_id, token
            )
        else:
            send_whatsapp_message(to, c.get("body", ""), phone_number_id, token)

    elif t == "time_delay":
        pass

    elif t == "rag":
        if body:
            send_whatsapp_message(to, body, phone_number_id, token)

    elif t == "message_event":
        event_id = c.get("event_id", "")
        proj_id = project_id or ""
        reg_url = f"{FRONTEND_URL}/event/{event_id}"

        # Send rich card — image + body + Register button (+ optional Call button)
        if c.get("banner_url"):
            send_whatsapp_media(
                to, "image",
                {"link": c["banner_url"], "caption": body},
                phone_number_id, token,
            )

        send_whatsapp_cta_url(
            to,
            body or "Register now — limited spots available!",
            c.get("button_text", "Register Now"),
            reg_url,
            phone_number_id,
            token,
        )

        # Optional second message — Call to Attend
        if c.get("contact_phone"):
            phone_clean = c["contact_phone"].replace(" ", "")
            send_whatsapp_cta_url(
                to,
                "Prefer to call instead?",
                "📞 Call to Attend",
                f"tel:{phone_clean}",
                phone_number_id,
                token,
            )

    elif t == "message_booking":
        proj_id = project_id or ""
        booking_url = f"{FRONTEND_URL}/book/{proj_id}?phone={to}"
        send_whatsapp_cta_url(
            to,
            c.get("body", "Book your appointment 📅\nChoose a date and time that works for you."),
            c.get("button_text", "Book Appointment"),
            booking_url,
            phone_number_id,
            token,
        )
        if proj_id:
            upsert_session(proj_id, to, {
                "mode": "shop_browsing",
                "metadata": {},
            })

    elif t == "message_shop":
        catalog_id = c.get("catalog_id", "")
        proj_id = project_id or ""
        shop_url = f"{FRONTEND_URL}/shop/{proj_id}?catalog={catalog_id}&phone={to}"

        send_whatsapp_cta_url(
            to,
            c.get("body", "Browse our menu and add items to your cart 🛒\nSelect multiple items at once"),
            c.get("button_text", "View Menu"),
            shop_url,
            phone_number_id,
            token,
        )

        if proj_id:
            upsert_session(proj_id, to, {
                "mode": "shop_browsing",
                "metadata": {"catalog_id": catalog_id},
            })


def send_back_to_menu_button(to: str, text: str, phone_number_id: str, token: str):
    # WhatsApp's interactive button body has a hard 1024-char limit — an AI
    # answer can be longer than that (e.g. a detailed policy list), and
    # WhatsApp rejects the ENTIRE message if it's over, so the customer
    # would get nothing at all. Send the full answer as a plain message
    # (no such limit in practice) and follow with a short separate button
    # message, instead of silently losing the whole reply.
    if len(text) <= 1024:
        send_whatsapp_buttons(
            to, text,
            [{"id": RESERVED_BACK, "title": "↩ Back to Menu"}],
            phone_number_id, token
        )
    else:
        send_whatsapp_message(to, text, phone_number_id, token)
        send_whatsapp_buttons(
            to, "Would you like to go back to the menu?",
            [{"id": RESERVED_BACK, "title": "↩ Back to Menu"}],
            phone_number_id, token
        )


# -------------------------------------------------
# FLOW EXECUTION
# -------------------------------------------------
def start_flow(flow: dict, project_id: str, phone_number: str, phone_number_id: str, token: str, chat_id: str = None):
    from chat import save_message
    start_node = get_start_node(flow["id"])
    if not start_node:
        print(f"No start node for flow {flow['id']}")
        return

    upsert_session(project_id, phone_number, {
        "flow_id": flow["id"],
        "current_node_id": start_node["id"],
        "mode": "flow",
    })

    body = start_node["content"].get("body", "")
    if flow.get("free_questions") and start_node["type"] in ("buttons", "list"):
        body = body + "\n\n💬 _Tap an option or type your question directly_"
        node_with_hint = {**start_node, "content": {**start_node["content"], "body": body}}
        send_node(node_with_hint, phone_number, phone_number_id, token, project_id=project_id)
    else:
        send_node(start_node, phone_number, phone_number_id, token, project_id=project_id)

    if chat_id and body:
        save_message(chat_id, "assistant", body)

    if start_node["type"] == "handoff":
        upsert_session(project_id, phone_number, {
            "flow_id": flow["id"],
            "current_node_id": start_node["id"],
            "mode": "human",
        })


# -------------------------------------------------
# TIME-DELAY NODES
# -------------------------------------------------
# A delay parks a customer mid-flow and resumes them later. Each one used to
# be a bare thread doing time.sleep() for up to 22 hours, one per customer
# per delay node, with nothing bounding how many could exist at once — a
# slow memory leak that a busy flow turns into a fast one.
#
# Known limitation, stated rather than hidden: these do NOT survive a
# restart. Render's free tier sleeps the process, so every pending thread
# dies and those customers simply never receive the next message. Making
# delays durable needs a scheduled-jobs table and a worker; until then the
# ceiling below at least keeps the failure bounded and predictable.
_delay_threads_lock = threading.Lock()
_delay_threads_active = 0

_DELAY_UNIT_MULTIPLIER = {"seconds": 1, "minutes": 60, "hours": 3600}
_MAX_DELAY_SECONDS = 22 * 3600  # 2h buffer before WhatsApp's 24h window shuts


def _resolve_delay_seconds(content: dict) -> int:
    """Coerce a node's delay to a sane number of seconds.

    `int(c.get("delay_seconds", 60))` raised ValueError on any non-numeric
    value, and the editor's clamp is client-side only — so a hand-crafted
    request could store a negative, huge, or non-numeric delay.
    """
    try:
        amount = int(float(content.get("delay_seconds", 60)))
    except (TypeError, ValueError):
        amount = 60
    unit = content.get("delay_unit", "seconds")
    multiplier = _DELAY_UNIT_MULTIPLIER.get(unit, 1)
    return max(0, min(amount * multiplier, _MAX_DELAY_SECONDS))


def _schedule_delayed_advance(flow_id, from_node_id, project_id, phone_number,
                              phone_number_id, token, chat_id, delay_secs):
    from chat import save_message
    global _delay_threads_active

    def advance():
        after_node = get_next_node(flow_id, from_node_id, "next")
        if not after_node:
            return
        # The customer may have moved on during the wait — typed a trigger
        # keyword, tapped Back to Menu, started an order. Resuming blindly
        # yanked them back into a branch they had already left. Only advance
        # if they are still parked on the delay node.
        current = get_session(project_id, phone_number)
        if not current or current.get("current_node_id") != from_node_id:
            return
        upsert_session(project_id, phone_number, {
            "flow_id": flow_id,
            "current_node_id": after_node["id"],
            "mode": "flow",
        })
        send_node(after_node, phone_number, phone_number_id, token, project_id=project_id)
        if chat_id:
            save_message(chat_id, "assistant", (after_node.get("content") or {}).get("body", ""))

    if delay_secs <= 0:
        advance()
        return

    with _delay_threads_lock:
        if _delay_threads_active >= MAX_FLOW_DELAY_THREADS:
            # At the ceiling, skip the wait rather than refusing to continue
            # — the customer gets the next message early instead of never.
            print(f"delay thread ceiling reached ({MAX_FLOW_DELAY_THREADS}); advancing immediately")
            advance()
            return
        _delay_threads_active += 1

    def delayed_advance():
        global _delay_threads_active
        import time
        try:
            time.sleep(delay_secs)
            advance()
        except Exception as e:
            sentry_sdk.capture_exception(e)
        finally:
            with _delay_threads_lock:
                _delay_threads_active -= 1

    threading.Thread(target=delayed_advance, daemon=True).start()


def handle_interactive(session: dict, trigger: str, phone_number: str, phone_number_id: str, token: str, project_id: str, chat_id: str = None):
    from chat import save_message

    if chat_id:
        save_message(chat_id, "user", f"[tapped: {trigger}]")

    # Handle appointment reschedule/cancel button triggers
    if trigger.startswith("reschedule_"):
        appointment_id = trigger.replace("reschedule_", "")
        booking_url = f"{FRONTEND_URL}/book/{project_id}?phone={phone_number}&reschedule={appointment_id}"
        send_whatsapp_cta_url(
            phone_number,
            "Tap below to pick a new date and time 📅\nYour previous appointment will be cancelled once you confirm the new one.",
            "Reschedule",
            booking_url,
            phone_number_id, token,
        )
        return

    if trigger.startswith("cancel_appt_"):
        from appointments import cancel_appointment
        appointment_id = trigger.replace("cancel_appt_", "")

        # FIX: previously cancelled whatever appointment_id was embedded in
        # the button trigger with no check the sender is actually that
        # appointment's customer — a forged/replayed trigger id (or a
        # button payload observed some other way) could cancel someone
        # else's booking. Confirm the appointment belongs to this exact
        # sender + project before touching it.
        appt_check = supabase.table("appointments").select("customer_phone").eq("id", appointment_id).eq("project_id", project_id).maybe_single().execute()
        appt_data = appt_check.data if appt_check else None
        if not appt_data or appt_data.get("customer_phone") != phone_number.replace("+", "").replace(" ", ""):
            return

        try:
            # notify_customer=False — the message right below already tells
            # them; reuses the same logic that also cleans up the Google
            # Calendar event, which this button previously skipped.
            cancel_appointment(appointment_id, notify_customer=False)
        except ValueError:
            pass
        send_whatsapp_message(
            phone_number,
            "✅ Your appointment has been cancelled.\n\nReply *book* to schedule a new one.",
            phone_number_id, token,
        )
        delete_session(project_id, phone_number)
        return

    if trigger == "cart_continue":
        handle_text(session, "continue", project_id, chat_id, phone_number, phone_number_id, token)
        return

    if trigger == "cart_add_more":
        handle_text(session, "add more", project_id, chat_id, phone_number, phone_number_id, token)
        return

    if trigger == "cart_clear":
        handle_text(session, "clear cart", project_id, chat_id, phone_number, phone_number_id, token)
        return

    if trigger == "skip_special_req":
        if session and session.get("mode") == "awaiting_special_request":
            handle_text(session, "skip", project_id, chat_id, phone_number, phone_number_id, token)
            return

    if trigger == "confirm_and_pay":
        handle_text(session, "confirm", project_id, chat_id, phone_number, phone_number_id, token)
        return

    if trigger == "change_order":
        handle_text(session, "change order", project_id, chat_id, phone_number, phone_number_id, token)
        return

    if trigger == "cancel_order":
        handle_text(session, "cancel", project_id, chat_id, phone_number, phone_number_id, token)
        return

    if trigger == RESERVED_BACK:
        flow = get_active_flow(project_id)
        if flow:
            start_flow(flow, project_id, phone_number, phone_number_id, token, chat_id)
        return

    if trigger == RESERVED_ASK_AI:
        upsert_session(project_id, phone_number, {
            "flow_id": session.get("flow_id"),
            "current_node_id": session.get("current_node_id"),
            "mode": "rag_question",
        })
        msg = "You can now ask me anything about our products and services!"
        send_whatsapp_buttons(
            phone_number, msg,
            [{"id": RESERVED_BACK, "title": "↩ Back to Menu"}],
            phone_number_id, token
        )
        if chat_id:
            save_message(chat_id, "assistant", msg)
        return

    if trigger == RESERVED_HANDOFF:
        upsert_session(project_id, phone_number, {
            "flow_id": session.get("flow_id"),
            "current_node_id": session.get("current_node_id"),
            "mode": "human",
        })
        msg = "Connecting you to our team. Please wait..."
        send_whatsapp_message(phone_number, msg, phone_number_id, token)
        if chat_id:
            save_message(chat_id, "assistant", msg)
        return

    flow_id = session.get("flow_id")
    current_node_id = session.get("current_node_id")

    if not flow_id or not current_node_id:
        return

    next_node = get_next_node(flow_id, current_node_id, trigger)
    if not next_node:
        current_node = get_node(current_node_id, flow_id=flow_id)
        if current_node:
            send_node(current_node, phone_number, phone_number_id, token, project_id=project_id)
            if chat_id:
                save_message(chat_id, "assistant", (current_node.get("content") or {}).get("body", ""))
        return

    upsert_session(project_id, phone_number, {
        "flow_id": flow_id,
        "current_node_id": next_node["id"],
        "mode": "human" if next_node["type"] in ("handoff", "talk_to_human") else
                "rag_question" if next_node["type"] == "ask_a_question" else "flow",
    })

    if next_node["type"] in ("handoff", "talk_to_human"):
        msg = next_node["content"].get("body", "Connecting you to our team...")
        send_whatsapp_message(phone_number, msg, phone_number_id, token)
        if chat_id:
            save_message(chat_id, "assistant", msg)
    elif next_node["type"] == "back_to_menu":
        flow = get_active_flow(project_id)
        if flow:
            start_flow(flow, project_id, phone_number, phone_number_id, token, chat_id)
    elif next_node["type"] == "time_delay":
        delay_secs = _resolve_delay_seconds(next_node.get("content") or {})
        _schedule_delayed_advance(
            flow_id, next_node["id"], project_id, phone_number,
            phone_number_id, token, chat_id, delay_secs,
        )
    else:
        send_node(next_node, phone_number, phone_number_id, token, project_id=project_id)
        if chat_id:
            save_message(chat_id, "assistant", next_node["content"].get("body", ""))

        outgoing = supabase.table("flow_edges") \
            .select("id") \
            .eq("flow_id", flow_id) \
            .eq("from_node_id", next_node["id"]) \
            .limit(1) \
            .execute()

        if not outgoing.data:
            if next_node["type"] == "ask_a_question":
                pass
            else:
                flow_row = supabase.table("flows").select("free_questions").eq("id", flow_id).single().execute()
                free_q = flow_row.data.get("free_questions", False) if flow_row.data else False

                if free_q:
                    upsert_session(project_id, phone_number, {
                        "flow_id": flow_id,
                        "current_node_id": next_node["id"],
                        "mode": "rag_question",
                    })
                    msg = "💬 Feel free to ask me anything!"
                    send_whatsapp_buttons(
                        phone_number, msg,
                        [{"id": RESERVED_BACK, "title": "↩ Back to Menu"}],
                        phone_number_id, token
                    )
                    if chat_id:
                        save_message(chat_id, "assistant", msg)
                else:
                    upsert_session(project_id, phone_number, {
                        "flow_id": flow_id,
                        "current_node_id": next_node["id"],
                        "mode": "flow",
                    })


def handle_text(session: Optional[dict], text: str, project_id: str, chat_id: str, phone_number: str, phone_number_id: str, token: str):
    """Handle a text message — behavior depends on session mode and free_questions toggle."""
    from chat import run_chat, get_history, save_message
    from usage import check_rate_limit, increment_usage

    if chat_id:
        save_message(chat_id, "user", text)

    if session and session.get("mode") == "appointment_confirmed":
        # Only unambiguous BUTTON taps (Reschedule/Cancel — handled in
        # handle_interactive by exact button ID) get special-cased for this
        # mode. Free text of ANY kind hands off cleanly to open AI chat,
        # which now has its own book_appointment/cancel_appointment tools
        # and can handle "reschedule"/"cancel"/a brand new booking through
        # normal conversation. Loosely keyword-matching free text here was
        # the actual bug: "book for thursday" (a brand new booking request)
        # got misread as "reschedule my last appointment" just because it
        # contained the word "book". The session gets fully cleared (not
        # just its mode) so no stale flow_id/current_node_id from earlier
        # in the conversation can leak back in on the next message.
        delete_session(project_id, phone_number)
        _rag_reply(project_id, chat_id, text, phone_number, phone_number_id, token)
        return

    if session and session.get("mode") == "shop_browsing":
        send_whatsapp_message(
            phone_number,
            "Please complete your order on the menu page 😊\nTap the link we sent to continue.",
            phone_number_id,
            token,
        )
        return

    if session and session.get("mode") == "awaiting_cart_confirm":
        order_id = (session.get("metadata") or {}).get("order_id")
        catalog_id = (session.get("metadata") or {}).get("catalog_id", "")
        # Pass order_id so the shop page can pre-load existing cart items
        shop_url = f"{FRONTEND_URL}/shop/{project_id}?catalog={catalog_id}&phone={phone_number}&order_id={order_id}"

        if "continue" in text.lower():
            send_whatsapp_buttons(
                phone_number,
                "📝 *Any special requests for your order?*\n\nOnly preparation or packing notes — e.g. less spice, no onion, less oil, pack separately.\n\nTo add another dish, use the *menu* please. Type your note below, or tap *Skip* to continue.",
                [{"id": "skip_special_req", "title": "Skip"}],
                phone_number_id, token,
            )
            upsert_session(project_id, phone_number, {
                "mode": "awaiting_special_request",
                "metadata": {"order_id": order_id, "catalog_id": catalog_id},
            })
        elif "add more" in text.lower():
            send_whatsapp_cta_url(
                phone_number,
                "Browse and add more items to your cart 🛍️",
                "View Menu",
                shop_url,
                phone_number_id, token,
            )
        elif "clear cart" in text.lower():
            if order_id:
                supabase.table("orders").delete().eq("id", order_id).execute()
            shop_url_fresh = f"{FRONTEND_URL}/shop/{project_id}?catalog={catalog_id}&phone={phone_number}"
            send_whatsapp_cta_url(
                phone_number,
                "Your cart has been cleared. Start fresh! 🛒",
                "View Menu",
                shop_url_fresh,
                phone_number_id, token,
            )
            upsert_session(project_id, phone_number, {
                "mode": "shop_browsing",
                "metadata": {"catalog_id": catalog_id},
            })
        else:
            send_whatsapp_buttons(
                phone_number,
                "Please choose one of the options below, or type *menu* to start over.",
                [
                    {"id": "cart_continue", "title": "Continue ➡️"},
                    {"id": "cart_add_more", "title": "Add More 🛍️"},
                    {"id": "cart_clear", "title": "Clear Cart 🗑️"},
                ],
                phone_number_id, token,
            )
        return

    if session and session.get("mode") == "awaiting_special_request":
        order_id = (session.get("metadata") or {}).get("order_id")
        catalog_id = (session.get("metadata") or {}).get("catalog_id", "")
        special_request = None if text.lower().strip() in ("skip", "skip_special_req") else text

        if order_id and special_request:
            supabase.table("orders").update({"special_request": special_request}).eq("id", order_id).execute()

        order_res = supabase.table("orders").select("*").eq("id", order_id).single().execute()
        order = order_res.data
        config_res = supabase.table("shop_config").select("*").eq("project_id", project_id).maybe_single().execute()
        config = (config_res.data if config_res else None) or {}
        currency = config.get("currency", "₹")
        store_name = config.get("store_name", "")
        store_phone = config.get("store_phone", "")
        terms_note = config.get("terms_note", "")
        delivery_type = order.get("delivery_type", "Takeaway")

        lines = []
        for i, item in enumerate(order["items"], 1):
            lines.append(f"{i}. {item['name']} x{item['quantity']} - {currency}{int(item['price'] * item['quantity'])}")
        items_text = "\n".join(lines)

        summary = f"🎉 *All set! You've picked your menu.*\nHere's a quick summary:\n\n"
        summary += f"🧾 *Booked Items*\n\n{items_text}\n\n"
        summary += f"{'─' * 20}\n"
        summary += f"Subtotal: {currency}{int(order['subtotal'])}\n"
        if order["gst_amount"] > 0:
            gst_pct = config.get("gst_percent", 0)
            summary += f"GST ({gst_pct}%): {currency}{order['gst_amount']:.2f}\n"
        summary += f"\n*Total Amount: {currency}{order['total']:.2f}*\n\n"
        if store_name:
            summary += f"🏪 Assigned Store Name - *{store_name}*\n"
        if store_phone:
            summary += f"📞 Contact - {store_phone}\n\n"
        if terms_note:
            summary += f"ℹ️ Note - {terms_note}\n"
        summary += f"🏃 {delivery_type}"

        send_whatsapp_buttons(
            phone_number,
            summary,
            [
                {"id": "confirm_and_pay", "title": "Confirm & Pay"},
                {"id": "change_order", "title": "✏️ Change Order"},
                {"id": "cancel_order", "title": "❌ Cancel"},
            ],
            phone_number_id, token,
        )
        upsert_session(project_id, phone_number, {
            "mode": "awaiting_payment_confirm",
            "metadata": {"order_id": order_id, "catalog_id": catalog_id},
        })
        return

    if session and session.get("mode") == "awaiting_payment_confirm":
        from shop import generate_razorpay_link
        order_id = (session.get("metadata") or {}).get("order_id")
        catalog_id = (session.get("metadata") or {}).get("catalog_id", "")

        if "confirm" in text.lower() or "pay" in text.lower():
            order_res = supabase.table("orders").select("*").eq("id", order_id).single().execute()
            order = order_res.data
            config_res = supabase.table("shop_config").select("*").eq("project_id", project_id).maybe_single().execute()
            config = (config_res.data if config_res else None) or {}
            currency = config.get("currency", "₹")

            payment_url = generate_razorpay_link(order, config)

            if payment_url:
                send_whatsapp_cta_url(
                    phone_number,
                    f"💳 *Complete Your Payment*\n\nYour order is confirmed! Please complete the payment of {currency}{order['total']:.2f} to proceed.\n\nTap the button below to pay securely.\n⏰ Link expires in 90 minutes",
                    "Pay Now",
                    payment_url,
                    phone_number_id, token,
                )
                upsert_session(project_id, phone_number, {
                    "mode": "awaiting_payment",
                    "metadata": {"order_id": order_id},
                })
            else:
                send_whatsapp_message(
                    phone_number,
                    "✅ *Order Confirmed!*\n\nThank you! We'll contact you shortly to arrange payment.",
                    phone_number_id, token,
                )
                upsert_session(project_id, phone_number, {"mode": "flow", "metadata": {}})
        elif "change" in text.lower() or "edit" in text.lower() or "add" in text.lower():
            # Same pre-load pattern "Add More" already uses one stage
            # earlier — an explicit, unambiguous link rather than trying to
            # guess what to modify from free text.
            shop_url = f"{FRONTEND_URL}/shop/{project_id}?catalog={catalog_id}&phone={phone_number}&order_id={order_id}"
            send_whatsapp_cta_url(
                phone_number,
                "Sure! Tap below to change your order 🛍️",
                "Edit Order",
                shop_url,
                phone_number_id, token,
            )
        elif "cancel" in text.lower():
            if order_id:
                supabase.table("orders").update({"status": "cancelled"}).eq("id", order_id).execute()
            send_whatsapp_message(
                phone_number,
                "No problem, your order has been cancelled. Feel free to ask me anything else! 😊",
                phone_number_id, token,
            )
            # Fully clear the session (not just its mode) so the next
            # message goes cleanly to open chat — same fix as the earlier
            # appointment-confirmed stale-session bug, not a partial reset.
            delete_session(project_id, phone_number)
        else:
            send_whatsapp_buttons(
                phone_number,
                "Please tap *Confirm & Pay* to proceed, *Change Order* to edit it, *Cancel* to cancel, or type *menu* to start over.",
                [
                    {"id": "confirm_and_pay", "title": "Confirm & Pay"},
                    {"id": "change_order", "title": "✏️ Change Order"},
                    {"id": "cancel_order", "title": "❌ Cancel"},
                ],
                phone_number_id, token,
            )
        return

    if session and session.get("mode") == "awaiting_payment":
        flow_check = get_active_flow(project_id)
        keywords = [k.lower() for k in (flow_check.get("trigger_keywords") or [])] if flow_check else []
        if text.lower().strip() in keywords and flow_check:
            start_flow(flow_check, project_id, phone_number, phone_number_id, token, chat_id)
            return
        send_whatsapp_message(
            phone_number,
            "⏳ Please complete your payment using the link we sent. Tap *Pay Now* to proceed, or type *menu* to start a new order.",
            phone_number_id, token,
        )
        return

    if not session:
        flow = get_active_flow(project_id)
        if flow:
            keywords = [k.lower() for k in (flow.get("trigger_keywords") or [])]
            if text.lower().strip() in keywords:
                start_flow(flow, project_id, phone_number, phone_number_id, token, chat_id)
                return
        _rag_reply(project_id, chat_id, text, phone_number, phone_number_id, token)
        return

    mode = session.get("mode", "flow")

    if mode == "human":
        return

    if mode == "rag_question":
        rate_check = check_rate_limit(project_id)
        if not rate_check["allowed"]:
            send_whatsapp_message(phone_number, "⚠️ Monthly message limit reached.", phone_number_id, token)
            return
        history = get_history(chat_id, limit=5)
        result = run_chat(project_id, chat_id, text, history)
        send_back_to_menu_button(phone_number, result["answer"], phone_number_id, token)
        increment_usage(project_id)
        return

    flow_id = session.get("flow_id")
    current_node_id = session.get("current_node_id")
    # Scoped to the session's own flow: a stale or tampered current_node_id
    # must not resolve to a node in someone else's flow.
    current_node = get_node(current_node_id, flow_id=flow_id) if current_node_id else None

    if not current_node:
        flow = get_active_flow(project_id)
        if flow:
            keywords = [k.lower() for k in (flow.get("trigger_keywords") or [])]
            if text.lower().strip() in keywords:
                start_flow(flow, project_id, phone_number, phone_number_id, token)
        return

    flow = supabase.table("flows").select("free_questions, trigger_keywords").eq("id", flow_id).single().execute()
    free_questions = flow.data.get("free_questions", False) if flow.data else False

    if current_node["type"] in ("buttons", "list"):
        if free_questions:
            rate_check = check_rate_limit(project_id)
            if not rate_check["allowed"]:
                send_whatsapp_message(phone_number, "⚠️ Monthly message limit reached.", phone_number_id, token)
                return
            history = get_history(chat_id, limit=5)
            result = run_chat(project_id, chat_id, text, history)
            send_whatsapp_message(phone_number, result["answer"], phone_number_id, token)
            send_node(current_node, phone_number, phone_number_id, token, project_id=project_id)
            increment_usage(project_id)
        else:
            send_node(current_node, phone_number, phone_number_id, token, project_id=project_id)

    elif current_node["type"] == "rag":
        rate_check = check_rate_limit(project_id)
        if not rate_check["allowed"]:
            send_whatsapp_message(phone_number, "⚠️ Monthly message limit reached.", phone_number_id, token)
            return
        history = get_history(chat_id, limit=5)
        result = run_chat(project_id, chat_id, text, history)
        send_back_to_menu_button(phone_number, result["answer"], phone_number_id, token)
        increment_usage(project_id)

    elif current_node["type"] == "text":
        next_node = get_next_node(flow_id, current_node_id, text.lower())
        if next_node:
            upsert_session(project_id, phone_number, {
                "flow_id": flow_id,
                "current_node_id": next_node["id"],
                "mode": "flow",
            })
            send_node(next_node, phone_number, phone_number_id, token, project_id=project_id)
        else:
            send_node(current_node, phone_number, phone_number_id, token, project_id=project_id)

    else:
        flow_data = get_active_flow(project_id)
        if flow_data:
            keywords = [k.lower() for k in (flow_data.get("trigger_keywords") or [])]
            if text.lower().strip() in keywords:
                start_flow(flow_data, project_id, phone_number, phone_number_id, token, chat_id)
                return

        flow_row = supabase.table("flows").select("free_questions").eq("id", flow_id).single().execute()
        free_q = flow_row.data.get("free_questions", False) if flow_row.data else False

        if free_q:
            rate_check = check_rate_limit(project_id)
            if not rate_check["allowed"]:
                send_whatsapp_message(phone_number, "⚠️ Monthly message limit reached.", phone_number_id, token)
                return
            history = get_history(chat_id, limit=5)
            result = run_chat(project_id, chat_id, text, history)
            send_back_to_menu_button(phone_number, result["answer"], phone_number_id, token)
            increment_usage(project_id)
        else:
            send_node(current_node, phone_number, phone_number_id, token, project_id=project_id)


def _rag_reply(project_id, chat_id, text, phone_number, phone_number_id, token):
    from chat import run_chat, get_history
    from usage import check_rate_limit, increment_usage

    rate_check = check_rate_limit(project_id)
    if not rate_check["allowed"]:
        send_whatsapp_message(phone_number, "⚠️ Monthly message limit reached.", phone_number_id, token)
        return
    history = get_history(chat_id, limit=5)
    result = run_chat(project_id, chat_id, text, history)
    send_whatsapp_message(phone_number, result["answer"], phone_number_id, token)
    increment_usage(project_id)


# -------------------------------------------------
# FLOW CRUD
# -------------------------------------------------
# Removed. This module used to expose a second, parallel implementation of
# every flow CRUD endpoint (list/create/update/delete flows, nodes, edges,
# and a sync handler), mounted publicly via main.py.
#
# Nothing called it: FlowsTab.js talks only to the Next.js routes under
# src/app/api/flows/. What it did do was duplicate every authorization
# decision — using require_project_role, which passes for ANY role, so an
# agent with no flows permission could drive it — and repeat the same
# destructive delete-then-reinsert sync that the Next.js route has now
# replaced with the transactional sync_flow_graph database function.
#
# Two divergent copies of the same authorization logic is strictly worse
# than one. The runtime above (which whatsapp.py imports) is what this
# module is actually for.
