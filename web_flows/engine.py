"""The website flow state machine.

Pure logic: all I/O goes through two injected objects, so the same engine
serves real visitors (store.SupabaseStore + store.LiveEffects), the editor's
Preview (nothing saved) and unit tests (in-memory fakes).

Store:   get_active_web_flow(project_id) -> flow | None
         get_start_node(flow_id) / get_node(node_id, flow_id) -> node | None
         get_edges(flow_id) -> [{from_node_id, trigger, to_node_id}]
         load_session(chat_id) -> session | None   (None if expired)
         create_session(session) -> session
         claim(chat_id, seq) -> session | None     (compare-and-swap seq -> seq+1)
         save_session(session)
Effects: record(session, entries, visitor_acted)   (transcript / chats rows)
         upsert_lead(project_id, visitor_id, fields, custom) -> lead_id | None
         log_events(events)
         now() -> aware datetime ; today() -> date ; random() -> float in [0,1)

One request = claim the session, apply the visitor's action to the node that
was waiting for it, then auto-advance until a node needs input again, the
flow ends, or a long delay starts. A hard cap per request (25 nodes) stops
a merchant's A->B->A loop from spinning forever.

Everything returned to the browser is built here from an explicit public
projection of each node, so internal settings never leak.
"""
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlsplit

from flow_common import option_id
from .templating import render_text, set_variable
from .validation import check_value, check_rating, INPUT_TYPES, FIELD_TYPES

SESSION_HOURS = 7 * 24        # matches the widget's 7-day chat session
MAX_AUTO_STEPS = 25
MAX_OPTIONS = 50
MAX_CARDS = 10
MAX_CARD_BUTTONS = 3
MAX_FORM_FIELDS = 10
IDLE_RESTART_HOURS = 2       # like WhatsApp: quiet this long -> next message opens the menu
INLINE_DELAY_MAX_S = 10      # up to this: a typing pause, no round trip
MAX_DELAY_S = 600            # web delays are capped at 10 minutes

# Removed from the product; may still sit in an old saved flow.
REMOVED_TYPES = {"condition", "set_variable", "random_split", "webhook"}

INPUT_NODE_TYPES = {"message_buttons", "quick_replies", "message_list", "carousel",
                    "ask_input", "form", "rating"}
RESTART_NOTICE = "This conversation was updated - let's start again."
REPROMPT = "Please choose one of the options above."


def _safe_url(url, schemes=("http", "https")):
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in schemes:
        return None
    if parts.scheme.lower() in ("http", "https") and not parts.netloc:
        return None
    return url


def _tel(phone):
    digits = "".join(c for c in str(phone or "") if c.isdigit() or c == "+")
    return f"tel:{digits}" if len([c for c in digits if c.isdigit()]) >= 5 else None


def _options(node):
    """Choice options of an input node as [{id, label}] (the handle ids)."""
    t, c = node.get("type"), node.get("content") or {}
    raw = []
    if t == "message_buttons":
        raw = c.get("buttons") or []
    elif t == "quick_replies":
        raw = c.get("options") or []
    elif t == "message_list":
        for sec in c.get("sections") or []:
            raw.extend(sec.get("rows") or [])
    elif t == "carousel":
        for card in (c.get("cards") or [])[:MAX_CARDS]:
            raw.extend(b for b in (card.get("buttons") or [])[:MAX_CARD_BUTTONS] if not b.get("url"))
    out, seen = [], set()
    for o in raw[:MAX_OPTIONS]:
        label = str(o.get("label") or o.get("title") or "").strip()
        oid = str(o.get("id") or "").strip() or option_id(label)
        if label and oid not in seen:
            seen.add(oid)
            out.append({"id": oid, "label": label[:80]})
    return out


class Ctx:
    """Per-request working state."""
    def __init__(self, flow):
        self.flow = flow
        self.edges = None
        self.nodes = {}
        self.messages = []      # public envelope messages
        self.transcript = []    # [{"role", "content"}] saved to the chat
        self.events = []
        self.steps = 0
        self.continue_after_ms = None
        self.error = None


class Engine:
    def __init__(self, store, effects, frontend_url=""):
        self.store = store
        self.fx = effects
        self.frontend_url = (frontend_url or "").rstrip("/")

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #
    def start(self, project_id, chat_id, visitor_id, via="open", page=None, text=None):
        flow = self.store.get_active_web_flow(project_id)
        if not flow:
            return self._bare(chat_id, 0, "inactive")
        existing = self.store.load_session(chat_id) if chat_id else None
        if existing and (existing.get("project_id") != project_id or existing.get("visitor_id") != visitor_id):
            existing, chat_id = None, None
        if existing:
            sess = self.store.claim(chat_id, existing["seq"])
            if not sess:
                return self._resync(existing)
        else:
            sess = self.store.create_session({
                "chat_id": chat_id or str(uuid.uuid4()), "project_id": project_id,
                "visitor_id": visitor_id, "flow_id": flow["id"], "mode": "flow",
                "variables": {}, "seq": 1, "pending_transcript": [],
                "started_via": str(via or "open")[:40],
                "expires_at": (self.fx.now() + timedelta(hours=SESSION_HOURS)).isoformat(),
            })
        self._set_page(sess, page)
        ctx = Ctx(flow)
        ctx.events.append(("start", None, {"via": str(via or "open")[:40]}))
        text = str(text or "").strip()[:4000]
        if text:
            # Their message opened the flow: keep it in the conversation.
            ctx.transcript.append({"role": "user", "content": text})
        self._restart(sess, ctx)
        return self._finish_request(sess, ctx, visitor_acted=bool(text))

    def resume(self, project_id, chat_id, visitor_id):
        sess = self._load_owned(project_id, chat_id, visitor_id)
        if not sess:
            return self._bare(chat_id, 0, "expired")
        flow = self.store.get_active_web_flow(project_id)
        if not flow or flow["id"] != sess.get("flow_id"):
            return self._bare(chat_id, sess["seq"], "inactive" if not flow else "expired")
        env = self._resync(sess)
        if not sess.get("chat_materialized"):
            # Never interacted (e.g. auto-opened), so history restore found
            # nothing: replay what the bot said.
            env["replay"] = list(sess.get("pending_transcript") or [])
        aw = sess.get("awaiting") or {}
        if aw.get("kind") == "delay" and sess.get("resume_at"):
            env["continueAfterMs"] = self._ms_until(sess["resume_at"])
        return env

    def step(self, project_id, chat_id, visitor_id, seq, node_id, action, page=None):
        loaded = self._load_owned(project_id, chat_id, visitor_id)
        if not loaded:
            return self._bare(chat_id, 0, "expired")
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            return self._resync(loaded)
        sess = self.store.claim(chat_id, seq)
        if not sess:
            return self._resync(loaded)   # double tap / second tab / race

        self._set_page(sess, page)
        flow = self.store.get_active_web_flow(project_id)
        if not flow:
            sess["mode"], sess["awaiting"] = "ended", None
            self.store.save_session(sess)
            return self._bare(chat_id, sess["seq"], "inactive")
        ctx = Ctx(flow)
        if flow["id"] != sess.get("flow_id"):
            ctx.messages.append({"kind": "text", "text": RESTART_NOTICE})
            self._restart(sess, ctx)
            return self._finish_request(sess, ctx, visitor_acted=True)

        action = action if isinstance(action, dict) else {}
        kind = action.get("type")
        # Like WhatsApp: after 2 hours of quiet (visitor and team), the next
        # message or tap starts the flow again - whatever mode it was in.
        if kind != "continue" and self._idle(loaded):
            if kind == "text" and str(action.get("text") or "").strip():
                ctx.transcript.append({"role": "user", "content": str(action["text"]).strip()[:4000]})
            ctx.events.append(("idle_restart", None, {}))
            self._restart(sess, ctx)
            return self._finish_request(sess, ctx, visitor_acted=True)
        if kind == "menu":
            ctx.transcript.append({"role": "user", "content": "Menu"})
            self._restart(sess, ctx)
            return self._finish_request(sess, ctx, visitor_acted=True)
        if kind == "text":
            return self._on_text(sess, ctx, str(action.get("text") or "").strip()[:4000])

        # Everything else answers the node that is waiting.
        aw = sess.get("awaiting") or {}
        if not aw or str(node_id or "") != str(aw.get("node_id") or ""):
            return self._resync(sess, saved=True)
        node = self._node(ctx, aw["node_id"])
        if not node:
            ctx.messages.append({"kind": "text", "text": RESTART_NOTICE})
            self._restart(sess, ctx)
            return self._finish_request(sess, ctx, visitor_acted=True)
        handled = self._apply_action(sess, ctx, node, aw, kind, action)
        if handled is None:
            return self._resync(sess, saved=True)
        return self._finish_request(sess, ctx, visitor_acted=handled)

    # ------------------------------------------------------------------ #
    # Visitor actions
    # ------------------------------------------------------------------ #
    def _apply_action(self, sess, ctx, node, aw, kind, action):
        """Returns True when handled (visitor acted), None when the action
        doesn't fit what's awaited (caller resyncs)."""
        c = node.get("content") or {}
        variables = sess.setdefault("variables", {})
        t = node.get("type")

        if kind == "choice" and aw.get("kind") in ("choices", "carousel"):
            opt = next((o for o in _options(node) if o["id"] == str(action.get("id") or "")), None)
            if not opt:
                return None
            if c.get("var"):
                set_variable(variables, c["var"], opt["label"])
            ctx.transcript.append({"role": "user", "content": opt["label"]})
            ctx.events.append(("choice", node["id"], {"option": opt["id"]}))
            self._advance(sess, ctx, node, opt["id"])
            return True

        if kind == "field" and aw.get("kind") == "field":
            return self._answer_field(sess, ctx, node, action.get("value"))

        if kind == "form" and aw.get("kind") == "form" and t == "form":
            values = action.get("values") if isinstance(action.get("values"), dict) else {}
            errors, clean = {}, {}
            for f in self._form_fields(node):
                ok, value, err = check_value(f["type"], values.get(f["name"]), f, self.fx.today())
                if ok:
                    clean[f["name"]] = value
                else:
                    errors[f["name"]] = err
            if errors:
                ctx.error = {"code": "invalid", "message": "Please check the highlighted fields.", "fields": errors}
                ctx.events.append(("invalid", node["id"], {"fields": list(errors)[:10]}))
                self._wait_on(sess, ctx, node)
                return True
            for name, value in clean.items():
                set_variable(variables, name, value)
            labels = {f["name"]: f["label"] for f in self._form_fields(node)}
            summary = ", ".join(
                f"{labels.get(k) or k}: {'yes' if v is True else ('no' if v is False else v)}"
                for k, v in clean.items() if v not in ("", None))
            ctx.transcript.append({"role": "user", "content": summary or "(form submitted)"})
            ctx.events.append(("submit", node["id"], {}))
            if c.get("save_lead"):
                lead_fields = {k: clean.get(k) for k in ("name", "email", "phone") if clean.get(k)}
                custom = {k: v for k, v in clean.items() if k not in ("name", "email", "phone") and v not in ("", None)}
                if lead_fields.get("email") or lead_fields.get("phone"):
                    lead_id = self.fx.upsert_lead(sess["project_id"], sess.get("visitor_id"), lead_fields, custom)
                    if lead_id:
                        sess["lead_id"] = lead_id
            self._advance(sess, ctx, node, "next")
            return True

        if kind == "rating" and aw.get("kind") == "rating":
            style = c.get("style") if c.get("style") in ("stars", "nps") else "stars"
            ok, value, err = check_rating(action.get("value"), style)
            if not ok:
                ctx.error = {"code": "invalid", "message": err, "fields": {}}
                self._wait_on(sess, ctx, node)
                return True
            set_variable(variables, c.get("var") or "rating", value)
            ctx.transcript.append({"role": "user", "content": f"Rated {value}/{10 if style == 'nps' else 5}"})
            ctx.events.append(("submit", node["id"], {"rating": value}))
            self._advance(sess, ctx, node, "next")
            return True

        if kind == "continue" and aw.get("kind") == "delay":
            remaining = self._ms_until(sess.get("resume_at")) if sess.get("resume_at") else 0
            if remaining > 2000:
                ctx.continue_after_ms = remaining
                return True
            sess["resume_at"] = None
            ctx.events.append(("pass", node["id"], {}))
            self._advance(sess, ctx, node, "next")
            return False
        return None

    def _answer_field(self, sess, ctx, node, raw):
        c = node.get("content") or {}
        ftype = c.get("input_type") if c.get("input_type") in INPUT_TYPES else "text"
        ok, value, err = check_value(ftype, raw, {**c, "required": c.get("required", True)}, self.fx.today())
        if not ok:
            ctx.error = {"code": "invalid", "message": err, "fields": {"value": err}}
            ctx.events.append(("invalid", node["id"], {}))
            self._wait_on(sess, ctx, node)
            return True
        set_variable(sess.setdefault("variables", {}), c.get("var") or "answer", value)
        ctx.transcript.append({"role": "user", "content": str(value) if value not in ("", None) else "(skipped)"})
        ctx.events.append(("submit", node["id"], {}))
        self._advance(sess, ctx, node, "next")
        return True

    def _on_text(self, sess, ctx, text):
        if not text:
            return self._resync(sess, saved=True)
        mode = sess.get("mode")
        free = bool(ctx.flow.get("free_questions"))

        # Like WhatsApp: a menu keyword ("hi", "menu", ...) starts the flow
        # again - except while a person is handling the chat, and unless it
        # is exactly one of the options currently on screen.
        if mode != "human" and self._is_keyword(ctx, text) and not self._matches_option(sess, ctx, text):
            ctx.transcript.append({"role": "user", "content": text})
            self._restart(sess, ctx)
            return self._finish_request(sess, ctx, visitor_acted=True)

        if mode == "human":
            ctx.transcript.append({"role": "user", "content": text})
            self.store.save_session(sess)
            self.fx.record(sess, ctx.transcript, True)
            return self._envelope(sess, ctx, status="human")
        if mode == "ai":
            self.store.save_session(sess)
            return self._envelope(sess, ctx, status="ai", delegate="ai")
        if mode == "ended" or not sess.get("awaiting"):
            if free:
                sess["mode"] = "ai"
                self.store.save_session(sess)
                return self._envelope(sess, ctx, status="ai", delegate="ai")
            ctx.transcript.append({"role": "user", "content": text})
            self._restart(sess, ctx)
            return self._finish_request(sess, ctx, visitor_acted=True)

        aw = sess["awaiting"]
        if aw.get("kind") == "delay":
            # Mid-wait: the next message is already scheduled; let the AI
            # answer if allowed, otherwise just keep waiting.
            self.store.save_session(sess)
            if free:
                return self._envelope(sess, ctx, status="flow", delegate="ai")
            env = self._envelope(sess, ctx, status="flow")
            env["continueAfterMs"] = self._ms_until(sess.get("resume_at")) if sess.get("resume_at") else 0
            return env
        node = self._node(ctx, aw.get("node_id"))
        if not node:
            ctx.transcript.append({"role": "user", "content": text})
            ctx.messages.append({"kind": "text", "text": RESTART_NOTICE})
            self._restart(sess, ctx)
            return self._finish_request(sess, ctx, visitor_acted=True)
        if aw.get("kind") == "field":
            self._answer_field(sess, ctx, node, text)
            return self._finish_request(sess, ctx, visitor_acted=True)
        if aw.get("kind") in ("choices", "carousel"):
            # Typing an option's exact label counts as tapping it.
            opt = next((o for o in _options(node) if o["label"].lower() == text.lower()), None)
            if opt:
                self._apply_action(sess, ctx, node, aw, "choice", {"id": opt["id"]})
                return self._finish_request(sess, ctx, visitor_acted=True)
        if free:
            # The widget asks /public/chat (which saves both turns), then
            # shows this same input again.
            self.store.save_session(sess)
            return self._envelope(sess, ctx, status="flow", delegate="ai", input=self._public_input(sess, node))
        ctx.transcript.append({"role": "user", "content": text})
        prompt = {"form": "Please fill in the form above.",
                  "rating": "Please pick a rating above."}.get(aw.get("kind"), REPROMPT)
        ctx.messages.append({"kind": "text", "text": prompt})
        ctx.transcript.append({"role": "assistant", "content": prompt})
        self._wait_on(sess, ctx, node)
        return self._finish_request(sess, ctx, visitor_acted=True)

    # ------------------------------------------------------------------ #
    # Running nodes
    # ------------------------------------------------------------------ #
    def _idle(self, sess):
        """True when neither the visitor nor the team has done anything for
        IDLE_RESTART_HOURS (updated_at moves on every visitor action)."""
        latest = None
        for key in ("updated_at", "last_agent_msg_at"):
            value = sess.get(key)
            if not value:
                continue
            try:
                when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            latest = when if latest is None or when > latest else latest
        return latest is not None and self.fx.now() - latest > timedelta(hours=IDLE_RESTART_HOURS)

    @staticmethod
    def _is_keyword(ctx, text):
        words = [str(k).strip().lower() for k in (ctx.flow.get("trigger_keywords") or []) if str(k).strip()]
        return text.strip().lower() in words

    def _matches_option(self, sess, ctx, text):
        aw = sess.get("awaiting") or {}
        if aw.get("kind") not in ("choices", "carousel"):
            return False
        node = self._node(ctx, aw.get("node_id"))
        return bool(node) and any(o["label"].lower() == text.strip().lower() for o in _options(node))

    def _restart(self, sess, ctx):
        sess.update({"flow_id": ctx.flow["id"], "mode": "flow", "awaiting": None,
                     "current_node_id": None, "resume_at": None})
        start = self.store.get_start_node(ctx.flow["id"])
        if not start:
            self._end(sess, ctx, None)
            return
        ctx.nodes[start["id"]] = start
        self._run_from(sess, ctx, start)

    def _advance(self, sess, ctx, node, handle):
        sess["awaiting"] = None
        target = self._next(ctx, node, handle)
        if target:
            self._run_from(sess, ctx, target)
        else:
            self._end(sess, ctx, node)

    def _run_from(self, sess, ctx, node):
        variables = sess.setdefault("variables", {})
        while node:
            ctx.steps += 1
            if ctx.steps > MAX_AUTO_STEPS:
                ctx.events.append(("loop_guard", node["id"], {}))
                self._end(sess, ctx, node)
                return
            sess["current_node_id"] = node["id"]
            ctx.events.append(("enter", node["id"], {}))
            t, c = node.get("type"), node.get("content") or {}

            if t in INPUT_NODE_TYPES:
                body = render_text(c.get("body"), variables)
                if body:
                    self._say(ctx, body)
                self._wait_on(sess, ctx, node)
                return
            if t in REMOVED_TYPES:
                # Condition / set variable / random split / webhook were
                # removed from the product. A test flow saved with one ends
                # here quietly instead of showing its raw settings.
                self._end(sess, ctx, node)
                return
            if t == "time_delay":
                secs = self._delay_seconds(c)
                if secs <= INLINE_DELAY_MAX_S:
                    if secs > 0:
                        ctx.messages.append({"kind": "pause", "ms": secs * 1000})
                    node = self._next(ctx, node, "next")
                    continue
                sess["resume_at"] = (self.fx.now() + timedelta(seconds=secs)).isoformat()
                sess["awaiting"] = {"kind": "delay", "node_id": node["id"]}
                ctx.continue_after_ms = secs * 1000
                return
            if t == "back_to_menu":
                start = self.store.get_start_node(ctx.flow["id"])
                node = start
                continue
            if t == "ask_a_question":
                self._say(ctx, render_text(c.get("body") or "You can now ask me anything!", variables))
                sess.update({"mode": "ai", "awaiting": None})
                return
            if t in ("talk_to_human", "handoff"):
                self._say(ctx, render_text(c.get("body") or "Connecting you to our team. Please wait...", variables))
                sess.update({"mode": "human", "awaiting": None})
                return
            if t == "end":
                if c.get("body"):
                    self._say(ctx, render_text(c.get("body"), variables))
                sess.update({"mode": "ended", "awaiting": None})
                return

            # Output-only nodes: say something, then continue on "next".
            self._output(sess, ctx, node)
            node = self._next(ctx, node, "next")
            if node is None:
                self._end(sess, ctx, None)
                return
        self._end(sess, ctx, None)

    def _end(self, sess, ctx, node):
        """The flow ran out of connections. Like WhatsApp: with "AI answers
        typed messages" on, the visitor's next message goes to AI; otherwise
        their next message starts the flow again."""
        sess["awaiting"] = None
        sess["mode"] = "ai" if ctx.flow.get("free_questions") else "ended"

    def _wait_on(self, sess, ctx, node):
        kind = {"ask_input": "field", "form": "form", "rating": "rating", "carousel": "carousel"}.get(node["type"], "choices")
        sess["awaiting"] = {"kind": kind, "node_id": node["id"]}
        sess["current_node_id"] = node["id"]
        sess["mode"] = "flow"
        ctx.nodes[node["id"]] = node

    @staticmethod
    def _delay_seconds(c):
        try:
            amount = int(float(c.get("delay_seconds", 5)))
        except (TypeError, ValueError):
            amount = 5
        unit = {"seconds": 1, "minutes": 60, "hours": 3600}.get(c.get("delay_unit"), 1)
        return max(0, min(amount * unit, MAX_DELAY_S))

    # ------------------------------------------------------------------ #
    # What the visitor sees
    # ------------------------------------------------------------------ #
    def _say(self, ctx, text):
        ctx.messages.append({"kind": "text", "text": text})
        ctx.transcript.append({"role": "assistant", "content": text})

    def _output(self, sess, ctx, node):
        t, c = node.get("type"), node.get("content") or {}
        v = sess.get("variables") or {}
        body = render_text(c.get("body"), v)
        media_kinds = {"message_media": ("image", "media_url"), "message_video": ("video", "video_url"),
                       "message_audio": ("audio", "audio_url"), "message_document": ("file", "document_url")}
        if t in media_kinds:
            kind, key = media_kinds[t]
            url = _safe_url(c.get(key), ("https",))
            if url:
                msg = {"kind": kind, "url": url, "caption": body}
                if kind == "file":
                    msg["name"] = str(c.get("filename") or "Document")[:120]
                ctx.messages.append(msg)
                ctx.transcript.append({"role": "assistant", "content": (body + "\n" if body else "") + url})
            elif body:
                self._say(ctx, body)
            return
        if t == "message_location":
            name = render_text(c.get("name"), v, 200)
            address = render_text(c.get("address"), v, 300)
            lat, lng = c.get("latitude"), c.get("longitude")
            try:
                lat, lng = float(lat), float(lng)
                query = f"{lat},{lng}" if -90 <= lat <= 90 and -180 <= lng <= 180 else None
            except (TypeError, ValueError):
                query = None
            query = query or (address or name)
            text = "\n".join(x for x in (body, name, address) if x) or "Our location"
            if query:
                self._link(ctx, text, "Open in Maps",
                           "https://www.google.com/maps/search/?api=1&query=" + quote(str(query), safe=","))
            else:
                self._say(ctx, text)
            return
        if t == "call_us":
            url = _tel(c.get("phone"))
            if url:
                self._link(ctx, body or "Need help? Call us directly!", c.get("button_text") or "Call us", url)
            elif body:
                self._say(ctx, body)
            return
        if t == "open_url":
            url = _safe_url(render_text(c.get("url"), v, 2000), ("http", "https", "mailto"))
            if url:
                self._link(ctx, body, c.get("button_text") or "Open", url)
            elif body:
                self._say(ctx, body)
            return
        if t == "message_shop":
            url = f"{self.frontend_url}/shop/{sess['project_id']}"
            if c.get("catalog_id"):
                url += "?catalog=" + quote(str(c["catalog_id"]), safe="")
            self._link(ctx, body or "Browse our menu", c.get("button_text") or "View Menu", url)
            return
        if t == "message_booking":
            url = f"{self.frontend_url}/book/{sess['project_id']}"
            phone = "".join(ch for ch in str(v.get("phone") or "") if ch.isdigit())
            if 7 <= len(phone) <= 15:
                url += "?phone=" + phone
            self._link(ctx, body or "Book your appointment", c.get("button_text") or "Book Appointment", url)
            return
        if t == "message_event":
            links = []
            if c.get("event_id"):
                links.append({"label": str(c.get("button_text") or "Register Now")[:40],
                              "url": f"{self.frontend_url}/event/{quote(str(c['event_id']), safe='')}"})
            tel = _tel(c.get("contact_phone"))
            if tel:
                links.append({"label": "Call to attend", "url": tel})
            card = {"kind": "card", "image": _safe_url(c.get("banner_url"), ("https",)),
                    "title": "", "text": body or "Register now - limited spots available!", "links": links}
            ctx.messages.append(card)
            ctx.transcript.append({"role": "assistant", "content": card["text"] + "".join(
                f"\n{l['label']}: {l['url']}" for l in links)})
            return
        # message / legacy text / anything unknown: plain text if any.
        if body:
            self._say(ctx, body)

    def _link(self, ctx, text, label, url):
        ctx.messages.append({"kind": "link", "text": text or "", "label": str(label)[:40], "url": url})
        ctx.transcript.append({"role": "assistant", "content": (text + "\n" if text else "") + f"{label}: {url}"})

    def _form_fields(self, node):
        out = []
        for f in ((node.get("content") or {}).get("fields") or [])[:MAX_FORM_FIELDS]:
            name = str((f or {}).get("name") or "").strip()
            ftype = (f or {}).get("type") if (f or {}).get("type") in FIELD_TYPES else "text"
            if not name:
                continue
            out.append({"name": name, "type": ftype, "label": str(f.get("label") or name)[:80],
                        "required": bool(f.get("required", True)),
                        "placeholder": str(f.get("placeholder") or "")[:80],
                        "options": [str(o)[:80] for o in (f.get("options") or [])[:20]],
                        "min": f.get("min"), "max": f.get("max")})
        return out

    def _public_input(self, sess, node):
        if not node:
            return None
        t, c = node.get("type"), node.get("content") or {}
        v = sess.get("variables") or {}
        if t in ("message_buttons", "quick_replies", "message_list"):
            out = {"kind": "choices", "layout": {"message_buttons": "buttons", "quick_replies": "chips",
                                                 "message_list": "list"}[t], "options": _options(node)}
            if t == "message_list":
                out["buttonText"] = str(c.get("button_text") or "View options")[:40]
            return out
        if t == "carousel":
            cards = []
            for card in (c.get("cards") or [])[:MAX_CARDS]:
                buttons = []
                for b in (card.get("buttons") or [])[:MAX_CARD_BUTTONS]:
                    label = str(b.get("label") or "").strip()[:40]
                    if not label:
                        continue
                    if b.get("url"):
                        url = _safe_url(render_text(b.get("url"), v, 2000), ("http", "https", "tel", "mailto"))
                        if url:
                            buttons.append({"label": label, "url": url})
                    else:
                        buttons.append({"label": label, "id": str(b.get("id") or "") or option_id(label)})
                cards.append({"image": _safe_url(card.get("image"), ("https",)),
                              "title": render_text(card.get("title"), v, 80),
                              "text": render_text(card.get("text"), v, 300), "buttons": buttons})
            return {"kind": "carousel", "cards": cards}
        if t == "ask_input":
            return {"kind": "field", "type": c.get("input_type") if c.get("input_type") in INPUT_TYPES else "text",
                    "placeholder": str(c.get("placeholder") or "")[:80],
                    "required": bool(c.get("required", True)), "min": c.get("min"), "max": c.get("max")}
        if t == "form":
            return {"kind": "form", "title": render_text(c.get("title"), v, 120),
                    "submitLabel": str(c.get("submit_label") or "Send")[:30],
                    "fields": [{k: f[k] for k in ("name", "type", "label", "required", "placeholder", "options", "min", "max")}
                               for f in self._form_fields(node)]}
        if t == "rating":
            return {"kind": "rating", "style": c.get("style") if c.get("style") in ("stars", "nps") else "stars"}
        return None

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    def _node(self, ctx, node_id):
        if not node_id:
            return None
        if node_id not in ctx.nodes:
            ctx.nodes[node_id] = self.store.get_node(node_id, ctx.flow["id"])
        return ctx.nodes[node_id]

    def _next(self, ctx, node, handle):
        if ctx.edges is None:
            ctx.edges = {}
            for e in self.store.get_edges(ctx.flow["id"]) or []:
                ctx.edges[(e["from_node_id"], e.get("trigger") or "next")] = e["to_node_id"]
        to = ctx.edges.get((node["id"], handle))
        return self._node(ctx, to) if to else None

    def _load_owned(self, project_id, chat_id, visitor_id):
        if not chat_id:
            return None
        sess = self.store.load_session(chat_id)
        if not sess or sess.get("project_id") != project_id or sess.get("visitor_id") != visitor_id:
            return None
        return sess

    def _set_page(self, sess, page):
        if not isinstance(page, dict):
            return
        v = sess.setdefault("variables", {})
        for key, name, n in (("url", "page_url", 500), ("path", "page_path", 300), ("title", "page_title", 200)):
            if page.get(key):
                v[name] = str(page[key])[:n]

    def _ms_until(self, iso):
        from datetime import datetime
        try:
            when = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        except ValueError:
            return 0
        return max(0, int((when - self.fx.now()).total_seconds() * 1000))

    def _finish_request(self, sess, ctx, visitor_acted):
        sess["expires_at"] = (self.fx.now() + timedelta(hours=SESSION_HOURS)).isoformat()
        self.store.save_session(sess)
        if ctx.transcript or visitor_acted:
            self.fx.record(sess, ctx.transcript, visitor_acted)
        if ctx.events:
            self.fx.log_events([
                {"project_id": sess["project_id"], "flow_id": ctx.flow["id"], "chat_id": sess["chat_id"],
                 "node_id": n, "event": e, "meta": m} for e, n, m in ctx.events])
        aw = sess.get("awaiting") or {}
        node = self._node(ctx, aw.get("node_id")) if aw.get("kind") not in (None, "delay") else None
        status = {"ai": "ai", "human": "human", "ended": "ended"}.get(sess.get("mode"), "flow")
        return self._envelope(sess, ctx, status=status, input=self._public_input(sess, node))

    def _envelope(self, sess, ctx, status, delegate=None, input=None):
        aw = sess.get("awaiting") or {}
        return {
            "sessionId": sess["chat_id"], "seq": sess["seq"], "status": status,
            "nodeId": aw.get("node_id"), "messages": ctx.messages if ctx else [],
            "input": input, "continueAfterMs": ctx.continue_after_ms if ctx else None,
            "delegate": delegate, "error": ctx.error if ctx else None,
            "menuChip": sess.get("mode") == "ai",
        }

    def _resync(self, sess, saved=False):
        """The visitor's view is stale: return the current input, no effects."""
        flow = self.store.get_active_web_flow(sess["project_id"])
        ctx = Ctx(flow or {"id": sess.get("flow_id")})
        if saved:
            self.store.save_session(sess)
        aw = sess.get("awaiting") or {}
        node = self._node(ctx, aw.get("node_id")) if flow and aw.get("kind") not in (None, "delay") else None
        status = {"ai": "ai", "human": "human", "ended": "ended"}.get(sess.get("mode"), "flow")
        env = self._envelope(sess, ctx, status=status, input=self._public_input(sess, node))
        env["resync"] = True
        return env

    @staticmethod
    def _bare(chat_id, seq, status):
        return {"sessionId": chat_id, "seq": seq, "status": status, "nodeId": None, "messages": [],
                "input": None, "continueAfterMs": None, "delegate": None, "error": None, "menuChip": False}
