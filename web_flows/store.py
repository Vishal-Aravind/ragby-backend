"""Supabase-backed Store and Effects for the website flow engine."""
import random
from datetime import datetime, timedelta, timezone

import sentry_sdk

from clients import supabase

MAX_PENDING_TRANSCRIPT = 40
_SESSION_COLUMNS = ("flow_id", "current_node_id", "mode", "awaiting", "variables", "seq",
                    "resume_at", "pending_transcript", "chat_materialized", "lead_id",
                    "last_agent_msg_at", "expires_at")


def _parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class SupabaseStore:
    def get_active_web_flow(self, project_id):
        res = supabase.table("flows").select("id, free_questions, revision, web_settings") \
            .eq("project_id", project_id).eq("is_active", True).eq("channel", "web") \
            .limit(1).execute()
        return res.data[0] if res.data else None

    def get_start_node(self, flow_id):
        res = supabase.table("flow_nodes").select("id, type, content") \
            .eq("flow_id", flow_id).eq("is_start", True).limit(1).execute()
        return res.data[0] if res.data else None

    def get_node(self, node_id, flow_id):
        # Always scoped to the flow (same cross-tenant rule as flows.get_node).
        if not node_id or not flow_id:
            return None
        res = supabase.table("flow_nodes").select("id, type, content") \
            .eq("id", node_id).eq("flow_id", flow_id).limit(1).execute()
        return res.data[0] if res.data else None

    def get_edges(self, flow_id):
        res = supabase.table("flow_edges").select("from_node_id, trigger, to_node_id") \
            .eq("flow_id", flow_id).execute()
        return res.data or []

    def load_session(self, chat_id):
        res = supabase.table("web_flow_sessions").select("*").eq("chat_id", chat_id).limit(1).execute()
        if not res.data:
            return None
        sess = res.data[0]
        exp = _parse_ts(sess.get("expires_at"))
        if exp and exp < datetime.now(timezone.utc):
            return None
        return sess

    def create_session(self, sess):
        # The id becomes chats.id once the visitor interacts. Never adopt an
        # id that belongs to another project or channel (same rule as
        # /public/chat); reuse it only if it's this project's public chat.
        existing = supabase.table("chats").select("id, project_id, channel").eq("id", sess["chat_id"]).execute()
        if existing.data:
            row = existing.data[0]
            if row.get("project_id") == sess["project_id"] and row.get("channel") == "public":
                sess["chat_materialized"] = True
            else:
                import uuid
                sess["chat_id"] = str(uuid.uuid4())
        supabase.table("web_flow_sessions").upsert(sess, on_conflict="chat_id").execute()
        return sess

    def claim(self, chat_id, seq):
        # Compare-and-swap: only one request per seq value wins, so a double
        # tap or a second tab can't run a node twice.
        res = supabase.table("web_flow_sessions").update({
            "seq": seq + 1, "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("chat_id", chat_id).eq("seq", seq).execute()
        return res.data[0] if res.data else None

    def save_session(self, sess):
        supabase.table("web_flow_sessions").update(
            {**{k: sess.get(k) for k in _SESSION_COLUMNS},
             "updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("chat_id", sess["chat_id"]).execute()


class LiveEffects:
    def __init__(self, project_id):
        self.project_id = project_id

    def record(self, sess, entries, visitor_acted):
        """Save what was said. Until the visitor does something, messages
        wait in pending_transcript (no chats row), so auto-opened chats
        nobody touched never reach the Conversations inbox."""
        entries = [e for e in (entries or []) if (e.get("content") or "").strip()]
        try:
            if not sess.get("chat_materialized") and not visitor_acted:
                pending = (sess.get("pending_transcript") or []) + entries
                sess["pending_transcript"] = pending[-MAX_PENDING_TRANSCRIPT:]
                supabase.table("web_flow_sessions").update(
                    {"pending_transcript": sess["pending_transcript"]}
                ).eq("chat_id", sess["chat_id"]).execute()
                return
            if not sess.get("chat_materialized"):
                clash = supabase.table("chats").select("id, project_id, channel").eq("id", sess["chat_id"]).execute()
                if not clash.data:
                    supabase.table("chats").insert({
                        "id": sess["chat_id"], "project_id": sess["project_id"],
                        "title": "Public Chat", "channel": "public",
                    }).execute()
                elif clash.data[0].get("project_id") != sess["project_id"]:
                    return  # never write into someone else's chat
                entries = (sess.get("pending_transcript") or []) + entries
                sess["chat_materialized"] = True
                sess["pending_transcript"] = []
                supabase.table("web_flow_sessions").update(
                    {"chat_materialized": True, "pending_transcript": []}
                ).eq("chat_id", sess["chat_id"]).execute()
            if entries:
                supabase.table("chat_messages").insert([
                    {"chat_id": sess["chat_id"], "role": e["role"], "content": e["content"][:4000]}
                    for e in entries
                ]).execute()
        except Exception as e:
            sentry_sdk.capture_exception(e)

    def upsert_lead(self, project_id, visitor_id, fields, custom):
        try:
            from leads import upsert_web_flow_lead
            return upsert_web_flow_lead(project_id, visitor_id, fields, custom)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return None

    def log_events(self, events):
        try:
            supabase.table("web_flow_events").insert(events[:60]).execute()
        except Exception as e:
            sentry_sdk.capture_exception(e)

    def now(self):
        return datetime.now(timezone.utc)

    def today(self):
        # India-first product; the date a visitor sees is IST.
        return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).date()

    def random(self):
        return random.random()


def prune_expired():
    """Daily job: drop finished sessions and old analytics."""
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        supabase.table("web_flow_sessions").delete().lt("expires_at", cutoff).execute()
        old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        supabase.table("web_flow_events").delete().lt("created_at", old).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
