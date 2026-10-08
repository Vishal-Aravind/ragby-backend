"""HTTP endpoints for website flows.

Public (called by widget-flows.js from merchants' sites; CORS * already
applies to /public/*). Each one applies the WIDGET rules from /public/chat:
the project's Allowed websites list (empty = refused), suspension, and
rate limits. Flow steps don't use the monthly message allowance — only AI
answers do, and those still go through the unchanged /public/chat.

Authenticated (editor): preview, agent reply on a website chat.
"""
import base64
import hashlib
import hmac
import json
import re
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from auth import verify_token, require_project_access
from clients import supabase
from config import FRONTEND_URL
from ratelimit import is_rate_limited, client_ip
from .engine import Engine
from .store import SupabaseStore, LiveEffects

router = APIRouter()
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

_CONFIG_TTL = 60
_CONFIG_MAX = 500
_config_cache: dict = {}


# --------------------------------------------------------------------------- #
# Shared gate
# --------------------------------------------------------------------------- #
def _project_row(project_id: str) -> dict:
    res = supabase.table("projects").select("id, suspended, allowed_domains") \
        .eq("id", project_id).limit(1).execute()
    return res.data[0] if res.data else {}


def _origin_ok(request: Request, project: dict) -> bool:
    from chat import _origin_allowed
    return _origin_allowed(request, project.get("allowed_domains") or [])


def _gate(request: Request, project_id: str, session_id: Optional[str], kind: str):
    if not _UUID_RE.match(project_id or ""):
        raise HTTPException(status_code=400, detail="Invalid project")
    if session_id is not None and not _UUID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session")
    ip = client_ip(request)
    limits = {"start": (10, 300), "step": (30, 300), "resume": (30, 300), "poll": (20, 600)}[kind]
    if (is_rate_limited(f"wf-{kind}-ip:{project_id}:{ip}", limits[0])
            or (session_id and is_rate_limited(f"wf-{kind}-s:{session_id}", limits[0]))
            or is_rate_limited(f"wf-{kind}-p:{project_id}", limits[1])):
        raise HTTPException(status_code=429, detail="Too many requests - please wait a moment.")
    project = _project_row(project_id)
    if not project or project.get("suspended"):
        raise HTTPException(status_code=404, detail="This assistant isn't available right now.")
    if not _origin_ok(request, project):
        raise HTTPException(status_code=403, detail="This assistant isn't available on this site.")


def _engine(project_id: str) -> Engine:
    return Engine(SupabaseStore(), LiveEffects(project_id), FRONTEND_URL)


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
_TRIGGER_TYPES = {"time_on_page", "url_match", "exit_intent", "scroll_depth"}


def _int_or(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _clean_settings(raw) -> dict:
    """Only the fields the widget needs, bounded."""
    raw = raw if isinstance(raw, dict) else {}
    triggers = []
    for t in (raw.get("triggers") or [])[:10]:
        if not isinstance(t, dict) or t.get("type") not in _TRIGGER_TYPES:
            continue
        triggers.append({
            "type": t["type"],
            "seconds": max(1, min(_int_or(t.get("seconds"), 10), 600)) if t["type"] == "time_on_page" else None,
            "percent": max(10, min(_int_or(t.get("percent"), 50), 100)) if t["type"] == "scroll_depth" else None,
            "match": t.get("match") if t.get("match") in ("contains", "equals", "starts_with") else "contains",
            "value": str(t.get("value") or "")[:300],
        })
    return {
        "startOnOpen": raw.get("start_on_open", True) is not False,
        "teaser": str(raw.get("teaser") or "")[:140],
        "display": raw.get("display") if raw.get("display") in ("open", "teaser") else "open",
        # 0 is a real choice ("no cooldown"); only a missing value defaults.
        "cooldownHours": max(0, min(_int_or(raw.get("cooldown_hours"), 24), 24 * 30)),
        "suppressDays": max(0, min(_int_or(raw.get("suppress_days"), 7), 90)),
        "triggers": triggers,
    }


def _cached_config(project_id: str) -> dict:
    now = time.time()
    hit = _config_cache.get(project_id)
    if hit and now - hit[0] < _CONFIG_TTL:
        return hit[1]
    project = _project_row(project_id)
    flow = None
    if project and not project.get("suspended"):
        res = supabase.table("flows").select("id, revision, web_settings") \
            .eq("project_id", project_id).eq("is_active", True).eq("channel", "web").limit(1).execute()
        flow = res.data[0] if res.data else None
    value = {"project": project, "flow": flow}
    if len(_config_cache) >= _CONFIG_MAX:
        _config_cache.pop(min(_config_cache, key=lambda k: _config_cache[k][0]), None)
    _config_cache[project_id] = (now, value)
    return value


@router.get("/public/flow-config/{project_id}")
def flow_config(project_id: str, request: Request):
    if not _UUID_RE.match(project_id or ""):
        return {"active": False}
    if is_rate_limited(f"wf-cfg:{project_id}:{client_ip(request)}", 60):
        return {"active": False}
    try:
        cfg = _cached_config(project_id)
    except Exception:
        return {"active": False}
    flow = cfg["flow"]
    if not flow:
        return {"active": False}
    return {"active": True, "flowId": flow["id"], "revision": flow.get("revision"),
            "allowedHere": _origin_ok(request, cfg["project"]),
            **_clean_settings(flow.get("web_settings"))}


class Page(BaseModel):
    url: Optional[str] = Field(default=None, max_length=1000)
    path: Optional[str] = Field(default=None, max_length=500)
    title: Optional[str] = Field(default=None, max_length=300)


class StartReq(BaseModel):
    projectId: str
    sessionId: Optional[str] = None
    visitorId: str = Field(min_length=1, max_length=64)
    via: str = Field(default="open", max_length=40)
    page: Optional[Page] = None


class StepReq(BaseModel):
    projectId: str
    sessionId: str
    visitorId: str = Field(min_length=1, max_length=64)
    seq: int
    nodeId: Optional[str] = Field(default=None, max_length=64)
    action: dict
    page: Optional[Page] = None


class ResumeReq(BaseModel):
    projectId: str
    sessionId: str
    visitorId: str = Field(min_length=1, max_length=64)


class PollReq(ResumeReq):
    after: Optional[str] = Field(default=None, max_length=40)


def _page(p: Optional[Page]):
    return p.model_dump() if p else None


@router.post("/public/flow/start")
def flow_start(req: StartReq, request: Request):
    _gate(request, req.projectId, req.sessionId, "start")
    return _engine(req.projectId).start(req.projectId, req.sessionId, req.visitorId, req.via, _page(req.page))


@router.post("/public/flow/step")
def flow_step(req: StepReq, request: Request):
    _gate(request, req.projectId, req.sessionId, "step")
    if len(json.dumps(req.action)) > 20000:
        raise HTTPException(status_code=413, detail="Too much data")
    return _engine(req.projectId).step(req.projectId, req.sessionId, req.visitorId, req.seq,
                                       req.nodeId, req.action, _page(req.page))


@router.post("/public/flow/resume")
def flow_resume(req: ResumeReq, request: Request):
    _gate(request, req.projectId, req.sessionId, "resume")
    return _engine(req.projectId).resume(req.projectId, req.sessionId, req.visitorId)


@router.post("/public/flow/poll")
def flow_poll(req: PollReq, request: Request):
    """Human mode: agent replies written from Conversations since `after`."""
    _gate(request, req.projectId, req.sessionId, "poll")
    sess = SupabaseStore().load_session(req.sessionId)
    if not sess or sess.get("project_id") != req.projectId or sess.get("visitor_id") != req.visitorId:
        return {"status": "expired", "messages": [], "cursor": req.after}
    q = supabase.table("chat_messages").select("content, created_at").eq("chat_id", req.sessionId) \
        .eq("role", "assistant").like("content", "[Human] %").order("created_at").limit(20)
    if req.after:
        q = q.gt("created_at", req.after)
    rows = q.execute().data or []
    status = {"ai": "ai", "human": "human", "ended": "ended"}.get(sess.get("mode"), "flow")
    return {
        "status": status,
        "messages": [{"kind": "text", "text": r["content"][len("[Human] "):], "agent": True} for r in rows],
        "cursor": rows[-1]["created_at"] if rows else req.after,
    }


# --------------------------------------------------------------------------- #
# Editor: preview (real engine, nothing saved)
# --------------------------------------------------------------------------- #
def _preview_secret() -> bytes:
    from chat import _chat_access_secret
    return hashlib.sha256(b"web-flow-preview:" + _chat_access_secret()).digest()


def _sign(data: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode()
    sig = hmac.new(_preview_secret(), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def _unsign(token: str):
    if not token or "." not in token:
        return None
    raw, _, sig = token.rpartition(".")
    if not hmac.compare_digest(sig, hmac.new(_preview_secret(), raw.encode(), hashlib.sha256).hexdigest()):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(raw.encode()))
    except ValueError:
        return None
    return data if data.get("exp", 0) > time.time() else None


class PreviewStore(SupabaseStore):
    """Runs the flow being edited (active or not); the one session lives in
    memory and travels back to the editor as a signed token."""
    def __init__(self, flow):
        self.flow = flow
        self.session = None

    def get_active_web_flow(self, project_id):
        return self.flow

    def load_session(self, chat_id):
        return dict(self.session) if self.session and self.session["chat_id"] == chat_id else None

    def create_session(self, sess):
        sess["chat_materialized"] = True
        self.session = dict(sess)
        return dict(sess)

    def claim(self, chat_id, seq):
        if not self.session or self.session["seq"] != seq:
            return None
        self.session["seq"] += 1
        return dict(self.session)

    def save_session(self, sess):
        self.session = dict(sess)


class PreviewEffects(LiveEffects):

    def record(self, sess, entries, visitor_acted):
        pass

    def upsert_lead(self, project_id, visitor_id, fields, custom):
        return None

    def log_events(self, events):
        pass



class PreviewReq(BaseModel):
    projectId: str
    flowId: str
    token: Optional[str] = Field(default=None, max_length=100000)
    action: Optional[dict] = None     # None = (re)start
    nodeId: Optional[str] = None


@router.post("/web-flows/preview")
def web_flow_preview(req: PreviewReq, user=Depends(verify_token)):
    require_project_access(user.id, req.projectId, tab="flows")
    if is_rate_limited(f"wf-preview:{user.id}", 60):
        raise HTTPException(status_code=429, detail="Slow down a little.")
    flow = supabase.table("flows").select("id, project_id, channel, free_questions") \
        .eq("id", req.flowId).limit(1).execute().data
    if not flow or flow[0]["project_id"] != req.projectId or flow[0].get("channel") != "web":
        raise HTTPException(status_code=404, detail="Flow not found")
    store = PreviewStore({"id": flow[0]["id"], "free_questions": flow[0].get("free_questions")})
    engine = Engine(store, PreviewEffects(req.projectId), FRONTEND_URL)
    visitor = f"preview:{user.id}"[:64]

    prior = _unsign(req.token) if req.token else None
    if prior and prior.get("flow") == req.flowId and prior.get("uid") == user.id and req.action is not None:
        store.session = prior["session"]
        sess = prior["session"]
        if (req.action or {}).get("type") == "continue":
            # The tester asked to skip the wait; a visitor can't do this.
            store.session["resume_at"] = None
        env = engine.step(req.projectId, sess["chat_id"], visitor, sess["seq"], req.nodeId, req.action)
    else:
        env = engine.start(req.projectId, None, visitor, "preview")
    sess = store.session or {}
    if env.get("delegate") == "ai":
        env["messages"] = env.get("messages", []) + [{"kind": "text", "text": "(The AI would answer this from your documents.)"}]
    env["token"] = _sign({"flow": req.flowId, "uid": user.id, "session": sess, "exp": time.time() + 3600})
    env["debug"] = {"variables": sess.get("variables") or {}, "currentNodeId": sess.get("current_node_id"),
                    "mode": sess.get("mode")}
    return env


# --------------------------------------------------------------------------- #
# Conversations: agent replies to a website chat
# --------------------------------------------------------------------------- #
class WebReplyReq(BaseModel):
    project_id: str
    chat_id: str
    message: str = Field(min_length=1, max_length=4000)


@router.post("/web-flows/reply")
def web_flow_reply(req: WebReplyReq, user=Depends(verify_token)):
    require_project_access(user.id, req.project_id, tab="conversations", min_role="admin")
    if is_rate_limited(f"wf-reply:{req.project_id}:{user.id}", 30):
        raise HTTPException(status_code=429, detail="Too many messages - please wait a moment.")
    chat = supabase.table("chats").select("id, project_id, channel").eq("id", req.chat_id).limit(1).execute().data
    if not chat or chat[0]["project_id"] != req.project_id or chat[0]["channel"] != "public":
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Any website chat (flow or plain AI): the widget / hosted page polls for
    # "[Human]" messages while the chat is in human mode. Replying puts the
    # chat in human mode so the bot doesn't answer over the person.
    supabase.table("chat_messages").insert({
        "chat_id": req.chat_id, "role": "assistant", "content": f"[Human] {req.message.strip()}",
    }).execute()
    supabase.table("chats").update({
        "human_mode": True, "last_agent_msg_at": "now()",
    }).eq("id", req.chat_id).execute()
    supabase.table("web_flow_sessions").update({
        "mode": "human", "last_agent_msg_at": "now()",
    }).eq("chat_id", req.chat_id).execute()
    return {"status": "sent"}
