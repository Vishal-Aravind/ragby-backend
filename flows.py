"""
Interactive Message Flows for WhatsApp
- Everything driven by button IDs, no global keywords
- "ask_a_question" button ID → RAG mode + [Back to Menu] after every answer
- "back_to_menu" button ID → restart flow
- "handoff" button ID → human handoff
- Free questions toggle → if ON, text on buttons node → RAG + resend buttons
"""
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends
from clients import supabase
from auth import verify_token
from whatsapp import (
    send_whatsapp_message,
    send_whatsapp_buttons,
    send_whatsapp_list,
    send_whatsapp_cta_url,
)

router = APIRouter()

# Reserved button IDs — builder cannot use these for custom nodes
RESERVED_ASK_AI   = "ask_a_question"
RESERVED_BACK     = "back_to_menu"
RESERVED_HANDOFF  = "talk_to_human"


# -------------------------------------------------
# SESSION MANAGEMENT
# -------------------------------------------------
def get_session(project_id: str, phone_number: str) -> Optional[dict]:
    """Get existing non-expired session."""
    try:
        res = supabase.table("whatsapp_sessions") \
            .select("*") \
            .eq("project_id", project_id) \
            .eq("phone_number", phone_number) \
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
        print(f"get_session error: {e}")
        return None


def upsert_session(project_id: str, phone_number: str, data: dict):
    """Create or update session."""
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


def get_node(node_id: str) -> Optional[dict]:
    res = supabase.table("flow_nodes") \
        .select("*") \
        .eq("id", node_id) \
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
    return get_node(edge.data[0]["to_node_id"])


# -------------------------------------------------
# NODE SENDER
# -------------------------------------------------
def send_node(node: dict, to: str, phone_number_id: str, token: str):
    """Send the right WhatsApp message type for a node."""
    t = node["type"]
    c = node["content"]

    if t == "text":
        send_whatsapp_message(to, c["body"], phone_number_id, token)

    elif t == "buttons":
        send_whatsapp_buttons(to, c["body"], c.get("buttons", []), phone_number_id, token)

    elif t == "list":
        send_whatsapp_list(
            to, c["body"], c.get("button_text", "View Options"),
            c.get("sections", []), phone_number_id, token
        )

    elif t == "cta_url":
        send_whatsapp_cta_url(
            to, c["body"], c.get("button_text", "Click Here"),
            c.get("url", ""), phone_number_id, token
        )

    elif t == "handoff":
        send_whatsapp_message(to, c["body"], phone_number_id, token)

    elif t == "rag":
        send_whatsapp_message(to, c["body"], phone_number_id, token)


def send_back_to_menu_button(to: str, text: str, phone_number_id: str, token: str):
    """Send a single [Back to Menu] button after RAG answers."""
    send_whatsapp_buttons(
        to, text,
        [{"id": RESERVED_BACK, "title": "↩ Back to Menu"}],
        phone_number_id, token
    )


# -------------------------------------------------
# FLOW EXECUTION
# -------------------------------------------------
def start_flow(flow: dict, project_id: str, phone_number: str, phone_number_id: str, token: str):
    """Start flow from start node."""
    start_node = get_start_node(flow["id"])
    if not start_node:
        print(f"No start node for flow {flow['id']}")
        return

    upsert_session(project_id, phone_number, {
        "flow_id": flow["id"],
        "current_node_id": start_node["id"],
        "mode": "flow",
    })

    # If free_questions ON, append hint to body
    body = start_node["content"].get("body", "")
    if flow.get("free_questions") and start_node["type"] in ("buttons", "list"):
        body = body + "\n\n💬 _Tap an option or type your question directly_"
        node_with_hint = {**start_node, "content": {**start_node["content"], "body": body}}
        send_node(node_with_hint, phone_number, phone_number_id, token)
    else:
        send_node(start_node, phone_number, phone_number_id, token)

    # If start node is handoff
    if start_node["type"] == "handoff":
        upsert_session(project_id, phone_number, {
            "flow_id": flow["id"],
            "current_node_id": start_node["id"],
            "mode": "human",
        })


def handle_interactive(session: dict, trigger: str, phone_number: str, phone_number_id: str, token: str, project_id: str):
    """Handle button/list click."""

    # Back to Menu → restart flow
    if trigger == RESERVED_BACK:
        flow = get_active_flow(project_id)
        if flow:
            start_flow(flow, project_id, phone_number, phone_number_id, token)
        return

    # Ask a Question → send confirmation + set rag_question mode
    if trigger == RESERVED_ASK_AI:
        upsert_session(project_id, phone_number, {
            "flow_id": session.get("flow_id"),
            "current_node_id": session.get("current_node_id"),
            "mode": "rag_question",
        })
        send_whatsapp_buttons(
            phone_number,
            "You can now ask me anything about our products and services!",
            [{"id": RESERVED_BACK, "title": "↩ Back to Menu"}],
            phone_number_id, token
        )
        return

    # Talk to Human → handoff
    if trigger == RESERVED_HANDOFF:
        upsert_session(project_id, phone_number, {
            "flow_id": session.get("flow_id"),
            "current_node_id": session.get("current_node_id"),
            "mode": "human",
        })
        send_whatsapp_message(
            phone_number,
            "Connecting you to our team. Please wait...",
            phone_number_id, token
        )
        return

    # Normal button → advance flow
    flow_id = session.get("flow_id")
    current_node_id = session.get("current_node_id")

    if not flow_id or not current_node_id:
        return

    next_node = get_next_node(flow_id, current_node_id, trigger)
    if not next_node:
        # No edge — resend current node
        current_node = get_node(current_node_id)
        if current_node:
            send_node(current_node, phone_number, phone_number_id, token)
        return

    upsert_session(project_id, phone_number, {
        "flow_id": flow_id,
        "current_node_id": next_node["id"],
        "mode": "human" if next_node["type"] == "handoff" else "flow",
    })

    if next_node["type"] == "handoff":
        send_whatsapp_message(
            phone_number,
            next_node["content"].get("body", "Connecting you to our team..."),
            phone_number_id, token
        )
    else:
        send_node(next_node, phone_number, phone_number_id, token)


def handle_text(session: Optional[dict], text: str, project_id: str, chat_id: str, phone_number: str, phone_number_id: str, token: str):
    """Handle a text message — behavior depends on session mode and free_questions toggle."""
    from chat import run_chat, get_history
    from usage import check_rate_limit, increment_usage

    # ── No session → check if flow should start ──────
    if not session:
        flow = get_active_flow(project_id)
        if flow:
            keywords = [k.lower() for k in (flow.get("trigger_keywords") or [])]
            if text.lower().strip() in keywords:
                start_flow(flow, project_id, phone_number, phone_number_id, token)
                return
        # Pure RAG fallback
        _rag_reply(project_id, chat_id, text, phone_number, phone_number_id, token)
        return

    mode = session.get("mode", "flow")

    # ── Human handoff mode → ignore, queue for human ─
    if mode == "human":
        return

    # ── RAG question mode → always answer with RAG ───
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

    # ── Flow mode ─────────────────────────────────────
    flow_id = session.get("flow_id")
    current_node_id = session.get("current_node_id")
    current_node = get_node(current_node_id) if current_node_id else None

    if not current_node:
        # No current node — try to start flow
        flow = get_active_flow(project_id)
        if flow:
            keywords = [k.lower() for k in (flow.get("trigger_keywords") or [])]
            if text.lower().strip() in keywords:
                start_flow(flow, project_id, phone_number, phone_number_id, token)
        return

    # Check free_questions toggle
    flow = supabase.table("flows").select("free_questions, trigger_keywords").eq("id", flow_id).single().execute()
    free_questions = flow.data.get("free_questions", False) if flow.data else False

    if current_node["type"] in ("buttons", "list"):
        if free_questions:
            # Answer with RAG then resend buttons
            rate_check = check_rate_limit(project_id)
            if not rate_check["allowed"]:
                send_whatsapp_message(phone_number, "⚠️ Monthly message limit reached.", phone_number_id, token)
                return
            history = get_history(chat_id, limit=5)
            result = run_chat(project_id, chat_id, text, history)
            # Send RAG answer first
            send_whatsapp_message(phone_number, result["answer"], phone_number_id, token)
            # Resend current buttons node
            send_node(current_node, phone_number, phone_number_id, token)
            increment_usage(project_id)
        else:
            # Strict mode — resend buttons, ignore text
            send_node(current_node, phone_number, phone_number_id, token)

    elif current_node["type"] == "rag":
        # RAG node — always answer
        rate_check = check_rate_limit(project_id)
        if not rate_check["allowed"]:
            send_whatsapp_message(phone_number, "⚠️ Monthly message limit reached.", phone_number_id, token)
            return
        history = get_history(chat_id, limit=5)
        result = run_chat(project_id, chat_id, text, history)
        send_back_to_menu_button(phone_number, result["answer"], phone_number_id, token)
        increment_usage(project_id)

    elif current_node["type"] == "text":
        # Text node receiving text — advance by keyword match or just move on
        next_node = get_next_node(flow_id, current_node_id, text.lower())
        if next_node:
            upsert_session(project_id, phone_number, {
                "flow_id": flow_id,
                "current_node_id": next_node["id"],
                "mode": "flow",
            })
            send_node(next_node, phone_number, phone_number_id, token)
        else:
            send_node(current_node, phone_number, phone_number_id, token)


def _rag_reply(project_id, chat_id, text, phone_number, phone_number_id, token):
    """Pure RAG reply with no flow context."""
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
# FLOW CRUD API
# -------------------------------------------------
@router.get("/flows")
def list_flows(project_id: str, user=Depends(verify_token)):
    res = supabase.table("flows") \
        .select("id, name, is_active, trigger_keywords, free_questions, created_at") \
        .eq("project_id", project_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data


@router.post("/flows")
def create_flow(data: dict, user=Depends(verify_token)):
    res = supabase.table("flows").insert({
        "project_id": data["project_id"],
        "name": data["name"],
        "is_active": data.get("is_active", False),
        "trigger_keywords": data.get("trigger_keywords", ["hi", "hello", "hey", "start", "menu"]),
        "free_questions": data.get("free_questions", False),
    }).execute()
    return res.data[0]


@router.put("/flows/{flow_id}")
def update_flow(flow_id: str, data: dict, user=Depends(verify_token)):
    update = {}
    if "name" in data: update["name"] = data["name"]
    if "is_active" in data: update["is_active"] = data["is_active"]
    if "trigger_keywords" in data: update["trigger_keywords"] = data["trigger_keywords"]
    if "free_questions" in data: update["free_questions"] = data["free_questions"]

    if data.get("is_active"):
        flow = supabase.table("flows").select("project_id").eq("id", flow_id).single().execute()
        if flow.data:
            supabase.table("flows") \
                .update({"is_active": False}) \
                .eq("project_id", flow.data["project_id"]) \
                .neq("id", flow_id) \
                .execute()

    res = supabase.table("flows").update(update).eq("id", flow_id).execute()
    return res.data[0]


@router.delete("/flows/{flow_id}")
def delete_flow(flow_id: str, user=Depends(verify_token)):
    supabase.table("flows").delete().eq("id", flow_id).execute()
    return {"status": "deleted"}


@router.get("/flows/{flow_id}/nodes")
def get_flow_nodes(flow_id: str, user=Depends(verify_token)):
    nodes = supabase.table("flow_nodes").select("*").eq("flow_id", flow_id).order("created_at").execute()
    edges = supabase.table("flow_edges").select("*").eq("flow_id", flow_id).execute()
    return {"nodes": nodes.data, "edges": edges.data}


@router.post("/flows/{flow_id}/nodes")
def create_node(flow_id: str, data: dict, user=Depends(verify_token)):
    res = supabase.table("flow_nodes").insert({
        "flow_id": flow_id,
        "type": data["type"],
        "content": data["content"],
        "is_start": data.get("is_start", False),
    }).execute()
    return res.data[0]


@router.put("/flows/nodes/{node_id}")
def update_node(node_id: str, data: dict, user=Depends(verify_token)):
    update = {}
    if "type" in data: update["type"] = data["type"]
    if "content" in data: update["content"] = data["content"]
    if "is_start" in data: update["is_start"] = data["is_start"]
    res = supabase.table("flow_nodes").update(update).eq("id", node_id).execute()
    return res.data[0]


@router.delete("/flows/nodes/{node_id}")
def delete_node(node_id: str, user=Depends(verify_token)):
    supabase.table("flow_nodes").delete().eq("id", node_id).execute()
    return {"status": "deleted"}


@router.post("/flows/{flow_id}/edges")
def create_edge(flow_id: str, data: dict, user=Depends(verify_token)):
    res = supabase.table("flow_edges").insert({
        "flow_id": flow_id,
        "from_node_id": data["from_node_id"],
        "trigger": data["trigger"],
        "to_node_id": data["to_node_id"],
    }).execute()
    return res.data[0]


@router.delete("/flows/edges/{edge_id}")
def delete_edge(edge_id: str, user=Depends(verify_token)):
    supabase.table("flow_edges").delete().eq("id", edge_id).execute()
    return {"status": "deleted"}