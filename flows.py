"""
Interactive Message Flows for WhatsApp
Handles: flow execution, session management, button/list routing
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from clients import supabase
from auth import verify_token
from whatsapp import (
    send_whatsapp_message,
    send_whatsapp_buttons,
    send_whatsapp_list,
    send_whatsapp_cta_url,
)

router = APIRouter()


# -------------------------------------------------
# ESCAPE & HANDOFF KEYWORDS
# -------------------------------------------------
ESCAPE_KEYWORDS = {"menu", "back", "start", "main menu", "home"}
HANDOFF_KEYWORDS = {"human", "agent", "support", "talk to human", "talk to agent"}
OPT_OUT_KEYWORDS = {"stop", "unsubscribe", "quit"}

def is_escape_keyword(text: str) -> bool:
    return text.lower().strip() in ESCAPE_KEYWORDS

def is_handoff_keyword(text: str) -> bool:
    return text.lower().strip() in HANDOFF_KEYWORDS

def is_opt_out_keyword(text: str) -> bool:
    return text.lower().strip() in OPT_OUT_KEYWORDS


# -------------------------------------------------
# SESSION MANAGEMENT
# -------------------------------------------------
def get_or_create_session(project_id: str, phone_number: str) -> Optional[dict]:
    """Get existing non-expired session or return None."""
    try:
        res = supabase.table("whatsapp_sessions") \
            .select("*") \
            .eq("project_id", project_id) \
            .eq("phone_number", phone_number) \
            .execute()

        if not res.data:
            return None

        session = res.data[0]

        # Check expiry
        expires_at = session.get("expires_at")
        if expires_at:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp < datetime.now(timezone.utc):
                # Expired — delete and return None
                supabase.table("whatsapp_sessions") \
                    .delete() \
                    .eq("id", session["id"]) \
                    .execute()
                return None

        return session

    except Exception as e:
        print(f"get_or_create_session error: {e}")
        return None


def create_session(project_id: str, phone_number: str, flow_id: str, start_node_id: str) -> dict:
    """Create a new flow session."""
    res = supabase.table("whatsapp_sessions").upsert({
        "project_id": project_id,
        "phone_number": phone_number,
        "flow_id": flow_id,
        "current_node_id": start_node_id,
        "mode": "flow",
    }, on_conflict="project_id,phone_number").execute()
    return res.data[0]


def update_session_node(session_id: str, node_id: str):
    supabase.table("whatsapp_sessions") \
        .update({"current_node_id": node_id}) \
        .eq("id", session_id) \
        .execute()


def delete_session(project_id: str, phone_number: str):
    supabase.table("whatsapp_sessions") \
        .delete() \
        .eq("project_id", project_id) \
        .eq("phone_number", phone_number) \
        .execute()


# -------------------------------------------------
# FLOW LOOKUP
# -------------------------------------------------
def get_active_flow(project_id: str) -> Optional[dict]:
    """Get the active flow for a project."""
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
    """Get the start node of a flow."""
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
    """Get next node by matching edge trigger."""
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
def send_flow_node(node: dict, to: str, phone_number_id: str, token: str):
    """Send the appropriate message type for a flow node."""
    node_type = node["type"]
    content = node["content"]

    if node_type == "text":
        send_whatsapp_message(to, content["body"], phone_number_id, token)

    elif node_type == "buttons":
        send_whatsapp_buttons(to, content["body"], content["buttons"], phone_number_id, token)

    elif node_type == "list":
        send_whatsapp_list(
            to, content["body"], content["button_text"],
            content["sections"], phone_number_id, token
        )

    elif node_type == "cta_url":
        send_whatsapp_cta_url(
            to, content["body"], content["button_text"],
            content["url"], phone_number_id, token
        )

    elif node_type == "handoff":
        send_whatsapp_message(to, content["body"], phone_number_id, token)
        # Switch session to human mode
        supabase.table("whatsapp_sessions").upsert({
            "project_id": None,  # will be set by caller
            "phone_number": to,
            "mode": "human",
        }, on_conflict="project_id,phone_number").execute()

    elif node_type == "rag":
        send_whatsapp_message(to, content["body"], phone_number_id, token)


# -------------------------------------------------
# FLOW EXECUTION
# -------------------------------------------------
def start_flow(flow: dict, project_id: str, phone_number: str, phone_number_id: str, token: str):
    """Start a flow from its start node."""
    start_node = get_start_node(flow["id"])
    if not start_node:
        print(f"No start node found for flow {flow['id']}")
        return

    create_session(project_id, phone_number, flow["id"], start_node["id"])

    if start_node["type"] == "handoff":
        supabase.table("whatsapp_sessions").upsert({
            "project_id": project_id,
            "phone_number": phone_number,
            "mode": "human",
        }, on_conflict="project_id,phone_number").execute()

    send_flow_node(start_node, phone_number, phone_number_id, token)


def advance_flow(session: dict, trigger: str, phone_number_id: str, token: str, phone_number: str):
    """Advance the flow based on a button/list trigger."""
    current_node_id = session.get("current_node_id")
    flow_id = session.get("flow_id")

    if not current_node_id or not flow_id:
        return

    next_node = get_next_node(flow_id, current_node_id, trigger)
    if not next_node:
        # No edge found — send fallback
        send_whatsapp_message(phone_number, "I didn't understand that. Type *menu* to start over.", phone_number_id, token)
        return

    update_session_node(session["id"], next_node["id"])

    if next_node["type"] == "handoff":
        supabase.table("whatsapp_sessions") \
            .update({"mode": "human"}) \
            .eq("id", session["id"]) \
            .execute()

    send_flow_node(next_node, phone_number, phone_number_id, token)


def handle_flow_text(session: dict, text: str, project_id: str, chat_id: str, phone_number: str, phone_number_id: str, token: str):
    """Handle a text message while inside a flow."""
    current_node = get_node(session.get("current_node_id"))

    if not current_node:
        return

    node_type = current_node["type"]

    if node_type == "rag":
        # RAG node — answer with AI
        rag_exit_keywords = current_node["content"].get("rag_exit_keywords", [])
        if text.lower() in [k.lower() for k in rag_exit_keywords]:
            # Exit RAG, restart flow
            flow = get_active_flow(project_id)
            if flow:
                start_flow(flow, project_id, phone_number, phone_number_id, token)
            return

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

    elif node_type in ("buttons", "list"):
        # Text received when buttons/list expected — resend the node
        send_flow_node(current_node, phone_number, phone_number_id, token)

    else:
        # Text node or anything else — try to advance by text match
        next_node = get_next_node(session["flow_id"], current_node["id"], text.lower())
        if next_node:
            update_session_node(session["id"], next_node["id"])
            send_flow_node(next_node, phone_number, phone_number_id, token)
        else:
            send_whatsapp_message(phone_number, "Type *menu* to see options.", phone_number_id, token)


# -------------------------------------------------
# FLOW MANAGEMENT API ENDPOINTS
# -------------------------------------------------
@router.get("/flows")
def list_flows(project_id: str, user=Depends(verify_token)):
    res = supabase.table("flows") \
        .select("id, name, is_active, trigger_keywords, created_at") \
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
    }).execute()
    return res.data[0]


@router.put("/flows/{flow_id}")
def update_flow(flow_id: str, data: dict, user=Depends(verify_token)):
    update = {}
    if "name" in data:
        update["name"] = data["name"]
    if "is_active" in data:
        update["is_active"] = data["is_active"]
    if "trigger_keywords" in data:
        update["trigger_keywords"] = data["trigger_keywords"]

    # If activating this flow, deactivate others for the same project
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
    nodes = supabase.table("flow_nodes") \
        .select("*") \
        .eq("flow_id", flow_id) \
        .order("created_at") \
        .execute()
    edges = supabase.table("flow_edges") \
        .select("*") \
        .eq("flow_id", flow_id) \
        .execute()
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
    if "type" in data:
        update["type"] = data["type"]
    if "content" in data:
        update["content"] = data["content"]
    if "is_start" in data:
        update["is_start"] = data["is_start"]

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