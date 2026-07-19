import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from clients import supabase, openai_client, embeddings, qdrant
from config import QDRANT_COLLECTION
from auth import verify_token
from usage import check_rate_limit, increment_usage
from qdrant_client import models

router = APIRouter()


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class ChatRequest(BaseModel):
    projectId: str
    chatId: str
    message: str

class PublicChatRequest(BaseModel):
    projectId: str
    message: str
    sessionId: Optional[str] = None


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful RAG AI assistant. Use ONLY the provided context and recent conversation if relevant. "
    "If the user asks about you, introduce yourself politely.\n\n"

    "Context source guidance:\n"
    "- Context chunks labeled [Source: gsheets] or [Source: database] contain structured data like records, names, values, dates.\n"
    "- Context chunks labeled [Source: document] contain policies, procedures, or explanatory content.\n"
    "- Always prefer the source that best matches the question type.\n\n"

    "Style:\n"
    "- Simple question → short answer\n"
    "- Complex question → structured answer, no heavy formatting, medium length\n"
    "- Be concise, do not over-explain unless necessary\n\n"
    "- Pricing/packages/plans questions → always list ALL options, never just one\n"

    "Logic:\n"
    "- One clear answer → answer directly\n"
    "- Multiple answers → ask a clarification question briefly\n"
    "- Partial info → answer + mention missing briefly\n"
    "- No answer → say you couldn't find the information and ask if they have more details\n"
    "- Match answer length to question complexity\n\n"

    "Formatting rules:\n"
    "- Use bullet points ONLY when needed\n\n"

    "Rules:\n"
    "- No hallucination\n"
    "- No external knowledge\n"
    "- Use previous conversation history for context when relevant"
)


# -------------------------------------------------
# HISTORY HELPERS
# -------------------------------------------------
def get_history(chat_id: str, limit: int = 7):
    res = supabase.table("chat_messages") \
        .select("role, content") \
        .eq("chat_id", chat_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return list(reversed(res.data)) if res.data else []

def save_message(chat_id: str, role: str, content: str):
    supabase.table("chat_messages").insert({
        "chat_id": chat_id,
        "role": role,
        "content": content
    }).execute()


# -------------------------------------------------
# AGENTIC ACTIONS — bot-can-book (opt-in, see appointment_settings.bot_can_book)
# -------------------------------------------------
def get_appointment_settings_if_bookable(project_id: str):
    """None unless the merchant has explicitly turned on in-chat booking."""
    res = supabase.table("appointment_settings").select("*").eq("project_id", project_id).maybe_single().execute()
    data = res.data if res else None
    return data if data and data.get("bot_can_book") else None


APPOINTMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_appointment_availability",
            "description": "Check which appointment time slots are free on a given date. Always call this before proposing a specific time to the customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date to check, format YYYY-MM-DD"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book a confirmed appointment slot. Only call this AFTER the customer has explicitly "
                "agreed to a specific date and time you already checked and proposed to them. Never call "
                "this on the first mention of wanting an appointment, and never guess or assume a "
                "date/time the customer hasn't confirmed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Confirmed date, format YYYY-MM-DD"},
                    "time": {"type": "string", "description": "Confirmed start time, format HH:MM (24-hour)"},
                    "customer_name": {"type": "string", "description": "The customer's name — only ask for this if it isn't already known from this conversation channel"},
                    "customer_phone": {"type": "string", "description": "Customer's phone number — only ask for this if it isn't already known from this conversation channel"},
                },
                "required": ["date", "time"],
            },
        },
    },
]


SHOP_READONLY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Look up the customer's own recent orders (status, payment, items, total) by their phone number. Read-only — cannot place or change an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_phone": {"type": "string", "description": "Customer's phone number — only ask for this if it isn't already known from this conversation channel"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_shop_catalog",
            "description": "Get the list of available products (name, price, description) to answer questions like 'what do you sell' or recommend an item. Read-only — cannot add items to a cart or place an order.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Only ever added to active_tools when channel == "whatsapp" (see run_chat) —
# the confirm/payment flow this hands off to (Continue/Add More/Clear Cart
# buttons + Razorpay) only exists on WhatsApp today.
SHOP_ORDER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Create a new order from items the customer has explicitly confirmed. Always call "
                "browse_shop_catalog first if you haven't already, so item names and prices are accurate. "
                "Read back the exact items, quantities, and total price to the customer and get their "
                "explicit yes BEFORE calling this — never call it on a first mention of wanting to order, "
                "and never guess an item or quantity they haven't confirmed. After this runs, the customer "
                "gets WhatsApp buttons to finish confirming and pay — tell them to check those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_name": {"type": "string", "description": "Exact product name as shown by browse_shop_catalog"},
                                "quantity": {"type": "integer"},
                            },
                            "required": ["product_name", "quantity"],
                        },
                    },
                    "delivery_type": {"type": "string", "description": "One of the store's delivery options, only if the customer specified one"},
                },
                "required": ["items"],
            },
        },
    },
]


def _get_known_customer_name(project_id: str, channel: str, external_id: str) -> Optional[str]:
    """WhatsApp already knows the sender's real profile name (captured at
    message time — see whatsapp.py) — use that instead of making the model
    ask for or guess a name."""
    if channel != "whatsapp" or not external_id:
        return None
    res = supabase.table("leads").select("name").eq("project_id", project_id).eq("phone", external_id).maybe_single().execute()
    data = res.data if res else None
    return (data or {}).get("name")


def execute_appointment_tool(name: str, args: dict, project_id: str, channel: str, external_id: str) -> dict:
    """Never trusts the model's parameters as final — create_appointment
    re-validates the slot is actually still free."""
    from appointments import get_available_slots, create_appointment

    try:
        if name == "check_appointment_availability":
            slots = get_available_slots(project_id, args["date"])
            return {"date": args["date"], "available_slots": slots}

        if name == "book_appointment":
            # WhatsApp's own sender number is authoritative — never rely on
            # the model to transcribe a phone number correctly when we
            # already know it for certain from the channel itself.
            phone = external_id if channel == "whatsapp" else args.get("customer_phone")
            if not phone:
                return {"error": "Still need a phone number from the customer to confirm this booking."}

            customer_name = _get_known_customer_name(project_id, channel, external_id) or args.get("customer_name")
            if not customer_name:
                return {"error": "Still need the customer's name to confirm this booking."}

            return create_appointment(
                project_id=project_id,
                customer_name=customer_name,
                customer_phone=phone,
                appointment_date=args["date"],
                start_time=args["time"],
                notes=None,
            )

        return {"error": f"Unknown tool {name}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"execute_appointment_tool error: {e}")
        return {"error": "Something went wrong trying to do that — please try again."}


def execute_shop_tool(name: str, args: dict, project_id: str, channel: str, external_id: str) -> dict:
    from shop import get_recent_orders_for_phone, get_active_catalog_summary, create_order_from_chat

    try:
        if name == "check_order_status":
            phone = external_id if channel == "whatsapp" else args.get("customer_phone")
            if not phone:
                return {"error": "Still need the customer's phone number to look up their order."}
            orders = get_recent_orders_for_phone(project_id, phone)
            if not orders:
                return {"message": "No orders found for this phone number."}
            return {"orders": orders}

        if name == "browse_shop_catalog":
            products = get_active_catalog_summary(project_id)
            if not products:
                return {"message": "No products currently available."}
            return {"products": products}

        if name == "place_order":
            # Belt-and-suspenders: only ever reachable when channel ==
            # "whatsapp" since that's the only case SHOP_ORDER_TOOLS gets
            # added to active_tools (see run_chat), but never trust that
            # alone — re-check here too.
            if channel != "whatsapp":
                return {"error": "Ordering in chat is currently only available on WhatsApp — direct the customer to the shop link instead."}
            if not external_id:
                return {"error": "Missing the customer's phone number."}
            return create_order_from_chat(project_id, external_id, args["items"], args.get("delivery_type"))

        return {"error": f"Unknown tool {name}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"execute_shop_tool error: {e}")
        return {"error": "Something went wrong trying to do that — please try again."}


def execute_tool(name: str, args: dict, project_id: str, channel: str, external_id: str) -> dict:
    if name in ("check_appointment_availability", "book_appointment"):
        return execute_appointment_tool(name, args, project_id, channel, external_id)
    if name in ("check_order_status", "browse_shop_catalog", "place_order"):
        return execute_shop_tool(name, args, project_id, channel, external_id)
    return {"error": f"Unknown tool {name}"}


def run_completion(messages: list, tools: list, project_id: str, channel: str, external_id: str, temperature: float, max_tokens: int) -> str:
    """Runs one OpenAI completion, transparently looping through any tool
    calls the model makes (max 3 rounds — a real conversation never needs
    more than that, and it caps the blast radius of a runaway loop)."""
    kwargs = {"model": "gpt-4o-mini", "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    for _ in range(3):
        completion = openai_client.chat.completions.create(messages=messages, **kwargs)
        msg = completion.choices[0].message

        if not msg.tool_calls:
            return (msg.content or "").strip()

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            import json as _json
            args = _json.loads(tc.function.arguments or "{}")
            result = execute_tool(tc.function.name, args, project_id, channel, external_id)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _json.dumps(result),
            })

    # Ran out of rounds — ask the model for a final plain answer, no more tools.
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


# -------------------------------------------------
# INTENT DETECTION
# -------------------------------------------------
def classify_intent(message: str) -> str:
    msg = message.lower().strip()
    words = msg.split()

    if len(words) <= 3 and msg in {"hi", "hello", "hey", "hi there", "hello there"}:
        return "greeting"

    if len(words) <= 4 and any(w in msg for w in ["thanks", "thank you", "thx"]):
        return "thanks"

    if any(k in msg for k in [
        "earlier", "previous", "you said", "we talked",
        "last message", "first question"
    ]):
        return "conversational"

    return "document_query"


def get_project_domain(project_id: str):
    res = supabase.table("projects") \
        .select("domain") \
        .eq("id", project_id) \
        .single() \
        .execute()
    return res.data.get("domain") if res.data else None


def classify_source_intent(message: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classify this question as either 'structured' or 'conceptual'.

'structured' = looking up a specific record, person, value, date, status, or list
Examples: "what are John's remarks", "show sales for March", "find order status for ID 123", "list all employees"

'conceptual' = asking about a process, policy, explanation, or general knowledge
Examples: "how does the refund process work", "what is the leave policy", "explain the onboarding steps"

Question: {message}
Reply with only one word: structured or conceptual"""
        }],
        temperature=0,
        max_tokens=10,
    )
    result = resp.choices[0].message.content.strip().lower()
    return result if result in ("structured", "conceptual") else "conceptual"


# -------------------------------------------------
# CORE CHAT LOGIC
# -------------------------------------------------
def run_chat(project_id: str, chat_id: str, message: str, history: list):
    try:
        from sources.postgres import run_text_to_sql

        history = history or []
        domain = get_project_domain(project_id)

        system_prompt = SYSTEM_PROMPT
        if domain:
            system_prompt += f"\n\nDomain:\n- You are specialized in {domain}."

        active_tools = []

        # Fetched once — used both to decide which tools are even offered
        # (e.g. ordering is WhatsApp-only) and later to execute them.
        chat_row = supabase.table("chats").select("channel, external_id").eq("id", chat_id).maybe_single().execute()
        chat_data = (chat_row.data if chat_row else None) or {}
        channel = chat_data.get("channel")
        external_id = chat_data.get("external_id")

        appointment_settings = get_appointment_settings_if_bookable(project_id)
        if appointment_settings:
            active_tools += APPOINTMENT_TOOLS
            # The model has no built-in sense of "today" — without this,
            # relative dates like "tomorrow" get resolved to an arbitrary
            # guess, which silently produces wrong availability checks.
            today_ist = (datetime.utcnow() + timedelta(hours=5, minutes=30))
            system_prompt += (
                "\n\nBooking:\n"
                f"- Today's date is {today_ist.strftime('%A, %Y-%m-%d')} (India time). Always resolve relative "
                "dates like 'tomorrow', 'next Monday', or 'this weekend' into an actual YYYY-MM-DD date "
                "yourself, based on today's date, before calling any booking tool — never pass a relative "
                "phrase as the date.\n"
                "- You can check appointment availability and book one using the tools provided.\n"
                "- Always check availability before proposing a time.\n"
                "- If the customer confirms a time without repeating the date (e.g. just says 'book 9am'), "
                "book it against the date from the MOST RECENT availability check or proposal in this "
                "conversation — never an earlier date mentioned before that. If it's ambiguous which date "
                "they mean, ask them to confirm the date explicitly rather than guessing.\n"
                "- Confirmation is MANDATORY and always takes two separate messages, no exceptions: first you "
                "state the exact date and time back to the customer and ask them to confirm; only on their "
                "NEXT reply, if it's a clear yes, do you call book_appointment. A message that merely names "
                "a date/time (even one containing the word 'book') is the REQUEST, not the confirmation — "
                "always propose it back and wait for their next reply before booking.\n"
                "- If a slot turns out to be unavailable, apologize briefly and offer to check another time."
            )

        from shop import get_shop_settings_if_assistable
        shop_settings = get_shop_settings_if_assistable(project_id)
        if shop_settings:
            active_tools += SHOP_READONLY_TOOLS
            system_prompt += (
                "\n\nShop:\n"
                "- You can look up the customer's own past orders and browse the product catalog using the tools provided."
            )
            if shop_settings.get("bot_can_order") and channel == "whatsapp":
                active_tools += SHOP_ORDER_TOOLS
                system_prompt += (
                    "\n- You can also place a new order once the customer has explicitly confirmed the exact "
                    "items, quantities, and total price — always check the catalog and read the order back "
                    "to them first, never guess or assume what they want."
                )
            else:
                system_prompt += (
                    "\n- These tools are read-only — you cannot place, change, or cancel an order this way. "
                    "If a customer wants to place a new order, point them to the shop link instead."
                )

        intent = classify_intent(message)

        if intent == "greeting":
            return {"answer": "Hey! 👋 What can I help you with?", "sources": []}

        if intent == "thanks":
            return {"answer": "You're welcome! 😊", "sources": []}

        save_message(chat_id, "user", message)

        if intent == "conversational":
            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-7:]:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": message})

            answer = run_completion(messages, active_tools, project_id, channel, external_id, temperature=0.3, max_tokens=500)
            save_message(chat_id, "assistant", answer)
            return {"answer": answer, "sources": []}

        source_intent = classify_source_intent(message)

        query_for_embedding = message
        if len(message.split()) <= 4 and history:
            last_user_msgs = [m for m in history if m["role"] == "user"]
            if last_user_msgs:
                query_for_embedding = last_user_msgs[-1]["content"] + " " + message

        q = embeddings.embed_query(query_for_embedding)
        context = None

        if source_intent == "structured":
            res = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q,
                limit=7,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id)),
                        models.FieldCondition(key="source_type", match=models.MatchAny(any=["gsheets", "excel"])),
                    ]
                )
            )
            hits = res.points

            if hits:
                context = "\n\n---\n\n".join(
                    f"[Source: gsheets]\n{h.payload.get('text', '')}" for h in hits
                )
            else:
                pg_source = supabase.table("data_sources") \
                    .select("config, allowed_schema") \
                    .eq("project_id", project_id) \
                    .eq("type", "postgres") \
                    .limit(1) \
                    .execute()

                if pg_source.data:
                    db_url = pg_source.data[0]["config"]["url"]
                    allowed_schema = pg_source.data[0].get("allowed_schema")
                    sql_result = run_text_to_sql(message, db_url, openai_client, allowed_schema)
                    context = f"[Source: database]\n{sql_result}"
                else:
                    source_intent = "conceptual"

        if source_intent == "conceptual":
            res = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q,
                limit=7,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id)),
                        models.FieldCondition(key="source_type", match=models.MatchAny(any=["document", "website"])),
                    ]
                )
            )
            hits = res.points

            if hits:
                context = "\n\n---\n\n".join(
                    f"[Source: document]\n{h.payload.get('text', '')}" for h in hits
                )

        if not context and not active_tools:
            answer = "I couldn't find that in your documents or data sources."
            save_message(chat_id, "assistant", answer)
            return {"answer": answer, "sources": []}

        messages = [{"role": "system", "content": system_prompt}]
        for h in history[-7:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({
            "role": "user",
            "content": f"Context:\n{context or '(none — this may be a booking/order/catalog request rather than a document question)'}\n\nQuestion:\n{message}"
        })

        answer = run_completion(messages, active_tools, project_id, channel, external_id, temperature=0.2, max_tokens=300)
        save_message(chat_id, "assistant", answer)
        return {"answer": answer, "sources": []}

    except Exception as e:
        print(f"ERROR IN RUN_CHAT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# CHAT ENDPOINTS
# -------------------------------------------------
@router.post("/chat")
def chat(req: ChatRequest, user=Depends(verify_token)):
    rate_check = check_rate_limit(req.projectId)
    if not rate_check["allowed"]:
        raise HTTPException(status_code=429, detail=rate_check["reason"])

    history = get_history(req.chatId, limit=7)
    result = run_chat(req.projectId, req.chatId, req.message, history)
    increment_usage(req.projectId)
    return result


@router.post("/public/chat")
def public_chat(req: PublicChatRequest):
    session_id = req.sessionId or str(uuid.uuid4())

    existing = supabase.table("chats").select("id").eq("id", session_id).execute()
    if not existing.data:
        supabase.table("chats").insert({
            "id": session_id,
            "project_id": req.projectId,
            "title": "Public Chat",
            "channel": "public",
        }).execute()

    rate_check = check_rate_limit(req.projectId)
    if not rate_check["allowed"]:
        return {
            "answer": "Sorry, this assistant has reached its monthly limit. Please try again next month.",
            "sessionId": session_id,
        }

    history = get_history(session_id, limit=7) if req.sessionId else []
    result = run_chat(req.projectId, session_id, req.message, history)
    result["sessionId"] = session_id
    increment_usage(req.projectId)
    return result