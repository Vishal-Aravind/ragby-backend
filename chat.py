import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from clients import supabase, openai_client, embeddings, qdrant
from config import QDRANT_COLLECTION, FRONTEND_URL
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
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": (
                "Cancel the customer's upcoming appointment. If the customer has more than one upcoming "
                "appointment, you MUST call check_my_appointments first (if you haven't already) and pass the "
                "exact date and time of the ONE they mean — never omit date/time when more than one exists, "
                "and never guess which one based on the customer's wording alone. Always state which "
                "appointment (date, time, service) you're about to cancel and get the customer's explicit yes "
                "on THIS SPECIFIC action before calling this — cancelling is its own separate confirmation, "
                "never reuse a 'yes' from earlier in the conversation that was about something else (like "
                "booking). CRITICAL: once you've stated a specific date AND time back to the customer (e.g. "
                "'you have an appointment at 3:30 PM on Thursday'), you already know both — when they confirm, "
                "pass BOTH that exact date and that exact time in this same call. Never call with only the "
                "date once you've already stated a specific time out loud; dropping the time you just "
                "mentioned is a bug, not a safe default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_phone": {"type": "string", "description": "Customer's phone number — only ask for this if it isn't already known from this conversation channel"},
                    "date": {"type": "string", "description": "The exact date (YYYY-MM-DD) of the appointment to cancel — required if the customer has more than one upcoming appointment"},
                    "time": {"type": "string", "description": "The exact start time of the appointment to cancel — required if the customer has more than one appointment on that date. If you already stated a specific time to the customer, always include it here."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_my_appointments",
            "description": "Look up the customer's own upcoming appointments (date, time, service). Read-only — cannot book, change, or cancel anything.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_phone": {"type": "string", "description": "Customer's phone number — only ask for this if it isn't already known from this conversation channel"},
                },
                "required": [],
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
            "description": (
                "Get the shop link plus the list of available products (name, price, description). Always "
                "call this before answering any catalog-related question — never guess or invent an item or "
                "price. Read-only — cannot add items to a cart or place an order."
            ),
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


# Only ever added when the project has at least one event with
# bot_can_register on (see get_bot_registrable_events in events.py) —
# per-event opt-in, not project-wide, since a merchant may want automation
# on a public webinar but not an invite-only event running the same week.
EVENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browse_upcoming_events",
            "description": (
                "Get the list of events currently open for registration via chat (title, date, day of week, "
                "time, location, spots left if capacity-limited, and any extra fields required to register) to "
                "answer questions like 'what events do you have coming up' or to find a specific event by name "
                "before registering. Only shows events the merchant has enabled for chat registration — if the "
                "customer names an event not in this list, don't guess; tell them it may not be available for "
                "chat registration. Read-only — cannot register or cancel anything."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_event_registration_link",
            "description": (
                "Get the direct registration link for a specific event, to send to a customer who wants to "
                "register or sign up. This does NOT register them itself — it only looks up the correct link. "
                "Call browse_upcoming_events first if you haven't already, so the event title is accurate — "
                "never invent an event name. No confirmation needed, just send the link back to the customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_title": {
                        "type": "string",
                        "description": "The event's title, as close as possible to how browse_upcoming_events listed it — never invent or guess a title the customer didn't say and browse hasn't shown.",
                    },
                },
                "required": ["event_title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_my_event_registrations",
            "description": "Look up the customer's own event registrations (event title, date, day of week, status) across all events. Read-only — cannot register or cancel anything.",
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
            "name": "cancel_event_registration",
            "description": (
                "Cancel the customer's registration for a specific event. If the customer is registered for "
                "more than one event, you MUST call check_my_event_registrations first (if you haven't already) "
                "and pass the exact event title of the ONE they mean — never omit it when more than one exists, "
                "and never guess based on wording alone. Always state which event's registration you're about "
                "to cancel and get the customer's explicit yes on THIS SPECIFIC action before calling this — "
                "never reuse a 'yes' from earlier in the conversation about something else (like registering)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_phone": {"type": "string", "description": "Customer's phone number — only ask for this if it isn't already known from this conversation channel"},
                    "event_title": {"type": "string", "description": "The exact title of the event whose registration to cancel — required if the customer has more than one registration. If you already stated a specific event to the customer, always include it here."},
                },
                "required": [],
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


def _normalize_time_str(time_str: str) -> str:
    """The model is told to always send 24-hour 'HH:MM', but doesn't
    reliably do so. A silent format mismatch here (e.g. '4:30 PM' or
    '16:30:00' instead of exactly '16:30') would make a genuinely available
    slot fail the availability re-check below and silently reject a real
    booking — so normalize common variants instead of trusting the model."""
    cleaned = time_str.strip()
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%H:%M")
        except ValueError:
            continue
    raise ValueError(f"Couldn't understand the time '{time_str}' — ask the customer for a clear time like 3:30 PM.")


def _with_day_of_week(appts: list) -> list:
    """LLMs are unreliable at computing which weekday an arbitrary date
    falls on — left to guess, the model will happily echo back whatever
    day name the customer said, even when it doesn't match the actual
    date. Compute it deterministically here instead of trusting the model."""
    result = []
    for a in appts:
        day_name = datetime.strptime(a["appointment_date"], "%Y-%m-%d").strftime("%A")
        result.append({
            "date": a["appointment_date"],
            "day_of_week": day_name,
            "time": a["start_time"],
            "service": a["service_name"],
            "status": a["status"],
        })
    return result


def _event_day_of_week(date_str: Optional[str]) -> Optional[str]:
    """Same reasoning as _with_day_of_week — never let the model compute a
    weekday itself. event_date is optional (some events are undated/ongoing),
    so this tolerates a missing or unparseable date."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
    except Exception:
        return None


def _shape_event_for_ai(e: dict) -> dict:
    """Strips internal ids/raw form_schema before handing an event to the
    model — only what's needed to browse and to build a registration link.
    Registration itself always happens on the real page (get_event_
    registration_link), so custom form fields aren't surfaced here."""
    return {
        "title": e.get("title"),
        "description": e.get("description"),
        "date": e.get("event_date"),
        "day_of_week": _event_day_of_week(e.get("event_date")),
        "time": e.get("event_time"),
        "location": e.get("location"),
        "spots_left": e.get("spots_left"),
    }


def _shape_registration_for_ai(r: dict) -> dict:
    return {
        "event_title": r.get("event_title"),
        "date": r.get("event_date"),
        "day_of_week": _event_day_of_week(r.get("event_date")),
        "time": r.get("event_time"),
        "location": r.get("location"),
        "status": r.get("status"),
    }


def execute_appointment_tool(name: str, args: dict, project_id: str, channel: str, external_id: str) -> dict:
    """Never trusts the model's parameters as final — create_appointment
    re-validates the slot is actually still free."""
    from appointments import get_available_slots, create_appointment, get_latest_upcoming_appointment, get_upcoming_appointments, cancel_appointment as cancel_appointment_fn

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
                start_time=_normalize_time_str(args["time"]),
                notes=None,
            )

        if name == "cancel_appointment":
            phone = external_id if channel == "whatsapp" else args.get("customer_phone")
            if not phone:
                return {"error": "Still need the customer's phone number to find their appointment."}

            appts = get_upcoming_appointments(project_id, phone, limit=20)
            if not appts:
                return {"message": "No upcoming appointment found for this customer."}

            if len(appts) == 1:
                target = appts[0]
            else:
                date_arg = args.get("date")
                if not date_arg:
                    return {
                        "error": "This customer has more than one upcoming appointment, and no date was given. "
                                 "If you already stated a specific date/time to the customer in your last message, "
                                 "retry this exact call now with that date and time — do not ask the customer "
                                 "again. Otherwise, call check_my_appointments and ask which one they mean.",
                        "appointments": _with_day_of_week(appts),
                    }
                time_arg = _normalize_time_str(args["time"]) if args.get("time") else None
                matches = [
                    a for a in appts
                    if a["appointment_date"] == date_arg and (not time_arg or a["start_time"][:5] == time_arg)
                ]
                if not matches:
                    return {
                        "error": f"No upcoming appointment found matching {date_arg}" + (f" at {time_arg}" if time_arg else "") + ".",
                        "appointments": _with_day_of_week(appts),
                    }
                if len(matches) > 1:
                    return {
                        "error": "More than one appointment matches that date, and no time was given. If you "
                                 "already stated a specific time to the customer in your last message, retry "
                                 "this exact call now with that time included — do not ask the customer again.",
                        "appointments": _with_day_of_week(matches),
                    }
                target = matches[0]

            # notify_customer=False — they're the one cancelling it right
            # here in this conversation, a second templated WhatsApp message
            # on top of the AI's own reply would just be noise.
            result = cancel_appointment_fn(target["id"], notify_customer=False)
            return {
                "status": "cancelled",
                "date": result["appointment_date"],
                "day_of_week": datetime.strptime(result["appointment_date"], "%Y-%m-%d").strftime("%A"),
                "time": result["start_time"],
                "service": result["service_name"],
            }

        if name == "check_my_appointments":
            phone = external_id if channel == "whatsapp" else args.get("customer_phone")
            if not phone:
                return {"error": "Still need the customer's phone number to look up their appointments."}

            appts = get_upcoming_appointments(project_id, phone)
            if not appts:
                return {"message": "No upcoming appointments found for this customer."}
            return {"appointments": _with_day_of_week(appts)}

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
            shop_link = f"{FRONTEND_URL}/shop/{project_id}"
            if channel == "whatsapp" and external_id:
                shop_link += f"?phone={external_id}"
            if not products:
                return {"shop_link": shop_link, "message": "No products currently available."}
            return {"shop_link": shop_link, "products": products}

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


def execute_event_tool(name: str, args: dict, project_id: str, channel: str, external_id: str) -> dict:
    from events import (
        get_upcoming_events_for_ai, find_event_by_title, get_registrations_for_phone,
        cancel_registration_core,
    )

    try:
        if name == "browse_upcoming_events":
            events = get_upcoming_events_for_ai(project_id)
            if not events:
                return {"message": "No events currently open for registration."}
            return {"events": [_shape_event_for_ai(e) for e in events]}

        if name == "get_event_registration_link":
            matches = find_event_by_title(project_id, args.get("event_title", ""))
            if not matches:
                return {"error": f"Couldn't find an open event matching '{args.get('event_title', '')}'. Call browse_upcoming_events and confirm the exact title with the customer."}
            if len(matches) > 1:
                return {
                    "error": "More than one open event matches that name — if you already stated a specific "
                             "event's exact title to the customer, retry this exact call now with that exact "
                             "title. Otherwise ask the customer which one they mean.",
                    "candidates": [_shape_event_for_ai(e) for e in matches],
                }
            event = matches[0]
            return {
                "event_title": event["title"],
                "registration_link": f"{FRONTEND_URL}/event/{event['id']}",
            }

        if name == "check_my_event_registrations":
            phone = external_id if channel == "whatsapp" else args.get("customer_phone")
            if not phone:
                return {"error": "Still need the customer's phone number to look up their registrations."}
            regs = get_registrations_for_phone(project_id, phone)
            if not regs:
                return {"message": "No event registrations found for this customer."}
            return {"registrations": [_shape_registration_for_ai(r) for r in regs]}

        if name == "cancel_event_registration":
            phone = external_id if channel == "whatsapp" else args.get("customer_phone")
            if not phone:
                return {"error": "Still need the customer's phone number to find their registration."}

            regs = get_registrations_for_phone(project_id, phone)
            if not regs:
                return {"message": "No event registrations found for this customer."}

            if len(regs) == 1:
                target = regs[0]
            else:
                title_arg = (args.get("event_title") or "").strip().lower()
                if not title_arg:
                    return {
                        "error": "This customer has more than one event registration, and no event title was "
                                 "given. If you already stated a specific event's exact title to the customer in "
                                 "your last message, retry this exact call now with that title — do not ask the "
                                 "customer again. Otherwise, call check_my_event_registrations and ask which one "
                                 "they mean.",
                        "registrations": [_shape_registration_for_ai(r) for r in regs],
                    }
                matches = [
                    r for r in regs
                    if title_arg == r["event_title"].strip().lower() or title_arg in r["event_title"].strip().lower()
                ]
                if not matches:
                    return {
                        "error": f"No registration found matching '{args.get('event_title')}'.",
                        "registrations": [_shape_registration_for_ai(r) for r in regs],
                    }
                if len(matches) > 1:
                    return {
                        "error": "More than one registration matches that title — ask the customer to be more specific.",
                        "registrations": [_shape_registration_for_ai(r) for r in matches],
                    }
                target = matches[0]

            # notify_customer=False — they're the one cancelling it right
            # here in this conversation, a second templated WhatsApp message
            # on top of the AI's own reply would just be noise.
            cancel_registration_core(target["registration_id"], notify_customer=False)
            return {
                "status": "cancelled",
                "event_title": target["event_title"],
                "date": target["event_date"],
                "day_of_week": _event_day_of_week(target["event_date"]),
            }

        return {"error": f"Unknown tool {name}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"execute_event_tool error: {e}")
        return {"error": "Something went wrong trying to do that — please try again."}


def execute_tool(name: str, args: dict, project_id: str, channel: str, external_id: str) -> dict:
    if name in ("check_appointment_availability", "book_appointment", "cancel_appointment", "check_my_appointments"):
        return execute_appointment_tool(name, args, project_id, channel, external_id)
    if name in ("check_order_status", "browse_shop_catalog", "place_order"):
        return execute_shop_tool(name, args, project_id, channel, external_id)
    if name in ("browse_upcoming_events", "get_event_registration_link", "check_my_event_registrations", "cancel_event_registration"):
        return execute_event_tool(name, args, project_id, channel, external_id)
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

    # A bare confirmation ("okay", "yes", "sure") has no real content to
    # search documents for — running it through document search anyway
    # risks pulling in unrelated content and derailing a pending
    # confirmation (e.g. for a booking). Treat it as conversational instead
    # — same lightweight path, but it still has the tools and full history.
    if msg.strip(".!") in {
        "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "confirmed",
        "correct", "proceed", "go ahead", "sounds good", "book it", "do it", "that works"
    }:
        return "conversational"

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

        # The model has no built-in sense of "today" — without this, relative
        # dates ("tomorrow", "next Monday") get resolved to an arbitrary
        # guess. Hoisted here (not just inside Booking) so Registrations can
        # use it too even when Appointments isn't enabled for this project.
        today_ist = (datetime.utcnow() + timedelta(hours=5, minutes=30))

        appointment_settings = get_appointment_settings_if_bookable(project_id)
        if appointment_settings:
            active_tools += APPOINTMENT_TOOLS
            system_prompt += (
                "\n\nBooking:\n"
                "- IMPORTANT: appointment/booking information (whether a slot is free, what the customer has "
                "booked, etc.) NEVER lives in your documents — it only exists through the tools below. If a "
                "message is about checking, booking, or cancelling an appointment, always use the matching "
                "tool. Never respond with 'I couldn't find that in your documents' or ask generic clarifying "
                "questions for this category — that fallback is for document questions, not this.\n"
                "- If the customer asks to see/check their existing bookings or appointments (e.g. 'show my "
                "bookings', 'what do I have booked'), call check_my_appointments immediately — it's read-only, "
                "needs no confirmation, and needs no other information from the customer first on WhatsApp.\n"
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
                "- Confirmation is MANDATORY and always takes two separate messages, no exceptions — this "
                "applies EVERY time, including when the customer just picks one of several times YOU offered "
                "(e.g. after telling them a time was unavailable and listing alternatives). Picking an option "
                "is still only the request. First you state the exact date and time back to the customer and "
                "ask them to confirm; only on their NEXT reply, if it's a clear yes, do you call "
                "book_appointment. A message that merely names or selects a date/time (even one containing "
                "the word 'book') is the REQUEST, not the confirmation.\n"
                "- If a slot turns out to be unavailable, apologize briefly and offer to check another time.\n"
                "- You can also cancel the customer's upcoming appointment using the tool provided. This needs "
                "its OWN separate two-message confirmation, exactly like booking — state which appointment "
                "you're about to cancel, wait for their next reply to be a clear yes, then call "
                "cancel_appointment. If the customer has multiple upcoming appointments, you MUST pass the "
                "exact date and time to cancel_appointment — never omit them, never guess which one they mean.\n"
                "- Never treat a 'yes' about booking as also confirming a cancellation, or vice versa — they "
                "are different actions and each needs its own confirmation.\n"
                "- When the tools return appointment information, they include a 'day_of_week' field (e.g. "
                "'Wednesday'). Always use this computed field when speaking back to the customer — do NOT "
                "compute the day of week yourself from the date, because you will almost certainly get it wrong. "
                "Use the tool's day_of_week value verbatim."
            )

        from shop import get_shop_settings_if_assistable
        shop_settings = get_shop_settings_if_assistable(project_id)
        if shop_settings:
            active_tools += SHOP_READONLY_TOOLS
            system_prompt += (
                "\n\nShop:\n"
                "- You can look up the customer's own past orders and browse the product catalog using the "
                "tools provided.\n"
                "- If the customer asks broadly what you sell, to see the menu, or to browse (no specific "
                "product or preference stated), call browse_shop_catalog and just send them the shop_link — "
                "don't enumerate every item in chat text, the real shop page is a much better browsing "
                "experience (photos, categories, cart).\n"
                "- If the customer asks about a specific product, or wants a recommendation with any stated "
                "preference (budget, dietary need, occasion, 'what's cheapest', 'what's good for X'), call "
                "browse_shop_catalog and answer directly using its product data — recommend by name and price, "
                "don't just deflect to the link. Only ever mention products that actually appear in that "
                "result — never invent an item or price.\n"
                "- Keep any single recommendation reply to at most 3-5 items even if more match — offer to "
                "share more if they want.\n"
                "- If a recommendation request is genuinely vague (just 'what's good?' with no stated "
                "preference at all), ask ONE clarifying question (budget, dietary need, occasion) before "
                "recommending, rather than immediately listing items."
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

        from events import get_bot_registrable_events
        if get_bot_registrable_events(project_id):
            active_tools += EVENT_TOOLS
            system_prompt += (
                "\n\nRegistrations:\n"
                "- IMPORTANT: event/registration information (which events are open, what a customer has "
                "registered for, etc.) NEVER lives in your documents — it only exists through the tools below. "
                "If a message is about browsing events, getting a registration link, checking, or cancelling an "
                "event registration, always use the matching tool, never the document fallback.\n"
                f"- Today's date is {today_ist.strftime('%A, %Y-%m-%d')} (India time) — use it to judge whether "
                "an event the customer names is still upcoming.\n"
                "- Always call browse_upcoming_events (or let get_event_registration_link tell you if a title "
                "doesn't match) before assuming an event exists — never invent an event name, date, or location.\n"
                "- You do NOT register the customer yourself — actual registration always happens on the "
                "event's own page, which can collect whatever information that specific event needs. If a "
                "customer wants to register or sign up for an event, use get_event_registration_link and send "
                "them that link — no confirmation needed for this, it's just a lookup.\n"
                "- If more than one open event matches what the customer said, or the customer has "
                "registrations for more than one event when cancelling, the tool will tell you so — ask the "
                "customer to clarify, or if you already stated one specific event's exact title in your last "
                "message, retry with that exact title included, don't ask again.\n"
                "- Cancelling IS an action that needs confirmation — state which registration you're about to "
                "cancel and get an explicit yes on THIS SPECIFIC action on the customer's NEXT reply before "
                "calling cancel_event_registration; never reuse a 'yes' from a different action (like getting "
                "a registration link).\n"
                "- When tool results include a 'day_of_week' field, always use it verbatim — never compute the "
                "weekday yourself."
            )

        if active_tools:
            system_prompt += (
                "\n\nTool honesty — this overrides everything else:\n"
                "- NEVER tell the customer an action succeeded (booked, cancelled, ordered) unless the tool's "
                "result actually confirms success. If a tool call returns an error, tell the customer honestly "
                "that it didn't work and why (in plain terms), then offer to try again — do not claim it "
                "worked, do not make up a confirmation message or booking ID."
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