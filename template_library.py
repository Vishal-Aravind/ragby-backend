"""
Pre-approved WhatsApp template library.
These are common business templates that Meta typically approves quickly.
Businesses can add these to their WABA with one click.
"""
from fastapi import APIRouter, Depends, HTTPException
from clients import supabase
from auth import verify_token, require_project_role
from config import WHATSAPP_TOKEN
import requests

router = APIRouter()

# -------------------------------------------------
# PRE-BUILT TEMPLATE LIBRARY
# -------------------------------------------------
TEMPLATE_LIBRARY = [
    {
        "id": "appointment_reminder",
        "category": "UTILITY",
        "industry": "Healthcare / Services",
        "name": "appointment_reminder",
        "display_name": "Appointment Reminder",
        "description": "Remind customers about upcoming appointments",
        "preview": "Hi {Name}! Your appointment is confirmed for {Date} at {Time}. Reply CONFIRM to confirm or CANCEL to cancel.",
        "components": [
            {
                "type": "BODY",
                "text": "Hi {{1}}! Your appointment is confirmed for {{2}} at {{3}}. Reply CONFIRM to confirm or CANCEL to cancel.",
            }
        ],
        "example_body": ["John", "25 Dec 2026", "10:00 AM"],
        "variables": ["Customer name", "Date", "Time"],
        "language": "en_US",
    },
    {
        "id": "order_confirmation",
        "category": "UTILITY",
        "industry": "E-commerce / Retail",
        "name": "order_confirmation",
        "display_name": "Order Confirmation",
        "description": "Confirm customer orders automatically",
        "preview": "Hi {Name}! Your order {Order ID} has been confirmed. Total: ₹{Amount}. We will notify you when it ships!",
        "components": [
            {
                "type": "BODY",
                # Removed # before {{2}} — Meta rejects #{{ pattern
                "text": "Hi {{1}}! Your order {{2}} has been confirmed. Total amount: {{3}}. We will notify you when it ships!",
            }
        ],
        "example_body": ["John", "ORD123", "500"],
        "variables": ["Customer name", "Order ID", "Amount"],
        "language": "en_US",
    },
    {
        "id": "payment_received",
        "category": "UTILITY",
        "industry": "All businesses",
        "name": "payment_received",
        "display_name": "Payment Received",
        "description": "Confirm payment receipt to customers",
        "preview": "Hi {Name}! We have received your payment of ₹{Amount}. Transaction ID: {Transaction ID}. Thank you!",
        "components": [
            {
                "type": "BODY",
                "text": "Hi {{1}}! We have received your payment. Amount: {{2}}. Transaction ID: {{3}}. Thank you for your business!",
            }
        ],
        "example_body": ["John", "500", "TXN123456"],
        "variables": ["Customer name", "Amount", "Transaction ID"],
        "language": "en_US",
    },
    {
        "id": "delivery_update",
        "category": "UTILITY",
        "industry": "E-commerce / Delivery",
        "name": "delivery_update",
        "display_name": "Delivery Update",
        "description": "Keep customers updated on their delivery",
        "preview": "Hi {Name}! Your order {Order ID} is out for delivery. Expected by {Time}. Please be available to receive it.",
        "components": [
            {
                "type": "BODY",
                # Removed #{{ and URL variable — both cause Meta rejections
                "text": "Hi {{1}}! Your order {{2}} is out for delivery. Expected by {{3}}. Please be available to receive it.",
            }
        ],
        "example_body": ["John", "ORD123", "5:00 PM"],
        "variables": ["Customer name", "Order ID", "Expected time"],
        "language": "en_US",
    },
    {
        "id": "booking_confirmation",
        "category": "UTILITY",
        "industry": "Hotels / Travel / Services",
        "name": "booking_confirmation",
        "display_name": "Booking Confirmation",
        "description": "Confirm table, room, or service bookings",
        "preview": "Hi {Name}! Your booking is confirmed. Date: {Date}, Time: {Time}, Reference: {Booking ID}. See you soon!",
        "components": [
            {
                "type": "BODY",
                # Removed #{{ before {{4}}
                "text": "Hi {{1}}! Your booking is confirmed. Date: {{2}}, Time: {{3}}, Reference: {{4}}. See you soon!",
            }
        ],
        "example_body": ["John", "25 Dec 2026", "7:00 PM", "BK001"],
        "variables": ["Customer name", "Date", "Time", "Booking ID"],
        "language": "en_US",
    },
    {
        "id": "welcome_message",
        "category": "MARKETING",
        "industry": "All businesses",
        "name": "welcome_message",
        "display_name": "Welcome Message",
        "description": "Welcome new customers to your business",
        "preview": "Hi {Name}! Welcome to {Business}! We are excited to have you. Reply with any questions.",
        "components": [
            {
                "type": "BODY",
                "text": "Hi {{1}}! Welcome to {{2}}! We are excited to have you. Feel free to reply with any questions — we are here to help!",
            }
        ],
        "example_body": ["John", "Best Store"],
        "variables": ["Customer name", "Business name"],
        "language": "en_US",
    },
    {
        "id": "special_offer",
        "category": "MARKETING",
        "industry": "All businesses",
        "name": "special_offer",
        "display_name": "Special Offer",
        "description": "Send promotional offers to customers",
        "preview": "Hi {Name}! Special offer just for you — {Discount}% off on {Product}. Valid till {Date}. Do not miss out!",
        "components": [
            {
                "type": "BODY",
                "text": "Hi {{1}}! Special offer just for you — {{2}}% off on {{3}}. Valid till {{4}}. Do not miss out!",
            }
        ],
        "example_body": ["John", "20", "All Products", "31 Dec 2026"],
        "variables": ["Customer name", "Discount %", "Product/Service", "Expiry date"],
        "language": "en_US",
    },
    {
        "id": "feedback_request",
        "category": "UTILITY",
        "industry": "All businesses",
        "name": "feedback_request",
        "display_name": "Feedback Request",
        "description": "Ask customers for reviews and feedback",
        "preview": "Hi {Name}! How was your experience with {Business}? Reply with a rating from 1 to 5 and your comments.",
        "components": [
            {
                "type": "BODY",
                # Removed URL variable — Meta flags link variables in UTILITY templates
                "text": "Hi {{1}}! How was your experience with {{2}}? Please reply with a rating from 1 to 5 and share your feedback. We appreciate your time!",
            }
        ],
        "example_body": ["John", "Best Store"],
        "variables": ["Customer name", "Business name"],
        "language": "en_US",
    },
    {
        "id": "invoice_ready",
        "category": "UTILITY",
        "industry": "B2B / Services",
        "name": "invoice_ready",
        "display_name": "Invoice Ready",
        "description": "Notify customers when their invoice is ready",
        "preview": "Hi {Name}! Your invoice {Invoice ID} for ₹{Amount} is ready. Due date: {Date}. Please contact us to make payment.",
        "components": [
            {
                "type": "BODY",
                # Removed #{{ and URL variable, reduced to 4 variables
                "text": "Hi {{1}}! Your invoice {{2}} is ready. Amount: {{3}}. Due date: {{4}}. Please contact us to make payment.",
            }
        ],
        "example_body": ["John", "INV001", "5000", "31 Dec 2026"],
        "variables": ["Customer name", "Invoice ID", "Amount", "Due date"],
        "language": "en_US",
    },
    {
        "id": "reorder_reminder",
        "category": "MARKETING",
        "industry": "Retail / Pharmacy",
        "name": "reorder_reminder",
        "display_name": "Reorder Reminder",
        "description": "Remind customers to restock their products",
        "preview": "Hi {Name}! Time to restock {Product}? Order now and get it delivered by {Date}. Reply YES to reorder!",
        "components": [
            {
                "type": "BODY",
                "text": "Hi {{1}}! Time to restock {{2}}? Order now and get it delivered by {{3}}. Reply YES to reorder!",
            }
        ],
        "example_body": ["John", "Vitamin C Tablets", "28 Jun 2026"],
        "variables": ["Customer name", "Product name", "Delivery date"],
        "language": "en_US",
    },
]


# -------------------------------------------------
# GET TEMPLATE LIBRARY
# -------------------------------------------------
@router.get("/template-library")
def get_template_library():
    """Return all pre-built templates."""
    return TEMPLATE_LIBRARY


# -------------------------------------------------
# ADD TEMPLATE TO WABA
# -------------------------------------------------
@router.post("/template-library/add")
def add_template_to_waba(data: dict, user=Depends(verify_token)):
    """Submit a pre-built template to the customer's WABA."""
    project_id  = data["project_id"]
    template_id = data["template_id"]
    require_project_role(user.id, project_id)

    # Find template
    template = next((t for t in TEMPLATE_LIBRARY if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Get WABA ID
    wa = supabase.table("whatsapp_integrations") \
        .select("waba_id") \
        .eq("project_id", project_id) \
        .execute()

    if not wa.data or not wa.data[0].get("waba_id"):
        raise HTTPException(status_code=400, detail="WhatsApp not connected")

    waba_id = wa.data[0]["waba_id"]

    # Build components with examples
    components = []
    for comp in template["components"]:
        new_comp = dict(comp)
        if comp["type"] == "BODY" and template.get("example_body"):
            new_comp["example"] = {
                "body_text": [template["example_body"]]
            }
        components.append(new_comp)

    # Submit to Meta
    payload = {
        "name": template["name"],
        "category": template["category"],
        "language": template["language"],
        "components": components,
    }

    print(f"Submitting template to Meta: {payload}")

    res = requests.post(
        f"https://graph.facebook.com/v19.0/{waba_id}/message_templates",
        headers={
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
    )

    resp_data = res.json()
    print(f"Meta template response: {resp_data}")

    if res.ok:
        return {
            "status": "submitted",
            "template_id": resp_data.get("id"),
            "message": "Template submitted for approval. Usually approved within a few minutes to hours."
        }
    else:
        if "already exists" in str(resp_data).lower():
            return {"status": "exists", "message": "Template already exists in your account."}
        raise HTTPException(
            status_code=400,
            detail=resp_data.get("error", {}).get("message", "Failed to submit template")
        )


# -------------------------------------------------
# SYNC TEMPLATES FROM META
# -------------------------------------------------
@router.get("/template-library/sync/{project_id}")
def sync_templates_from_meta(project_id: str, user=Depends(verify_token)):
    """Fetch all templates from the merchant's WABA and return with approval status."""
    require_project_role(user.id, project_id)

    # Get WABA ID
    wa = supabase.table("whatsapp_integrations") \
        .select("waba_id") \
        .eq("project_id", project_id) \
        .maybe_single() \
        .execute()

    if not wa or not wa.data or not wa.data.get("waba_id"):
        raise HTTPException(status_code=400, detail="WhatsApp not connected")

    waba_id = wa.data["waba_id"]

    # Fetch all templates from Meta
    res = requests.get(
        f"https://graph.facebook.com/v19.0/{waba_id}/message_templates",
        headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
        params={"fields": "name,status,category,language,components", "limit": 100},
    )

    if not res.ok:
        raise HTTPException(status_code=400, detail=res.json().get("error", {}).get("message", "Failed to fetch templates"))

    templates = res.json().get("data", [])
    return {"templates": templates, "total": len(templates)}