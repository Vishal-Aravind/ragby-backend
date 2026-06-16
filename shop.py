from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from clients import supabase
from auth import verify_token
import os
import hmac
import hashlib
import json
import time

router = APIRouter()

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://ragby-frontend.vercel.app")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class ShopConfigUpdate(BaseModel):
    store_name: Optional[str] = None
    store_phone: Optional[str] = None
    gst_percent: Optional[float] = None
    currency: Optional[str] = None
    accent_color: Optional[str] = None
    delivery_types: Optional[List[str]] = None
    terms_note: Optional[str] = None
    razorpay_key_id: Optional[str] = None
    razorpay_key_secret: Optional[str] = None
    is_enabled: Optional[bool] = None

class CatalogCreate(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    is_active: bool = True

class CatalogUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

class ProductCreate(BaseModel):
    project_id: str
    catalog_id: str
    name: str
    description: Optional[str] = None
    price: float
    image_url: Optional[str] = None
    category: Optional[str] = "General"
    gst_percent: Optional[float] = 0
    is_available: bool = True
    sort_order: int = 0

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    gst_percent: Optional[float] = None
    is_available: Optional[bool] = None
    sort_order: Optional[int] = None
    catalog_id: Optional[str] = None

class CartItem(BaseModel):
    product_id: str
    name: str
    price: float
    quantity: int
    image_url: Optional[str] = None

class CartSubmit(BaseModel):
    phone: str
    project_id: str
    catalog_id: str
    items: List[CartItem]
    delivery_type: Optional[str] = "Takeaway"
    order_id: Optional[str] = None  # if present, UPDATE existing order instead of creating new

class OrderStatusUpdate(BaseModel):
    status: Optional[str] = None
    payment_status: Optional[str] = None


# ─────────────────────────────────────────────
# SHOP CONFIG
# ─────────────────────────────────────────────

@router.get("/shop-config/{project_id}")
async def get_shop_config(project_id: str, user=Depends(verify_token)):
    res = supabase.table("shop_config").select("*").eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        return {
            "project_id": project_id,
            "store_name": "",
            "store_phone": "",
            "gst_percent": 0,
            "currency": "₹",
            "accent_color": "#16a34a",
            "delivery_types": ["Takeaway"],
            "terms_note": "This order is not eligible for any kind of Discounts. T&C apply.",
            "razorpay_key_id": "",
            "razorpay_key_secret": "",
            "is_enabled": False,
        }
    return res.data

@router.put("/shop-config/{project_id}")
async def update_shop_config(project_id: str, body: ShopConfigUpdate, user=Depends(verify_token)):
    existing = supabase.table("shop_config").select("id").eq("project_id", project_id).maybe_single().execute()
    update = {k: v for k, v in body.dict().items() if v is not None}
    if existing and existing.data:
        supabase.table("shop_config").update(update).eq("project_id", project_id).execute()
    else:
        supabase.table("shop_config").insert({"project_id": project_id, **update}).execute()
    res = supabase.table("shop_config").select("*").eq("project_id", project_id).single().execute()
    return res.data


# ─────────────────────────────────────────────
# CATALOGS
# ─────────────────────────────────────────────

@router.get("/catalogs")
async def list_catalogs(project_id: str, user=Depends(verify_token)):
    res = supabase.table("catalogs").select("*").eq("project_id", project_id).order("created_at", desc=False).execute()
    return res.data or []

@router.post("/catalogs")
async def create_catalog(body: CatalogCreate, user=Depends(verify_token)):
    supabase.table("catalogs").insert({
        "project_id": body.project_id,
        "name": body.name,
        "description": body.description,
        "is_active": body.is_active,
    }).execute()
    res = supabase.table("catalogs").select("*").eq("project_id", body.project_id).order("created_at", desc=True).limit(1).execute()
    return res.data[0]

@router.put("/catalogs/{catalog_id}")
async def update_catalog(catalog_id: str, body: CatalogUpdate, user=Depends(verify_token)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    supabase.table("catalogs").update(update).eq("id", catalog_id).execute()
    res = supabase.table("catalogs").select("*").eq("id", catalog_id).single().execute()
    return res.data

@router.delete("/catalogs/{catalog_id}")
async def delete_catalog(catalog_id: str, user=Depends(verify_token)):
    supabase.table("catalogs").delete().eq("id", catalog_id).execute()
    return {"status": "deleted"}


# ─────────────────────────────────────────────
# PRODUCTS
# ─────────────────────────────────────────────

@router.get("/products")
async def list_products(project_id: str, catalog_id: Optional[str] = None, user=Depends(verify_token)):
    query = supabase.table("products").select("*").eq("project_id", project_id)
    if catalog_id:
        query = query.eq("catalog_id", catalog_id)
    res = query.order("sort_order", desc=False).execute()
    return res.data or []

@router.post("/products")
async def create_product(body: ProductCreate, user=Depends(verify_token)):
    supabase.table("products").insert({
        "project_id": body.project_id,
        "catalog_id": body.catalog_id,
        "name": body.name,
        "description": body.description,
        "price": body.price,
        "image_url": body.image_url,
        "category": body.category,
        "gst_percent": body.gst_percent,
        "is_available": body.is_available,
        "sort_order": body.sort_order,
    }).execute()
    res = supabase.table("products").select("*").eq("project_id", body.project_id).eq("catalog_id", body.catalog_id).order("created_at", desc=True).limit(1).execute()
    return res.data[0]

@router.put("/products/{product_id}")
async def update_product(product_id: str, body: ProductUpdate, user=Depends(verify_token)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    supabase.table("products").update(update).eq("id", product_id).execute()
    res = supabase.table("products").select("*").eq("id", product_id).single().execute()
    return res.data

@router.delete("/products/{product_id}")
async def delete_product(product_id: str, user=Depends(verify_token)):
    supabase.table("products").delete().eq("id", product_id).execute()
    return {"status": "deleted"}


# ─────────────────────────────────────────────
# PUBLIC — Shop page APIs (no auth needed)
# ─────────────────────────────────────────────

@router.get("/public/shop/{project_id}/config")
async def public_shop_config(project_id: str):
    res = supabase.table("shop_config").select(
        "store_name,store_phone,gst_percent,currency,accent_color,delivery_types,terms_note,is_enabled"
    ).eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        return {
            "gst_percent": 0,
            "currency": "₹",
            "accent_color": "#16a34a",
            "delivery_types": ["Takeaway"],
            "terms_note": "",
            "is_enabled": True,
        }
    return res.data

@router.get("/public/shop/{project_id}/catalogs")
async def public_catalogs(project_id: str):
    res = supabase.table("catalogs").select("*").eq("project_id", project_id).eq("is_active", True).order("created_at", desc=False).execute()
    return res.data or []

@router.get("/public/shop/{project_id}/products")
async def public_products(project_id: str, catalog_id: Optional[str] = None):
    query = supabase.table("products").select("*").eq("project_id", project_id).eq("is_available", True)
    if catalog_id:
        query = query.eq("catalog_id", catalog_id)
    res = query.order("sort_order", desc=False).execute()
    return res.data or []

@router.get("/public/shop/order/{order_id}")
async def public_get_order(order_id: str):
    """Fetch an existing order's items — used to pre-populate cart for 'Add More' flow."""
    res = supabase.table("orders").select("*").eq("id", order_id).maybe_single().execute()
    if not res or not res.data:
        return {"items": []}
    return {"items": res.data.get("items", []), "delivery_type": res.data.get("delivery_type", "Takeaway")}


# ─────────────────────────────────────────────
# CART SUBMIT — called from web shop page
# ─────────────────────────────────────────────

@router.post("/public/shop/submit-cart")
async def submit_cart(body: CartSubmit):
    from whatsapp import send_whatsapp_buttons
    from config import WHATSAPP_TOKEN

    project_id = body.project_id
    phone = body.phone.replace("+", "").replace(" ", "")

    # Get shop config
    try:
        config_res = supabase.table("shop_config").select("*").eq("project_id", project_id).maybe_single().execute()
        config = (config_res.data if config_res else None) or {}
    except Exception as e:
        print(f"shop_config fetch error: {e}")
        config = {}

    gst_percent = config.get("gst_percent", 0)
    currency = config.get("currency", "₹")

    # Calculate totals
    subtotal = sum(item.price * item.quantity for item in body.items)
    gst_amount = round(subtotal * gst_percent / 100, 2)
    total = round(subtotal + gst_amount, 2)
    items_data = [item.dict() for item in body.items]

    # ── If order_id is present, UPDATE the existing order (Add More flow) ──
    if body.order_id:
        supabase.table("orders").update({
            "items": items_data,
            "subtotal": subtotal,
            "gst_amount": gst_amount,
            "total": total,
            "delivery_type": body.delivery_type or "Takeaway",
        }).eq("id", body.order_id).execute()

        order_res = supabase.table("orders").select("*").eq("id", body.order_id).single().execute()
        order = order_res.data
    else:
        # ── Otherwise create a new order ──
        supabase.table("orders").insert({
            "project_id": project_id,
            "phone_number": phone,
            "items": items_data,
            "subtotal": subtotal,
            "gst_amount": gst_amount,
            "total": total,
            "status": "pending",
            "payment_status": "unpaid",
            "delivery_type": body.delivery_type or "Takeaway",
        }).execute()

        order_res = supabase.table("orders") \
            .select("*") \
            .eq("project_id", project_id) \
            .eq("phone_number", phone) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        order = order_res.data[0]

    # Build cart summary
    lines = []
    for i, item in enumerate(body.items, 1):
        lines.append(f"{i}. {item.name} x{item.quantity} - {currency}{int(item.price * item.quantity)}")
    items_text = "\n".join(lines)
    summary = f"🛒 *Your Cart*\n\n{items_text}\n\nSubtotal: {currency}{int(subtotal)}"
    if gst_amount > 0:
        summary += f"\nGST: {currency}{gst_amount}"
    summary += f"\n*Total: {currency}{total}*"

    # Get WhatsApp integration
    try:
        wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", project_id).maybe_single().execute()
        wa_data = (wa_res.data if wa_res else None)
    except Exception as e:
        print(f"whatsapp_integrations fetch error: {e}")
        wa_data = None

    if not wa_data:
        supabase.table("whatsapp_sessions").upsert({
            "project_id": project_id,
            "phone_number": phone,
            "mode": "awaiting_cart_confirm",
            "metadata": {
                "order_id": order["id"],
                "catalog_id": body.catalog_id,
            },
        }, on_conflict="project_id,phone_number").execute()
        return {"status": "ok", "order_id": order["id"], "warning": "WhatsApp not connected"}

    phone_number_id = wa_data["phone_number_id"]
    token = wa_data.get("access_token") or WHATSAPP_TOKEN

    send_whatsapp_buttons(
        to=phone,
        body=summary,
        buttons=[
            {"id": "cart_continue", "title": "Continue ➡️"},
            {"id": "cart_add_more", "title": "Add More 🛍️"},
            {"id": "cart_clear", "title": "Clear Cart 🗑️"},
        ],
        phone_number_id=phone_number_id,
        token=token,
    )

    supabase.table("whatsapp_sessions").upsert({
        "project_id": project_id,
        "phone_number": phone,
        "mode": "awaiting_cart_confirm",
        "metadata": {
            "order_id": order["id"],
            "catalog_id": body.catalog_id,
        },
    }, on_conflict="project_id,phone_number").execute()

    return {"status": "ok", "order_id": order["id"]}


# ─────────────────────────────────────────────
# ORDERS
# ─────────────────────────────────────────────

@router.get("/orders")
async def list_orders(project_id: str, user=Depends(verify_token)):
    res = supabase.table("orders").select("*").eq("project_id", project_id).order("created_at", desc=True).execute()
    return res.data or []

@router.put("/orders/{order_id}")
async def update_order(order_id: str, body: OrderStatusUpdate, user=Depends(verify_token)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    supabase.table("orders").update(update).eq("id", order_id).execute()
    res = supabase.table("orders").select("*").eq("id", order_id).single().execute()
    return res.data


# ─────────────────────────────────────────────
# RAZORPAY WEBHOOK
# ─────────────────────────────────────────────

@router.post("/webhook/razorpay")
async def razorpay_webhook(request: Request):
    from whatsapp import send_whatsapp_message
    from config import WHATSAPP_TOKEN

    body_bytes = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    payload = json.loads(body_bytes)
    event = payload.get("event")

    if event == "payment_link.paid":
        payment_link_id = payload["payload"]["payment_link"]["entity"]["id"]

        order_res = supabase.table("orders").select("*").eq("payment_id", payment_link_id).maybe_single().execute()
        if not order_res or not order_res.data:
            return {"status": "ok"}
        order = order_res.data

        try:
            config_res = supabase.table("shop_config").select("*").eq("project_id", order["project_id"]).maybe_single().execute()
            config = (config_res.data if config_res else None) or {}
        except:
            config = {}

        key_secret = config.get("razorpay_key_secret") or RAZORPAY_KEY_SECRET
        if key_secret and signature:
            expected = hmac.new(key_secret.encode(), body_bytes, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                raise HTTPException(status_code=400, detail="Invalid signature")

        currency = config.get("currency", "₹")
        store_phone = config.get("store_phone", "")

        supabase.table("orders").update({
            "payment_status": "paid",
            "status": "confirmed",
        }).eq("id", order["id"]).execute()

        try:
            wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", order["project_id"]).maybe_single().execute()
            wa_data = (wa_res.data if wa_res else None)
        except:
            wa_data = None

        if not wa_data:
            return {"status": "ok"}

        phone_number_id = wa_data["phone_number_id"]
        token = wa_data.get("access_token") or WHATSAPP_TOKEN

        msg = f"✅ *Payment Confirmed!*\n\n"
        msg += f"Thank you! Your order has been confirmed.\n\n"
        msg += f"Order ID: #{order['id'][:8].upper()}\n"
        msg += f"Total: {currency}{order['total']:.2f}\n\n"
        if store_phone:
            msg += f"📞 Contact: {store_phone}\n\n"
        msg += "We'll notify you when your order is ready! 🎉"

        send_whatsapp_message(to=order["phone_number"], text=msg, phone_number_id=phone_number_id, token=token)

        if store_phone:
            lines = []
            for i, item in enumerate(order["items"], 1):
                lines.append(f"{i}. {item['name']} x{item['quantity']} - {currency}{int(item['price'] * item['quantity'])}")

            owner_msg = f"🔔 *New Order Paid!*\n\nFrom: +{order['phone_number']}\n\n"
            owner_msg += "\n".join(lines) + "\n\n"
            owner_msg += f"Total: {currency}{order['total']:.2f}\n"
            owner_msg += f"Order ID: #{order['id'][:8].upper()}"

            owner_phone = store_phone.replace("+", "").replace(" ", "")
            send_whatsapp_message(to=owner_phone, text=owner_msg, phone_number_id=phone_number_id, token=token)

        _advance_flow_after_payment(order["project_id"], order["phone_number"], phone_number_id, token)

    return {"status": "ok"}


def _advance_flow_after_payment(project_id: str, phone: str, phone_number_id: str, token: str):
    try:
        from flows import get_next_node, send_node, upsert_session

        session_res = supabase.table("whatsapp_sessions").select("*").eq("project_id", project_id).eq("phone_number", phone).maybe_single().execute()
        if not session_res or not session_res.data:
            return

        session = session_res.data
        flow_id = session.get("flow_id")
        current_node_id = session.get("current_node_id")
        if not flow_id or not current_node_id:
            return

        next_node = get_next_node(flow_id, current_node_id, "next")
        if next_node:
            upsert_session(project_id, phone, {
                "flow_id": flow_id,
                "current_node_id": next_node["id"],
                "mode": "flow",
                "metadata": {},
            })
            send_node(next_node, phone, phone_number_id, token, project_id=project_id)
        else:
            upsert_session(project_id, phone, {
                "flow_id": flow_id,
                "current_node_id": current_node_id,
                "mode": "flow",
                "metadata": {},
            })
    except Exception as e:
        print(f"_advance_flow_after_payment error: {e}")


# ─────────────────────────────────────────────
# HELPER — called from flows.py
# ─────────────────────────────────────────────

def generate_razorpay_link(order: dict, config: dict) -> Optional[str]:
    try:
        import razorpay

        key_id = config.get("razorpay_key_id") or RAZORPAY_KEY_ID
        key_secret = config.get("razorpay_key_secret") or RAZORPAY_KEY_SECRET

        if not key_id or not key_secret:
            return None

        client = razorpay.Client(auth=(key_id, key_secret))
        link = client.payment_link.create({
            "amount": int(order["total"] * 100),
            "currency": "INR",
            "description": f"Order #{order['id'][:8].upper()}",
            "customer": {"contact": f"+{order['phone_number']}"},
            "notify": {"sms": False, "email": False},
            "reminder_enable": False,
            "expire_by": int(time.time()) + 5400,
        })

        supabase.table("orders").update({
            "payment_id": link["id"],
            "payment_status": "link_sent",
        }).eq("id", order["id"]).execute()

        return link["short_url"]

    except Exception as e:
        print(f"generate_razorpay_link error: {e}")
        return None