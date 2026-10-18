import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from clients import supabase
from webhook_dedup import already_processed
from auth import verify_token, require_project_access
from ratelimit import is_rate_limited, client_ip
from shopify_client import graphql as shopify_graphql
from config import RAZORPAY_WEBHOOK_SECRET, SUPABASE_URL
import phones
import shop_rules
import shop_identity
import math
import os
import uuid
from datetime import datetime, timedelta, timezone
import hmac
import hashlib
import json
import re
import time

router = APIRouter()

_UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

# The web shop checkout is public and unauthenticated by design, so these
# are the only thing bounding what one request can write or spend.
MAX_CART_LINES = 50
MAX_QUANTITY_PER_LINE = 100

# Written straight to the row and branched on elsewhere in this file, so a
# free string meant any value at all could be stored and then match nothing.
VALID_ORDER_STATUSES = set(shop_rules.ORDER_STATUSES)
# The dashboard used to write "delivered", which nothing else understood.
# Still accepted from old clients, stored as completed.
ORDER_STATUS_ALIASES = {"delivered": "completed"}
VALID_PAYMENT_STATUSES = {"unpaid", "link_sent", "paid", "underpaid", "refunded", "not_required"}

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://ragby-frontend.vercel.app")
# Legacy manual-key fallback — deprecated in favor of Razorpay Partner OAuth
# (razorpay_oauth.py) but kept alive so shops that haven't reconnected yet,
# and payment links created before cutover, keep working. See
# generate_razorpay_link() and _candidate_webhook_secrets() below.
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
    currency_code: Optional[str] = None
    accent_color: Optional[str] = None
    delivery_types: Optional[List[str]] = None
    terms_note: Optional[str] = None
    razorpay_key_id: Optional[str] = None
    razorpay_key_secret: Optional[str] = None
    is_enabled: Optional[bool] = None
    bot_can_assist: Optional[bool] = None
    bot_can_order: Optional[bool] = None

class CatalogCreate(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    is_active: bool = True
    branch_id: Optional[str] = Field(default=None, max_length=64)   # default branch when omitted

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
    """What the BROWSER sends. Only product_id and quantity are trusted.

    price, name, image_url and shopify_variant_id are all re-read from the
    products table in _price_cart_server_side below — they used to be taken
    verbatim, which let a shopper post price: 0.01 for any item and receive
    a Razorpay link for one rupee.
    """
    product_id: str = Field(max_length=64)
    quantity: int = Field(ge=1, le=MAX_QUANTITY_PER_LINE)
    # Accepted for backwards compatibility with the existing shop page and
    # widget payloads, then ignored.
    name: Optional[str] = Field(default=None, max_length=200)
    price: Optional[float] = None
    image_url: Optional[str] = Field(default=None, max_length=1000)
    shopify_variant_id: Optional[str] = Field(default=None, max_length=128)

class CartSubmit(BaseModel):
    """No prices, fees, totals, distance or open state: the server works
    all of those out (shop_branches.price_order)."""
    phone: str = Field(max_length=32)
    project_id: str = Field(max_length=64)
    branch_id: Optional[str] = Field(default=None, max_length=64)
    catalog_id: Optional[str] = Field(default=None, max_length=64)
    items: List[CartItem] = Field(max_length=MAX_CART_LINES)
    fulfilment: Optional[str] = Field(default="pickup", max_length=16)
    customer_name: Optional[str] = Field(default=None, max_length=400)
    delivery_address: Optional[str] = Field(default=None, max_length=2000)
    customer_lat: Optional[float] = None
    customer_lng: Optional[float] = None
    customer_pincode: Optional[Union[str, int]] = None
    # Legacy free text from old pages. Ignored: the label is derived from
    # fulfilment, or a shopper could pick "Home Delivery" and skip the fee.
    delivery_type: Optional[str] = Field(default=None, max_length=64)
    order_id: Optional[str] = Field(default=None, max_length=64)  # if present, UPDATE existing order

class OrderStatusUpdate(BaseModel):
    status: Optional[str] = Field(default=None, max_length=32)
    payment_status: Optional[str] = Field(default=None, max_length=32)
    notify: Optional[bool] = True   # WhatsApp the customer when the status changes


def shop_error(status: int, code: str, message: str) -> HTTPException:
    """Error shape for the shop endpoints. The frontend shows
    detail.message and branches on detail.code."""
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def load_shop_config(project_id: str) -> dict:
    """The full shop_config row (secrets included), for server use only.
    {} when there is none, or on a DB error (treated as defaults)."""
    try:
        res = supabase.table("shop_config").select("*").eq("project_id", project_id).maybe_single().execute()
        return (res.data if res else None) or {}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"shop_config fetch error: {e}")
        return {}


def _default_branch_catalog_ids(project_id: str) -> list:
    """The chat bot sells from the default branch only. After a menu copy
    every item exists once per branch, so a project-wide lookup would match
    each name twice and find_product_by_name would find nothing."""
    from shop_branches import get_default_branch, active_catalog_ids
    branch = get_default_branch(project_id)
    return active_catalog_ids(project_id, branch["id"]) if branch else []


# -------------------------------------------------
# AGENTIC ACTIONS — bot-can-assist (opt-in, see shop_config.bot_can_assist)
# Read-only: order lookup + catalog browsing only. See migration
# 20260716100000_shop_bot_assist.sql for why ordering itself isn't included.
# -------------------------------------------------
def get_shop_settings_if_assistable(project_id: str):
    res = supabase.table("shop_config").select("*").eq("project_id", project_id).maybe_single().execute()
    data = res.data if res else None
    return data if data and data.get("bot_can_assist") else None


def get_recent_orders_for_phone(project_id: str, phone: str, limit: int = 3) -> list:
    """Used by the in-chat 'check my order' tool — read-only, no writes."""
    clean_phone = phone.replace("+", "").replace(" ", "")
    res = supabase.table("orders") \
        .select("id, items, subtotal, gst_amount, total, status, payment_status, delivery_type, fulfilment, created_at") \
        .eq("project_id", project_id) \
        .eq("phone_number", clean_phone) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    orders = []
    for o in (res.data or []):
        orders.append({
            "order_id": f"#{o['id'][:8].upper()}",
            "items": [f"{it['name']} x{it['quantity']}" for it in (o.get("items") or [])],
            "total": o["total"],
            "status": o["status"],
            "payment_status": o["payment_status"],
            "delivery_type": o.get("delivery_type"),
            "fulfilment": o.get("fulfilment"),
            "placed_at": o["created_at"],
        })
    return orders


def get_active_catalog_summary(project_id: str) -> list:
    """Used by the in-chat 'what do you sell' tool — read-only, no writes."""
    catalog_ids = _default_branch_catalog_ids(project_id)
    if not catalog_ids:
        return []
    catalogs = supabase.table("catalogs").select("id, name").eq("project_id", project_id).in_("id", catalog_ids).execute()
    catalog_map = {c["id"]: c["name"] for c in (catalogs.data or [])}
    if not catalog_map:
        return []

    products = supabase.table("products") \
        .select("catalog_id, name, description, price") \
        .eq("project_id", project_id) \
        .eq("is_available", True) \
        .in_("catalog_id", list(catalog_map.keys())) \
        .order("sort_order", desc=False) \
        .execute()

    result = []
    for p in (products.data or []):
        result.append({
            "catalog": catalog_map.get(p["catalog_id"], "Menu"),
            "name": p["name"],
            "price": p["price"],
            "description": p.get("description") or "",
        })
    return result


# ─────────────────────────────────────────────
# SHOP CONFIG
# ─────────────────────────────────────────────

# FIX: shop_config.razorpay_key_secret was previously returned to the
# browser on every GET/PUT via select("*") — a real, live merchant secret
# key shipped into the dashboard's JS/network tab on every settings-page
# load. Explicit column list, excluding it, used by both handlers below.
# razorpay_key_id is kept (not sensitive on its own); razorpay_key_secret
# is now write-only from the frontend's perspective.
_SHOP_CONFIG_SAFE_COLUMNS = (
    "project_id, store_name, store_phone, gst_percent, currency, currency_code, "
    "accent_color, delivery_types, terms_note, razorpay_key_id, is_enabled, "
    "bot_can_assist, bot_can_order, cover_image_url, logo_url"
)


@router.get("/shop-config/{project_id}")
async def get_shop_config(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="shop")
    res = supabase.table("shop_config").select(_SHOP_CONFIG_SAFE_COLUMNS).eq("project_id", project_id).maybe_single().execute()
    if not res or not res.data:
        return {
            "project_id": project_id,
            "store_name": "",
            "store_phone": "",
            "gst_percent": 0,
            "currency": "₹",
            "currency_code": "INR",
            "accent_color": "#16a34a",
            "delivery_types": ["Takeaway"],
            "terms_note": "This order is not eligible for any kind of Discounts. T&C apply.",
            "razorpay_key_id": "",
            "is_enabled": False,
            "bot_can_assist": False,
            "bot_can_order": False,
            "cover_image_url": None,
            "logo_url": None,
        }
    return res.data

@router.put("/shop-config/{project_id}")
async def update_shop_config(project_id: str, body: ShopConfigUpdate, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="shop")
    existing = supabase.table("shop_config").select("id").eq("project_id", project_id).maybe_single().execute()
    update = {k: v for k, v in body.dict().items() if v is not None}
    if existing and existing.data:
        supabase.table("shop_config").update(update).eq("project_id", project_id).execute()
    else:
        supabase.table("shop_config").insert({"project_id": project_id, **update}).execute()
    res = supabase.table("shop_config").select(_SHOP_CONFIG_SAFE_COLUMNS).eq("project_id", project_id).single().execute()
    return res.data


class BrandingUpdate(BaseModel):
    cover_image_url: Optional[str] = Field(default=None, max_length=2000)
    logo_url: Optional[str] = Field(default=None, max_length=2000)


@router.put("/shop-config/{project_id}/branding")
def update_shop_branding(project_id: str, body: BrandingUpdate, user=Depends(verify_token)):
    """Cover image and logo. Only URLs of this project's own uploads (the
    flow-media bucket, shop-branding/products folder, png/jpg/webp) are
    accepted, so the public page never loads an arbitrary third-party URL."""
    if not _UUID_RE.match(project_id or ""):
        raise shop_error(400, "invalid_request", "Invalid project id")
    require_project_access(user.id, project_id, tab="shop", min_role="admin")
    data = body.model_dump(exclude_unset=True)
    fields = {}
    for key in ("cover_image_url", "logo_url"):
        if key not in data:
            continue
        url = data[key]
        if url is None or not str(url).strip():
            fields[key] = None
        elif shop_rules.valid_media_url(str(url).strip(), SUPABASE_URL):
            fields[key] = str(url).strip()
        else:
            raise shop_error(400, "invalid_image_url", "Upload the image from the dashboard (PNG, JPG or WEBP).")
    if fields:
        existing = supabase.table("shop_config").select("id").eq("project_id", project_id).maybe_single().execute()
        if existing and existing.data:
            supabase.table("shop_config").update(fields).eq("project_id", project_id).execute()
        else:
            supabase.table("shop_config").insert({"project_id": project_id, **fields}).execute()
    res = supabase.table("shop_config").select("cover_image_url, logo_url").eq("project_id", project_id).maybe_single().execute()
    row = (res.data if res else None) or {}
    return {"cover_image_url": row.get("cover_image_url"), "logo_url": row.get("logo_url")}


# ─────────────────────────────────────────────
# CATALOGS
# ─────────────────────────────────────────────

@router.get("/catalogs")
async def list_catalogs(project_id: str, branch_id: Optional[str] = None, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="shop")
    query = supabase.table("catalogs").select("*").eq("project_id", project_id)
    if branch_id:
        if not _UUID_RE.match(branch_id):
            raise shop_error(400, "invalid_request", "Invalid branch id")
        query = query.eq("branch_id", branch_id)
    res = query.order("created_at", desc=False).execute()
    return res.data or []

@router.post("/catalogs")
def create_catalog(body: CatalogCreate, user=Depends(verify_token)):
    from shop_branches import resolve_branch
    require_project_access(user.id, body.project_id, tab="shop")
    # Must be this project's branch; the default one when not given.
    branch = resolve_branch(body.project_id, body.branch_id)
    inserted = supabase.table("catalogs").insert({
        "project_id": body.project_id,
        "branch_id": branch["id"],
        "name": body.name,
        "description": body.description,
        "is_active": body.is_active,
    }).execute()
    if not inserted.data:
        raise shop_error(500, "invalid_request", "Could not create the catalog. Please try again.")
    return inserted.data[0]

def _require_role_for_catalog(user_id: str, catalog_id: str, min_role: str = None):
    res = supabase.table("catalogs").select("project_id").eq("id", catalog_id).maybe_single().execute()
    catalog = res.data if res else None
    if not catalog:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user_id, catalog["project_id"], tab="shop", min_role=min_role)

@router.put("/catalogs/{catalog_id}")
async def update_catalog(catalog_id: str, body: CatalogUpdate, user=Depends(verify_token)):
    _require_role_for_catalog(user.id, catalog_id)
    update = {k: v for k, v in body.dict().items() if v is not None}
    supabase.table("catalogs").update(update).eq("id", catalog_id).execute()
    res = supabase.table("catalogs").select("*").eq("id", catalog_id).single().execute()
    return res.data

@router.delete("/catalogs/{catalog_id}")
async def delete_catalog(catalog_id: str, user=Depends(verify_token)):
    _require_role_for_catalog(user.id, catalog_id, min_role="admin")
    supabase.table("catalogs").delete().eq("id", catalog_id).execute()
    return {"status": "deleted"}


# ─────────────────────────────────────────────
# PRODUCTS
# ─────────────────────────────────────────────

@router.get("/products")
async def list_products(project_id: str, catalog_id: Optional[str] = None, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="shop")
    query = supabase.table("products").select("*").eq("project_id", project_id)
    if catalog_id:
        query = query.eq("catalog_id", catalog_id)
    res = query.order("sort_order", desc=False).execute()
    return res.data or []

@router.post("/products")
async def create_product(body: ProductCreate, user=Depends(verify_token)):
    require_project_access(user.id, body.project_id, tab="shop")
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

def _require_role_for_product(user_id: str, product_id: str, min_role: str = None):
    res = supabase.table("products").select("project_id").eq("id", product_id).maybe_single().execute()
    product = res.data if res else None
    if not product:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user_id, product["project_id"], tab="shop", min_role=min_role)

@router.put("/products/{product_id}")
async def update_product(product_id: str, body: ProductUpdate, user=Depends(verify_token)):
    _require_role_for_product(user.id, product_id)
    update = {k: v for k, v in body.dict().items() if v is not None}
    supabase.table("products").update(update).eq("id", product_id).execute()
    res = supabase.table("products").select("*").eq("id", product_id).single().execute()
    return res.data

@router.delete("/products/{product_id}")
async def delete_product(product_id: str, user=Depends(verify_token)):
    _require_role_for_product(user.id, product_id, min_role="admin")
    supabase.table("products").delete().eq("id", product_id).execute()
    return {"status": "deleted"}


# ─────────────────────────────────────────────
# PUBLIC — Shop page APIs (no auth needed)
# ─────────────────────────────────────────────

def _public_shop_guard(project_id: str, request: Request, bucket: str, limit: int = 60):
    """Shared preamble for the unauthenticated shop endpoints.

    None of these were rate limited, and project_id went straight into a
    uuid column so a non-UUID surfaced as a 500 with driver detail.
    """
    if not _UUID_RE.match(project_id or ""):
        raise shop_error(400, "invalid_request", "Invalid project id")
    if is_rate_limited(f"{bucket}:{project_id}:{client_ip(request)}", limit=limit, window_seconds=60):
        raise shop_error(429, "rate_limited", "Too many requests — please wait a moment.")


def _assert_shop_enabled(project_id: str):
    """A merchant who switches their shop off should stop being a shop.

    is_enabled was read and returned but never enforced, so a disabled shop
    still served its whole catalogue publicly and still took orders. Mirrors
    appointments.py's public_settings, which already does this.
    """
    res = supabase.table("shop_config").select("is_enabled").eq("project_id", project_id).maybe_single().execute()
    data = res.data if res else None
    # No config row at all is the pre-setup default, which public_shop_config
    # already treats as enabled — keep the two consistent.
    if data and data.get("is_enabled") is False:
        raise shop_error(403, "shop_disabled", "This shop isn't available right now.")


def _public_branch_summary(project_id: str, config: dict) -> dict:
    """How many active branches the page has to choose from. 404/403 for
    an unknown project or one with no shop (see public_default_branch);
    otherwise best-effort: on a DB error the page behaves like a
    single-branch shop."""
    from shop_branches import public_default_branch
    default_id = public_default_branch(project_id, config)["id"]
    try:
        res = supabase.table("shop_branches").select("id").eq("project_id", project_id).eq("is_active", True).execute()
        return {"branch_count": len(res.data or []), "default_branch_id": default_id}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {"branch_count": 1, "default_branch_id": default_id}


@router.get("/public/shop/{project_id}/config")
def public_shop_config(project_id: str, request: Request):
    _public_shop_guard(project_id, request, "shop-config")
    res = supabase.table("shop_config").select(
        "store_name,store_phone,gst_percent,currency,accent_color,delivery_types,terms_note,is_enabled,"
        "cover_image_url,logo_url"
    ).eq("project_id", project_id).maybe_single().execute()
    if res and res.data and res.data.get("is_enabled") is False:
        raise shop_error(403, "shop_disabled", "This shop isn't available right now.")
    if not res or not res.data:
        data = {
            "gst_percent": 0,
            "currency": "₹",
            "accent_color": "#16a34a",
            "delivery_types": ["Takeaway"],
            "terms_note": "",
            "is_enabled": True,
            "cover_image_url": None,
            "logo_url": None,
        }
    else:
        data = dict(res.data)
    data["online_payment"] = online_payments_available(project_id, load_shop_config(project_id))
    return {**data, **_public_branch_summary(project_id, (res.data if res else None) or {})}

@router.get("/public/shop/{project_id}/catalogs")
def public_catalogs(project_id: str, request: Request, branch_id: Optional[str] = None):
    from shop_branches import resolve_branch
    _public_shop_guard(project_id, request, "shop-catalogs")
    _assert_shop_enabled(project_id)
    branch = resolve_branch(project_id, branch_id, public=True)
    # Named columns rather than select("*") — this is an anonymous read.
    res = supabase.table("catalogs").select("id,name,description").eq("project_id", project_id) \
        .eq("branch_id", branch["id"]).eq("is_active", True).order("created_at", desc=False).execute()
    return res.data or []

@router.get("/public/shop/{project_id}/products")
def public_products(project_id: str, request: Request, branch_id: Optional[str] = None, catalog_id: Optional[str] = None):
    from shop_branches import resolve_branch, active_catalog_ids
    _public_shop_guard(project_id, request, "shop-products")
    _assert_shop_enabled(project_id)
    if catalog_id and not _UUID_RE.match(catalog_id):
        raise shop_error(400, "invalid_request", "Invalid catalog id")
    if catalog_id and not branch_id:
        # Old links carry only ?catalog=, so the branch comes from the catalog.
        cat = supabase.table("catalogs").select("branch_id").eq("id", catalog_id).eq("project_id", project_id).limit(1).execute()
        if not cat.data:
            return []
        branch_id = cat.data[0].get("branch_id")
    branch = resolve_branch(project_id, branch_id, public=True)
    # Only the branch's ACTIVE catalogs: products of a switched-off catalog
    # used to stay publicly listed.
    catalog_ids = active_catalog_ids(project_id, branch["id"])
    if catalog_id:
        catalog_ids = [c for c in catalog_ids if c == catalog_id]
    if not catalog_ids:
        return []
    # select("*") shipped every column to anonymous callers, including
    # internal fields like shopify_variant_id.
    res = supabase.table("products").select(
        "id,catalog_id,name,description,price,image_url,category,gst_percent,sort_order"
    ).eq("project_id", project_id).eq("is_available", True).in_("catalog_id", catalog_ids) \
        .order("sort_order", desc=False).execute()
    return res.data or []

@router.get("/public/shop/order/{order_id}")
def public_get_order(order_id: str, request: Request, phone: str = ""):
    """Fetch an existing order's items — used to pre-populate cart for 'Add More' flow."""
    # Keyed by IP alone (not order_id) — the point is slowing down someone
    # probing many DIFFERENT order ids, not just repeated hits on one.
    ip = client_ip(request)
    if is_rate_limited(f"order-lookup:{ip}", limit=20, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")

    # Requires the phone that owns the order. This returned any order's
    # contents to anyone holding its UUID, with no scoping at all — the IP
    # rate limit slowed enumeration but didn't restrict who could read what.
    # submit_cart's update path already checks ownership the same way.
    clean_phone = (phone or "").replace("+", "").replace(" ", "").replace("-", "")
    if not clean_phone:
        raise HTTPException(status_code=400, detail="phone is required")

    if not _UUID_RE.match(order_id or ""):
        return {"items": []}
    res = supabase.table("orders").select(
        "items, delivery_type, branch_id, fulfilment, customer_name, delivery_address, "
        "customer_pincode, customer_lat, customer_lng, status, payment_status, "
        "address_line1, address_line2, address_city, sold_out_preference, special_request"
    ).eq("id", order_id).eq("phone_number", clean_phone).maybe_single().execute()
    if not res or not res.data:
        return {"items": []}
    o = res.data
    return {
        "items": o.get("items", []),
        "delivery_type": o.get("delivery_type", "Takeaway"),
        "branch_id": o.get("branch_id"),
        "fulfilment": o.get("fulfilment") or "pickup",
        "customer_name": o.get("customer_name"),
        "delivery_address": o.get("delivery_address"),
        "customer_pincode": o.get("customer_pincode"),
        "customer_lat": o.get("customer_lat"),
        "customer_lng": o.get("customer_lng"),
        "address_line1": o.get("address_line1"),
        "address_line2": o.get("address_line2"),
        "address_city": o.get("address_city"),
        "sold_out_preference": o.get("sold_out_preference"),
        "note": o.get("special_request"),
        # Matches submit_cart's edit rule, so the page can tell the shopper
        # up front instead of failing at checkout.
        "editable": o.get("status") == "pending" and o.get("payment_status") == "unpaid",
    }


# ─────────────────────────────────────────────
# CHECKOUT MODE, OFFER LIFECYCLE, PAYMENT LINK CANCEL (v3)
# ─────────────────────────────────────────────

# Written on every edit path that can't carry an offer (legacy submit-cart).
NO_OFFER_COLUMNS = {
    "offer_id": None, "offer_code": None, "offer_title": None, "discount_type": None,
    "items_discount": 0, "delivery_discount": 0, "discount_amount": 0,
}


def online_payments_available(project_id: str, config: dict) -> bool:
    """A Razorpay OAuth connection, or both legacy keys. `config` must be
    the full row (load_shop_config), since the secret is never selected
    for the browser."""
    config = config or {}
    if config.get("razorpay_key_id") and config.get("razorpay_key_secret"):
        return True
    try:
        res = supabase.table("razorpay_connections").select("project_id").eq("project_id", project_id).limit(1).execute()
        return bool(res.data)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return False


def checkout_mode(project_id: str, config: dict, verified: bool) -> str:
    """pay_online | place_offline | confirm_in_whatsapp. Placing without
    payment alerts the owner, so it needs a verified customer token."""
    if online_payments_available(project_id, config):
        return "pay_online"
    if verified:
        return "place_offline"
    return "confirm_in_whatsapp"


def _offer_rpc(name: str, order_id: str):
    try:
        supabase.rpc(name, {"p_order_id": order_id}).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)


def confirm_offer(order_id: str):
    """held -> confirmed (payment taken or order placed). Best-effort."""
    _offer_rpc("shop_confirm_offer", order_id)


def release_offer(order_id: str):
    """Frees the order's offer redemption (cancel, offer removed). Best-effort."""
    _offer_rpc("shop_release_offer", order_id)


def claim_offer(order_id: str, offer_id: str, phone: str, verified: bool, hold_until: Optional[str],
                discount: float) -> dict:
    """The authoritative claim (DB function shop_claim_offer). {"ok": bool, "reason"}."""
    try:
        res = supabase.rpc("shop_claim_offer", {
            "p_order_id": order_id, "p_offer_id": offer_id, "p_phone": phone, "p_verified": bool(verified),
            "p_hold_until": hold_until, "p_discount": round(float(discount or 0), 2),
        }).execute()
        data = res.data
        if isinstance(data, list):
            data = data[0] if data else None
        if isinstance(data, dict) and "ok" in data:
            return data
        return {"ok": False, "reason": "offer_unavailable"}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {"ok": False, "reason": "offer_unavailable"}


def cancel_payment_link(project_id: str, link_id: Optional[str], config: Optional[dict] = None) -> bool:
    """Best-effort POST /payment_links/{id}/cancel, so a cancelled order's
    link can't be paid. OAuth first, else the merchant's legacy keys."""
    if not link_id:
        return False
    try:
        from razorpay_oauth import _razorpay_api_request
        try:
            res = _razorpay_api_request("POST", f"/payment_links/{link_id}/cancel", project_id)
            return res.status_code < 300
        except ValueError:
            pass   # no OAuth connection: legacy keys below
        config = config if config is not None else load_shop_config(project_id)
        key_id, key_secret = config.get("razorpay_key_id"), config.get("razorpay_key_secret")
        if not key_id or not key_secret:
            return False
        import razorpay
        razorpay.Client(auth=(key_id, key_secret)).payment_link.cancel(link_id)
        return True
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return False


def order_status_label(order: dict) -> str:
    status = order.get("status")
    pickup = (order.get("fulfilment") or "pickup") != "delivery"
    if status == "pending" and order.get("payment_status") == "link_sent":
        return "Awaiting payment"
    return {
        "pending": "Placed",
        "confirmed": "Confirmed",
        "preparing": "Being prepared",
        "ready": "Ready for pickup" if pickup else "Ready",
        "out_for_delivery": "Out for delivery",
        "completed": "Picked up" if pickup else "Delivered",
        "cancelled": "Cancelled",
    }.get(status, str(status or "").replace("_", " ").capitalize())


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def live_pay_link(order: dict, margin_seconds: int = 0):
    """(url, expires_at) while the order's link can still be paid, else (None, None)."""
    if order.get("status") != "pending" or order.get("payment_status") != "link_sent":
        return None, None
    url = order.get("payment_link_url")
    expires = shop_rules.parse_ts(order.get("payment_link_expires_at"))
    if not url or not expires:
        return None, None
    if datetime.now(timezone.utc) >= expires - timedelta(seconds=margin_seconds):
        return None, None
    return url, _iso_z(expires)


# ─────────────────────────────────────────────
# CART SUBMIT — called from web shop page
# ─────────────────────────────────────────────

def _price_items(project_id: str, branch_id: str, catalog_id: Optional[str], items: List[CartItem]):
    """Price the cart from the products table, collecting problems instead
    of raising (see _price_cart_server_side for why prices are never taken
    from the browser).

    A product must be available and in one of the branch's ACTIVE catalogs
    (and in catalog_id, when given). Returns (items_data, subtotal,
    price_changed, error_codes); items with a problem are left out.
    """
    wanted = {}
    sent_price = {}
    errors = []
    for item in items:
        pid = str(item.product_id or "")
        if not _UUID_RE.match(pid):
            # Not even an id: it can't be on any menu, and querying a uuid
            # column with it would be a driver error.
            errors.append("item_unavailable")
            continue
        # Two lines for the same product are merged rather than priced twice.
        wanted[pid] = wanted.get(pid, 0) + int(item.quantity)
        if item.price is not None:
            sent_price[pid] = float(item.price)
    if not wanted:
        return [], 0.0, False, errors

    cats = supabase.table("catalogs").select("id") \
        .eq("project_id", project_id).eq("branch_id", branch_id).eq("is_active", True).execute()
    on_menu = {c["id"] for c in (cats.data or [])}

    res = supabase.table("products") \
        .select("id, name, price, image_url, shopify_variant_id, is_available, catalog_id") \
        .eq("project_id", project_id) \
        .in_("id", list(wanted.keys())) \
        .execute()
    by_id = {row["id"]: row for row in (res.data or [])}

    items_data = []
    subtotal = 0.0
    price_changed = False
    for pid, qty in wanted.items():
        product = by_id.get(pid)
        if not product or not product.get("is_available"):
            errors.append("item_unavailable")
            continue
        if product.get("catalog_id") not in on_menu or (catalog_id and product.get("catalog_id") != catalog_id):
            errors.append("item_not_on_menu")
            continue
        real_price = float(product.get("price") or 0)
        if pid in sent_price and abs(sent_price[pid] - real_price) > 0.001:
            price_changed = True
        quantity = max(1, min(qty, MAX_QUANTITY_PER_LINE))
        subtotal += real_price * quantity
        items_data.append({
            "product_id": product["id"],
            "name": product.get("name") or "Item",
            "price": real_price,
            "quantity": quantity,
            "image_url": product.get("image_url"),
            "shopify_variant_id": product.get("shopify_variant_id"),
        })
    return items_data, round(subtotal, 2), price_changed, errors


def _price_cart_server_side(project_id: str, branch_id: str, catalog_id: Optional[str], items: List[CartItem]):
    """Rebuild the cart from the products table, ignoring the browser.

    THE fix for this file. submit_cart used to compute the order total from
    CartItem.price — a float posted by the customer's browser — and that
    total became the Razorpay charge amount in generate_razorpay_link. So a
    shopper could post price: 0.01 for a five-thousand-rupee item, receive a
    payment link for one rupee, pay it, and the merchant would see a fully
    paid order. The endpoint is public by design; there was nothing else
    standing between a customer and their own pricing.

    Returns (items_data, subtotal, price_changed). price_changed is True
    when the stored price differs from what the browser sent, so the
    confirmation can say prices were updated rather than silently charging
    something the shopper never saw.
    """
    if not items:
        raise shop_error(400, "cart_empty", "Your cart is empty.")
    # Scoped to this project AND this branch's active menu: a product id from
    # another project, branch or catalog is refused rather than silently priced.
    items_data, subtotal, price_changed, errors = _price_items(project_id, branch_id, catalog_id, items)
    if "item_unavailable" in errors:
        raise shop_error(400, "item_unavailable", "One of the items in your cart is no longer available. Please refresh the menu.")
    if "item_not_on_menu" in errors:
        raise shop_error(400, "item_not_on_menu", "One of the items in your cart isn't on this menu. Please refresh and try again.")
    if not items_data:
        raise shop_error(400, "cart_empty", "Your cart is empty.")
    return items_data, subtotal, price_changed


def one_line(value) -> str:
    """Free text squeezed onto one line for a WhatsApp message, so a typed
    newline can't fake extra lines (a "Total", a "PAID") under it."""
    return " ".join(str(value).split()) if value is not None else ""


def _where(order: dict) -> str:
    return ", ".join(one_line(x) for x in (order.get("delivery_address"), order.get("customer_pincode")) if x)


def order_fulfilment_lines(order: dict, branch: Optional[dict], currency: str = "₹") -> str:
    """The branch + pickup/delivery block shared by the cart summary, the
    pre-payment summary and the payment confirmation."""
    branch = branch or {}
    lines = []
    if branch.get("name"):
        lines.append(f"🏪 {branch['name']}")
    if order.get("fulfilment") == "delivery":
        where = _where(order)
        lines.append(f"🛵 Delivery to: {where}" if where else "🛵 Delivery")
        if order.get("delivery_distance_km") is not None:
            lines.append(f"📏 {float(order['delivery_distance_km']):g} km")
    else:
        prep = branch.get("prep_minutes")
        lines.append("🏃 Pickup" + (f" · ready in ~{prep} min" if prep else ""))
        if branch.get("address"):
            lines.append(f"📍 {branch['address']}")
    return "\n".join(lines)


def _send_cart_confirmation(project_id: str, phone: str, order: dict, items_data: list, currency: str, subtotal: float, gst_amount: float, total: float, catalog_id: Optional[str], price_changed: bool = False, branch: Optional[dict] = None) -> dict:
    """Shared by the web-shop-page checkout AND the in-chat ordering tool —
    single source of truth for the cart summary text + Continue/Add
    More/Clear Cart buttons + session handoff, so both paths land the
    customer in the exact same, already-proven confirm/payment flow."""
    from whatsapp import send_whatsapp_buttons
    from config import WHATSAPP_TOKEN

    lines = []
    for i, item in enumerate(items_data, 1):
        lines.append(f"{i}. {item['name']} x{item['quantity']} - {currency}{int(item['price'] * item['quantity'])}")
    items_text = "\n".join(lines)
    header = f"🛒 *Your Cart* — {branch['name']}" if branch and branch.get("name") else "🛒 *Your Cart*"
    summary = f"{header}\n\n{items_text}\n\nSubtotal: {currency}{shop_rules.money(subtotal)}"
    if gst_amount > 0:
        summary += f"\nGST: {currency}{gst_amount}"
    is_delivery = order.get("fulfilment") == "delivery"
    if is_delivery:
        fee = float(order.get("delivery_fee") or 0)
        summary += f"\nDelivery: {currency}{shop_rules.money(fee)}" if fee > 0 else "\nDelivery: FREE"
    summary += f"\n*Total: {currency}{total}*"
    if is_delivery:
        summary += f"\n\n🛵 Delivery to {one_line(order.get('delivery_address')) or 'your address'}"
    else:
        summary += "\n\n🏃 Pickup"
    # Prices are read from the products table at checkout, so a merchant who
    # edited one mid-session would otherwise silently charge a total the
    # shopper never saw. Say so instead of quietly changing it.
    if price_changed:
        summary += "\n\n_Some prices were updated since you added these items — the total above is current._"

    try:
        wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", project_id).maybe_single().execute()
        wa_data = (wa_res.data if wa_res else None)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"whatsapp_integrations fetch error: {e}")
        wa_data = None

    # branch_id rides along so every later link (Add More, Change Order)
    # reopens the same branch.
    metadata = {"order_id": order["id"], "catalog_id": catalog_id, "branch_id": order.get("branch_id")}

    if not wa_data:
        supabase.table("whatsapp_sessions").upsert({
            "project_id": project_id,
            "phone_number": phone,
            "mode": "awaiting_cart_confirm",
            "metadata": metadata,
        }, on_conflict="project_id,phone_number").execute()
        return {"status": "ok", "order_id": order["id"], "warning": "WhatsApp not connected", "summary": summary}

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
        "metadata": metadata,
    }, on_conflict="project_id,phone_number").execute()

    return {"status": "ok", "order_id": order["id"], "summary": summary}


def find_product_by_name(project_id: str, name: str) -> Optional[dict]:
    """Case-insensitive, substring-tolerant match against real catalog
    products — used by the in-chat ordering tool. Never trusts a product
    name/price supplied by the model as final; this looks up the real row.
    Returns None if there's no match OR more than one equally-good match
    (ambiguous — caller should ask the customer to clarify rather than guess)."""
    catalog_ids = _default_branch_catalog_ids(project_id)
    if not catalog_ids:
        return None
    res = supabase.table("products").select("*").eq("project_id", project_id).eq("is_available", True) \
        .in_("catalog_id", catalog_ids).execute()
    products = res.data or []
    name_lower = name.strip().lower()

    exact = [p for p in products if p["name"].strip().lower() == name_lower]
    if len(exact) == 1:
        return exact[0]

    partial = [p for p in products if name_lower in p["name"].lower() or p["name"].lower() in name_lower]
    if len(partial) == 1:
        return partial[0]

    return None


def create_order_from_chat(project_id: str, phone: str, requested_items: list, delivery_type: Optional[str] = None) -> dict:
    """
    In-chat equivalent of submit_cart — builds a fresh order from items the
    AI understood in conversation, then hands off to the SAME
    confirm/payment flow the web shop page already uses. Every item is
    re-resolved against the real product catalog here — the model only
    supplies a product name + quantity, never a price.

    Always the default branch, always pickup (delivery needs a map pin, so
    it goes through the menu page). delivery_type is accepted for the tool
    signature and ignored.
    """
    from shop_branches import get_default_branch, status_for

    branch = get_default_branch(project_id)
    if not branch:
        raise ValueError("The shop isn't set up for ordering yet.")
    config = load_shop_config(project_id)
    status = status_for(branch, config)
    if not status["can_order"]:
        raise ValueError(status["label"])
    if not branch.get("pickup_enabled", True):
        raise ValueError("Please order using the menu link")

    unmatched = []
    items_data = []
    catalog_id = None

    for req in requested_items:
        product = find_product_by_name(project_id, req["product_name"])
        if not product:
            unmatched.append(req["product_name"])
            continue
        try:
            qty = int(req.get("quantity", 1))
        except (TypeError, ValueError):
            qty = 1
        qty = max(1, min(qty, MAX_QUANTITY_PER_LINE))
        items_data.append({
            "product_id": product["id"],
            "name": product["name"],
            "price": product["price"],
            "quantity": qty,
            "image_url": product.get("image_url"),
            "shopify_variant_id": product.get("shopify_variant_id"),
        })
        catalog_id = catalog_id or product.get("catalog_id")

    if unmatched:
        raise ValueError(f"Couldn't find these items in the menu: {', '.join(unmatched)}. Ask the customer to confirm the exact item name.")
    if not items_data:
        raise ValueError("No valid items to order.")

    gst_percent = config.get("gst_percent", 0) or 0
    currency = config.get("currency", "₹")

    subtotal = round(sum(it["price"] * it["quantity"] for it in items_data), 2)
    min_order = float(branch.get("min_order_value") or 0)
    if subtotal < min_order:
        raise ValueError(f"Minimum order is {currency}{shop_rules.money(min_order)} — "
                         f"add {currency}{shop_rules.money(min_order - subtotal)} more.")
    gst_amount = round(subtotal * gst_percent / 100, 2)
    total = round(subtotal + gst_amount, 2)

    inserted = supabase.table("orders").insert({
        "project_id": project_id,
        "branch_id": branch["id"],
        "phone_number": phone,
        "items": items_data,
        "subtotal": subtotal,
        "gst_amount": gst_amount,
        "total": total,
        "status": "pending",
        "payment_status": "unpaid",
        "fulfilment": "pickup",
        "delivery_type": "Pickup",
    }).execute()

    # Same fix as the web checkout: use the insert's own row rather than
    # re-reading "newest order for this phone", which two concurrent
    # orders from one number could resolve to each other's.
    if not inserted.data:
        raise ValueError("Could not create the order. Please try again.")
    order = inserted.data[0]

    result = _send_cart_confirmation(project_id, phone, order, items_data, currency, subtotal, gst_amount, total, catalog_id, branch=branch)
    result["items"] = [f"{it['name']} x{it['quantity']}" for it in items_data]
    result["total"] = total
    result["currency"] = currency
    return result


@router.post("/public/shop/submit-cart")
def submit_cart(body: CartSubmit, request: Request):
    # Plain def: it is all blocking DB and WhatsApp calls.
    from shop_branches import resolve_branch, price_order, raise_first_error, parse_customer_location, shop_is_enabled

    project_id = body.project_id
    phone = body.phone.replace("+", "").replace(" ", "")

    if not _UUID_RE.match(project_id or ""):
        raise shop_error(400, "invalid_request", "Invalid project id")

    ip = client_ip(request)
    if is_rate_limited(f"cart:{project_id}:{ip}", limit=5):
        raise shop_error(429, "rate_limited", "Too many attempts — please wait a moment and try again.")

    config = load_shop_config(project_id)
    # A merchant who switches their shop off should not still be taking
    # orders through it.
    if not shop_is_enabled(config):
        raise shop_error(403, "shop_disabled", "This shop isn't taking orders right now.")

    fulfilment = body.fulfilment or "pickup"
    if fulfilment not in ("pickup", "delivery"):
        raise shop_error(400, "invalid_request", "Choose pickup or delivery.")
    catalog_id = body.catalog_id or None     # old links send ?catalog= empty
    if catalog_id and not _UUID_RE.match(catalog_id):
        raise shop_error(400, "invalid_request", "Invalid catalog id")
    if body.branch_id and not _UUID_RE.match(body.branch_id):
        raise shop_error(400, "invalid_request", "Invalid branch id")

    # ── Existing order (Add More / Change Order) ──
    existing = None
    branch_id = body.branch_id
    if body.order_id:
        # FIX: this endpoint is public/unauthenticated by design (it's the
        # web shop checkout) — previously it updated ANY order by id with
        # no check that it actually belongs to this project_id/phone,
        # letting anyone who learned an order UUID overwrite its items/
        # price and (via _send_cart_confirmation below) trigger a WhatsApp
        # message from a project they don't own to an arbitrary phone.
        if not _UUID_RE.match(body.order_id):
            raise shop_error(404, "order_not_found", "Order not found")
        owned_order = supabase.table("orders").select("*") \
            .eq("id", body.order_id).eq("project_id", project_id).eq("phone_number", phone) \
            .maybe_single().execute()
        existing = owned_order.data if owned_order else None
        if not existing:
            raise shop_error(404, "order_not_found", "Order not found")
        # A stale "Add More" link used to be able to change an order after
        # its payment link went out (it then read as underpaid).
        if existing.get("status") != "pending" or existing.get("payment_status") != "unpaid":
            raise shop_error(409, "order_not_editable", "This order is already being processed. Please place a new order.")
        if body.branch_id and existing.get("branch_id") and body.branch_id != existing["branch_id"]:
            raise shop_error(400, "invalid_request", "An order can't move to another branch. Please place a new order.")
        branch_id = existing.get("branch_id") or body.branch_id

    if not branch_id and catalog_id:
        cat = supabase.table("catalogs").select("branch_id").eq("id", catalog_id).eq("project_id", project_id).limit(1).execute()
        if not cat.data:
            raise shop_error(400, "invalid_request", "This menu isn't available any more. Please refresh.")
        branch_id = cat.data[0].get("branch_id")
    branch = resolve_branch(project_id, branch_id, public=True)
    if catalog_id:
        cat = supabase.table("catalogs").select("id").eq("id", catalog_id).eq("project_id", project_id) \
            .eq("branch_id", branch["id"]).limit(1).execute()
        if not cat.data:
            raise shop_error(400, "invalid_request", "This menu isn't part of that branch. Please refresh.")

    # ── Customer details ──
    try:
        # One line: the name is shown inside the owner alert and the cart
        # summary, so a newline could fake extra lines (e.g. a "Total").
        customer_name = shop_rules.clean_text(" ".join(str(body.customer_name).split()) if body.customer_name else None, 80)
    except ValueError:
        raise shop_error(400, "invalid_request", "Name must be at most 80 characters.")
    address = lat = lng = pincode = None
    if fulfilment == "delivery":
        if not customer_name:
            raise shop_error(400, "invalid_request", "Please enter your name for delivery.")
        try:
            address = shop_rules.clean_text(body.delivery_address, 300)
        except ValueError:
            raise shop_error(400, "invalid_request", "Address must be at most 300 characters.")
        if not address or len(address) < 10:
            raise shop_error(400, "invalid_request", "Please enter your full delivery address (house / flat, street, landmark).")
        lat, lng, pincode = parse_customer_location(
            body.customer_lat, body.customer_lng, body.customer_pincode,
            phones.default_region_for_project(project_id))
        if lat is None or lng is None:
            raise shop_error(400, "location_required", "Set your delivery location on the map.")
        if not pincode:
            raise shop_error(400, "invalid_request", "Please enter your pincode.")

    # Prices, fee, open state: all from the server. See price_order.
    priced = price_order(project_id, branch, body.items, fulfilment, lat, lng, pincode,
                         catalog_id=catalog_id, config=config)
    raise_first_error(priced)
    items_data = priced["items_data"]
    subtotal, gst_amount, total = priced["subtotal"], priced["gst_amount"], priced["total"]
    currency = priced["currency"]

    row = {
        "branch_id": branch["id"],
        "items": items_data,
        "subtotal": subtotal,
        "gst_amount": gst_amount,
        "total": total,
        "fulfilment": fulfilment,
        "delivery_type": "Delivery" if fulfilment == "delivery" else "Pickup",
        "delivery_fee": priced["delivery_fee"],
        "delivery_distance_km": priced["delivery_distance_km"] if fulfilment == "delivery" else None,
        # Location and address are kept only for delivery orders.
        "delivery_address": address,
        "customer_lat": lat,
        "customer_lng": lng,
        "customer_pincode": pincode,
        "customer_name": customer_name,
    }

    if existing:
        # Old pages can't apply offers, so an edit through them drops any
        # offer the order had (and frees its redemption below).
        row.update(NO_OFFER_COLUMNS)
        # Conditional on still being pending/unpaid, so a payment link sent
        # between our check and this write can't be undercut.
        updated = supabase.table("orders").update(row) \
            .eq("id", existing["id"]).eq("project_id", project_id) \
            .eq("status", "pending").eq("payment_status", "unpaid").execute()
        if not updated.data:
            raise shop_error(409, "order_not_editable", "This order is already being processed. Please place a new order.")
        order = updated.data[0]
        release_offer(order["id"])
    else:
        # Uses the insert's OWN returned row. This used to re-read "newest
        # order for this phone", so two concurrent submits from one number
        # could each describe the other's order in the confirmation and in
        # the payment link.
        inserted = supabase.table("orders").insert({
            "project_id": project_id,
            "phone_number": phone,
            "status": "pending",
            "payment_status": "unpaid",
            **row,
        }).execute()
        if not inserted.data:
            raise shop_error(500, "invalid_request", "Could not create the order. Please try again.")
        order = inserted.data[0]

    result = _send_cart_confirmation(project_id, phone, order, items_data, currency, subtotal, gst_amount, total,
                                     catalog_id, priced["price_changed"], branch=branch)
    return {
        "status": result["status"],
        "order_id": order["id"],
        "subtotal": subtotal,
        "gst_amount": gst_amount,
        "delivery_fee": priced["delivery_fee"],
        "total": total,
        **({"warning": result["warning"]} if "warning" in result else {}),
    }


# ─────────────────────────────────────────────
# PUBLIC — customer identity, My Orders, order status / cancel (v3)
# ─────────────────────────────────────────────

class TokenBody(BaseModel):
    customer_token: Optional[str] = Field(default=None, max_length=1024)


class MyOrdersBody(BaseModel):
    customer_token: Optional[str] = Field(default=None, max_length=1024)
    limit: Optional[int] = 20


class OrderAuthBody(BaseModel):
    receipt: Optional[str] = Field(default=None, max_length=128)
    customer_token: Optional[str] = Field(default=None, max_length=1024)


@router.post("/public/shop/{project_id}/me")
def public_shop_me(project_id: str, body: TokenBody, request: Request):
    _public_shop_guard(project_id, request, "shop-me", limit=30)
    _assert_shop_enabled(project_id)
    phone, reason = shop_identity.verify_customer_token(body.customer_token, project_id)
    if not phone:
        return {"verified": False, "reason": reason}

    last = None
    res = supabase.table("orders").select(
        "customer_name, address_line1, address_line2, address_city, customer_pincode, "
        "customer_lat, customer_lng, delivery_address"
    ).eq("project_id", project_id).eq("phone_number", phone).eq("fulfilment", "delivery") \
        .not_.is_("placed_at", "null").order("created_at", desc=True).limit(1).execute()
    if res.data:
        o = res.data[0]
        last = {
            "customer_name": o.get("customer_name"),
            # Orders from before v3 only have the composed address.
            "address_line1": o.get("address_line1") or ((o.get("delivery_address") or "")[:110] or None),
            "address_line2": o.get("address_line2"),
            "address_city": o.get("address_city"),
            "customer_pincode": o.get("customer_pincode"),
            "customer_lat": o.get("customer_lat"),
            "customer_lng": o.get("customer_lng"),
        }
    else:
        named = supabase.table("orders").select("customer_name").eq("project_id", project_id) \
            .eq("phone_number", phone).not_.is_("customer_name", "null") \
            .order("created_at", desc=True).limit(1).execute()
        if named.data:
            last = {"customer_name": named.data[0].get("customer_name")}

    return {
        "verified": True,
        "reason": None,
        "phone": phone,
        "phone_masked": shop_identity.mask_phone(phone),
        "expires_at": shop_identity.token_expiry(body.customer_token, project_id),
        "last_details": last,
    }


def _branch_names(project_id: str, ids) -> dict:
    ids = [i for i in set(ids) if i]
    if not ids:
        return {}
    try:
        res = supabase.table("shop_branches").select("id, name").eq("project_id", project_id).in_("id", ids).execute()
        return {b["id"]: b.get("name") for b in (res.data or [])}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {}


@router.post("/public/shop/{project_id}/my-orders")
def public_my_orders(project_id: str, body: MyOrdersBody, request: Request):
    _public_shop_guard(project_id, request, "shop-my-orders", limit=20)
    _assert_shop_enabled(project_id)
    phone, _ = shop_identity.verify_customer_token(body.customer_token, project_id)
    if not phone:
        raise shop_error(401, "token_required", "Open the menu from your WhatsApp chat to see your orders.")
    limit = max(1, min(int(body.limit or 20), 20))
    res = supabase.table("orders").select(
        "id, created_at, placed_at, status, payment_status, payment_method, branch_id, fulfilment, items, "
        "total, discount_amount, offer_code, payment_link_url, payment_link_expires_at"
    ).eq("project_id", project_id).eq("phone_number", phone) \
        .or_("placed_at.not.is.null,status.neq.pending") \
        .order("created_at", desc=True).limit(limit).execute()
    rows = res.data or []
    names = _branch_names(project_id, [o.get("branch_id") for o in rows])
    out = []
    for o in rows:
        pay_url, pay_expires = live_pay_link(o)
        out.append({
            "order_id": o["id"],
            "short_id": o["id"][:8].upper(),
            "created_at": o.get("created_at"),
            "placed_at": o.get("placed_at"),
            "status": o.get("status"),
            "payment_status": o.get("payment_status"),
            "payment_method": o.get("payment_method"),
            "status_label": order_status_label(o),
            "branch_id": o.get("branch_id"),
            "branch_name": names.get(o.get("branch_id")),
            "fulfilment": o.get("fulfilment") or "pickup",
            "items": [{"product_id": it.get("product_id"), "name": it.get("name"),
                       "quantity": it.get("quantity"), "price": it.get("price")} for it in (o.get("items") or [])],
            "total": float(o.get("total") or 0),
            "discount_amount": float(o.get("discount_amount") or 0),
            "offer_code": o.get("offer_code"),
            "pay_url": pay_url,
            "pay_expires_at": pay_expires,
        })
    return {"orders": out}


def _order_for_customer(project_id: str, order_id: str, body: OrderAuthBody) -> dict:
    """The order, if the caller holds its receipt or a token for its phone.
    Every failure is the same 404, so nothing is learned about other orders."""
    not_found = shop_error(404, "order_not_found", "Order not found")
    if not _UUID_RE.match(order_id or ""):
        raise not_found
    res = supabase.table("orders").select("*").eq("id", order_id).eq("project_id", project_id).limit(1).execute()
    order = res.data[0] if res.data else None
    if not order:
        raise not_found
    if body.receipt and shop_identity.verify_receipt(order["id"], body.receipt):
        return order
    phone, _ = shop_identity.verify_customer_token(body.customer_token, project_id)
    if phone and phone == order.get("phone_number"):
        return order
    raise not_found


@router.post("/public/shop/{project_id}/orders/{order_id}/status")
def public_order_status(project_id: str, order_id: str, body: OrderAuthBody, request: Request):
    _public_shop_guard(project_id, request, "shop-order-status", limit=40)
    order = _order_for_customer(project_id, order_id, body)
    pay_url, pay_expires = live_pay_link(order)
    names = _branch_names(project_id, [order.get("branch_id")])
    return {
        "order_id": order["id"],
        "short_id": order["id"][:8].upper(),
        "status": order.get("status"),
        "payment_status": order.get("payment_status"),
        "payment_method": order.get("payment_method"),
        "status_label": order_status_label(order),
        "total": float(order.get("total") or 0),
        "fulfilment": order.get("fulfilment") or "pickup",
        "branch_name": names.get(order.get("branch_id")),
        "pay_url": pay_url,
        "pay_expires_at": pay_expires,
    }


@router.post("/public/shop/{project_id}/orders/{order_id}/cancel")
def public_order_cancel(project_id: str, order_id: str, body: OrderAuthBody, request: Request):
    # Plain def: may call Razorpay.
    _public_shop_guard(project_id, request, "shop-order-cancel", limit=10)
    order = _order_for_customer(project_id, order_id, body)
    if order.get("status") != "pending" or order.get("payment_status") not in ("unpaid", "link_sent"):
        raise shop_error(409, "order_not_cancellable", "This order can't be cancelled any more. Please contact the store.")
    updated = supabase.table("orders").update({"status": "cancelled"}) \
        .eq("id", order["id"]).eq("project_id", project_id).eq("status", "pending") \
        .in_("payment_status", ["unpaid", "link_sent"]).execute()
    if not updated.data:
        raise shop_error(409, "order_not_cancellable", "This order can't be cancelled any more. Please contact the store.")
    release_offer(order["id"])
    if order.get("payment_status") == "link_sent":
        cancel_payment_link(project_id, order.get("payment_id"))
    return {"status": "cancelled"}


# ─────────────────────────────────────────────
# PUBLIC — place order (v3: the page's only order-writing call)
# ─────────────────────────────────────────────

class AddressIn(BaseModel):
    line1: Optional[str] = Field(default=None, max_length=400)
    line2: Optional[str] = Field(default=None, max_length=400)
    city: Optional[str] = Field(default=None, max_length=200)
    pincode: Optional[Union[str, int]] = None


class PlaceOrderBody(BaseModel):
    """No prices, discounts or totals: the server computes all money.
    expected_total only REJECTS a mismatch (409 price_changed)."""
    customer_token: Optional[str] = Field(default=None, max_length=1024)
    phone: Optional[str] = Field(default=None, max_length=32)            # legacy, only without a valid token
    idempotency_key: Optional[str] = Field(default=None, max_length=64)
    order_id: Optional[str] = Field(default=None, max_length=64)
    branch_id: Optional[str] = Field(default=None, max_length=64)
    catalog_id: Optional[str] = Field(default=None, max_length=64)
    items: List[CartItem] = Field(default_factory=list, max_length=MAX_CART_LINES)
    fulfilment: Optional[str] = Field(default="pickup", max_length=16)
    customer_name: Optional[str] = Field(default=None, max_length=400)
    address: Optional[AddressIn] = None
    customer_lat: Optional[float] = None
    customer_lng: Optional[float] = None
    coupon_code: Optional[str] = Field(default=None, max_length=64)
    sold_out_preference: Optional[str] = Field(default=None, max_length=32)
    note: Optional[str] = Field(default=None, max_length=2000)
    expected_total: Optional[float] = None


class _Restart(Exception):
    """client_ref insert race: another request with the same key won."""


def _shop_error_extra(status: int, code: str, message: str, **extra) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, **extra})


def _bad_request(message: str) -> HTTPException:
    return shop_error(400, "invalid_request", message)


def _one_line_field(value, max_len: int, label: str, min_len: int = 0, required: bool = False) -> Optional[str]:
    text = " ".join(str(value).split()) if value is not None else ""
    if not text:
        if required:
            raise _bad_request(f"Please enter your {label}.")
        return None
    try:
        text = shop_rules.clean_text(text, max_len)
    except ValueError:
        raise _bad_request(f"{label.capitalize()} must be at most {max_len} characters.")
    if text is None or len(text) < min_len:
        if required:
            raise _bad_request(f"Please enter your {label}.")
        return None
    return text


def _validate_place_body(project_id: str, body: PlaceOrderBody) -> dict:
    """Cleaned customer fields (all 400s happen here, before any write)."""
    import phones as _phones
    fulfilment = body.fulfilment or "pickup"
    if fulfilment not in ("pickup", "delivery"):
        raise _bad_request("Choose pickup or delivery.")
    try:
        key = str(uuid.UUID(str(body.idempotency_key or "")))
    except ValueError:
        raise _bad_request("Missing or invalid idempotency key. Please refresh and try again.")
    if body.order_id and not _UUID_RE.match(body.order_id):
        raise shop_error(404, "order_not_found", "Order not found")
    if body.branch_id and not _UUID_RE.match(body.branch_id):
        raise _bad_request("Invalid branch id")
    catalog_id = body.catalog_id or None
    if catalog_id and not _UUID_RE.match(catalog_id):
        raise _bad_request("Invalid catalog id")

    name = _one_line_field(body.customer_name, 80, "name", min_len=1, required=True)
    out = {"key": key, "fulfilment": fulfilment, "catalog_id": catalog_id, "name": name,
           "line1": None, "line2": None, "city": None, "pincode": None, "lat": None, "lng": None,
           "delivery_address": None}
    if fulfilment == "delivery":
        a = body.address or AddressIn()
        out["line1"] = _one_line_field(a.line1, 110, "address (house / flat, street)", min_len=3, required=True)
        out["line2"] = _one_line_field(a.line2, 110, "landmark")
        out["city"] = _one_line_field(a.city, 50, "city", min_len=2, required=True)
        region = _phones.default_region_for_project(project_id)
        if a.pincode is None or not str(a.pincode).strip():
            raise _bad_request("Please enter your pincode.")
        pin = shop_rules.normalize_pincode(a.pincode, region)
        if not pin:
            raise _bad_request("That doesn't look like a valid pincode.")
        out["pincode"] = pin
        if body.customer_lat is None or body.customer_lng is None:
            raise shop_error(400, "location_required", "Set your delivery location on the map.")
        try:
            out["lat"], out["lng"] = shop_rules.validate_latlng(body.customer_lat, body.customer_lng)
        except ValueError as e:
            raise _bad_request(str(e))
        composed = ", ".join(x for x in (out["line1"], out["line2"], out["city"]) if x)
        if len(composed) < 10:
            raise _bad_request("Please enter your full delivery address (house / flat, street, landmark).")
        out["delivery_address"] = composed

    pref = body.sold_out_preference or "remove_item"
    if pref not in shop_rules.SOLD_OUT_PREFS:
        raise _bad_request("Invalid sold-out preference.")
    out["sold_out_preference"] = pref
    try:
        note = shop_rules.clean_text(body.note, 2000)
    except ValueError:
        note = None
    note = one_line(note) if note else None
    if note and len(note) > 300:
        raise _bad_request("Instructions must be at most 300 characters.")
    out["note"] = note or None

    if body.expected_total is None or not math.isfinite(float(body.expected_total)):
        raise _bad_request("Missing expected total. Please refresh and try again.")
    out["expected_total"] = float(body.expected_total)
    return out


def _totals(order: dict) -> dict:
    def f(k):
        return float(order.get(k) or 0)
    return {"subtotal": f("subtotal"), "items_discount": f("items_discount"), "gst_amount": f("gst_amount"),
            "delivery_fee": f("delivery_fee"), "delivery_discount": f("delivery_discount"),
            "discount_amount": f("discount_amount"), "total": f("total")}


def _place_summary(order: dict, state: str, branch_name: Optional[str] = None, payment_url: Optional[str] = None,
                   payment_expires_at: Optional[str] = None, notice: Optional[str] = None,
                   whatsapp: Optional[str] = None) -> dict:
    if branch_name is None and order.get("branch_id"):
        branch_name = _branch_names(order["project_id"], [order["branch_id"]]).get(order["branch_id"])
    offer = None
    if order.get("offer_id"):
        offer = {"id": order["offer_id"], "title": order.get("offer_title"), "code": order.get("offer_code"),
                 "kind": order.get("discount_type")}
    return {
        "state": state,
        "order_id": order["id"],
        "short_id": order["id"][:8].upper(),
        "payment_url": payment_url if state == "payment_required" else None,
        "payment_expires_at": payment_expires_at if state == "payment_required" else None,
        "payment_method": order.get("payment_method") or ("online" if state == "payment_required" else None),
        "receipt": shop_identity.order_receipt(order["id"]),
        "branch_name": branch_name,
        "fulfilment": order.get("fulfilment") or "pickup",
        "totals": _totals(order),
        "offer": offer,
        "notice": notice,
        "whatsapp": whatsapp,
    }


def _money_columns(priced: dict) -> dict:
    offer = priced.get("offer") or {}
    return {
        "subtotal": priced["subtotal"],
        "gst_amount": priced["gst_amount"],
        "total": priced["total"],
        "delivery_fee": priced["delivery_fee"],
        "offer_id": offer.get("id"),
        "offer_code": offer.get("code") if offer.get("application") == "code" else None,
        "offer_title": offer.get("title"),
        "discount_type": offer.get("kind"),
        "items_discount": priced["items_discount"],
        "delivery_discount": priced["delivery_discount"],
        "discount_amount": priced["discount_amount"],
    }


@router.post("/public/shop/{project_id}/place-order")
def place_order(project_id: str, body: PlaceOrderBody, request: Request):
    # Plain def: blocking DB, WhatsApp and Razorpay calls.
    _public_shop_guard(project_id, request, "shop-place", limit=10)
    _assert_shop_enabled(project_id)
    config = load_shop_config(project_id)

    ident = shop_identity.resolve_identity(project_id, body.customer_token, body.phone)
    if not ident["phone"]:
        raise _bad_request("Open this menu from your WhatsApp chat to order.")
    if shop_identity.require_customer_token() and not ident["verified"]:
        raise shop_error(401, "token_required", "Open this menu from your WhatsApp chat to order.")
    # Unverified (legacy ?phone=) attempts get their own bucket: anyone can
    # type a number, so they must not be able to use up the real customer's
    # verified allowance and lock them out of ordering.
    bucket = "place-phone" if ident["verified"] else "place-phone-u"
    if is_rate_limited(f"{bucket}:{project_id}:{ident['phone']}", 10, 3600):
        raise shop_error(429, "rate_limited", "Too many orders from this number — please try again later.")
    v = _validate_place_body(project_id, body)

    for attempt in range(2):
        try:
            return _place_order_attempt(project_id, body, request, config, ident, v)
        except _Restart:
            if attempt == 1:
                raise shop_error(409, "idempotency_conflict", "This order is already being placed. Please wait a moment.")
    raise shop_error(409, "idempotency_conflict", "This order is already being placed. Please wait a moment.")


def _place_order_attempt(project_id: str, body: PlaceOrderBody, request: Request, config: dict,
                         ident: dict, v: dict) -> dict:
    from shop_branches import resolve_branch, price_with_mode, raise_first_error, quote_view

    phone, verified, key = ident["phone"], ident["verified"], v["key"]
    now = datetime.now(timezone.utc)

    # ── 1. Existing order: Add More link, or a retry with the same key ──
    if body.order_id:
        res = supabase.table("orders").select("*").eq("id", body.order_id).eq("project_id", project_id) \
            .eq("phone_number", phone).limit(1).execute()
        existing = res.data[0] if res.data else None
        if not existing:
            raise shop_error(404, "order_not_found", "Order not found")
    else:
        res = supabase.table("orders").select("*").eq("project_id", project_id).eq("client_ref", key).limit(1).execute()
        existing = res.data[0] if res.data else None
        if existing and existing.get("phone_number") != phone:
            raise shop_error(409, "idempotency_conflict", "This checkout belongs to another order. Please refresh.")

    branch_for_fp = (existing or {}).get("branch_id") or body.branch_id
    fp = shop_rules.checkout_fingerprint({
        "branch_id": branch_for_fp, "items": [i.model_dump() for i in body.items], "fulfilment": v["fulfilment"],
        "name": v["name"], "line1": v["line1"], "line2": v["line2"], "city": v["city"], "pincode": v["pincode"],
        "lat": v["lat"], "lng": v["lng"], "coupon": body.coupon_code,
        "sold_out_preference": v["sold_out_preference"], "note": v["note"],
    })

    regen_from = None
    if existing:
        done = existing.get("status") != "pending" or \
            existing.get("payment_status") in ("paid", "underpaid", "refunded", "not_required")
        if done:
            # Pure retry (same key, unchanged checkout): no side effects. The
            # page reuses its key across edits of an unplaced order, so a
            # changed checkout on a placed order is a NEW order, not a retry.
            if existing.get("client_ref") == key and existing.get("checkout_hash") in (None, fp):
                return _place_summary(existing, "already_placed")
            raise shop_error(409, "order_not_editable", "This order is already being processed. Please place a new order.")
        if existing.get("payment_status") == "link_sent":
            live_url, live_expires = live_pay_link(existing, margin_seconds=60)
            if fp == existing.get("checkout_hash") and live_url:
                return _place_summary(existing, "payment_required", payment_url=live_url,
                                      payment_expires_at=live_expires)   # SAME link, never a second one
            if fp != existing.get("checkout_hash"):
                raise _shop_error_extra(409, "payment_in_progress", "You already started paying for this order.",
                                        order_id=existing["id"], payment_url=live_url,
                                        payment_expires_at=live_expires)
            regen_from = existing.get("payment_id")   # same cart, link expired: new link on the same order

    # ── Branch ──
    if existing and body.branch_id and existing.get("branch_id") and body.branch_id != existing["branch_id"]:
        raise _bad_request("An order can't move to another branch. Please place a new order.")
    branch_id = (existing or {}).get("branch_id") or body.branch_id
    catalog_id = v["catalog_id"]
    if not branch_id and catalog_id:
        cat = supabase.table("catalogs").select("branch_id").eq("id", catalog_id).eq("project_id", project_id).limit(1).execute()
        if not cat.data:
            raise _bad_request("This menu isn't available any more. Please refresh.")
        branch_id = cat.data[0].get("branch_id")
    branch = resolve_branch(project_id, branch_id, public=True)

    # ── Price (server is the only source of money) ──
    price_args = dict(catalog_id=catalog_id, config=config, identity=ident, coupon_code=body.coupon_code,
                      ip=client_ip(request), exclude_order_id=existing["id"] if existing else None)
    priced, mode = price_with_mode(project_id, branch, body.items, v["fulfilment"], v["lat"], v["lng"],
                                   v["pincode"], **price_args)
    raise_first_error(priced)
    coupon = priced.get("coupon")
    if body.coupon_code and coupon and coupon.get("status") == "rejected":
        raise _shop_error_extra(409, "coupon_invalid", coupon.get("message") or "This code can't be used.",
                                reason=coupon.get("reason"), quote=quote_view(priced, mode, verified))
    if abs(priced["total"] - v["expected_total"]) > 0.009:
        raise _shop_error_extra(409, "price_changed", "Prices or offers changed — please check the total.",
                                quote=quote_view(priced, mode, verified))

    delivery = v["fulfilment"] == "delivery"
    row = {
        "branch_id": branch["id"],
        "items": priced["items_data"],
        "fulfilment": v["fulfilment"],
        "delivery_type": "Delivery" if delivery else "Pickup",
        "delivery_distance_km": priced["delivery_distance_km"] if delivery else None,
        "delivery_address": v["delivery_address"],
        "address_line1": v["line1"], "address_line2": v["line2"], "address_city": v["city"],
        "customer_lat": v["lat"], "customer_lng": v["lng"], "customer_pincode": v["pincode"],
        "customer_name": v["name"],
        "special_request": v["note"],
        "sold_out_preference": v["sold_out_preference"],
        "checkout_hash": fp,
        **_money_columns(priced),
    }
    if not existing or not existing.get("client_ref"):
        row["client_ref"] = key

    # ── 2. Write the order (conditional) or insert it ──
    if existing:
        q = supabase.table("orders").update(row).eq("id", existing["id"]).eq("project_id", project_id) \
            .eq("status", "pending").eq("payment_status", existing.get("payment_status"))
        if regen_from:
            q = q.eq("payment_id", regen_from)
        written = q.execute()
        if not written.data:
            raise shop_error(409, "order_not_editable", "This order is already being processed. Please place a new order.")
        order = written.data[0]
    else:
        try:
            inserted = supabase.table("orders").insert({
                "project_id": project_id, "phone_number": phone,
                "status": "pending", "payment_status": "unpaid", **row,
            }).execute()
        except Exception as e:
            text = str(e)
            if "23505" in text or "duplicate key" in text:
                raise _Restart()
            raise
        if not inserted.data:
            raise shop_error(500, "invalid_request", "Could not create the order. Please try again.")
        order = inserted.data[0]

    # ── 3. Claim the offer (authoritative, in the DB) ──
    offer = priced.get("offer")
    if offer:
        hold = None if mode == "place_offline" else (now + timedelta(minutes=OFFER_HOLD_MINUTES)).isoformat()
        claim = claim_offer(order["id"], offer["id"], phone, verified, hold, priced["discount_amount"])
        if not claim.get("ok"):
            reason = claim.get("reason") or "offer_unavailable"
            p2, mode2 = price_with_mode(project_id, branch, body.items, v["fulfilment"], v["lat"], v["lng"],
                                        v["pincode"], exclude_offer_ids=(offer["id"],), **price_args)
            message = shop_rules.offer_reason_message(reason) or "This offer is no longer available."
            if p2.get("coupon") and p2["coupon"].get("code") and p2["coupon"].get("status") == "rejected":
                p2["coupon"] = {**p2["coupon"], "reason": reason, "message": message}
            if not p2["errors"]:
                supabase.table("orders").update(_money_columns(p2)).eq("id", order["id"]) \
                    .eq("project_id", project_id).eq("status", "pending") \
                    .eq("payment_status", order.get("payment_status")).execute()
            release_offer(order["id"])
            raise _shop_error_extra(409, "offer_unavailable", message, reason=reason,
                                    quote=quote_view(p2, mode2, verified))
    else:
        release_offer(order["id"])     # a code removed on an edit / retry

    notice = "best_offer_applied" if coupon and coupon.get("status") == "not_better" else None
    currency = priced["currency"]

    # ── 4. Act on the mode ──
    if mode == "confirm_in_whatsapp":
        if existing and existing.get("client_ref") == key and existing.get("checkout_hash") == fp:
            # A pure retry (same key, unchanged checkout): the cart message
            # already went out, so don't send it again.
            return _place_summary(order, "confirm_in_whatsapp", branch_name=branch.get("name"), notice=notice,
                                  whatsapp="skipped")
        res = _send_cart_confirmation(project_id, phone, order, priced["items_data"], currency,
                                      priced["subtotal"], priced["gst_amount"], priced["total"],
                                      catalog_id, priced["price_changed"], branch=branch)
        return _place_summary(order, "confirm_in_whatsapp", branch_name=branch.get("name"), notice=notice,
                              whatsapp="skipped" if res.get("warning") else "text")

    if mode == "place_offline":
        free = priced["total"] == 0
        placed = {"status": "confirmed", "payment_status": "not_required" if free else "unpaid",
                  "payment_method": "none" if free else "offline", "placed_at": now.isoformat()}
        u = supabase.table("orders").update(placed).eq("id", order["id"]).eq("project_id", project_id) \
            .eq("status", "pending").eq("payment_status", "unpaid").execute()
        if not u.data:
            again = supabase.table("orders").select("*").eq("id", order["id"]).limit(1).execute()
            return _place_summary((again.data or [order])[0], "already_placed", branch_name=branch.get("name"))
        order = u.data[0]
        confirm_offer(order["id"])
        wa = send_order_placed_message(order, branch, config)
        phone_number_id, token = _wa_creds(project_id)
        if phone_number_id:
            send_owner_order_alert(order, branch, {**config, "project_id": project_id}, phone_number_id, token,
                                   header="🔔 *New Order*" if free else "🔔 *New Order (collect payment)*")
        _set_session(project_id, phone, {"mode": "flow", "metadata": {}})
        return _place_summary(order, "placed", branch_name=branch.get("name"), notice=notice, whatsapp=wa)

    # pay_online
    try:
        link = create_payment_link(order, config, callback_url=shop_identity.build_return_url(project_id, order["id"]),
                                   regenerate_from=regen_from)
    except OrderChangedError:
        raise shop_error(409, "order_not_editable", "This order just changed. Please review and try again.")
    if not link:
        raise _shop_error_extra(502, "payment_unavailable", "Couldn't start the payment. Please try again.",
                                order_id=order["id"])
    _set_session(project_id, phone, {"mode": "awaiting_payment",
                                     "metadata": {"order_id": order["id"], "branch_id": branch["id"], "source": "web"}})
    order = {**order, "payment_method": "online", "payment_status": "link_sent"}
    return _place_summary(order, "payment_required", branch_name=branch.get("name"), payment_url=link["url"],
                          payment_expires_at=link["expires_at"], notice=notice)


def _set_session(project_id: str, phone: str, data: dict):
    """Session handoff that keeps flow_id / current_node_id. Best-effort:
    the order is already placed."""
    try:
        from flows import upsert_session
        upsert_session(project_id, phone, data)
    except Exception as e:
        sentry_sdk.capture_exception(e)


# ─────────────────────────────────────────────
# ORDERS
# ─────────────────────────────────────────────

@router.get("/orders")
async def list_orders(project_id: str, branch_id: Optional[str] = None, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="shop")
    query = supabase.table("orders").select("*").eq("project_id", project_id)
    if branch_id and _UUID_RE.match(branch_id):
        query = query.eq("branch_id", branch_id)
    res = query.order("created_at", desc=True).execute()
    return res.data or []

@router.put("/orders/{order_id}")
def update_order(order_id: str, body: OrderStatusUpdate, user=Depends(verify_token)):
    # Plain def: a status change may send a WhatsApp message (blocking).
    if not _UUID_RE.match(order_id or ""):
        raise shop_error(404, "order_not_found", "Order not found")
    existing = supabase.table("orders").select("*").eq("id", order_id).maybe_single().execute()
    order = existing.data if existing else None
    if not order:
        raise shop_error(404, "order_not_found", "Order not found")
    require_project_access(user.id, order["project_id"], tab="shop")

    new_status = None
    if body.status is not None:
        new_status = ORDER_STATUS_ALIASES.get(body.status, body.status)
        if new_status not in VALID_ORDER_STATUSES:
            raise shop_error(400, "invalid_status", "Invalid order status")
    if body.payment_status is not None and body.payment_status not in VALID_PAYMENT_STATUSES:
        raise shop_error(400, "invalid_request", "Invalid payment status")

    current = ORDER_STATUS_ALIASES.get(order.get("status"), order.get("status"))
    status_changed = new_status is not None and new_status != current
    if status_changed:
        # Closed orders stay closed: a reopened one would re-message the
        # customer and could be "completed" again after a refund.
        if current in shop_rules.FINAL_STATUSES:
            raise shop_error(409, "invalid_transition", "This order is already closed")
        if new_status == "out_for_delivery" and (order.get("fulfilment") or "pickup") != "delivery":
            raise shop_error(409, "invalid_transition", "Pickup orders can't go out for delivery")

    update = {}
    if status_changed:
        update["status"] = new_status
    if body.payment_status is not None and body.payment_status != order.get("payment_status"):
        update["payment_status"] = body.payment_status

    if update:
        query = supabase.table("orders").update(update).eq("id", order_id)
        if status_changed:
            # Only from the status we checked: two staff tapping at once
            # must not both pass the checks above and both notify.
            query = query.eq("status", order.get("status"))
        written = query.execute()
        if status_changed and not written.data:
            raise shop_error(409, "invalid_transition", "This order was just updated. Refresh and try again.")
        if status_changed and new_status == "cancelled":
            release_offer(order_id)
            if order.get("payment_status") == "link_sent":
                cancel_payment_link(order["project_id"], order.get("payment_id"))

    notification = None
    if status_changed and body.notify is not False:
        try:
            notification = notify_order_status({**order, **update}, new_status)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            notification = "failed"

    res = supabase.table("orders").select("*").eq("id", order_id).single().execute()
    return {**res.data, "notification": notification}


# ─────────────────────────────────────────────
# ORDER MESSAGES (status updates, owner alerts)
# ─────────────────────────────────────────────

ORDER_STATUS_MSG_CAP_PER_HOUR = 500


def _send_order_template(to: str, template: str, params: list, phone_number_id: str, token: str) -> bool:
    """An order message as an approved template. False when the template
    isn't in the account or WhatsApp refuses it (the caller then falls back
    to plain text). Same shape as events._send_registration_template."""
    from whatsapp import http
    clean = [shop_rules.sanitize_template_param(p) for p in params]
    try:
        res = http.post(
            f"https://graph.facebook.com/v25.0/{phone_number_id}/messages",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "messaging_product": "whatsapp", "to": to, "type": "template",
                "template": {
                    "name": template, "language": {"code": "en_US"},
                    "components": [{"type": "body", "parameters": [{"type": "text", "text": p} for p in clean]}],
                },
            },
        )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return False
    if not res.ok:
        print(f"Order template {template} not sent ({res.status_code}); falling back to text")
    return res.ok


def notify_order_status(order: dict, new_status: str) -> str:
    """WhatsApp the customer about a status change the merchant made.
    Template first (reaches everyone), plain text as the fallback (only
    reaches people inside WhatsApp's 24-hour window). Returns "template",
    "text", "skipped" or "failed". Never raises for a send problem."""
    from whatsapp import send_whatsapp_message
    from config import WHATSAPP_TOKEN
    from shop_branches import get_branch, contact_phone

    project_id = order["project_id"]
    wa_res = supabase.table("whatsapp_integrations").select("phone_number_id, access_token") \
        .eq("project_id", project_id).maybe_single().execute()
    wa = wa_res.data if wa_res else None
    if not wa or not order.get("phone_number"):
        return "skipped"

    config = load_shop_config(project_id)
    branch = get_branch(project_id, order.get("branch_id")) if order.get("branch_id") else None
    contact = contact_phone(branch, config)
    msg = shop_rules.order_status_message(new_status, order.get("fulfilment") or "pickup", {
        "name": order.get("customer_name"),
        "oid": order["id"][:8].upper(),
        "branch": (branch or {}).get("name") or config.get("store_name") or "our store",
        "contact": phones.pretty(contact) if contact else "our store",
        "prep": (branch or {}).get("prep_minutes"),
        "address": (branch or {}).get("address"),
    })
    if not msg:
        return "skipped"

    # Each message is billable; bound a runaway script or bulk edit.
    if is_rate_limited(f"order-status-msg:{project_id}", ORDER_STATUS_MSG_CAP_PER_HOUR, 3600):
        sentry_sdk.capture_message(f"Order status messages capped for project {project_id}")
        return "skipped"

    token = wa.get("access_token") or WHATSAPP_TOKEN
    if _send_order_template(order["phone_number"], msg["template"], msg["params"], wa["phone_number_id"], token):
        return "template"
    res = send_whatsapp_message(to=order["phone_number"], text=msg["text"],
                                phone_number_id=wa["phone_number_id"], token=token)
    return "text" if res is not None and res.ok else "failed"


SOLD_OUT_LABELS = {"remove_item": "Remove it", "call_me": "Call me", "merchant_choice": "Your recommendation"}


def order_money_lines(order: dict, currency: str = "₹") -> str:
    """The shared money block (C.1) for customer messages."""
    lines = [f"Item total: {currency}{float(order.get('subtotal') or 0):.2f}"]
    items_discount = float(order.get("items_discount") or 0)
    if items_discount > 0:
        name = one_line(order.get("offer_code") or order.get("offer_title") or "")
        lines.append(f"Discount{f' ({name})' if name else ''}: −{currency}{items_discount:.2f}")
    if float(order.get("gst_amount") or 0) > 0:
        lines.append(f"GST: {currency}{float(order['gst_amount']):.2f}")
    if order.get("fulfilment") == "delivery":
        fee = float(order.get("delivery_fee") or 0)
        delivery_discount = float(order.get("delivery_discount") or 0)
        if delivery_discount > 0:
            lines.append(f"Delivery: {currency}{fee:.2f}")
            lines.append(f"Free delivery offer: −{currency}{delivery_discount:.2f}")
        elif fee <= 0:
            lines.append("Delivery: FREE")
        else:
            lines.append(f"Delivery: {currency}{fee:.2f}")
    return "\n".join(lines)


def _wa_creds(project_id: str):
    """(phone_number_id, token) of the project's WhatsApp, or (None, None)."""
    from config import WHATSAPP_TOKEN
    try:
        res = supabase.table("whatsapp_integrations").select("phone_number_id, access_token") \
            .eq("project_id", project_id).maybe_single().execute()
        wa = res.data if res else None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        wa = None
    if not wa or not wa.get("phone_number_id"):
        return None, None
    return wa["phone_number_id"], wa.get("access_token") or WHATSAPP_TOKEN


def send_order_placed_message(order: dict, branch: Optional[dict], config: dict) -> str:
    """C.3: an order placed without online payment. Text first, then the
    order_status_update template. "text" | "template" | "failed" | "skipped"."""
    from whatsapp import send_whatsapp_message
    from shop_branches import contact_phone

    project_id = order["project_id"]
    phone_number_id, token = _wa_creds(project_id)
    if not phone_number_id or not order.get("phone_number"):
        return "skipped"
    if is_rate_limited(f"order-status-msg:{project_id}", ORDER_STATUS_MSG_CAP_PER_HOUR, 3600):
        sentry_sdk.capture_message(f"Order status messages capped for project {project_id}")
        return "skipped"

    currency = config.get("currency") or "₹"
    config = {**config, "project_id": project_id}
    contact = contact_phone(branch, config)
    oid = order["id"][:8].upper()
    free = order.get("payment_status") == "not_required" or float(order.get("total") or 0) <= 0
    pickup = (order.get("fulfilment") or "pickup") != "delivery"
    if free:
        pay_text, status_text = "Nothing to pay", "Placed"
    elif pickup:
        pay_text, status_text = "pay at pickup", "Placed (pay at pickup)"
    else:
        pay_text, status_text = "pay on delivery", "Placed (pay on delivery)"

    msg = "✅ *Order Placed!*\n\n"
    msg += f"Order ID: #{oid}\n"
    msg += order_money_lines(order, currency) + "\n"
    msg += f"*Total: {currency}{float(order.get('total') or 0):.2f}* — {pay_text}\n"
    msg += order_fulfilment_lines(order, branch, currency) + "\n"
    if contact:
        msg += f"📞 Contact: {phones.pretty(contact)}\n"
    msg += f"📦 Your orders: {shop_identity.build_shop_url(project_id, order['phone_number'], branch_id=order.get('branch_id'), view='orders')}\n"
    msg += "\nThe store will contact you if anything changes."

    try:
        res = send_whatsapp_message(to=order["phone_number"], text=msg, phone_number_id=phone_number_id, token=token)
        if res is not None and getattr(res, "ok", False):
            return "text"
    except Exception as e:
        sentry_sdk.capture_exception(e)
    branch_name = (branch or {}).get("name") or config.get("store_name") or "our store"
    if _send_order_template(order["phone_number"], shop_rules.ORDER_STATUS_TEMPLATE,
                            [order.get("customer_name") or "there", oid, branch_name, status_text],
                            phone_number_id, token):
        return "template"
    return "failed"


def _owner_window_open(project_id: str, phone: str) -> bool:
    """True when this phone messaged the business number in the last 24
    hours - the only time a free-text message reaches it. Any doubt (no
    chat, a lookup error) counts as closed, so the template goes out."""
    digits = re.sub(r"\D", "", str(phone or ""))
    if not digits:
        return False
    try:
        chat = supabase.table("chats").select("id") \
            .eq("project_id", project_id).eq("external_id", digits).eq("channel", "whatsapp") \
            .limit(1).execute()
        if not chat.data:
            return False
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        last = supabase.table("chat_messages").select("id") \
            .eq("chat_id", chat.data[0]["id"]).eq("role", "user").gte("created_at", since) \
            .limit(1).execute()
        return bool(last.data)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return False


def send_owner_order_alert(order: dict, branch: Optional[dict], config: dict, phone_number_id: str, token: str,
                           header: str = "🔔 *New Order Paid!*", payment_note: Optional[str] = None) -> None:
    """New-order alert to the BRANCH's phone. Best-effort.
    Inside the 24-hour window: the full text (free). Outside it: the approved
    shop_new_order_alert template, which reaches the owner regardless; if the
    template isn't added/approved yet, the text is still tried. Each branch
    has its own phone and its own on/off switch (off = the dashboard bell
    only); a branch without a phone sends nothing."""
    from whatsapp import send_whatsapp_message
    from shop_branches import contact_phone

    if branch is not None:
        # Missing key (column not added yet) means on.
        if branch.get("notify_owner_whatsapp") is False:
            return
        owner_phone = branch.get("phone")
    else:
        owner_phone = contact_phone(None, config)
    if not owner_phone:
        return
    currency = config.get("currency") or "₹"
    title = f"{header} — {branch['name']}" if branch and branch.get("name") else header
    who = f"+{order['phone_number']}" + (f" ({one_line(order['customer_name'])})" if order.get("customer_name") else "")

    lines = []
    for i, item in enumerate(order.get("items") or [], 1):
        lines.append(f"{i}. {item['name']} x{item['quantity']} - {currency}{int(item['price'] * item['quantity'])}")

    is_delivery = order.get("fulfilment") == "delivery"
    money_parts = [f"Subtotal {currency}{shop_rules.money(order.get('subtotal'))}"]
    items_discount = float(order.get("items_discount") or 0)
    delivery_discount = float(order.get("delivery_discount") or 0)
    offer_name = one_line(order.get("offer_code") or order.get("offer_title") or "")
    if items_discount > 0:
        money_parts.append(f"Discount {currency}{items_discount:.2f}" + (f" ({offer_name})" if offer_name else ""))
    if float(order.get("gst_amount") or 0) > 0:
        money_parts.append(f"GST {currency}{float(order['gst_amount']):.2f}")
    if is_delivery:
        fee = float(order.get("delivery_fee") or 0)
        money_parts.append(f"Delivery {currency}{shop_rules.money(fee)}" if fee > 0 else "Delivery FREE")
        if delivery_discount > 0:
            money_parts.append(f"Free delivery offer −{currency}{delivery_discount:.2f}")

    if order.get("payment_status") == "paid":
        pay_label = "Paid online"
    elif order.get("payment_status") == "not_required":
        pay_label = "Free"
    else:
        pay_label = "Collect payment"

    msg = f"{title}\n\nFrom: {who}\n\n" + "\n".join(lines) + "\n\n"
    msg += " · ".join(money_parts) + "\n"
    msg += f"Total: {currency}{float(order['total']):.2f}  ·  {pay_label}\n"
    if is_delivery:
        dist = order.get("delivery_distance_km")
        msg += "🛵 Delivery" + (f" · {float(dist):g} km" if dist is not None else "") + "\n"
        where = _where(order)
        if where:
            msg += f"📍 {where}\n"
        if order.get("customer_lat") is not None and order.get("customer_lng") is not None:
            # Numbers only, so nothing customer-typed ends up in the link.
            msg += f"🗺️ https://maps.google.com/?q={float(order['customer_lat']):.6f},{float(order['customer_lng']):.6f}\n"
    else:
        msg += "🏃 Pickup\n"
    pref = SOLD_OUT_LABELS.get(order.get("sold_out_preference"))
    if pref:
        msg += f"🍽️ If an item is sold out: {pref}\n"
    if order.get("special_request"):
        msg += f"📝 Note: {one_line(order['special_request'])}\n"
    msg += f"Order ID: #{order['id'][:8].upper()}"

    try:
        if order.get("project_id") and not _owner_window_open(order["project_id"], owner_phone):
            branch_name = (branch or {}).get("name") or config.get("store_name") or "your store"
            sent = _send_order_template(
                owner_phone, shop_rules.OWNER_ALERT_TEMPLATE,
                [order["id"][:8].upper(), branch_name, order.get("customer_name") or "a customer",
                 f"{currency}{float(order['total']):.2f}", "delivery" if is_delivery else "pickup",
                 payment_note or pay_label],
                phone_number_id, token)
            if sent:
                return
        send_whatsapp_message(to=owner_phone, text=msg, phone_number_id=phone_number_id, token=token)
    except Exception as e:
        sentry_sdk.capture_exception(e)


# ─────────────────────────────────────────────
# RAZORPAY WEBHOOK
# ─────────────────────────────────────────────

def _candidate_webhook_secrets(project_id: str) -> list:
    """Every secret worth trying to verify a webhook against, in preference
    order: the shared Razorpay Partner-app secret (covers every OAuth-
    connected merchant, one config for the whole app) first, then the
    project's own legacy manual key_secret / global env fallback (covers
    shops that haven't reconnected via OAuth yet). Trying both — instead of
    picking one — is what lets OAuth-connected and legacy shops coexist
    correctly during the transition."""
    candidates = []
    if RAZORPAY_WEBHOOK_SECRET:
        candidates.append(RAZORPAY_WEBHOOK_SECRET)
    try:
        config_res = supabase.table("shop_config").select("razorpay_key_secret").eq("project_id", project_id).maybe_single().execute()
        legacy_secret = ((config_res.data if config_res else None) or {}).get("razorpay_key_secret")
    except Exception as e:
        # A DB hiccup here silently narrows the candidate list, which can
        # make a genuine Razorpay payment webhook fail signature
        # verification and get rejected as unauthorized — with nothing
        # distinguishing that from an actually-forged request. Single call
        # per webhook delivery, not a loop.
        sentry_sdk.capture_exception(e)
        legacy_secret = None
    # Deliberately NOT falling back to the global RAZORPAY_KEY_SECRET here.
    # As a candidate for every project it meant one shared secret could sign
    # a forged payment event for any tenant, and now that payment links are
    # never created with the global keys (see generate_razorpay_link) no
    # legitimate event can be signed with them either.
    if legacy_secret:
        candidates.append(legacy_secret)
    return candidates


def _verify_razorpay_signature(body_bytes: bytes, signature: str, project_id: str) -> bool:
    """Fail-closed: no signature, no resolvable secret, or no matching
    secret all mean 'reject'. Replaces the old `if key_secret and signature:`
    check, which silently skipped verification entirely (and accepted the
    payload unverified) whenever either side was empty."""
    if not signature:
        return False
    for secret in _candidate_webhook_secrets(project_id):
        expected = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, signature):
            return True
    return False


@router.post("/webhook/razorpay")
async def razorpay_webhook(request: Request):
    from whatsapp import send_whatsapp_message
    from config import WHATSAPP_TOKEN

    body_bytes = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    # Parsing before verifying meant malformed JSON from an unauthenticated
    # caller raised an uncaught 500, and the differing responses per branch
    # leaked whether a given payment_link id existed in our database.
    try:
        payload = json.loads(body_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    event = payload.get("event")

    if event == "account.app.authorization_revoked":
        # This event isn't tied to any order/project — only the shared
        # Partner-app secret can possibly apply, since there's no per-
        # project fallback to resolve without already knowing the account.
        if not RAZORPAY_WEBHOOK_SECRET or not signature or not hmac.compare_digest(
            hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), body_bytes, hashlib.sha256).hexdigest(), signature
        ):
            raise HTTPException(status_code=400, detail="Invalid signature")
        from razorpay_oauth import handle_authorization_revoked
        handle_authorization_revoked(payload.get("account_id"))
        return {"status": "ok"}

    if event == "payment_link.paid":
        try:
            link_entity = payload["payload"]["payment_link"]["entity"]
            payment_link_id = link_entity["id"]
        except (KeyError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid payload")

        # Dedup deliberately happens AFTER the signature is verified, below.
        # It used to run here, before any verification — so an unauthenticated
        # caller who knew a payment_link_id (the shopper always does, it is in
        # their own WhatsApp link) could POST an unsigned body, mark that id
        # processed, and the genuine Razorpay webhook would then return
        # duplicate_ignored. The order would never be marked paid.

        # Payment Link ids are Razorpay-global unique identifiers, so
        # checking orders then appointments carries no collision risk.
        order_res = supabase.table("orders").select("*").eq("payment_id", payment_link_id).maybe_single().execute()
        order = order_res.data if order_res else None
        if not order:
            # An older link of an order whose link was regenerated.
            hist = supabase.table("orders").select("*").contains("payment_link_ids", [payment_link_id]) \
                .limit(1).execute()
            order = hist.data[0] if hist.data else None

        if not order:
            appt_res = supabase.table("appointments").select("*").eq("payment_id", payment_link_id).maybe_single().execute()
            appointment = appt_res.data if appt_res else None
            if not appointment:
                return {"status": "ok"}

            if not _verify_razorpay_signature(body_bytes, signature, appointment["project_id"]):
                raise HTTPException(status_code=400, detail="Invalid signature")

            # Signature proven — now it is safe to claim this delivery.
            # Razorpay retries, and this sends WhatsApp confirmations and
            # advances the flow, so a replay must not re-run any of it.
            if already_processed("razorpay-payment", payment_link_id):
                return {"status": "duplicate_ignored"}

            from appointments import handle_appointment_payment_paid
            handle_appointment_payment_paid(appointment)
            return {"status": "ok"}

        if not _verify_razorpay_signature(body_bytes, signature, order["project_id"]):
            raise HTTPException(status_code=400, detail="Invalid signature")

        # Signature proven — see the note above on why this is not earlier.
        if already_processed("razorpay-payment", payment_link_id):
            return {"status": "duplicate_ignored"}

        # Meaningful only because order["total"] is now derived from the
        # products table (see _price_cart_server_side). While the browser set
        # the price, this compared the payment against the shopper's own
        # chosen number and so proved nothing.
        # Confirm the amount actually paid matches what we asked for, rather
        # than marking the order paid purely on the event arriving. The
        # signature proves the payload came from Razorpay; it says nothing
        # about the link having been created for this order's total, which
        # matters if a link is ever reused or regenerated at a stale price.
        try:
            paid_amount = int(link_entity.get("amount_paid") or link_entity.get("amount") or 0)
            expected_amount = int(round(float(order["total"]) * 100))
            if paid_amount and paid_amount < expected_amount:
                sentry_sdk.capture_message(
                    f"Razorpay underpayment for order {order['id']}: paid {paid_amount}, expected {expected_amount}"
                )
                supabase.table("orders").update({"payment_status": "underpaid"}).eq("id", order["id"]).execute()
                return {"status": "amount_mismatch"}
        except (TypeError, ValueError, KeyError) as e:
            # Never block a real payment on our own parsing being fussy.
            sentry_sdk.capture_exception(e)

        try:
            config_res = supabase.table("shop_config").select("*").eq("project_id", order["project_id"]).maybe_single().execute()
            config = (config_res.data if config_res else None) or {}
        except Exception as e:
            # Falls back to currency/store_phone defaults, so the paid
            # confirmation below still sends — but silently, with no record
            # that the merchant's own currency/contact settings were lost
            # for this specific confirmation. Single call per webhook, real
            # money already confirmed at this point.
            sentry_sdk.capture_exception(e)
            config = {}

        currency = config.get("currency", "₹")

        # Deliberately not notify_order_status: the payment confirmation
        # below already tells the customer, so that would double-send.
        # Never revives a cancelled order: conditional on not cancelled.
        paid_fields = {
            "payment_status": "paid",
            "status": "confirmed",
            "placed_at": order.get("placed_at") or datetime.now(timezone.utc).isoformat(),
        }
        written = None
        if order.get("status") != "cancelled":
            written = supabase.table("orders").update(paid_fields) \
                .eq("id", order["id"]).neq("status", "cancelled").execute()
        if not written or not written.data:
            _handle_paid_after_cancel(order, config)
            return {"status": "ok"}
        order = {**order, **paid_fields}
        confirm_offer(order["id"])

        try:
            push_order_to_shopify(order["id"])
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Shopify order write-back error for order {order['id']}: {e}")

        try:
            wa_res = supabase.table("whatsapp_integrations").select("*").eq("project_id", order["project_id"]).maybe_single().execute()
            wa_data = (wa_res.data if wa_res else None)
        except Exception as e:
            # wa_data=None below skips both the customer's payment
            # confirmation and the merchant's new-order notification — a
            # customer who just paid gets silent nothing, and nothing here
            # said why. Single call per webhook, not a loop.
            sentry_sdk.capture_exception(e)
            wa_data = None

        if not wa_data:
            return {"status": "ok"}

        phone_number_id = wa_data["phone_number_id"]
        token = wa_data.get("access_token") or WHATSAPP_TOKEN

        from shop_branches import get_branch, get_default_branch, contact_phone
        try:
            branch = (get_branch(order["project_id"], order.get("branch_id")) if order.get("branch_id")
                      else get_default_branch(order["project_id"]))
        except Exception as e:
            # Money is already taken: send the messages without branch lines
            # rather than not at all.
            sentry_sdk.capture_exception(e)
            branch = None
        config.setdefault("project_id", order["project_id"])
        contact = contact_phone(branch, config)

        oid = order['id'][:8].upper()
        msg = f"✅ *Payment Confirmed!*\n\n"
        msg += f"Thank you! Your order has been confirmed.\n\n"
        msg += f"Order ID: #{oid}\n"
        msg += order_money_lines(order, currency) + "\n"
        msg += f"*Total paid: {currency}{float(order['total']):.2f}*\n"
        msg += order_fulfilment_lines(order, branch, currency) + "\n"
        if contact:
            msg += f"📞 Contact: {phones.pretty(contact)}\n"
        orders_url = shop_identity.build_shop_url(order["project_id"], order["phone_number"],
                                                  branch_id=order.get("branch_id"), view="orders")
        msg += f"📦 Your orders: {orders_url}\n"
        msg += "\nWe'll keep you posted on WhatsApp! 🎉"

        sent_ok = False
        try:
            sent = send_whatsapp_message(to=order["phone_number"], text=msg, phone_number_id=phone_number_id, token=token)
            sent_ok = sent is not None and getattr(sent, "ok", False)
        except Exception as e:
            sentry_sdk.capture_exception(e)
        if not sent_ok:
            branch_name = (branch or {}).get("name") or config.get("store_name") or "our store"
            _send_order_template(order["phone_number"], shop_rules.ORDER_STATUS_TEMPLATE,
                                 [order.get("customer_name") or "there", oid, branch_name, "Paid and confirmed"],
                                 phone_number_id, token)

        send_owner_order_alert(order, branch, config, phone_number_id, token)

        _advance_flow_after_payment(order["project_id"], order["phone_number"], phone_number_id, token)

    return {"status": "ok"}


def _handle_paid_after_cancel(order: dict, config: dict):
    """A link paid after its order was cancelled: record the money, keep the
    order cancelled, and tell the owner to refund or fulfil."""
    from whatsapp import send_whatsapp_message
    from shop_branches import get_branch

    supabase.table("orders").update({"payment_status": "paid"}).eq("id", order["id"]).execute()
    sentry_sdk.capture_message(f"paid after cancel: order {order['id']}")
    phone_number_id, token = _wa_creds(order["project_id"])
    if not phone_number_id:
        return
    try:
        branch = get_branch(order["project_id"], order.get("branch_id")) if order.get("branch_id") else None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        branch = None
    config = {**(config or {}), "project_id": order["project_id"]}
    paid = {**order, "payment_status": "paid"}
    send_owner_order_alert(paid, branch, config, phone_number_id, token,
                           header="⚠️ *Paid after cancel — refund or fulfil*",
                           payment_note="paid after the order was cancelled - refund or fulfil")
    try:
        send_whatsapp_message(
            to=order["phone_number"],
            text=(f"We received your payment for order #{order['id'][:8].upper()}, which was cancelled. "
                  "The store will contact you about a refund."),
            phone_number_id=phone_number_id, token=token)
    except Exception as e:
        sentry_sdk.capture_exception(e)


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
        sentry_sdk.capture_exception(e)
        print(f"_advance_flow_after_payment error: {e}")


# ─────────────────────────────────────────────
# HELPER — called from flows.py
# ─────────────────────────────────────────────

PAYMENT_LINK_TTL_SECONDS = 5400
# Online offer holds outlive the link by 15 minutes (link 90 + 15).
OFFER_HOLD_MINUTES = 105


class OrderChangedError(Exception):
    """The order changed (edited, paid, cancelled) while its payment link
    was being created. The link is not recorded and must not be sent."""


def generate_razorpay_link(order: dict, config: dict, callback_url: Optional[str] = None,
                           regenerate_from: Optional[str] = None) -> Optional[str]:
    """The link's short URL, or None. Raises OrderChangedError (see
    create_payment_link)."""
    link = create_payment_link(order, config, callback_url=callback_url, regenerate_from=regenerate_from)
    return link["url"] if link else None


def create_payment_link(order: dict, config: dict, callback_url: Optional[str] = None,
                        regenerate_from: Optional[str] = None) -> Optional[dict]:
    """{"id", "url", "expires_at"} or None. Raises OrderChangedError when the
    order moved on meanwhile.

    regenerate_from: the expired link this one replaces (same order, still
    pending/link_sent with that payment_id). The old id stays in
    payment_link_ids so a late payment on it is still found by the webhook.

    Prefers the project's Razorpay Partner OAuth connection; falls back
    to the merchant's legacy manually-entered keys (or the global env-var
    pair) if no OAuth connection exists yet — this is what lets shops keep
    accepting payments uninterrupted while they migrate to OAuth."""
    project_id = config.get("project_id") or order.get("project_id")
    payload = {
        # round(), not int(): int() truncates, so a total like 19.99 became
        # 1998 paise (19.99 * 100 == 1998.9999999999998). The customer was
        # charged a paisa short, and the paid-amount check in the webhook
        # (which rounds) then read it as an underpayment and never marked the
        # order paid. About 1 in 22 totals was affected.
        "amount": int(round(float(order["total"]) * 100)),
        # FIX: was hardcoded "INR" regardless of the store's real
        # currency — harmless for every existing (INR-only) store, but
        # would have silently charged a Shopify-sourced multi-currency
        # catalog in the wrong currency. currency_code is a real
        # ISO-4217 code; shop_config.currency is just a display symbol.
        "currency": config.get("currency_code") or "INR",
        "description": f"Order #{order['id'][:8].upper()}",
        "customer": {"contact": f"+{order['phone_number']}"},
        "notify": {"sms": False, "email": False},
        "reminder_enable": False,
        "expire_by": int(time.time()) + PAYMENT_LINK_TTL_SECONDS,
    }
    expire_by = payload["expire_by"]
    if callback_url:
        payload["callback_url"] = callback_url
        payload["callback_method"] = "get"

    try:
        link = None

        has_oauth_connection = False
        if project_id:
            from razorpay_oauth import _razorpay_api_request
            try:
                res = _razorpay_api_request("POST", "/payment_links", project_id, json=payload)
                has_oauth_connection = True
                if res.status_code < 300:
                    link = res.json()
                else:
                    sentry_sdk.capture_message(f"Razorpay OAuth payment_link.create failed ({res.status_code}) for project {project_id}: {res.text[:300]}")
                    # Do NOT fall through to the legacy/global keys here. This
                    # project HAS an OAuth connection that simply failed (token
                    # expired, Razorpay 5xx). Falling back would create the
                    # payment link on whichever account the global env keys
                    # point at, so the customer's money would land in the wrong
                    # Razorpay account. Failing the link is the safe outcome.
                    return None
            except ValueError:
                pass  # no OAuth connection for this project — fall through to legacy keys

        if link is None and not has_oauth_connection:
            import razorpay
            # The global env-var pair is only a sane default for a merchant
            # who configured their OWN legacy keys. Without per-project keys
            # this would silently bill into our account, so require them.
            key_id = config.get("razorpay_key_id")
            key_secret = config.get("razorpay_key_secret")
            if not key_id or not key_secret:
                return None
            client = razorpay.Client(auth=(key_id, key_secret))
            link = client.payment_link.create(payload)

        # Only if the order is still exactly what the link was made for: an
        # "Add More" edit (or a merchant status change) landing in between
        # would otherwise leave a link for the old total on the new order.
        expires_at = _iso_z(datetime.fromtimestamp(expire_by, tz=timezone.utc))
        q = supabase.table("orders").update({
            "payment_id": link["id"],
            "payment_status": "link_sent",
            "payment_method": "online",
            "payment_link_url": link["short_url"],
            "payment_link_expires_at": expires_at,
            "payment_link_ids": list(order.get("payment_link_ids") or []) + [link["id"]],
            "placed_at": order.get("placed_at") or datetime.now(timezone.utc).isoformat(),
        }).eq("id", order["id"]).eq("status", "pending").eq("total", order["total"])
        if regenerate_from:
            q = q.eq("payment_status", "link_sent").eq("payment_id", regenerate_from)
        else:
            q = q.eq("payment_status", "unpaid")
        updated = q.execute()
        if not updated.data:
            # Never leave a payable link for an order it no longer matches.
            cancel_payment_link(project_id, link.get("id"), config)
            raise OrderChangedError(order["id"])

        return {"id": link["id"], "url": link["short_url"], "expires_at": expires_at}

    except OrderChangedError:
        raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"generate_razorpay_link error: {e}")
        return None


# ─────────────────────────────────────────────
# SHOPIFY ORDER WRITE-BACK — called from razorpay_webhook above
# ─────────────────────────────────────────────

def push_order_to_shopify(order_id: str):
    """Pushes a completed WhatsApp/Razorpay order into Shopify as an order
    record, purely for the merchant's unified sales reporting. This is
    Shopify's own documented "importing orders from an external system" use
    case for orderCreate — the actual payment already happened via Razorpay,
    not a live Shopify Checkout session, which is what keeps this compliant
    (see the Shopify integration plan for why a widget-based Razorpay
    checkout would NOT be compliant the same way).

    No-op if the project has no connected Shopify store, or if this order
    was already pushed — shopify_order_id doubles as the idempotency check,
    so a retried Razorpay webhook can never create a duplicate Shopify order."""
    order_res = supabase.table("orders").select("*").eq("id", order_id).maybe_single().execute()
    order = order_res.data if order_res else None
    if not order or order.get("shopify_order_id"):
        return

    integration_res = supabase.table("shopify_integrations").select("*").eq("project_id", order["project_id"]).maybe_single().execute()
    integration = integration_res.data if integration_res else None
    if not integration:
        return

    config_res = supabase.table("shop_config").select("currency_code").eq("project_id", order["project_id"]).maybe_single().execute()
    currency_code = ((config_res.data if config_res else None) or {}).get("currency_code") or "INR"

    line_items = []
    for item in (order.get("items") or []):
        variant_id = item.get("shopify_variant_id")
        if variant_id:
            line_items.append({"variantId": variant_id, "quantity": item["quantity"]})
        else:
            # Manually-added product (or an item placed before this column
            # existed) — record it as a custom, non-catalog line item
            # rather than dropping it, so the Shopify order total still
            # matches what the customer actually paid.
            line_items.append({
                "title": item.get("name", "Item"),
                "quantity": item["quantity"],
                "priceSet": {"shopMoney": {"amount": f"{item['price']:.2f}", "currencyCode": currency_code}},
            })

    # Line items alone only sum to the subtotal — without this, the order's
    # line-item total wouldn't reconcile with the transaction amount below
    # (which is the customer's actual total, GST included), leaving a
    # visibly mismatched-looking order in the merchant's Shopify admin.
    tax_lines = []
    if order.get("gst_amount"):
        tax_lines.append({
            "title": "GST",
            "priceSet": {"shopMoney": {"amount": f"{order['gst_amount']:.2f}", "currencyCode": currency_code}},
        })
    # Same reason for the delivery fee: the paid total includes it.
    shipping_lines = []
    delivery_fee = float(order.get("delivery_fee") or 0)
    if delivery_fee > 0:
        shipping_lines.append({
            "title": "Delivery",
            "priceSet": {"shopMoney": {"amount": f"{delivery_fee:.2f}", "currencyCode": currency_code}},
        })

    # Offer discount (v3). Tax lines already carry the discounted GST and the
    # transaction stays order.total. Check this shape on a dev store first.
    discount_code = None
    if float(order.get("discount_amount") or 0) > 0:
        items_discount = float(order.get("items_discount") or 0)
        if items_discount > 0:
            discount_code = {"itemFixedDiscountCode": {
                "code": order.get("offer_code") or "OFFER",
                "amountSet": {"shopMoney": {"amount": f"{items_discount:.2f}", "currencyCode": currency_code}},
            }}
        elif float(order.get("delivery_discount") or 0) > 0:
            discount_code = {"freeShippingDiscountCode": {"code": order.get("offer_code") or "FREEDELIVERY"}}

    # NOTE: this mutation shape follows Shopify's documented OrderCreateOrderInput
    # conventions as closely as I can verify without a live store to test
    # against — worth a quick check against Shopify's GraphQL schema explorer
    # on your dev store before this runs against a real merchant's data.
    mutation = """
        mutation orderCreate($order: OrderCreateOrderInput!) {
          orderCreate(order: $order) {
            order { id }
            userErrors { field message }
          }
        }
    """
    variables = {
        "order": {
            "currency": currency_code,
            "lineItems": line_items,
            "taxLines": tax_lines,
            **({"shippingLines": shipping_lines} if shipping_lines else {}),
            **({"discountCode": discount_code} if discount_code else {}),
            "financialStatus": "PAID",
            "transactions": [{
                "kind": "SALE",
                "status": "SUCCESS",
                "gateway": "Razorpay",
                "amountSet": {"shopMoney": {"amount": f"{order['total']:.2f}", "currencyCode": currency_code}},
            }],
        }
    }

    try:
        data = shopify_graphql(integration["shop_domain"], integration["access_token"], mutation, variables)
        result = data["orderCreate"]
        errors = result.get("userErrors") or []
        if errors:
            sentry_sdk.capture_message(f"Shopify orderCreate userErrors for order {order_id}: {errors}")
            return
        shopify_order_id = (result.get("order") or {}).get("id")
        if shopify_order_id:
            supabase.table("orders").update({"shopify_order_id": shopify_order_id}).eq("id", order_id).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"push_order_to_shopify error for order {order_id}: {e}")