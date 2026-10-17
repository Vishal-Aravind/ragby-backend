import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from clients import supabase
from webhook_dedup import already_processed
from auth import verify_token, require_project_access
from ratelimit import is_rate_limited, client_ip
from shopify_client import graphql as shopify_graphql
from config import RAZORPAY_WEBHOOK_SECRET
import phones
import shop_rules
import os
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
    "bot_can_assist, bot_can_order"
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
        "store_name,store_phone,gst_percent,currency,accent_color,delivery_types,terms_note,is_enabled"
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
        }
    else:
        data = dict(res.data)
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
        "customer_pincode, customer_lat, customer_lng, status, payment_status"
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
        # Matches submit_cart's edit rule, so the page can tell the shopper
        # up front instead of failing at checkout.
        "editable": o.get("status") == "pending" and o.get("payment_status") == "unpaid",
    }


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
        # Conditional on still being pending/unpaid, so a payment link sent
        # between our check and this write can't be undercut.
        updated = supabase.table("orders").update(row) \
            .eq("id", existing["id"]).eq("project_id", project_id) \
            .eq("status", "pending").eq("payment_status", "unpaid").execute()
        if not updated.data:
            raise shop_error(409, "order_not_editable", "This order is already being processed. Please place a new order.")
        order = updated.data[0]
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


def send_owner_order_alert(order: dict, branch: Optional[dict], config: dict, phone_number_id: str, token: str,
                           header: str = "🔔 *New Order Paid!*") -> None:
    """New-order alert to the branch phone (else store_phone). Best-effort.
    Free text, so it only reaches an owner inside the 24-hour window, as
    it always has."""
    from whatsapp import send_whatsapp_message
    from shop_branches import contact_phone

    owner_phone = contact_phone(branch, config)
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
    if float(order.get("gst_amount") or 0) > 0:
        money_parts.append(f"GST {currency}{float(order['gst_amount']):.2f}")
    if is_delivery:
        fee = float(order.get("delivery_fee") or 0)
        money_parts.append(f"Delivery {currency}{shop_rules.money(fee)}" if fee > 0 else "Delivery FREE")

    msg = f"{title}\n\nFrom: {who}\n\n" + "\n".join(lines) + "\n\n"
    msg += " · ".join(money_parts) + "\n"
    msg += f"Total: {currency}{float(order['total']):.2f}\n"
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
    if order.get("special_request"):
        msg += f"📝 {one_line(order['special_request'])}\n"
    msg += f"Order ID: #{order['id'][:8].upper()}"

    try:
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
        supabase.table("orders").update({
            "payment_status": "paid",
            "status": "confirmed",
        }).eq("id", order["id"]).execute()

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

        msg = f"✅ *Payment Confirmed!*\n\n"
        msg += f"Thank you! Your order has been confirmed.\n\n"
        msg += f"Order ID: #{order['id'][:8].upper()}\n"
        msg += f"Total: {currency}{float(order['total']):.2f}\n"
        msg += order_fulfilment_lines(order, branch, currency) + "\n"
        if contact:
            msg += f"📞 Contact: {phones.pretty(contact)}\n"
        msg += "\nWe'll keep you posted on WhatsApp! 🎉"

        send_whatsapp_message(to=order["phone_number"], text=msg, phone_number_id=phone_number_id, token=token)

        send_owner_order_alert(order, branch, config, phone_number_id, token)

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
        sentry_sdk.capture_exception(e)
        print(f"_advance_flow_after_payment error: {e}")


# ─────────────────────────────────────────────
# HELPER — called from flows.py
# ─────────────────────────────────────────────

class OrderChangedError(Exception):
    """The order changed (edited, paid, cancelled) while its payment link
    was being created. The link is not recorded and must not be sent."""


def generate_razorpay_link(order: dict, config: dict) -> Optional[str]:
    """Raises OrderChangedError when the order moved on meanwhile.

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
        "expire_by": int(time.time()) + 5400,
    }

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
        updated = supabase.table("orders").update({
            "payment_id": link["id"],
            "payment_status": "link_sent",
        }).eq("id", order["id"]).eq("status", "pending").eq("payment_status", "unpaid") \
            .eq("total", order["total"]).execute()
        if not updated.data:
            raise OrderChangedError(order["id"])

        return link["short_url"]

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