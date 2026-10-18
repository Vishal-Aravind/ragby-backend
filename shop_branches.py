"""Shop branches: per-branch menus, opening hours, pickup/delivery.

Dashboard CRUD for branches plus the public endpoints the customer page
uses to pick a branch, check it is open and get a server-side quote.

Every number that ends up in an order (subtotal, GST, delivery fee, total)
and every open/closed decision comes from price_order below, which is the
single code path for both /quote and /submit-cart. The browser only shows
what the server worked out.
"""
import math
from datetime import datetime, timezone
from typing import List, Optional, Union

import sentry_sdk
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

import phones
import shop_rules
from auth import verify_token, require_project_access
from clients import supabase
from config import SUPABASE_URL
from ratelimit import is_rate_limited, client_ip
from shop import (
    CartItem, MAX_CART_LINES, shop_error, _public_shop_guard, _assert_shop_enabled,
    _price_items, load_shop_config, checkout_mode,
)
from shop_identity import resolve_identity
from shop_offers import evaluate_offers

router = APIRouter()

MAX_BRANCHES_PER_PROJECT = 50
MAX_PINCODES = 500
_MONEY_MAX = 99_999_999          # numeric(10,2)


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class BranchUpsert(BaseModel):
    project_id: Optional[str] = Field(default=None, max_length=64)
    name: Optional[str] = Field(default=None, max_length=500)
    address: Optional[str] = Field(default=None, max_length=2000)
    phone: Optional[str] = Field(default=None, max_length=40)
    lat: Optional[float] = None
    lng: Optional[float] = None
    pincodes: Optional[List[Union[str, int]]] = Field(default=None, max_length=2000)
    hours: Optional[dict] = None
    timezone: Optional[str] = Field(default=None, max_length=64)
    is_active: Optional[bool] = None
    accepting_orders: Optional[bool] = None
    pickup_enabled: Optional[bool] = None
    delivery_enabled: Optional[bool] = None
    delivery_per_km: Optional[float] = None
    delivery_base_fee: Optional[float] = None
    free_delivery_above: Optional[float] = None
    max_delivery_km: Optional[float] = None
    min_order_value: Optional[float] = None
    prep_minutes: Optional[int] = None
    sort_order: Optional[int] = None
    make_default: Optional[bool] = None
    image_url: Optional[str] = Field(default=None, max_length=2000)


class CopyMenuBody(BaseModel):
    source_branch_id: str = Field(max_length=64)
    mode: Optional[str] = Field(default="append", max_length=16)


class BranchSearch(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None
    pincode: Optional[Union[str, int]] = None
    source: Optional[str] = Field(default=None, max_length=16)   # "gps" | "pin"; treated the same


class QuoteRequest(BaseModel):
    branch_id: Optional[str] = Field(default=None, max_length=64)
    catalog_id: Optional[str] = Field(default=None, max_length=64)
    items: List[CartItem] = Field(default_factory=list, max_length=MAX_CART_LINES)
    fulfilment: Optional[str] = Field(default="pickup", max_length=16)
    customer_lat: Optional[float] = None
    customer_lng: Optional[float] = None
    customer_pincode: Optional[Union[str, int]] = None
    customer_token: Optional[str] = Field(default=None, max_length=1024)
    coupon_code: Optional[str] = Field(default=None, max_length=64)


# ─────────────────────────────────────────────
# HELPERS (also used by shop.py and flows.py)
# ─────────────────────────────────────────────

def _bad(message: str):
    return shop_error(400, "invalid_request", message)


def ensure_default_branch(project_id: str) -> Optional[str]:
    """The project's default branch id, creating "Main" if there is none.
    The SQL function is idempotent (one default per project, enforced by a
    unique index), so concurrent callers can't create two."""
    res = supabase.rpc("shop_default_branch_id", {"p_project_id": project_id}).execute()
    return res.data if isinstance(res.data, str) else None


def get_branch(project_id: str, branch_id: str) -> Optional[dict]:
    """The branch row, only if it belongs to project_id."""
    if not branch_id or not shop_rules.UUID_RE.match(str(branch_id)):
        return None
    res = supabase.table("shop_branches").select("*") \
        .eq("id", branch_id).eq("project_id", project_id).limit(1).execute()
    return res.data[0] if res.data else None


def get_default_branch(project_id: str) -> Optional[dict]:
    res = supabase.table("shop_branches").select("*") \
        .eq("project_id", project_id).eq("is_default", True).limit(1).execute()
    if res.data:
        return res.data[0]
    branch_id = ensure_default_branch(project_id)
    return get_branch(project_id, branch_id) if branch_id else None


def public_default_branch(project_id: str, config: Optional[dict] = None) -> dict:
    """The default branch for anonymous endpoints. Unlike get_default_branch
    it only creates "Main" for a project that has set up a shop (a
    shop_config row): 404 for an unknown project id, 403 shop_disabled for
    a project with no shop, never a 500 from the insert's project FK."""
    res = supabase.table("shop_branches").select("*") \
        .eq("project_id", project_id).eq("is_default", True).limit(1).execute()
    if res.data:
        return res.data[0]
    if config is None:
        config = load_shop_config(project_id)
    if config:
        try:
            branch_id = ensure_default_branch(project_id)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            branch_id = None
        branch = get_branch(project_id, branch_id) if branch_id else None
        if branch:
            return branch
    project = supabase.table("projects").select("id").eq("id", project_id).limit(1).execute()
    if not project.data:
        raise shop_error(404, "branch_not_found", "This shop doesn't exist.")
    raise shop_error(403, "shop_disabled", "This shop isn't available right now.")


def resolve_branch(project_id: str, branch_id: Optional[str], public: bool = False) -> dict:
    """The named branch (404 if it isn't this project's) or the default.
    public=True: anonymous caller, see public_default_branch."""
    if branch_id:
        if not shop_rules.UUID_RE.match(str(branch_id)):
            raise _bad("Invalid branch id")
        branch = get_branch(project_id, branch_id)
    elif public:
        branch = public_default_branch(project_id)
    else:
        branch = get_default_branch(project_id)
    if not branch:
        raise shop_error(404, "branch_not_found", "This branch doesn't exist.")
    return branch


def active_catalog_ids(project_id: str, branch_id: str) -> list:
    res = supabase.table("catalogs").select("id") \
        .eq("project_id", project_id).eq("branch_id", branch_id).eq("is_active", True).execute()
    return [c["id"] for c in (res.data or [])]


def shop_is_enabled(config: dict) -> bool:
    # No config row is the pre-setup default, treated as enabled everywhere.
    return not (config and config.get("is_enabled") is False)


def status_for(branch: dict, config: dict, now_utc: Optional[datetime] = None) -> dict:
    return shop_rules.branch_status(branch, shop_is_enabled(config), now_utc or datetime.now(timezone.utc))


def contact_phone(branch: Optional[dict], config: dict) -> Optional[str]:
    """Digits of the number customers should call / the owner alert goes to:
    the branch phone, else the shop's store_phone."""
    if branch and branch.get("phone"):
        return branch["phone"]
    raw = (config or {}).get("store_phone") or ""
    if not raw.strip():
        return None
    project_id = (config or {}).get("project_id")
    region = phones.default_region_for_project(project_id) if project_id else phones.FALLBACK_REGION
    digits, _ = phones.normalize(raw, region)
    if digits:
        return digits
    fallback = "".join(ch for ch in raw if ch.isdigit())
    return fallback or None


def _err(code: str, message: str) -> dict:
    return {"code": code, "message": message}


_ITEM_MESSAGES = {
    "item_unavailable": "Some items in your cart are no longer available. Please remove them and try again.",
    "item_not_on_menu": "Some items in your cart aren't on this branch's menu. Please refresh and try again.",
}

# Which HTTP status each business-rule error becomes at submit-cart.
SUBMIT_ERROR_STATUS = {
    "shop_disabled": 403,
    "branch_inactive": 403,
    "branch_closed": 409,
    "branch_paused": 409,
}


def price_order(project_id: str, branch: dict, items: list, fulfilment: str,
                cust_lat=None, cust_lng=None, pincode=None, catalog_id: Optional[str] = None,
                config: Optional[dict] = None, now_utc: Optional[datetime] = None,
                coupon_code: Optional[str] = None, identity: Optional[dict] = None, ip: Optional[str] = None,
                offers_enabled: bool = False, exclude_order_id: Optional[str] = None,
                exclude_offer_ids=()) -> dict:
    """Server-side pricing and every order rule, in one place.

    Never raises for business rules: they come back in `errors`, in check
    order (shop, branch, open state, fulfilment, items, minimum, delivery).
    /quote returns them as-is; /submit-cart raises the first one.

    Offers (v3) apply only with offers_enabled=True; offer problems never
    go into `errors`, they are reported in `coupon` / `available_offers`.
    GST is charged on the discounted item value.
    """
    if config is None:
        config = load_shop_config(project_id)
    now = now_utc or datetime.now(timezone.utc)
    currency = config.get("currency") or "₹"
    gst_percent = float(config.get("gst_percent") or 0)
    status = status_for(branch, config, now)
    errors = []

    if not shop_is_enabled(config):
        errors.append(_err("shop_disabled", "This shop isn't taking orders right now."))
    if not branch.get("is_active", True):
        errors.append(_err("branch_inactive", "This branch isn't taking orders right now."))
    elif status["reason"] == "closed":
        errors.append(_err("branch_closed", status["label"]))
    elif status["reason"] == "paused":
        errors.append(_err("branch_paused", status["label"]))

    if fulfilment == "delivery" and not branch.get("delivery_enabled"):
        errors.append(_err("fulfilment_unavailable", "Delivery isn't available at this branch."))
    elif fulfilment == "pickup" and not branch.get("pickup_enabled", True):
        errors.append(_err("fulfilment_unavailable", "Pickup isn't available at this branch."))

    items_data, subtotal, price_changed, item_errors = [], 0.0, False, []
    if not items:
        errors.append(_err("cart_empty", "Your cart is empty."))
    else:
        items_data, subtotal, price_changed, item_errors = _price_items(project_id, branch["id"], catalog_id, items)
        for code in dict.fromkeys(item_errors):
            errors.append(_err(code, _ITEM_MESSAGES[code]))
        if not items_data and not item_errors:
            errors.append(_err("cart_empty", "Your cart is empty."))

    min_order = float(branch.get("min_order_value") or 0)
    if items_data and subtotal < min_order:
        short = min_order - subtotal
        errors.append(_err("below_minimum",
                           f"Minimum order is {currency}{shop_rules.money(min_order)} — "
                           f"add {currency}{shop_rules.money(short)} more."))

    delivery = shop_rules.quote_delivery(branch, subtotal, cust_lat, cust_lng, pincode)
    delivery_fee = 0
    distance = None
    if fulfilment == "delivery":
        distance = delivery.get("distance_km")
        if delivery["available"]:
            delivery_fee = delivery["fee"]
        elif branch.get("delivery_enabled"):
            reason = delivery["reason"]
            if reason == "delivery_off":
                msg = "Delivery isn't available at this branch."
            elif reason == "location_required" and cust_lat is not None and cust_lng is not None:
                msg = "This branch hasn't set its location yet."
            elif reason == "location_required":
                msg = "Set your delivery location on the map."
            else:
                msg = "This branch doesn't deliver to your location."
            errors.append(_err(reason, msg))

    ev = evaluate_offers(project_id, branch, fulfilment, subtotal if items_data else 0,
                         delivery_fee, identity or {}, coupon_code, ip, now,
                         exclude_order_id, exclude_offer_ids, enabled=offers_enabled, currency=currency)
    t = shop_rules.apply_offer_totals(subtotal, gst_percent, delivery_fee,
                                      ev["items_discount"], ev["delivery_discount"])

    free_above = branch.get("free_delivery_above")
    amount_to_free = None
    if free_above is not None and shop_rules.delivery_available(branch):
        amount_to_free = round(max(0.0, float(free_above) - subtotal), 2)

    return {
        "currency": currency,
        "items_data": items_data,
        "items": [{
            "product_id": it["product_id"], "name": it["name"], "price": it["price"],
            "quantity": it["quantity"], "line_total": round(it["price"] * it["quantity"], 2),
        } for it in items_data],
        "price_changed": price_changed,
        "subtotal": subtotal,
        "gst_percent": gst_percent,
        "gst_amount": t["gst_amount"],
        "delivery_fee": delivery_fee,
        "delivery_distance_km": distance,
        "total": t["total"],
        "taxable_subtotal": t["taxable_subtotal"],
        "items_discount": ev["items_discount"],
        "delivery_discount": ev["delivery_discount"],
        "discount_amount": round(ev["items_discount"] + ev["delivery_discount"], 2),
        "offer": ev["offer"],
        "coupon": ev["coupon"],
        "available_offers": ev["available_offers"],
        "min_order_value": min_order,
        "free_delivery_above": float(free_above) if free_above is not None else None,
        "amount_to_free_delivery": amount_to_free,
        "status": status,
        "delivery": {"available": delivery["available"], "reason": delivery["reason"]},
        "errors": errors,
    }


def raise_first_error(priced: dict):
    """/submit-cart: the first business-rule error as its HTTP error."""
    if priced["errors"]:
        e = priced["errors"][0]
        raise shop_error(SUBMIT_ERROR_STATUS.get(e["code"], 400), e["code"], e["message"])


# ─────────────────────────────────────────────
# BRANCH VALIDATION
# ─────────────────────────────────────────────

def _finite(name: str, value, lo: float, hi: float, nullable: bool, integer: bool = False):
    if value is None:
        if nullable:
            return None
        raise _bad(f"{name} is required")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise _bad(f"{name} must be a number")
    if not math.isfinite(v) or v < lo or v > hi:
        raise _bad(f"{name} must be between {shop_rules.money(lo)} and {shop_rules.money(hi)}")
    if integer:
        if v != int(v):
            raise _bad(f"{name} must be a whole number")
        return int(v)
    return round(v, 2)


def _branch_fields(data: dict, project_id: str, creating: bool) -> dict:
    """Cleaned DB columns for the keys present in `data`. 400 on anything
    the DB checks would reject, with a message the merchant can act on."""
    out = {}
    region = None

    def _region():
        nonlocal region
        if region is None:
            region = phones.default_region_for_project(project_id)
        return region

    if creating or "name" in data:
        try:
            name = shop_rules.clean_text(data.get("name"), 80)
        except ValueError:
            raise _bad("Branch name must be at most 80 characters")
        if not name:
            raise _bad("Branch name is required")
        out["name"] = name

    if "address" in data:
        try:
            out["address"] = shop_rules.clean_text(data.get("address"), 300)
        except ValueError:
            raise _bad("Address must be at most 300 characters")

    if "phone" in data:
        raw = data.get("phone")
        if raw is None or not str(raw).strip():
            out["phone"] = None
        else:
            digits, reason = phones.normalize(raw, _region())
            if not digits:
                raise _bad(f"Phone: {reason}")
            out["phone"] = digits

    if "lat" in data or "lng" in data:
        lat, lng = data.get("lat"), data.get("lng")
        if lat is None and lng is None:
            out["lat"], out["lng"] = None, None
        elif lat is None or lng is None:
            raise _bad("Send both latitude and longitude, or neither")
        else:
            try:
                out["lat"], out["lng"] = shop_rules.validate_latlng(lat, lng)
            except ValueError as e:
                raise _bad(str(e))

    if "pincodes" in data:
        cleaned, bad = [], []
        for raw in (data.get("pincodes") or []):
            pin = shop_rules.normalize_pincode(raw, _region())
            if pin is None:
                bad.append(str(raw))
            elif pin not in cleaned:
                cleaned.append(pin)
        if bad:
            raise _bad(f"Invalid pincode: {', '.join(bad[:5])}")
        if len(cleaned) > MAX_PINCODES:
            raise _bad(f"At most {MAX_PINCODES} pincodes per branch")
        out["pincodes"] = cleaned

    if "hours" in data:
        try:
            out["hours"] = shop_rules.validate_hours(data.get("hours"))
        except ValueError as e:
            raise _bad(str(e))

    if "timezone" in data:
        try:
            out["timezone"] = shop_rules.validate_timezone(data.get("timezone"))
        except ValueError as e:
            raise _bad(str(e))

    for key in ("is_active", "accepting_orders", "pickup_enabled", "delivery_enabled"):
        if key in data:
            if not isinstance(data[key], bool):
                raise _bad(f"{key} must be true or false")
            out[key] = data[key]

    if "image_url" in data:
        url = data.get("image_url")
        if url is None or (isinstance(url, str) and not url.strip()):
            out["image_url"] = None
        elif not shop_rules.valid_media_url(str(url).strip(), SUPABASE_URL):
            raise shop_error(400, "invalid_image_url", "Upload the image from the dashboard (PNG, JPG or WEBP).")
        else:
            out["image_url"] = str(url).strip()

    limits = {
        # key: (label, lo, hi, nullable, integer)
        "delivery_per_km": ("Fee per km", 0, 10000, False, False),
        "delivery_base_fee": ("Base delivery fee", 0, 100000, False, False),
        "free_delivery_above": ("Free delivery above", 0, _MONEY_MAX, True, False),
        "max_delivery_km": ("Max delivery distance", 0.01, 100, True, False),
        "min_order_value": ("Minimum order value", 0, _MONEY_MAX, False, False),
        "prep_minutes": ("Prep time", 0, 600, True, True),
        "sort_order": ("Sort order", -1_000_000, 1_000_000, False, True),
    }
    for key, (label, lo, hi, nullable, integer) in limits.items():
        if key in data:
            value = data[key]
            # Money fields that can't be null: an emptied input means 0.
            if value is None and key in ("delivery_per_km", "delivery_base_fee", "min_order_value", "sort_order"):
                value = 0
            out[key] = _finite(label, value, lo, hi, nullable, integer)
    return out


def _decorate(branch: dict, config: dict, catalog_count: int, now_utc: datetime) -> dict:
    """Dashboard view: the full row plus status, counts and warnings."""
    has_location = branch.get("lat") is not None and branch.get("lng") is not None
    configured = has_location and shop_rules.delivery_area_set(branch)
    warnings = []
    if not has_location:
        warnings.append("no_location")
    if branch.get("delivery_enabled") and not configured:
        warnings.append("delivery_not_configured")
    if not branch.get("phone"):
        warnings.append("no_phone")
    if catalog_count == 0:
        warnings.append("no_menu")
    return {
        **branch,
        "catalog_count": catalog_count,
        "status": status_for(branch, config, now_utc),
        "delivery_configured": configured,
        "warnings": warnings,
    }


def _load_branch_or_404(branch_id: str) -> dict:
    if not shop_rules.UUID_RE.match(branch_id or ""):
        raise _bad("Invalid branch id")
    res = supabase.table("shop_branches").select("*").eq("id", branch_id).limit(1).execute()
    if not res.data:
        raise shop_error(404, "branch_not_found", "This branch doesn't exist.")
    return res.data[0]


def _reload(branch_id: str) -> dict:
    res = supabase.table("shop_branches").select("*").eq("id", branch_id).limit(1).execute()
    if not res.data:
        raise shop_error(404, "branch_not_found", "This branch doesn't exist.")
    return res.data[0]


def _with_status(branch: dict, project_id: str) -> dict:
    """POST/PUT responses: the full row plus the same extras GET adds, so
    the dashboard can drop the returned object straight into its list."""
    config = load_shop_config(project_id)
    count_res = supabase.table("catalogs").select("id").eq("project_id", project_id) \
        .eq("branch_id", branch["id"]).execute()
    return _decorate(branch, config, len(count_res.data or []), datetime.now(timezone.utc))


# ─────────────────────────────────────────────
# DASHBOARD ENDPOINTS
# ─────────────────────────────────────────────

@router.get("/shop/branches")
def list_branches(project_id: str, user=Depends(verify_token)):
    if not shop_rules.UUID_RE.match(project_id or ""):
        raise _bad("Invalid project id")
    require_project_access(user.id, project_id, tab="shop")
    ensure_default_branch(project_id)

    res = supabase.table("shop_branches").select("*").eq("project_id", project_id) \
        .order("sort_order", desc=False).order("created_at", desc=False).execute()
    branches = res.data or []
    cats = supabase.table("catalogs").select("branch_id").eq("project_id", project_id).execute()
    counts = {}
    for c in (cats.data or []):
        counts[c.get("branch_id")] = counts.get(c.get("branch_id"), 0) + 1

    config = load_shop_config(project_id)
    now = datetime.now(timezone.utc)
    return [_decorate(b, config, counts.get(b["id"], 0), now) for b in branches]


@router.post("/shop/branches")
def create_branch(body: BranchUpsert, user=Depends(verify_token)):
    project_id = body.project_id or ""
    if not shop_rules.UUID_RE.match(project_id):
        raise _bad("Invalid project id")
    require_project_access(user.id, project_id, tab="shop", min_role="admin")

    data = body.model_dump(exclude_unset=True)
    data.pop("project_id", None)
    data.pop("make_default", None)   # a new branch never starts as the default
    fields = _branch_fields(data, project_id, creating=True)

    # Main first, so the new branch is never accidentally the only (and so
    # implicitly default-less) branch.
    default_id = ensure_default_branch(project_id)
    existing = supabase.table("shop_branches").select("id, timezone, is_default").eq("project_id", project_id).execute()
    rows = existing.data or []
    if len(rows) >= MAX_BRANCHES_PER_PROJECT:
        raise _bad(f"A shop can have at most {MAX_BRANCHES_PER_PROJECT} branches")
    if "timezone" not in fields:
        default_row = next((r for r in rows if r["id"] == default_id or r.get("is_default")), None)
        fields["timezone"] = (default_row or {}).get("timezone") or shop_rules.DEFAULT_TIMEZONE

    inserted = supabase.table("shop_branches").insert({"project_id": project_id, **fields}).execute()
    if not inserted.data:
        raise shop_error(500, "invalid_request", "Could not create the branch. Please try again.")
    return _with_status(inserted.data[0], project_id)


@router.put("/shop/branches/{branch_id}")
def update_branch(branch_id: str, body: BranchUpsert, user=Depends(verify_token)):
    branch = _load_branch_or_404(branch_id)
    project_id = branch["project_id"]
    data = body.model_dump(exclude_unset=True)
    data.pop("project_id", None)

    # Pausing orders is a day-to-day staff action, so agents with the shop
    # tab can do it; anything else about a branch is an admin setting.
    if set(data.keys()) == {"accepting_orders"}:
        require_project_access(user.id, project_id, tab="shop")
    else:
        require_project_access(user.id, project_id, tab="shop", min_role="admin")

    make_default = data.pop("make_default", None) is True
    fields = _branch_fields(data, project_id, creating=False)
    if fields:
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        supabase.table("shop_branches").update(fields).eq("id", branch_id).eq("project_id", project_id).execute()

    if make_default and not branch.get("is_default"):
        # Clear the old default first: the unique index allows one per project.
        now_iso = datetime.now(timezone.utc).isoformat()
        supabase.table("shop_branches").update({"is_default": False, "updated_at": now_iso}) \
            .eq("project_id", project_id).eq("is_default", True).execute()
        supabase.table("shop_branches").update({"is_default": True, "updated_at": now_iso}) \
            .eq("id", branch_id).eq("project_id", project_id).execute()

    return _with_status(_reload(branch_id), project_id)


@router.delete("/shop/branches/{branch_id}")
def delete_branch(branch_id: str, user=Depends(verify_token)):
    branch = _load_branch_or_404(branch_id)
    project_id = branch["project_id"]
    require_project_access(user.id, project_id, tab="shop", min_role="admin")

    if branch.get("is_default"):
        raise shop_error(409, "cannot_delete_default", "The default branch can't be deleted. Make another branch the default first.")
    orders = supabase.table("orders").select("id").eq("project_id", project_id).eq("branch_id", branch_id).limit(1).execute()
    if orders.data:
        raise shop_error(409, "branch_has_orders", "This branch has orders. Turn it off instead.")

    # products.catalog_id doesn't cascade, so products go first; the branch
    # delete then cascades to its catalogs.
    cats = supabase.table("catalogs").select("id").eq("project_id", project_id).eq("branch_id", branch_id).execute()
    cat_ids = [c["id"] for c in (cats.data or [])]
    if cat_ids:
        supabase.table("products").delete().eq("project_id", project_id).in_("catalog_id", cat_ids).execute()
    supabase.table("shop_branches").delete().eq("id", branch_id).eq("project_id", project_id).execute()
    return {"status": "deleted"}


@router.post("/shop/branches/{target_branch_id}/copy-menu")
def copy_menu(target_branch_id: str, body: CopyMenuBody, user=Depends(verify_token)):
    target = _load_branch_or_404(target_branch_id)
    project_id = target["project_id"]
    require_project_access(user.id, project_id, tab="shop", min_role="admin")

    if not shop_rules.UUID_RE.match(body.source_branch_id or ""):
        raise _bad("Invalid source branch id")
    mode = body.mode or "append"
    if mode not in ("append", "replace"):
        raise _bad("mode must be 'append' or 'replace'")
    if is_rate_limited(f"copy-menu:{project_id}", 10, 3600):
        raise shop_error(429, "rate_limited", "Too many menu copies — please try again in a while.")

    try:
        res = supabase.rpc("copy_shop_menu", {
            "p_project_id": project_id,
            "p_source_branch": body.source_branch_id,
            "p_target_branch": target_branch_id,
            "p_replace": mode == "replace",
        }).execute()
    except Exception as e:
        text = str(e)
        if "same_branch" in text:
            raise shop_error(400, "same_branch", "Pick a different branch to copy from.")
        if "branch_not_found" in text:
            raise shop_error(404, "branch_not_found", "That branch doesn't exist.")
        if "too_large" in text:
            raise shop_error(400, "menu_too_large", "That menu is too large to copy (over 100 catalogs or 5000 items).")
        sentry_sdk.capture_exception(e)
        raise shop_error(500, "invalid_request", "Could not copy the menu. Please try again.")

    data = res.data if isinstance(res.data, dict) else {}
    return {
        "catalogs_copied": int(data.get("catalogs_copied") or 0),
        "products_copied": int(data.get("products_copied") or 0),
        "mode": mode,
    }


# ─────────────────────────────────────────────
# PUBLIC ENDPOINTS
# ─────────────────────────────────────────────

def _delivers_here(branch: dict, pincode: Optional[str], distance: Optional[float]):
    """True / False when we can tell, None when we can't."""
    if not shop_rules.delivery_available(branch):
        return False
    pins = branch.get("pincodes") or []
    max_km = branch.get("max_delivery_km")
    has_radius = max_km is not None and float(max_km) > 0
    checks, unknown = [], False
    if pins:
        if pincode is not None:
            checks.append(pincode in pins)
        else:
            unknown = True
    if has_radius:
        if distance is not None:
            checks.append(distance <= float(max_km))
        else:
            unknown = True
    if any(checks):
        return True
    return None if unknown else False


@router.post("/public/shop/{project_id}/branches")
def public_branches(project_id: str, body: BranchSearch, request: Request):
    # Plain def: may call Nominatim (blocking, up to ~8 s).
    _public_shop_guard(project_id, request, "shop-branches", limit=30)
    _assert_shop_enabled(project_id)

    region = phones.default_region_for_project(project_id)
    pincode = None
    if body.pincode is not None and str(body.pincode).strip():
        pincode = shop_rules.normalize_pincode(body.pincode, region)
        if pincode is None:
            raise _bad("That doesn't look like a valid pincode.")

    origin = None
    if body.lat is not None or body.lng is not None:
        if body.lat is None or body.lng is None:
            raise _bad("Send both latitude and longitude.")
        try:
            lat, lng = shop_rules.validate_latlng(body.lat, body.lng)
        except ValueError as e:
            raise _bad(str(e))
        origin = {"lat": lat, "lng": lng, "source": "gps"}
    elif pincode:
        coords = None
        try:
            from geocode import pincode_to_latlng
            coords = pincode_to_latlng(region, pincode, ip=client_ip(request))
        except Exception as e:
            sentry_sdk.capture_exception(e)
        if coords:
            origin = {"lat": coords[0], "lng": coords[1], "source": "pincode"}

    config = load_shop_config(project_id)
    public_default_branch(project_id, config)
    res = supabase.table("shop_branches").select("*").eq("project_id", project_id) \
        .eq("is_active", True).execute()
    now = datetime.now(timezone.utc)

    out = []
    for b in (res.data or []):
        distance = None
        if origin and b.get("lat") is not None and b.get("lng") is not None:
            distance = round(shop_rules.haversine_km(origin["lat"], origin["lng"], b["lat"], b["lng"]), 2)
        out.append({
            "id": b["id"],
            "name": b["name"],
            "address": b.get("address"),
            "phone": phones.pretty(b["phone"]) if b.get("phone") else None,
            "lat": b.get("lat"),
            "lng": b.get("lng"),
            "distance_km": distance,
            "is_nearest": False,
            "delivers_here": _delivers_here(b, pincode, distance),
            "pickup_enabled": bool(b.get("pickup_enabled")),
            "delivery_available": shop_rules.delivery_available(b),
            "min_order_value": float(b.get("min_order_value") or 0),
            "free_delivery_above": float(b["free_delivery_above"]) if b.get("free_delivery_above") is not None else None,
            "prep_minutes": b.get("prep_minutes"),
            "image_url": b.get("image_url"),
            "status": status_for(b, config, now),
            "_sort": b.get("sort_order") or 0,
        })

    if any(o["distance_km"] is not None for o in out):
        out.sort(key=lambda o: (o["distance_km"] is None, o["distance_km"] or 0, o["_sort"]))
    else:
        out.sort(key=lambda o: (o["delivers_here"] is not True, o["_sort"]))

    nearest_id = None
    with_distance = [o for o in out if o["distance_km"] is not None]
    if with_distance:
        nearest = min(with_distance, key=lambda o: o["distance_km"])
        nearest["is_nearest"] = True
        nearest_id = nearest["id"]
    for o in out:
        o.pop("_sort", None)

    return {"origin": origin, "nearest_branch_id": nearest_id, "branches": out}


@router.get("/public/shop/{project_id}/branches/{branch_id}/status")
def public_branch_status(project_id: str, branch_id: str, request: Request):
    _public_shop_guard(project_id, request, "shop-status", limit=60)
    _assert_shop_enabled(project_id)
    branch = resolve_branch(project_id, branch_id, public=True)
    # An inactive branch reports reason "inactive" rather than 404, so a page
    # left open on it shows "not available" instead of breaking.
    return {"branch_id": branch["id"], **status_for(branch, load_shop_config(project_id))}


def parse_customer_location(lat, lng, pincode, region: str):
    """(lat, lng, pincode) validated; 400 on a malformed value. Missing
    values stay None (the quote then reports location_required)."""
    if lat is not None or lng is not None:
        if lat is None or lng is None:
            raise _bad("Send both latitude and longitude.")
        try:
            lat, lng = shop_rules.validate_latlng(lat, lng)
        except ValueError as e:
            raise _bad(str(e))
    pin = None
    if pincode is not None and str(pincode).strip():
        pin = shop_rules.normalize_pincode(pincode, region)
        if pin is None:
            raise _bad("That doesn't look like a valid pincode.")
    return lat, lng, pin


@router.post("/public/shop/{project_id}/quote")
def public_quote(project_id: str, body: QuoteRequest, request: Request):
    _public_shop_guard(project_id, request, "shop-quote", limit=30)
    _assert_shop_enabled(project_id)

    fulfilment = body.fulfilment or "pickup"
    if fulfilment not in ("pickup", "delivery"):
        raise _bad("fulfilment must be 'pickup' or 'delivery'")
    branch = resolve_branch(project_id, body.branch_id, public=True)
    catalog_id = None
    if body.catalog_id:
        if not shop_rules.UUID_RE.match(body.catalog_id):
            raise _bad("Invalid catalog id")
        catalog_id = body.catalog_id

    # Parsed for pickup too, so the quote can still say whether delivery
    # would be possible from where the customer is.
    lat, lng, pin = parse_customer_location(body.customer_lat, body.customer_lng, body.customer_pincode,
                                            phones.default_region_for_project(project_id))

    # Legacy phone not accepted here: only a signed token is an identity.
    identity = resolve_identity(project_id, body.customer_token, None)
    config = load_shop_config(project_id)
    priced, mode = price_with_mode(project_id, branch, body.items, fulfilment, lat, lng, pin,
                                   catalog_id=catalog_id, config=config, identity=identity,
                                   coupon_code=body.coupon_code, ip=client_ip(request))
    return quote_view(priced, mode, identity["verified"])


def price_with_mode(project_id: str, branch: dict, items: list, fulfilment: str, lat, lng, pin,
                    catalog_id: Optional[str], config: dict, identity: dict, coupon_code: Optional[str],
                    ip: Optional[str], exclude_order_id: Optional[str] = None, exclude_offer_ids=()):
    """(priced, checkout_mode), shared by /quote and /place-order so both
    always agree. Offers are on unless the mode is confirm_in_whatsapp. A
    free order (total 0) needs a verified token like an offline one; an
    unverified free checkout falls back to confirm_in_whatsapp, re-priced
    with offers off (that flow never applies offers)."""
    mode = checkout_mode(project_id, config, identity.get("verified"))
    kw = dict(catalog_id=catalog_id, config=config, coupon_code=coupon_code, identity=identity, ip=ip,
              exclude_order_id=exclude_order_id, exclude_offer_ids=exclude_offer_ids)
    priced = price_order(project_id, branch, items, fulfilment, lat, lng, pin,
                         offers_enabled=mode != "confirm_in_whatsapp", **kw)
    if not priced["errors"] and priced["total"] == 0 and mode == "pay_online":
        if identity.get("verified"):
            mode = "place_offline"
        else:
            mode = "confirm_in_whatsapp"
            priced = price_order(project_id, branch, items, fulfilment, lat, lng, pin, offers_enabled=False, **kw)
    return priced, mode


def quote_view(priced: dict, mode: str, verified: bool) -> dict:
    """The /quote response shape (also the `quote` in place-order 409s)."""
    return {
        "ok": not priced["errors"],
        "checkout_mode": mode,
        "verified": verified,
        "currency": priced["currency"],
        "items": priced["items"],
        "subtotal": priced["subtotal"],
        "gst_percent": priced["gst_percent"],
        "gst_amount": priced["gst_amount"],
        "delivery_fee": priced["delivery_fee"],
        "delivery_distance_km": priced["delivery_distance_km"],
        "total": priced["total"],
        "min_order_value": priced["min_order_value"],
        "free_delivery_above": priced["free_delivery_above"],
        "amount_to_free_delivery": priced["amount_to_free_delivery"],
        "status": priced["status"],
        "delivery": priced["delivery"],
        "errors": priced["errors"],
        "taxable_subtotal": priced["taxable_subtotal"],
        "items_discount": priced["items_discount"],
        "delivery_discount": priced["delivery_discount"],
        "discount_amount": priced["discount_amount"],
        "offer": priced["offer"],
        "coupon": priced["coupon"],
        "available_offers": priced["available_offers"],
    }
