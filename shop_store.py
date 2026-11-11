"""The shop's store: opening hours, pickup/delivery, minimum order.

Every shop is one store. Its settings live on shop_config (location,
hours, delivery area and fees, minimum order, ready-in time, pause), saved
through PUT /shop-config and validated by store_fields below.

Every number that ends up in an order (subtotal, GST, delivery fee, total)
and every open/closed decision comes from price_order below, which is the
single code path for both /quote and the order endpoints. The browser only
shows what the server worked out.
"""
import math
from datetime import datetime, timezone
from typing import List, Optional, Union

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

import phones
import shop_rules
from clients import supabase
from ratelimit import client_ip
from shop import (
    CartItem, MAX_CART_LINES, shop_error, _public_shop_guard, _assert_shop_enabled,
    _price_items, load_shop_config, checkout_mode,
)
from shop_identity import resolve_identity
from shop_offers import evaluate_offers

router = APIRouter()

MAX_PINCODES = 500
_MONEY_MAX = 99_999_999          # numeric(10,2)


class QuoteRequest(BaseModel):
    catalog_id: Optional[str] = Field(default=None, max_length=64)
    items: List[CartItem] = Field(default_factory=list, max_length=MAX_CART_LINES)
    fulfilment: Optional[str] = Field(default="pickup", max_length=16)
    customer_lat: Optional[float] = None
    customer_lng: Optional[float] = None
    customer_pincode: Optional[Union[str, int]] = None
    customer_token: Optional[str] = Field(default=None, max_length=1024)
    coupon_code: Optional[str] = Field(default=None, max_length=64)


# ─────────────────────────────────────────────
# HELPERS (also used by shop.py, flows.py and shop_invoice.py)
# ─────────────────────────────────────────────

def _bad(message: str):
    return shop_error(400, "invalid_request", message)


def shop_is_enabled(config: dict) -> bool:
    # No config row is the pre-setup default, treated as enabled everywhere.
    return not (config and config.get("is_enabled") is False)


def contact_phone(config: dict) -> Optional[str]:
    """Digits of the shop's phone: the number customers call and the one
    new-order alerts go to."""
    raw = (config or {}).get("store_phone") or ""
    if not str(raw).strip():
        return None
    project_id = (config or {}).get("project_id")
    region = phones.default_region_for_project(project_id) if project_id else phones.FALLBACK_REGION
    digits, _ = phones.normalize(raw, region)
    if digits:
        return digits
    fallback = "".join(ch for ch in str(raw) if ch.isdigit())
    return fallback or None


def store_view(config: dict) -> dict:
    """The shop settings as the store the order rules read: name, phone
    digits and the pickup/delivery/hours columns (defaults for a shop that
    has not saved them yet)."""
    config = config or {}
    return {
        **config,
        "name": (config.get("store_name") or "").strip(),
        "phone": contact_phone(config),
        "timezone": config.get("timezone") or shop_rules.DEFAULT_TIMEZONE,
        "accepting_orders": config.get("accepting_orders", True) is not False,
        "pickup_enabled": config.get("pickup_enabled", True) is not False,
        "delivery_enabled": bool(config.get("delivery_enabled")),
        "pincodes": config.get("pincodes") or [],
    }


def get_store(project_id: str, config: Optional[dict] = None) -> dict:
    if config is None:
        config = load_shop_config(project_id)
    return store_view({**config, "project_id": project_id})


def public_store(project_id: str, config: Optional[dict] = None) -> dict:
    """The store for anonymous endpoints: 404 for an unknown project, 403
    shop_disabled for one that has not set up a shop."""
    if config is None:
        config = load_shop_config(project_id)
    if config:
        return get_store(project_id, config)
    project = supabase.table("projects").select("id").eq("id", project_id).limit(1).execute()
    if not project.data:
        raise shop_error(404, "shop_not_found", "This shop doesn't exist.")
    raise shop_error(403, "shop_disabled", "This shop isn't available right now.")


def active_catalog_ids(project_id: str) -> list:
    res = supabase.table("catalogs").select("id") \
        .eq("project_id", project_id).eq("is_active", True).execute()
    return [c["id"] for c in (res.data or [])]


def status_for(store: dict, config: dict, now_utc: Optional[datetime] = None) -> dict:
    return shop_rules.store_status(store, shop_is_enabled(config), now_utc or datetime.now(timezone.utc))


def _err(code: str, message: str) -> dict:
    return {"code": code, "message": message}


_ITEM_MESSAGES = {
    "item_unavailable": "Some items in your cart are no longer available. Please remove them and try again.",
    "item_not_on_menu": "Some items in your cart aren't available any more. Please refresh and try again.",
}

# Which HTTP status each business-rule error becomes when placing an order.
SUBMIT_ERROR_STATUS = {
    "shop_disabled": 403,
    "shop_closed": 409,
    "shop_paused": 409,
}


def price_order(project_id: str, store: dict, items: list, fulfilment: str,
                cust_lat=None, cust_lng=None, pincode=None, catalog_id: Optional[str] = None,
                config: Optional[dict] = None, now_utc: Optional[datetime] = None,
                coupon_code: Optional[str] = None, identity: Optional[dict] = None, ip: Optional[str] = None,
                offers_enabled: bool = False, exclude_order_id: Optional[str] = None,
                exclude_offer_ids=()) -> dict:
    """Server-side pricing and every order rule, in one place.

    Never raises for business rules: they come back in `errors`, in check
    order (shop, open state, fulfilment, items, minimum, delivery). /quote
    returns them as-is; placing an order raises the first one.

    Offers apply only with offers_enabled=True; offer problems never go
    into `errors`, they are reported in `coupon` / `available_offers`.
    GST is charged on the discounted item value.
    """
    if config is None:
        config = load_shop_config(project_id)
    now = now_utc or datetime.now(timezone.utc)
    currency = config.get("currency") or "₹"
    gst_percent = float(config.get("gst_percent") or 0)
    status = status_for(store, config, now)
    errors = []

    if not shop_is_enabled(config):
        errors.append(_err("shop_disabled", "This shop isn't taking orders right now."))
    elif status["reason"] == "closed":
        errors.append(_err("shop_closed", status["label"]))
    elif status["reason"] == "paused":
        errors.append(_err("shop_paused", status["label"]))

    if fulfilment == "delivery" and not store.get("delivery_enabled"):
        errors.append(_err("fulfilment_unavailable", "Delivery isn't available."))
    elif fulfilment == "pickup" and not store.get("pickup_enabled", True):
        errors.append(_err("fulfilment_unavailable", "Pickup isn't available."))

    items_data, subtotal, price_changed, item_errors = [], 0.0, False, []
    if not items:
        errors.append(_err("cart_empty", "Your cart is empty."))
    else:
        items_data, subtotal, price_changed, item_errors = _price_items(project_id, catalog_id, items)
        for code in dict.fromkeys(item_errors):
            errors.append(_err(code, _ITEM_MESSAGES[code]))
        if not items_data and not item_errors:
            errors.append(_err("cart_empty", "Your cart is empty."))

    min_order = float(store.get("min_order_value") or 0)
    if items_data and subtotal < min_order:
        short = min_order - subtotal
        errors.append(_err("below_minimum",
                           f"Minimum order is {currency}{shop_rules.money(min_order)} — "
                           f"add {currency}{shop_rules.money(short)} more."))

    delivery = shop_rules.quote_delivery(store, subtotal, cust_lat, cust_lng, pincode)
    delivery_fee = 0
    distance = None
    if fulfilment == "delivery":
        distance = delivery.get("distance_km")
        if delivery["available"]:
            delivery_fee = delivery["fee"]
        elif store.get("delivery_enabled"):
            reason = delivery["reason"]
            if reason == "delivery_off":
                msg = "Delivery isn't available."
            elif reason == "location_required" and cust_lat is not None and cust_lng is not None:
                msg = "This shop hasn't set its location yet."
            elif reason == "location_required":
                msg = "Set your delivery location on the map."
            else:
                msg = "This shop doesn't deliver to your location."
            errors.append(_err(reason, msg))

    ev = evaluate_offers(project_id, store, fulfilment, subtotal if items_data else 0,
                         delivery_fee, identity or {}, coupon_code, ip, now,
                         exclude_order_id, exclude_offer_ids, enabled=offers_enabled, currency=currency)
    # GST per item rate (an item without its own rate uses the shop rate).
    t = shop_rules.compute_totals(
        [{"line_total": round(it["price"] * it["quantity"], 2), "gst_percent": it.get("gst_percent")}
         for it in items_data],
        gst_percent, delivery_fee, ev["items_discount"], ev["delivery_discount"])
    for it, rate in zip(items_data, t["line_rates"]):
        it["gst_percent"] = rate          # saved with the order: the rate actually charged

    free_above = store.get("free_delivery_above")
    amount_to_free = None
    if free_above is not None and shop_rules.delivery_available(store):
        amount_to_free = round(max(0.0, float(free_above) - subtotal), 2)

    return {
        "currency": currency,
        "items_data": items_data,
        "items": [{
            "product_id": it["product_id"], "name": it["name"], "price": it["price"],
            "quantity": it["quantity"], "line_total": round(it["price"] * it["quantity"], 2),
            "gst_percent": it.get("gst_percent"),
        } for it in items_data],
        "price_changed": price_changed,
        "subtotal": subtotal,
        "gst_percent": gst_percent,
        "gst_breakdown": t["gst_breakdown"],
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
    """Placing an order: the first business-rule error as its HTTP error."""
    if priced["errors"]:
        e = priced["errors"][0]
        raise shop_error(SUBMIT_ERROR_STATUS.get(e["code"], 400), e["code"], e["message"])


# ─────────────────────────────────────────────
# STORE SETTINGS VALIDATION (PUT /shop-config)
# ─────────────────────────────────────────────

STORE_KEYS = (
    "address", "lat", "lng", "pincodes", "hours", "timezone", "accepting_orders",
    "pickup_enabled", "delivery_enabled", "delivery_per_km", "delivery_base_fee",
    "free_delivery_above", "max_delivery_km", "min_order_value", "prep_minutes",
)


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


def store_fields(data: dict, project_id: str) -> dict:
    """Cleaned shop_config columns for the store keys present in `data`.
    400 on anything the DB checks would reject, with a message the merchant
    can act on."""
    out = {}
    region = None

    def _region():
        nonlocal region
        if region is None:
            region = phones.default_region_for_project(project_id)
        return region

    if "address" in data:
        try:
            out["address"] = shop_rules.clean_text(data.get("address"), 300)
        except ValueError:
            raise _bad("Address must be at most 300 characters")

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
            raise _bad(f"At most {MAX_PINCODES} pincodes")
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

    for key in ("accepting_orders", "pickup_enabled", "delivery_enabled"):
        if key in data:
            if not isinstance(data[key], bool):
                raise _bad(f"{key} must be true or false")
            out[key] = data[key]

    limits = {
        # key: (label, lo, hi, nullable, integer)
        "delivery_per_km": ("Fee per km", 0, 10000, False, False),
        "delivery_base_fee": ("Base delivery fee", 0, 100000, False, False),
        "free_delivery_above": ("Free delivery above", 0, _MONEY_MAX, True, False),
        "max_delivery_km": ("Max delivery distance", 0.01, 100, True, False),
        "min_order_value": ("Minimum order value", 0, _MONEY_MAX, False, False),
        "prep_minutes": ("Ready-in time", 0, 600, True, True),
    }
    for key, (label, lo, hi, nullable, integer) in limits.items():
        if key in data:
            value = data[key]
            # Money fields that can't be null: an emptied input means 0.
            if value is None and key in ("delivery_per_km", "delivery_base_fee", "min_order_value"):
                value = 0
            out[key] = _finite(label, value, lo, hi, nullable, integer)
    return out


def store_warnings(store: dict) -> list:
    """Setup problems shown in the dashboard."""
    has_location = store.get("lat") is not None and store.get("lng") is not None
    warnings = []
    if store.get("delivery_enabled") and not (shop_rules.delivery_area_set(store)
                                              and (has_location or not store.get("max_delivery_km"))):
        warnings.append("delivery_not_configured")
    if not store.get("phone"):
        warnings.append("no_phone")
    return warnings


# ─────────────────────────────────────────────
# PUBLIC ENDPOINTS
# ─────────────────────────────────────────────

def public_store_info(store: dict, config: dict) -> dict:
    """What the customer page shows about the store."""
    return {
        "address": store.get("address"),
        "phone": phones.pretty(store["phone"]) if store.get("phone") else None,
        "lat": store.get("lat"),
        "lng": store.get("lng"),
        "pickup_enabled": bool(store.get("pickup_enabled")),
        "delivery_available": shop_rules.delivery_available(store),
        "delivery_by_distance": bool(store.get("max_delivery_km")),
        "delivery_by_pincode": bool(store.get("pincodes")),
        "min_order_value": float(store.get("min_order_value") or 0),
        "free_delivery_above": float(store["free_delivery_above"]) if store.get("free_delivery_above") is not None else None,
        "prep_minutes": store.get("prep_minutes"),
        "status": status_for(store, config),
    }


@router.get("/public/shop/{project_id}/status")
def public_store_status(project_id: str, request: Request):
    _public_shop_guard(project_id, request, "shop-status", limit=60)
    _assert_shop_enabled(project_id)
    config = load_shop_config(project_id)
    return status_for(public_store(project_id, config), config)


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
    config = load_shop_config(project_id)
    store = public_store(project_id, config)
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
    priced, mode = price_with_mode(project_id, store, body.items, fulfilment, lat, lng, pin,
                                   catalog_id=catalog_id, config=config, identity=identity,
                                   coupon_code=body.coupon_code, ip=client_ip(request))
    return quote_view(priced, mode, identity["verified"])


def price_with_mode(project_id: str, store: dict, items: list, fulfilment: str, lat, lng, pin,
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
    priced = price_order(project_id, store, items, fulfilment, lat, lng, pin,
                         offers_enabled=mode != "confirm_in_whatsapp", **kw)
    if not priced["errors"] and priced["total"] == 0 and mode == "pay_online":
        if identity.get("verified"):
            mode = "place_offline"
        else:
            mode = "confirm_in_whatsapp"
            priced = price_order(project_id, store, items, fulfilment, lat, lng, pin, offers_enabled=False, **kw)
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
        "gst_breakdown": priced.get("gst_breakdown") or [],
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
