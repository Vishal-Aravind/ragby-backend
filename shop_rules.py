"""Shop branch rules: opening hours, open/closed state, delivery quote and
order-status message wording. Pure functions, no database or network, so
every money and open-state decision the server makes can be unit tested
(tests/test_shop_rules.py).

The customer page only ever displays what these return. It never decides
by itself whether a branch is open or what delivery costs.
"""
import hashlib
import json
import math
import re
from datetime import datetime, time, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import sentry_sdk

DEFAULT_TIMEZONE = "Asia/Kolkata"

# Same pattern shop.py has always used for ids in public routes.
UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")
HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]   # index = weekday()
DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_PINCODE_IN_RE = re.compile(r"^[1-9]\d{5}$")
_PINCODE_OTHER_RE = re.compile(r"^[A-Z0-9][A-Z0-9 -]{1,8}[A-Z0-9]$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")

ORDER_STATUSES = ["pending", "confirmed", "preparing", "ready", "out_for_delivery", "completed", "cancelled"]
FINAL_STATUSES = {"completed", "cancelled"}


# ─────────────────────────────────────────────
# INPUT CLEANING
# ─────────────────────────────────────────────

def clean_text(value, max_len: int) -> Optional[str]:
    """Strip control characters and runs of blank lines. None for empty.
    Raises ValueError when over max_len (after cleaning)."""
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text).strip()
    if not text:
        return None
    if len(text) > max_len:
        raise ValueError(f"must be at most {max_len} characters")
    return text


def normalize_pincode(raw, region: str = "IN") -> Optional[str]:
    """The pincode in canonical form, or None when it isn't one."""
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if (region or "IN").upper() == "IN":
        text = text.replace(" ", "")
        return text if _PINCODE_IN_RE.match(text) else None
    text = re.sub(r"\s+", " ", text)
    return text if _PINCODE_OTHER_RE.match(text) else None


def validate_latlng(lat, lng) -> Tuple[float, float]:
    """(lat, lng) rounded to 6 dp, or ValueError. (0,0) is rejected: it is
    what a broken GPS reading or an unset form field looks like."""
    try:
        lat = float(lat)
        lng = float(lng)
    except (TypeError, ValueError):
        raise ValueError("Location must be numbers")
    if not (math.isfinite(lat) and math.isfinite(lng)):
        raise ValueError("Location must be numbers")
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("Location is out of range")
    if lat == 0 and lng == 0:
        raise ValueError("Location is not set")
    return round(lat, 6), round(lng, 6)


def validate_hours(hours) -> Optional[dict]:
    """The hours JSON (spec B.1) in canonical form, or ValueError.
    None means always open."""
    if hours is None:
        return None
    if not isinstance(hours, dict):
        raise ValueError("Opening hours must be an object")
    out = {}
    for key, day in hours.items():
        if key not in DAY_KEYS:
            raise ValueError(f"Unknown day '{key}' in opening hours")
        if not isinstance(day, dict):
            raise ValueError(f"Opening hours for {key} must be an object")
        extra = set(day.keys()) - {"closed", "open", "close"}
        if extra:
            raise ValueError(f"Unknown field in opening hours for {key}")
        closed = day.get("closed", False)
        if not isinstance(closed, bool):
            raise ValueError(f"'closed' for {key} must be true or false")
        open_t, close_t = day.get("open"), day.get("close")
        for label, val in (("open", open_t), ("close", close_t)):
            if not isinstance(val, str) or not HHMM_RE.match(val):
                raise ValueError(f"{key} {label} time must be HH:MM (24-hour)")
        out[key] = {"closed": closed, "open": open_t, "close": close_t}
    return out


def validate_timezone(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Timezone is required")
    try:
        ZoneInfo(name.strip())
    except Exception:
        raise ValueError("Unknown timezone")
    return name.strip()


# ─────────────────────────────────────────────
# OPEN / CLOSED
# ─────────────────────────────────────────────

def format_time(h: int, m: int) -> str:
    # Built by hand: strftime's %-I doesn't exist on Windows.
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _fmt_dt(dt: datetime) -> str:
    return format_time(dt.hour, dt.minute)


def _mins(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _at(day, hhmm: str, tz) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime.combine(day, time(h, m), tzinfo=tz)


def _is_open(hours, now: datetime):
    """(open_now, closes_at, kind) with kind "always" | "24h" | None."""
    # Only None is "always open"; {} means every day is missing = closed.
    if hours is None:
        return True, None, "always"
    t = now.hour * 60 + now.minute
    wd = now.weekday()
    tz = now.tzinfo

    # Yesterday's hours that run past midnight belong to yesterday.
    y = hours.get(DAY_KEYS[(wd - 1) % 7])
    if y and not y["closed"] and _mins(y["close"]) < _mins(y["open"]) and t < _mins(y["close"]):
        return True, _at(now.date(), y["close"], tz), None

    d = hours.get(DAY_KEYS[wd])
    if not d or d["closed"]:
        return False, None, None
    o, c = _mins(d["open"]), _mins(d["close"])
    if o == c:
        return True, None, "24h"
    if o < c:
        if o <= t < c:
            return True, _at(now.date(), d["close"], tz), None
        return False, None, None
    if t >= o:
        return True, _at(now.date() + timedelta(days=1), d["close"], tz), None
    return False, None, None


def _next_open(hours, now: datetime) -> Optional[datetime]:
    if not hours:
        return None
    for offset in range(0, 8):
        day = now.date() + timedelta(days=offset)
        d = hours.get(DAY_KEYS[day.weekday()])
        if not d or d["closed"]:
            continue
        start = _at(day, d["open"], now.tzinfo)
        if start > now:
            return start
    return None


def _opens_label(next_open: datetime, now: datetime) -> str:
    """"9:00 AM" today, "tomorrow at 9:00 AM", or "Mon at 9:00 AM"."""
    days = (next_open.date() - now.date()).days
    t = _fmt_dt(next_open)
    if days == 0:
        return t
    if days == 1:
        return f"tomorrow at {t}"
    return f"{DAY_LABELS[next_open.weekday()]} at {t}"


def _today_hours_label(hours, now: datetime) -> str:
    if hours is None:
        return "Open 24 hours"
    d = hours.get(DAY_KEYS[now.weekday()])
    if not d or d["closed"]:
        return "Closed today"
    if d["open"] == d["close"]:
        return "Open 24 hours"
    o = format_time(*map(int, d["open"].split(":")))
    c = format_time(*map(int, d["close"].split(":")))
    return f"Today {o} – {c}"


def resolve_timezone(name):
    try:
        return ZoneInfo(name or DEFAULT_TIMEZONE)
    except Exception:
        # Validated on save, so this is a row edited by hand or an old tz.
        sentry_sdk.capture_message(f"Shop branch has invalid timezone {name!r}; using {DEFAULT_TIMEZONE}")
        return ZoneInfo(DEFAULT_TIMEZONE)


def branch_status(branch: dict, shop_enabled: bool, now_utc: datetime) -> dict:
    tz = resolve_timezone(branch.get("timezone"))
    now = now_utc.astimezone(tz)
    hours = branch.get("hours")

    open_now, closes_at, kind = _is_open(hours, now)
    next_open = None if open_now else _next_open(hours, now)

    if not shop_enabled or not branch.get("is_active", True):
        reason = "inactive"
    elif not open_now:
        reason = "closed"
    elif not branch.get("accepting_orders", True):
        reason = "paused"
    else:
        reason = None

    opens_at_label = _opens_label(next_open, now) if next_open else None
    if reason == "inactive":
        label = "Not available right now"
    elif reason == "paused":
        label = "Not taking orders right now"
    elif open_now:
        if kind == "always":
            label = "Open now"
        elif kind == "24h":
            label = "Open 24 hours"
        else:
            label = f"Open now · closes {_fmt_dt(closes_at)}"
    elif next_open:
        if (next_open.date() - now.date()).days == 0:
            label = f"Closed now · opens at {opens_at_label}"
        else:
            label = f"Closed now · opens {opens_at_label}"
    else:
        label = "Closed"

    return {
        "is_open": open_now,
        "can_order": reason is None,
        "reason": reason,
        "label": label,
        "closes_at_label": _fmt_dt(closes_at) if closes_at else None,
        "opens_at_label": opens_at_label,
        "next_open_at": next_open.isoformat() if next_open else None,
        "timezone": str(tz),
        "today_hours_label": _today_hours_label(hours, now),
    }


# ─────────────────────────────────────────────
# DELIVERY
# ─────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = p2 - p1
    dl = math.radians(float(lng2) - float(lng1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def delivery_area_set(branch: dict) -> bool:
    has_pins = len(branch.get("pincodes") or []) > 0
    max_km = branch.get("max_delivery_km")
    has_radius = max_km is not None and float(max_km) > 0
    return has_pins or has_radius


def delivery_available(branch: dict) -> bool:
    """Delivery switched on AND a delivery area set. Without an area there
    is no rule for who may order, so delivery is off."""
    return bool(branch.get("delivery_enabled")) and delivery_area_set(branch)


def _unavailable(reason: str, distance=None) -> dict:
    return {"available": False, "reason": reason, "fee": 0, "distance_km": distance}


def quote_delivery(branch: dict, subtotal: float, cust_lat, cust_lng, pincode) -> dict:
    pins = branch.get("pincodes") or []
    has_pins = len(pins) > 0
    max_km = branch.get("max_delivery_km")
    has_radius = max_km is not None and float(max_km) > 0
    if not branch.get("delivery_enabled") or (not has_pins and not has_radius):
        return _unavailable("delivery_off")
    # Delivery always needs the customer's real pin, so the fee is never
    # worked out from a pincode's rough centre.
    if cust_lat is None or cust_lng is None:
        return _unavailable("location_required")

    distance = None
    if branch.get("lat") is not None and branch.get("lng") is not None:
        distance = round(haversine_km(branch["lat"], branch["lng"], cust_lat, cust_lng), 2)

    pin_ok = has_pins and pincode is not None and pincode in pins
    radius_ok = has_radius and distance is not None and distance <= float(max_km)
    if not (pin_ok or radius_ok):
        return _unavailable("out_of_area", distance)

    per_km = float(branch.get("delivery_per_km") or 0)
    base = float(branch.get("delivery_base_fee") or 0)
    if per_km > 0 and distance is None:
        # Pincode-only branch that charges per km but has no location yet.
        return _unavailable("location_required")

    free_above = branch.get("free_delivery_above")
    if free_above is not None and float(subtotal) >= float(free_above):
        fee = 0
    else:
        # round() first so float noise (20 + 10 * 2.1 = 41.00000001) doesn't
        # ceil up a whole rupee.
        fee = math.ceil(round(base + per_km * (distance or 0), 2))
    return {"available": True, "reason": None, "fee": int(fee), "distance_km": distance}


def money(value) -> str:
    """230 -> "230", 11.5 -> "11.50": whole rupees stay short."""
    v = round(float(value or 0), 2)
    return str(int(v)) if v == int(v) else f"{v:.2f}"


# ─────────────────────────────────────────────
# ORDER STATUS MESSAGES
# ─────────────────────────────────────────────

ORDER_STATUS_TEMPLATE = "order_status_update"
ORDER_CANCELLED_TEMPLATE = "order_cancelled"

_MULTISPACE_RE = re.compile(r" {4,}")


def sanitize_template_param(value) -> str:
    """Meta rejects template parameters with newlines, tabs or 4+ spaces."""
    text = str(value if value is not None else "")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = _MULTISPACE_RE.sub(" ", text).strip()[:200]
    return text or "-"


def order_status_message(new_status: str, fulfilment: str, ctx: dict) -> Optional[dict]:
    """{"template", "params", "text"} for a status change, or None when that
    status sends nothing. ctx: name, oid, branch, contact, prep, address."""
    oid = ctx.get("oid") or ""
    branch = ctx.get("branch") or "our store"
    name = ctx.get("name") or "there"
    contact = ctx.get("contact") or "our store"
    prep = ctx.get("prep")
    address = ctx.get("address")
    delivery = fulfilment == "delivery"

    if new_status == "cancelled":
        return {
            "template": ORDER_CANCELLED_TEMPLATE,
            "params": [name, oid, branch, contact],
            "text": (f"❌ *Order Cancelled* #{oid}\nYour order from {branch} has been cancelled. "
                     f"If you already paid, please contact {contact}."),
        }

    if new_status == "confirmed":
        status_text = "Confirmed"
        text = f"✅ *Order Confirmed* #{oid}\nYour order from {branch} is confirmed."
        if prep:
            text += f"\nReady in ~{prep} min."
    elif new_status == "preparing":
        status_text = "Being prepared"
        text = f"👨‍🍳 Your order #{oid} from {branch} is being prepared."
    elif new_status == "ready" and delivery:
        status_text = "Ready, dispatching soon"
        text = f"📦 Your order #{oid} is ready and will be dispatched soon."
    elif new_status == "ready":
        status_text = "Ready for pickup"
        text = f"🛍️ Your order #{oid} is *ready for pickup* at {branch}."
        if address:
            text += f"\n📍 {address}"
    elif new_status == "out_for_delivery":
        status_text = "Out for delivery"
        text = f"🛵 Your order #{oid} is *out for delivery*!"
    elif new_status == "completed" and delivery:
        status_text = "Delivered"
        text = f"🎉 Order #{oid} delivered. Enjoy your meal!"
    elif new_status == "completed":
        status_text = "Picked up"
        text = f"🎉 Order #{oid} picked up. Thank you for ordering from {branch}!"
    else:
        return None

    return {"template": ORDER_STATUS_TEMPLATE, "params": [name, oid, branch, status_text], "text": text}


# ─────────────────────────────────────────────
# OFFERS & COUPONS (pure maths; DB loading is in shop_offers.py)
# ─────────────────────────────────────────────

OFFER_KINDS = ("percent", "flat", "free_delivery")
OFFER_APPLICATIONS = ("automatic", "code")
OFFER_FULFILMENTS = ("any", "pickup", "delivery")
SOLD_OUT_PREFS = ("remove_item", "call_me", "merchant_choice")
CODE_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{3,19}$")

OFFER_REASON_MESSAGES = {
    "invalid_code": "This code isn't valid.",
    "expired": "This offer has expired.",
    "not_started": "This offer starts on {date}.",
    "paused": "This offer isn't available right now.",
    "branch": "This offer isn't valid at {branch}.",
    "fulfilment": "This offer is only for {fulfilment} orders.",
    "login_required": "Open the menu from your WhatsApp chat to use this offer.",
    "min_order": "Add {short} more to use this offer (min order {min}).",
    "limit_reached": "This offer has been fully claimed.",
    "already_used": "You've already used this offer.",
    "not_first_order": "This offer is for your first order only.",
    "no_saving": "This offer doesn't lower your total yet.",
    "not_better": "Your code saves {a}, but \"{title}\" saves {b} — we applied the better one.",
    "rate_limited": "Too many code attempts. Please try again in 15 minutes.",
}


def normalize_code(raw) -> Optional[str]:
    """Strip, remove inner spaces, uppercase. None when it isn't a valid
    code shape (4-20 of A-Z 0-9 _ -, not starting with _ or -)."""
    if raw is None:
        return None
    text = "".join(str(raw).split()).upper()
    return text if CODE_RE.match(text) else None


def parse_ts(value) -> Optional[datetime]:
    """An aware datetime from a DB/ISO value, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=ZoneInfo("UTC"))
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Postgres may send fractional seconds with other than 6 digits.
        m = re.match(r"^(.*T\d{2}:\d{2}:\d{2})\.(\d+)(.*)$", text)
        if not m:
            return None
        try:
            dt = datetime.fromisoformat(f"{m.group(1)}.{(m.group(2) + '000000')[:6]}{m.group(3)}")
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo("UTC"))


def _f(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def offer_label(offer: dict, currency: str = "₹") -> str:
    kind = offer.get("kind")
    if kind == "percent":
        label = f"{money(offer.get('value'))}% off"
        if offer.get("max_discount") is not None and _f(offer.get("max_discount")) > 0:
            label += f" up to {currency}{money(offer['max_discount'])}"
    elif kind == "flat":
        label = f"{currency}{money(offer.get('value'))} off"
    else:
        label = "Free delivery"
    if _f(offer.get("min_order_value")) > 0:
        label += f" on orders above {currency}{money(offer['min_order_value'])}"
    if offer.get("first_order_only"):
        label += " · first order"
    if offer.get("application") == "code":
        if offer.get("is_visible", True) and offer.get("code"):
            label += f" · use {offer['code']}"
    else:
        label += " · auto-applied"
    return label


def offer_saving(offer: dict, subtotal, delivery_fee) -> dict:
    subtotal = round(_f(subtotal), 2)
    kind = offer.get("kind")
    if kind == "percent":
        d = round(subtotal * _f(offer.get("value")) / 100, 2)
        if offer.get("max_discount") is not None and _f(offer.get("max_discount")) > 0:
            d = min(d, round(_f(offer["max_discount"]), 2))
        d = min(d, subtotal)
        return {"items_discount": round(max(d, 0.0), 2), "delivery_discount": 0.0}
    if kind == "flat":
        d = min(round(_f(offer.get("value")), 2), subtotal)
        return {"items_discount": round(max(d, 0.0), 2), "delivery_discount": 0.0}
    if kind == "free_delivery":
        return {"items_discount": 0.0, "delivery_discount": round(max(_f(delivery_fee), 0.0), 2)}
    return {"items_discount": 0.0, "delivery_discount": 0.0}


def saving_total(s: dict) -> float:
    return round(s["items_discount"] + s["delivery_discount"], 2)


def offer_ineligibility(offer: dict, ctx: dict) -> Optional[str]:
    """The first reason this offer can't apply, or None. ctx: now,
    branch_id, fulfilment, subtotal, delivery_fee, verified, used_total,
    used_by_customer, has_prior_order."""
    now = ctx.get("now")
    if not offer.get("is_active", True) or offer.get("archived_at"):
        return "paused"
    starts, ends = parse_ts(offer.get("starts_at")), parse_ts(offer.get("ends_at"))
    if starts and now and now < starts:
        return "not_started"
    if ends and now and now >= ends:
        return "expired"
    branch_ids = [str(b) for b in (offer.get("branch_ids") or [])]
    if branch_ids and str(ctx.get("branch_id")) not in branch_ids:
        return "branch"
    scope = offer.get("fulfilment") or "any"
    if scope != "any" and scope != ctx.get("fulfilment"):
        return "fulfilment"
    if (offer.get("first_order_only") or offer.get("per_customer_limit") is not None) and not ctx.get("verified"):
        return "login_required"
    if _f(ctx.get("subtotal")) < _f(offer.get("min_order_value")):
        return "min_order"
    if offer.get("usage_limit") is not None and int(ctx.get("used_total") or 0) >= int(offer["usage_limit"]):
        return "limit_reached"
    if offer.get("per_customer_limit") is not None and \
            int(ctx.get("used_by_customer") or 0) >= int(offer["per_customer_limit"]):
        return "already_used"
    if offer.get("first_order_only") and ctx.get("has_prior_order"):
        return "not_first_order"
    if saving_total(offer_saving(offer, ctx.get("subtotal"), ctx.get("delivery_fee"))) <= 0:
        return "no_saving"
    return None


def _format_local(dt: datetime, tz_name: Optional[str]) -> str:
    local = dt.astimezone(resolve_timezone(tz_name)) if tz_name else dt
    return f"{local.day} {local.strftime('%b')}, {format_time(local.hour, local.minute)}"


def offer_reason_message(reason: Optional[str], offer: Optional[dict] = None, ctx: Optional[dict] = None,
                         currency: str = "₹") -> Optional[str]:
    """The customer-facing text for a reason code (OFFER_REASON_MESSAGES)."""
    if not reason:
        return None
    template = OFFER_REASON_MESSAGES.get(reason)
    if not template:
        return None
    offer, ctx = offer or {}, ctx or {}
    if reason == "not_started":
        starts = parse_ts(offer.get("starts_at"))
        return template.format(date=_format_local(starts, ctx.get("timezone")) if starts else "a later date")
    if reason == "branch":
        return template.format(branch=ctx.get("branch_name") or "this branch")
    if reason == "fulfilment":
        return template.format(fulfilment="delivery" if offer.get("fulfilment") == "delivery" else "pickup")
    if reason == "min_order":
        min_v = _f(offer.get("min_order_value"))
        short = max(0.0, round(min_v - _f(ctx.get("subtotal")), 2))
        return template.format(short=f"{currency}{money(short)}", min=f"{currency}{money(min_v)}")
    if reason == "not_better":
        return template.format(a=f"{currency}{money(ctx.get('code_saving'))}",
                               title=" ".join(str(ctx.get("better_title") or "").split()),
                               b=f"{currency}{money(ctx.get('better_saving'))}")
    return template


_FAR_FUTURE = datetime(9999, 12, 31, tzinfo=ZoneInfo("UTC"))


def _tie_key(candidate: dict):
    offer = candidate["offer"]
    ends = parse_ts(offer.get("ends_at"))
    created = parse_ts(offer.get("created_at"))
    return (-candidate["saving"], ends is None, ends or _FAR_FUTURE, created or _FAR_FUTURE)


def choose_offer(auto_candidates: list, code_candidate: Optional[dict]):
    """(chosen candidate | None, coupon_status | None).

    Candidates are ELIGIBLE offers as {"offer", "saving", "items_discount",
    "delivery_discount"}. Best automatic = biggest saving; tie -> earliest
    ends_at (no end date last), then oldest. An eligible code wins when it
    saves at least as much as the best automatic offer ("applied"),
    otherwise the automatic offer is used and the coupon is "not_better".
    """
    best_auto = min(auto_candidates, key=_tie_key) if auto_candidates else None
    if code_candidate is not None:
        if best_auto is None or code_candidate["saving"] >= best_auto["saving"]:
            return code_candidate, "applied"
        return best_auto, "not_better"
    return best_auto, None


def apply_offer_totals(subtotal, gst_percent, delivery_fee, items_discount, delivery_discount) -> dict:
    """GST on the discounted item value; the delivery fee stays untaxed."""
    taxable = round(_f(subtotal) - _f(items_discount), 2)
    gst = round(taxable * _f(gst_percent) / 100, 2)
    total = round(taxable + gst + _f(delivery_fee) - _f(delivery_discount), 2)
    return {"taxable_subtotal": taxable, "gst_amount": gst, "total": max(total, 0.0)}


def checkout_fingerprint(d: dict) -> str:
    """sha256 hex of the canonical checkout: same cart, same details, same
    coupon = same fingerprint (an unexpired payment link can be reused)."""
    merged = {}
    for it in (d.get("items") or []):
        pid = str((it.get("product_id") if isinstance(it, dict) else getattr(it, "product_id", "")) or "")
        qty = int((it.get("quantity") if isinstance(it, dict) else getattr(it, "quantity", 0)) or 0)
        merged[pid] = merged.get(pid, 0) + qty

    def s(v):
        if v is None:
            return None
        text = " ".join(str(v).split())
        return text or None

    def num(v):
        return None if v is None else round(float(v), 6)

    delivery = d.get("fulfilment") == "delivery"
    canonical = {
        "branch_id": s(d.get("branch_id")),
        "items": sorted([[k, v] for k, v in merged.items()]),
        "fulfilment": s(d.get("fulfilment")),
        "name": s(d.get("name")),
        "line1": s(d.get("line1")) if delivery else None,
        "line2": s(d.get("line2")) if delivery else None,
        "city": s(d.get("city")) if delivery else None,
        "pincode": s(d.get("pincode")) if delivery else None,
        "lat": num(d.get("lat")) if delivery else None,
        "lng": num(d.get("lng")) if delivery else None,
        "coupon": normalize_code(d.get("coupon")) if d.get("coupon") else None,
        "sold_out_preference": s(d.get("sold_out_preference")),
        "note": s(d.get("note")),
    }
    blob = json.dumps(canonical, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def offer_state(offer: dict, now: datetime, used_total: int = 0) -> str:
    """Dashboard badge: active | scheduled | expired | paused | limit_reached | archived."""
    if offer.get("archived_at"):
        return "archived"
    ends = parse_ts(offer.get("ends_at"))
    if ends and now >= ends:
        return "expired"
    if not offer.get("is_active", True):
        return "paused"
    starts = parse_ts(offer.get("starts_at"))
    if starts and now < starts:
        return "scheduled"
    if offer.get("usage_limit") is not None and int(used_total or 0) >= int(offer["usage_limit"]):
        return "limit_reached"
    return "active"


def _num(v):
    if v is None:
        return None
    v = round(float(v), 2)
    return int(v) if v == int(v) else v


def public_offer(offer: dict, currency: str = "₹") -> dict:
    """The PublicOffer shape (no usage counts, limits or internal fields)."""
    return {
        "id": offer["id"],
        "title": offer.get("title"),
        "description": offer.get("description"),
        "kind": offer.get("kind"),
        "value": _num(offer.get("value")),
        "max_discount": _num(offer.get("max_discount")),
        "min_order_value": _num(offer.get("min_order_value") or 0),
        "application": offer.get("application"),
        "code": offer.get("code") if offer.get("application") == "code" else None,
        "fulfilment": offer.get("fulfilment") or "any",
        "first_order_only": bool(offer.get("first_order_only")),
        "ends_at": offer.get("ends_at"),
        "label": offer_label(offer, currency),
    }


def valid_media_url(url, supabase_url: Optional[str]) -> bool:
    """A public flow-media object of this project's Supabase, in the
    shop-branding or products folder, png/jpg/webp."""
    if not url or not supabase_url or not isinstance(url, str) or len(url) > 1000:
        return False
    pattern = (rf"^{re.escape(supabase_url.rstrip('/'))}/storage/v1/object/public/flow-media/"
               r"[0-9a-f-]{36}/(shop-branding|products)/\d{10,16}\.(png|jpe?g|webp)$")
    return re.match(pattern, url) is not None
