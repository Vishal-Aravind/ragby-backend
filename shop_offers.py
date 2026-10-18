"""Shop offers & coupons: dashboard CRUD, the public offers strip, and
evaluate_offers, which price_order (shop_branches.py) calls to work out the
one offer an order gets.

The maths (savings, eligibility order, best-of choice, GST on the discounted
value) is pure and lives in shop_rules.py. This module only loads rows and
usage counts. Usage limits are enforced authoritatively by the DB function
shop_claim_offer at placement; everything here is a read-only preview.
"""
import math
from datetime import datetime, timezone
from typing import List, Optional

import sentry_sdk
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

import shop_rules
from auth import verify_token, require_project_access
from clients import supabase
from ratelimit import is_rate_limited, is_blocked, client_ip
from shop import shop_error, _public_shop_guard, _assert_shop_enabled, load_shop_config
from shop_identity import mask_phone

router = APIRouter()

MAX_OFFERS_PER_PROJECT = 100
MAX_ACTIVE_AUTOMATIC = 20
MAX_BRANCH_IDS = 50

# Coupon brute-force buckets (spec G.1).
BAD_IP_LIMIT, BAD_IP_WINDOW = 8, 900
BAD_PROJECT_LIMIT, BAD_PROJECT_WINDOW = 300, 3600
TRY_IP_LIMIT, TRY_IP_WINDOW = 60, 600

_OFFER_COLUMNS = ("title", "description", "kind", "value", "max_discount", "min_order_value", "application",
                  "code", "is_visible", "starts_at", "ends_at", "is_active", "branch_ids", "fulfilment",
                  "first_order_only", "usage_limit", "per_customer_limit", "sort_order")


# ─────────────────────────────────────────────
# EVALUATION (called by shop_branches.price_order)
# ─────────────────────────────────────────────

def _empty(coupon=None) -> dict:
    return {"offer": None, "items_discount": 0.0, "delivery_discount": 0.0,
            "coupon": coupon, "available_offers": []}


def _coupon(code, status, reason=None, message=None) -> dict:
    return {"code": code, "status": status, "reason": reason, "message": message}


def _rejected(code, reason, offer=None, ctx=None, currency="₹") -> dict:
    return _coupon(code, "rejected", reason, shop_rules.offer_reason_message(reason, offer, ctx, currency))


def _count_bad_code(project_id: str, ip, identity: dict):
    is_rate_limited(f"coupon-bad:{project_id}:{ip}", BAD_IP_LIMIT, BAD_IP_WINDOW)
    is_rate_limited(f"coupon-bad-proj:{project_id}", BAD_PROJECT_LIMIT, BAD_PROJECT_WINDOW)
    if identity.get("verified") and identity.get("phone"):
        is_rate_limited(f"coupon-bad-phone:{project_id}:{identity['phone']}", BAD_IP_LIMIT, BAD_IP_WINDOW)


def _sort_key(o: dict):
    return (int(o.get("sort_order") or 0), str(o.get("created_at") or ""))


def offer_counts(project_id: str, offer_ids: list, phone: Optional[str], exclude_order_id=None) -> dict:
    """{offer_id: (used_total, used_by_customer)}. Empty on a DB error: the
    claim at placement re-checks every limit, so a preview never over-grants."""
    if not offer_ids:
        return {}
    try:
        res = supabase.rpc("shop_offer_counts", {
            "p_project_id": project_id, "p_offer_ids": list(offer_ids),
            "p_phone": phone, "p_exclude_order": exclude_order_id,
        }).execute()
        return {r["offer_id"]: (int(r.get("used_total") or 0), int(r.get("used_by_customer") or 0))
                for r in (res.data or [])}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {}


def _has_prior_order(project_id: str, phone: str, exclude_order_id=None) -> bool:
    try:
        res = supabase.rpc("shop_customer_has_prior_order", {
            "p_project_id": project_id, "p_phone": phone, "p_exclude_order": exclude_order_id,
        }).execute()
        return bool(res.data)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return False


def evaluate_offers(project_id, branch, fulfilment, subtotal, delivery_fee, identity, coupon_code,
                    ip, now_utc, exclude_order_id=None, exclude_offer_ids=(), enabled=True,
                    currency="₹") -> dict:
    """The one offer this order gets, the coupon verdict and the offers list.

    Returns {"offer": PublicOffer + {"saving"} | None, "items_discount",
    "delivery_discount", "coupon": {...} | None, "available_offers": [...]}.
    Offer problems never raise; they are reported in coupon/available_offers.
    """
    identity = identity or {}
    branch = branch or {}
    raw_code = coupon_code.strip() if isinstance(coupon_code, str) else coupon_code
    if not enabled or not subtotal or float(subtotal) <= 0:
        if raw_code and not enabled:
            code = shop_rules.normalize_code(raw_code) or str(raw_code).upper()[:20]
            return _empty(_rejected(code, "login_required", currency=currency))
        return _empty()

    now = now_utc or datetime.now(timezone.utc)
    code = shop_rules.normalize_code(raw_code) if raw_code else None
    coupon = None
    if raw_code and code is None:
        coupon = _rejected(str(raw_code).upper()[:20], "invalid_code", currency=currency)
        _count_bad_code(project_id, ip, identity)
    if code:
        # The per-phone bucket (counted in _count_bad_code) is enforced too,
        # so a verified customer rotating IPs is still locked out.
        bad_phone_blocked = bool(identity.get("verified") and identity.get("phone")) and \
            is_blocked(f"coupon-bad-phone:{project_id}:{identity['phone']}", BAD_IP_LIMIT, BAD_IP_WINDOW)
        if is_blocked(f"coupon-bad:{project_id}:{ip}", BAD_IP_LIMIT, BAD_IP_WINDOW) or \
                is_blocked(f"coupon-bad-proj:{project_id}", BAD_PROJECT_LIMIT, BAD_PROJECT_WINDOW) or \
                bad_phone_blocked:
            coupon = _rejected(code, "rate_limited", currency=currency)   # same answer for every code
            code = None
        elif is_rate_limited(f"coupon-try:{project_id}:{ip}", TRY_IP_LIMIT, TRY_IP_WINDOW):
            coupon = _rejected(code, "rate_limited", currency=currency)
            code = None

    # One query: live automatic offers, plus the typed code even when paused
    # (so the customer hears "isn't available right now", not "isn't valid").
    try:
        q = supabase.table("offers").select("*").eq("project_id", project_id).is_("archived_at", "null")
        if code:
            # code matched CODE_RE ([A-Z0-9_-] only), so it is safe inside or_().
            q = q.or_(f"and(application.eq.automatic,is_active.eq.true),code.eq.{code}")
        else:
            q = q.eq("application", "automatic").eq("is_active", True)
        rows = q.execute().data or []
    except Exception as e:
        sentry_sdk.capture_exception(e)
        rows = []
    def _is_code_row(r):
        return bool(code) and r.get("application") == "code" and r.get("code") == code

    code_known = any(_is_code_row(r) for r in rows)
    excluded = {str(x) for x in (exclude_offer_ids or ())}
    rows = [r for r in rows if str(r["id"]) not in excluded]

    code_row = next((r for r in rows if _is_code_row(r)), None)
    if code and not code_row:
        if code_known:
            # The code's offer was excluded after a failed claim (place-order
            # re-prices without it): the code is real, so not a bad try.
            coupon = _rejected(code, "limit_reached", currency=currency)
        else:
            _count_bad_code(project_id, ip, identity)
            coupon = _rejected(code, "invalid_code", currency=currency)
    if code_row is None:
        rows = [r for r in rows if r.get("application") == "automatic"]

    verified = bool(identity.get("verified"))
    phone = identity.get("phone") if verified else None
    counts = offer_counts(project_id, [r["id"] for r in rows], phone, exclude_order_id)
    prior = False
    if verified and phone and any(r.get("first_order_only") for r in rows):
        prior = _has_prior_order(project_id, phone, exclude_order_id)

    base_ctx = {
        "now": now, "branch_id": branch.get("id"), "branch_name": branch.get("name"),
        "timezone": branch.get("timezone"), "fulfilment": fulfilment,
        "subtotal": float(subtotal), "delivery_fee": float(delivery_fee or 0),
        "verified": verified, "has_prior_order": prior,
    }

    evaluated = []      # (row, reason, saving dict, ctx)
    for r in rows:
        used_total, used_mine = counts.get(r["id"], (0, 0))
        ctx = {**base_ctx, "used_total": used_total, "used_by_customer": used_mine}
        reason = shop_rules.offer_ineligibility(r, ctx)
        saving = shop_rules.offer_saving(r, subtotal, delivery_fee)
        evaluated.append((r, reason, saving, ctx))

    def cand(r, s):
        return {"offer": r, "saving": shop_rules.saving_total(s), **s}

    autos = [cand(r, s) for r, reason, s, _ in evaluated if r.get("application") == "automatic" and reason is None]
    code_cand = None
    if code_row is not None:
        _, reason, s, ctx = next(e for e in evaluated if e[0] is code_row)
        if reason is None:
            code_cand = cand(code_row, s)
        else:
            coupon = _rejected(code, reason, code_row, ctx, currency)

    chosen, status = shop_rules.choose_offer(autos, code_cand)
    if code_cand is not None:
        if status == "applied":
            coupon = _coupon(code, "applied")
        else:
            coupon = _coupon(code, "not_better", "not_better", shop_rules.offer_reason_message(
                "not_better", ctx={"code_saving": code_cand["saving"], "better_title": chosen["offer"].get("title"),
                                   "better_saving": chosen["saving"]}, currency=currency))

    available = []
    for r, reason, s, ctx in sorted(evaluated, key=lambda e: _sort_key(e[0])):
        visible = r.get("application") == "automatic" or r.get("is_visible", True) or r is code_row
        if not visible:
            continue
        available.append({
            **shop_rules.public_offer(r, currency),
            "eligible": reason is None,
            "reason": reason,
            "message": shop_rules.offer_reason_message(reason, r, ctx, currency),
            "saving": shop_rules.saving_total(s) if reason is None else 0,
        })

    if not chosen:
        return {**_empty(coupon), "available_offers": available}
    return {
        "offer": {**shop_rules.public_offer(chosen["offer"], currency), "saving": chosen["saving"]},
        "items_discount": chosen["items_discount"],
        "delivery_discount": chosen["delivery_discount"],
        "coupon": coupon,
        "available_offers": available,
    }


# ─────────────────────────────────────────────
# PUBLIC: offers strip
# ─────────────────────────────────────────────

@router.get("/public/shop/{project_id}/offers")
def public_offers(project_id: str, request: Request, branch_id: Optional[str] = None,
                  fulfilment: Optional[str] = None):
    _public_shop_guard(project_id, request, "shop-offers", limit=60)
    _assert_shop_enabled(project_id)
    if branch_id and not shop_rules.UUID_RE.match(branch_id):
        raise shop_error(400, "invalid_request", "Invalid branch id")
    if fulfilment and fulfilment not in ("pickup", "delivery"):
        raise shop_error(400, "invalid_request", "fulfilment must be 'pickup' or 'delivery'")

    config = load_shop_config(project_id)
    currency = config.get("currency") or "₹"
    now = datetime.now(timezone.utc)
    res = supabase.table("offers").select("*").eq("project_id", project_id) \
        .is_("archived_at", "null").eq("is_active", True).execute()
    rows = []
    for r in (res.data or []):
        if r.get("application") == "code" and not r.get("is_visible", True):
            continue
        starts, ends = shop_rules.parse_ts(r.get("starts_at")), shop_rules.parse_ts(r.get("ends_at"))
        if (starts and now < starts) or (ends and now >= ends):
            continue
        ids = [str(b) for b in (r.get("branch_ids") or [])]
        if branch_id and ids and branch_id not in ids:
            continue
        if fulfilment and (r.get("fulfilment") or "any") not in ("any", fulfilment):
            continue
        rows.append(r)

    limited = [r["id"] for r in rows if r.get("usage_limit") is not None]
    counts = offer_counts(project_id, limited, None)
    rows = [r for r in rows if r.get("usage_limit") is None
            or counts.get(r["id"], (0, 0))[0] < int(r["usage_limit"])]
    rows.sort(key=_sort_key)
    return {"offers": [shop_rules.public_offer(r, currency) for r in rows]}


# ─────────────────────────────────────────────
# DASHBOARD CRUD
# ─────────────────────────────────────────────

class OfferUpsert(BaseModel):
    project_id: Optional[str] = Field(default=None, max_length=64)
    title: Optional[str] = Field(default=None, max_length=500)
    description: Optional[str] = Field(default=None, max_length=2000)
    kind: Optional[str] = Field(default=None, max_length=32)
    value: Optional[float] = None
    max_discount: Optional[float] = None
    min_order_value: Optional[float] = None
    application: Optional[str] = Field(default=None, max_length=32)
    code: Optional[str] = Field(default=None, max_length=64)
    is_visible: Optional[bool] = None
    starts_at: Optional[str] = Field(default=None, max_length=64)
    ends_at: Optional[str] = Field(default=None, max_length=64)
    is_active: Optional[bool] = None
    branch_ids: Optional[List[str]] = Field(default=None, max_length=200)
    fulfilment: Optional[str] = Field(default=None, max_length=16)
    first_order_only: Optional[bool] = None
    usage_limit: Optional[int] = None
    per_customer_limit: Optional[int] = None
    sort_order: Optional[int] = None


def _bad(message: str):
    return shop_error(400, "invalid_request", message)


def _money_field(label, value, lo, hi, lo_inclusive=True):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise _bad(f"{label} must be a number")
    if not math.isfinite(v) or v > hi or v < lo or (not lo_inclusive and v <= lo):
        raise _bad(f"{label} must be between {shop_rules.money(lo)} and {shop_rules.money(hi)}")
    return round(v, 2)


def _parse_dt(label, value) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise _bad(f"{label} must be an ISO 8601 date-time")
    if dt.tzinfo is None:
        raise _bad(f"{label} must include a timezone")
    return dt.astimezone(timezone.utc).isoformat()


def validate_offer(merged: dict, project_id: str, creating: bool, now: Optional[datetime] = None,
                   branch_lookup=None) -> dict:
    """The full set of editable columns, validated and normalised. `merged`
    is the existing row (on update) with the request's fields on top.
    branch_lookup(project_id, ids) -> set of ids that exist (injectable for tests)."""
    now = now or datetime.now(timezone.utc)
    out = {}

    try:
        title = shop_rules.clean_text(" ".join(str(merged.get("title")).split()) if merged.get("title") else None, 60)
    except ValueError:
        raise _bad("Title must be at most 60 characters")
    if not title:
        raise _bad("Title is required")
    out["title"] = title
    try:
        out["description"] = shop_rules.clean_text(merged.get("description"), 200)
    except ValueError:
        raise _bad("Description must be at most 200 characters")

    kind = merged.get("kind")
    if kind not in shop_rules.OFFER_KINDS:
        raise _bad("Type must be percent, flat or free_delivery")
    out["kind"] = kind
    if kind == "percent":
        out["value"] = _money_field("Percent", merged.get("value"), 0, 100, lo_inclusive=False)
    elif kind == "flat":
        out["value"] = _money_field("Discount amount", merged.get("value"), 0, 100000, lo_inclusive=False)
    else:
        out["value"] = 0

    md = merged.get("max_discount")
    if kind == "percent" and md is not None:
        out["max_discount"] = _money_field("Maximum discount", md, 0, 100000, lo_inclusive=False)
    else:
        out["max_discount"] = None

    mov = merged.get("min_order_value")
    out["min_order_value"] = _money_field("Minimum order value", 0 if mov is None else mov, 0, 99_999_999)

    application = merged.get("application")
    if application not in shop_rules.OFFER_APPLICATIONS:
        raise _bad("Choose automatic or coupon code")
    out["application"] = application
    if application == "code":
        raw = merged.get("code")
        code = shop_rules.normalize_code(raw) if raw else None
        if not code:
            raise _bad("Code must be 4-20 letters, digits, - or _")
        visible = merged.get("is_visible")
        visible = True if visible is None else bool(visible)
        if not visible and len(code) < 6:
            raise _bad("A hidden code needs at least 6 characters")
        out["code"], out["is_visible"] = code, visible
    else:
        out["code"], out["is_visible"] = None, True

    starts = _parse_dt("Start", merged.get("starts_at"))
    ends = _parse_dt("End", merged.get("ends_at"))
    if starts and ends and shop_rules.parse_ts(ends) <= shop_rules.parse_ts(starts):
        raise _bad("The end must be after the start")
    if creating and ends and shop_rules.parse_ts(ends) <= now:
        raise _bad("The end must be in the future")
    out["starts_at"], out["ends_at"] = starts, ends

    active = merged.get("is_active")
    out["is_active"] = True if active is None else bool(active)

    ids = []
    for b in (merged.get("branch_ids") or []):
        b = str(b).lower()
        if not shop_rules.UUID_RE.match(b):
            raise _bad("Invalid branch id")
        if b not in ids:
            ids.append(b)
    if len(ids) > MAX_BRANCH_IDS:
        raise _bad(f"At most {MAX_BRANCH_IDS} branches per offer")
    if ids:
        found = (branch_lookup or _existing_branch_ids)(project_id, ids)
        if set(ids) - set(found):
            raise _bad("One of the branches isn't part of this shop")
    out["branch_ids"] = ids

    scope = merged.get("fulfilment") or "any"
    if scope not in shop_rules.OFFER_FULFILMENTS:
        raise _bad("Fulfilment must be any, pickup or delivery")
    out["fulfilment"] = "delivery" if kind == "free_delivery" else scope

    out["first_order_only"] = bool(merged.get("first_order_only"))
    for key, label, hi in (("usage_limit", "Total uses", 1_000_000), ("per_customer_limit", "Uses per customer", 1000)):
        v = merged.get(key)
        if v is None:
            out[key] = None
        else:
            if isinstance(v, bool) or not isinstance(v, int) or v < 1 or v > hi:
                raise _bad(f"{label} must be a whole number from 1 to {hi}")
            out[key] = v
    so = merged.get("sort_order")
    so = 0 if so is None else so
    if isinstance(so, bool) or not isinstance(so, int) or abs(so) > 1_000_000:
        raise _bad("Sort order must be a whole number")
    out["sort_order"] = so
    return out


def _existing_branch_ids(project_id: str, ids: list) -> set:
    res = supabase.table("shop_branches").select("id").eq("project_id", project_id).in_("id", ids).execute()
    return {str(r["id"]).lower() for r in (res.data or [])}


def _usage(project_id: str) -> dict:
    try:
        res = supabase.rpc("shop_offer_usage", {"p_project_id": project_id}).execute()
        return {r["offer_id"]: r for r in (res.data or [])}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {}


def _decorate(offer: dict, usage: dict, currency: str, now: datetime) -> dict:
    u = usage.get(offer["id"]) or {}
    confirmed, held = int(u.get("confirmed_count") or 0), int(u.get("held_count") or 0)
    return {
        **offer,
        "label": shop_rules.offer_label(offer, currency),
        "state": shop_rules.offer_state(offer, now, confirmed + held),
        "usage": {"confirmed": confirmed, "held": held, "total_discount": float(u.get("total_discount") or 0)},
    }


def _one(offer: dict) -> dict:
    project_id = offer["project_id"]
    currency = load_shop_config(project_id).get("currency") or "₹"
    return _decorate(offer, _usage(project_id), currency, datetime.now(timezone.utc))


def _load_offer_or_404(offer_id: str) -> dict:
    if not shop_rules.UUID_RE.match(offer_id or ""):
        raise shop_error(404, "offer_not_found", "This offer doesn't exist.")
    res = supabase.table("offers").select("*").eq("id", offer_id).limit(1).execute()
    if not res.data:
        raise shop_error(404, "offer_not_found", "This offer doesn't exist.")
    return res.data[0]


def _check_limits(project_id: str, fields: dict, offer_id: Optional[str] = None):
    res = supabase.table("offers").select("id, application, is_active") \
        .eq("project_id", project_id).is_("archived_at", "null").execute()
    others = [r for r in (res.data or []) if r["id"] != offer_id]
    if offer_id is None and len(others) >= MAX_OFFERS_PER_PROJECT:
        raise shop_error(400, "too_many_offers", f"A shop can have at most {MAX_OFFERS_PER_PROJECT} offers. Delete some old ones first.")
    if fields["application"] == "automatic" and fields["is_active"]:
        active_auto = [r for r in others if r.get("application") == "automatic" and r.get("is_active")]
        if len(active_auto) >= MAX_ACTIVE_AUTOMATIC:
            raise shop_error(400, "too_many_offers", f"At most {MAX_ACTIVE_AUTOMATIC} automatic offers can be active at once.")


def _is_unique_violation(e: Exception) -> bool:
    text = str(e)
    return "23505" in text or "duplicate key" in text


@router.get("/shop/offers")
def list_offers(project_id: str, include_archived: bool = False, user=Depends(verify_token)):
    if not shop_rules.UUID_RE.match(project_id or ""):
        raise _bad("Invalid project id")
    require_project_access(user.id, project_id, tab="shop")
    q = supabase.table("offers").select("*").eq("project_id", project_id)
    if not include_archived:
        q = q.is_("archived_at", "null")
    rows = q.execute().data or []
    rows.sort(key=lambda o: (o.get("archived_at") is not None, *_sort_key(o)))
    currency = load_shop_config(project_id).get("currency") or "₹"
    usage = _usage(project_id)
    now = datetime.now(timezone.utc)
    return [_decorate(o, usage, currency, now) for o in rows]


@router.post("/shop/offers")
def create_offer(body: OfferUpsert, user=Depends(verify_token)):
    project_id = body.project_id or ""
    if not shop_rules.UUID_RE.match(project_id):
        raise _bad("Invalid project id")
    require_project_access(user.id, project_id, tab="shop", min_role="admin")
    data = body.model_dump(exclude_unset=True)
    data.pop("project_id", None)
    fields = validate_offer(data, project_id, creating=True)
    _check_limits(project_id, fields)
    try:
        inserted = supabase.table("offers").insert({"project_id": project_id, "created_by": user.id, **fields}).execute()
    except Exception as e:
        if _is_unique_violation(e):
            raise shop_error(409, "code_taken", "Another offer already uses this code")
        raise
    if not inserted.data:
        raise shop_error(500, "invalid_request", "Could not create the offer. Please try again.")
    return _one(inserted.data[0])


@router.put("/shop/offers/{offer_id}")
def update_offer(offer_id: str, body: OfferUpsert, user=Depends(verify_token)):
    offer = _load_offer_or_404(offer_id)
    project_id = offer["project_id"]
    require_project_access(user.id, project_id, tab="shop", min_role="admin")
    if offer.get("archived_at"):
        raise shop_error(409, "invalid_request", "This offer is archived and can't be edited.")
    data = body.model_dump(exclude_unset=True)
    data.pop("project_id", None)
    merged = {k: offer.get(k) for k in _OFFER_COLUMNS}
    merged.update(data)
    fields = validate_offer(merged, project_id, creating=False)
    _check_limits(project_id, fields, offer_id=offer_id)
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        updated = supabase.table("offers").update(fields).eq("id", offer_id).eq("project_id", project_id).execute()
    except Exception as e:
        if _is_unique_violation(e):
            raise shop_error(409, "code_taken", "Another offer already uses this code")
        raise
    if not updated.data:
        raise shop_error(404, "offer_not_found", "This offer doesn't exist.")
    return _one(updated.data[0])


@router.delete("/shop/offers/{offer_id}")
def delete_offer(offer_id: str, user=Depends(verify_token)):
    offer = _load_offer_or_404(offer_id)
    project_id = offer["project_id"]
    require_project_access(user.id, project_id, tab="shop", min_role="admin")
    used = supabase.table("offer_redemptions").select("id").eq("offer_id", offer_id) \
        .eq("project_id", project_id).limit(1).execute()
    if used.data:
        now_iso = datetime.now(timezone.utc).isoformat()
        supabase.table("offers").update({"archived_at": now_iso, "is_active": False, "updated_at": now_iso}) \
            .eq("id", offer_id).eq("project_id", project_id).execute()
        return {"status": "archived"}
    supabase.table("offers").delete().eq("id", offer_id).eq("project_id", project_id).execute()
    return {"status": "deleted"}


@router.get("/shop/offers/{offer_id}/redemptions")
def offer_redemptions(offer_id: str, limit: int = 50, user=Depends(verify_token)):
    offer = _load_offer_or_404(offer_id)
    project_id = offer["project_id"]
    require_project_access(user.id, project_id, tab="shop")
    limit = max(1, min(int(limit or 50), 200))
    res = supabase.table("offer_redemptions") \
        .select("order_id, customer_phone, status, discount_amount, created_at") \
        .eq("offer_id", offer_id).eq("project_id", project_id) \
        .order("created_at", desc=True).limit(limit).execute()
    rows = res.data or []
    orders = {}
    ids = [r["order_id"] for r in rows]
    if ids:
        o = supabase.table("orders").select("id, status, total").eq("project_id", project_id).in_("id", ids).execute()
        orders = {x["id"]: x for x in (o.data or [])}
    return [{
        "order_id": r["order_id"],
        "short_id": str(r["order_id"])[:8].upper(),
        "customer_phone_masked": mask_phone(r.get("customer_phone")),
        "status": r.get("status"),
        "discount_amount": float(r.get("discount_amount") or 0),
        "created_at": r.get("created_at"),
        "order_status": (orders.get(r["order_id"]) or {}).get("status"),
        "order_total": float((orders.get(r["order_id"]) or {}).get("total") or 0),
    } for r in rows]
