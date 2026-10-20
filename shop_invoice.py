"""GST tax invoices for shop orders.

An order gets an invoice number once it is PAID (Razorpay webhook) or
CONFIRMED without online payment (pay at pickup / delivery, free orders).
The number is allocated by the shop_issue_invoice SQL function: sequential
per project per Indian financial year, one per order, never re-issued.

Everything on the invoice is rendered from the order row and the frozen
seller_snapshot, never from anything the browser sends.

Tax split: CGST + SGST, half each (intra-state supply assumed: place of
supply = the seller's state). IGST is not handled in this version.
"""
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import sentry_sdk
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from auth import verify_token, require_project_access
from config import FRONTEND_URL
import shop_identity
import shop_rules
from shop import _public_shop_guard, _order_for_customer, OrderAuthBody, shop_error

router = APIRouter()

GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
IST = timezone(timedelta(hours=5, minutes=30))
SELLER_FIELDS = ("legal_name", "gstin", "registered_address", "state")
_UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def _db():
    """shop.py's client: one object, so the tests' patch of shop.supabase
    covers this module too (and nothing here reaches a live database)."""
    import shop
    return shop.supabase


# ─────────────────────────────────────────────
# PURE HELPERS (tested)
# ─────────────────────────────────────────────

def normalize_gstin(raw) -> Optional[str]:
    """Upper-cased GSTIN, None for empty. ValueError when malformed."""
    if raw is None:
        return None
    text = re.sub(r"\s+", "", str(raw)).upper()
    if not text:
        return None
    if not GSTIN_RE.match(text):
        raise ValueError("GSTIN must be 15 characters, like 29ABCDE1234F1Z5")
    return text


def financial_year(dt: datetime) -> str:
    """Indian financial year (Apr-Mar) of a moment, by the date in India:
    2026-04-01 IST -> '2026-27', 2027-03-31 IST -> '2026-27'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(IST)
    start = local.year - (1 if local.month < 4 else 0)
    return f"{start}-{(start + 1) % 100:02d}"


def format_invoice_number(fy: str, seq: int) -> str:
    """Same format as the SQL function: INV/2026-27/0001."""
    return f"INV/{fy}/{seq:04d}"


def _paise(value) -> int:
    return int(round(float(value or 0) * 100))


def split_tax(tax) -> tuple:
    """(cgst, sgst): halves that always add up to `tax` exactly. An odd
    paisa goes to SGST."""
    total = _paise(tax)
    cgst = total // 2
    return cgst / 100, (total - cgst) / 100


_ONES = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
         "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
         "Eighteen", "Nineteen"]
_TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]


def _below_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    return (_TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")).strip()


def _below_1000(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(_ONES[n // 100] + " Hundred")
        n %= 100
    if n:
        parts.append(_below_100(n))
    return " ".join(parts)


def number_in_words(n: int) -> str:
    """Indian numbering: 12345678 -> 'One Crore Twenty Three Lakh Forty Five
    Thousand Six Hundred Seventy Eight'."""
    n = int(n)
    if n == 0:
        return "Zero"
    parts = []
    crore, n = divmod(n, 10_000_000)
    if crore:
        parts.append(number_in_words(crore) + " Crore")
    lakh, n = divmod(n, 100_000)
    if lakh:
        parts.append(_below_100(lakh) + " Lakh")
    thousand, n = divmod(n, 1000)
    if thousand:
        parts.append(_below_100(thousand) + " Thousand")
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def amount_in_words(amount) -> str:
    """1234.5 -> 'Rupees One Thousand Two Hundred Thirty Four and Fifty Paise Only'."""
    total = max(_paise(amount), 0)
    rupees, paise = divmod(total, 100)
    text = f"Rupees {number_in_words(rupees)}"
    if paise:
        text += f" and {_below_100(paise)} Paise"
    return text + " Only"


def line_taxables(items: list, items_discount) -> list:
    """Each line's taxable value: the item discount shared in proportion to
    line value, the last line taking the rounding rest. Mirrors
    shop_rules.compute_totals, so the lines add up to the stored breakdown."""
    values = [round(float(it.get("price") or 0) * int(it.get("quantity") or 0), 2) for it in items]
    subtotal = round(sum(values), 2)
    discount = min(max(float(items_discount or 0), 0.0), subtotal)
    out, allocated = [], 0.0
    for i, value in enumerate(values):
        if i == len(values) - 1:
            share = round(discount - allocated, 2)
        else:
            share = round(discount * value / subtotal, 2) if subtotal > 0 else 0.0
            allocated += share
        out.append(max(round(value - share, 2), 0.0))
    return out


def seller_snapshot(config: dict, branch: Optional[dict]) -> dict:
    """Seller details frozen onto the invoice: the branch's own value for a
    field when set, else the shop's."""
    config, branch = config or {}, branch or {}
    snap = {}
    for f in SELLER_FIELDS:
        snap[f] = (branch.get(f) or "").strip() or (config.get(f) or "").strip() or None
    snap["legal_name"] = snap["legal_name"] or (config.get("store_name") or "").strip() or None
    snap["branch_name"] = branch.get("name")
    snap["branch_address"] = branch.get("address")
    return snap


def payment_label(order: dict) -> str:
    ps = order.get("payment_status")
    delivery = (order.get("fulfilment") or "pickup") == "delivery"
    if ps == "paid":
        return "Paid online" if order.get("payment_method") == "online" else "Paid"
    if ps == "refunded":
        return "Refunded"
    if ps == "not_required" or float(order.get("total") or 0) <= 0:
        return "Nothing to pay"
    return "To be paid on delivery" if delivery else "To be paid at pickup"


def build_invoice(order: dict, currency: str = "₹", public: bool = True) -> dict:
    """The invoice JSON, from the order row alone."""
    snap = order.get("seller_snapshot") or {}
    has_gstin = bool(snap.get("gstin"))
    items = list(order.get("items") or [])
    taxables = line_taxables(items, order.get("items_discount"))
    lines = []
    for it, taxable in zip(items, taxables):
        qty = int(it.get("quantity") or 0)
        rate = float(it.get("price") or 0)
        gst = it.get("gst_percent")
        lines.append({
            "name": it.get("name") or "Item",
            "quantity": qty,
            "rate": rate,
            "amount": round(rate * qty, 2),
            "taxable": taxable,
            "gst_percent": float(gst) if gst is not None else None,
        })

    rows = []
    for b in (order.get("gst_breakdown") or []):
        tax = float(b.get("tax") or 0)
        cgst, sgst = split_tax(tax)
        rows.append({"rate": float(b.get("rate") or 0), "taxable": float(b.get("taxable") or 0),
                     "tax": tax, "cgst": cgst, "sgst": sgst})
    if not rows and float(order.get("gst_amount") or 0) > 0:
        # Orders from before per-item GST: one row, rate unknown.
        tax = float(order["gst_amount"])
        cgst, sgst = split_tax(tax)
        rows.append({"rate": None, "taxable": round(sum(taxables), 2), "tax": tax, "cgst": cgst, "sgst": sgst})
    tax_total = round(sum(r["tax"] for r in rows), 2)
    cgst_total = round(sum(_paise(r["cgst"]) for r in rows) / 100, 2)
    sgst_total = round(tax_total - cgst_total, 2)

    phone = order.get("phone_number")
    delivery = (order.get("fulfilment") or "pickup") == "delivery"
    status = order.get("status")
    return {
        "title": "Tax Invoice" if has_gstin else "Receipt",
        "is_tax_invoice": has_gstin,
        "cancelled": status == "cancelled",
        "invoice_number": order.get("invoice_number"),
        "invoice_date": order.get("invoice_date"),
        "order_id": order["id"],
        "short_id": order["id"][:8].upper(),
        "seller": {
            "legal_name": snap.get("legal_name"),
            "branch_name": snap.get("branch_name"),
            "address": snap.get("registered_address") or snap.get("branch_address"),
            "gstin": snap.get("gstin"),
            "state": snap.get("state"),
        },
        "place_of_supply": snap.get("state"),
        "customer": {
            "name": order.get("customer_name"),
            "phone": shop_identity.mask_phone(phone) if public else (f"+{phone}" if phone else None),
            "address": order.get("delivery_address") if delivery else None,
        },
        "fulfilment": "delivery" if delivery else "pickup",
        "currency": currency or "₹",
        "items": lines,
        "subtotal": float(order.get("subtotal") or 0),
        "items_discount": float(order.get("items_discount") or 0),
        "discount_label": order.get("offer_code") or order.get("offer_title"),
        "taxable_total": round(sum(taxables), 2),
        "tax_rows": rows,
        "tax_total": tax_total,
        "cgst_total": cgst_total,
        "sgst_total": sgst_total,
        "delivery_fee": float(order.get("delivery_fee") or 0) if delivery else 0.0,
        "delivery_discount": float(order.get("delivery_discount") or 0) if delivery else 0.0,
        "total": float(order.get("total") or 0),
        "amount_in_words": amount_in_words(order.get("total")),
        "payment_label": payment_label(order),
        "tax_note": "CGST and SGST at half the GST rate each (intra-state supply)." if rows else None,
    }


# ─────────────────────────────────────────────
# ISSUE
# ─────────────────────────────────────────────

def invoice_eligible(order: dict) -> bool:
    if not order or order.get("status") == "cancelled":
        return False
    if order.get("payment_status") == "paid":
        return True
    return order.get("status") not in (None, "pending")


def issue_invoice(order: dict, config: Optional[dict] = None, branch: Optional[dict] = None) -> Optional[dict]:
    """Give a paid / confirmed order its invoice number. Idempotent (the SQL
    function returns the existing number) and best-effort: any failure is
    reported and swallowed, never blocking the order. Returns the order with
    invoice_number / invoice_date / seller_snapshot set, or None."""
    try:
        if not invoice_eligible(order):
            return None
        if order.get("invoice_number"):
            return order
        project_id = order["project_id"]
        if config is None:
            res = _db().table("shop_config").select("*").eq("project_id", project_id).maybe_single().execute()
            config = (res.data if res else None) or {}
        if branch is None and order.get("branch_id"):
            b = _db().table("shop_branches").select("*").eq("id", order["branch_id"]) \
                .eq("project_id", project_id).limit(1).execute()
            branch = b.data[0] if b.data else None
        snap = seller_snapshot(config, branch)
        res = _db().rpc("shop_issue_invoice", {"p_order_id": order["id"], "p_snapshot": snap}).execute()
        data = res.data
        row = (data[0] if isinstance(data, list) and data else data) or {}
        number = row.get("out_invoice_number")
        if not number:
            return None
        # Only the first call writes the snapshot; read the stored one back.
        stored = _db().table("orders").select("seller_snapshot").eq("id", order["id"]).limit(1).execute()
        stored_snap = (stored.data[0].get("seller_snapshot") if stored.data else None) or snap
        return {**order, "invoice_number": number, "invoice_date": row.get("out_invoice_date"),
                "seller_snapshot": stored_snap}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"issue_invoice failed for order {(order or {}).get('id')}: {e}")
        return None


def invoice_url(project_id: str, order_id: str) -> str:
    """Public invoice link; the receipt in `r` is the per-order credential."""
    return f"{FRONTEND_URL}/shop/{project_id}/invoice/{order_id}?r={shop_identity.order_receipt(order_id) or ''}"


def invoice_line(order: Optional[dict]) -> str:
    """'🧾 Invoice: <url>\\n' for a WhatsApp message, or '' when none."""
    if not order or not order.get("invoice_number"):
        return ""
    return f"🧾 Invoice: {invoice_url(order['project_id'], order['id'])}\n"


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

def _currency(project_id: str) -> str:
    try:
        res = _db().table("shop_config").select("currency").eq("project_id", project_id).maybe_single().execute()
        return ((res.data if res else None) or {}).get("currency") or "₹"
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return "₹"


@router.post("/public/shop/{project_id}/orders/{order_id}/invoice")
def public_invoice(project_id: str, order_id: str, body: OrderAuthBody, request: Request):
    _public_shop_guard(project_id, request, "shop-invoice", limit=30)
    order = _order_for_customer(project_id, order_id, body)
    if not order.get("invoice_number"):
        raise shop_error(404, "invoice_not_ready", "The invoice isn't ready yet.")
    return build_invoice(order, _currency(project_id), public=True)


@router.get("/orders/{order_id}/invoice")
def dashboard_invoice(order_id: str, user=Depends(verify_token)):
    not_found = shop_error(404, "order_not_found", "Order not found")
    if not _UUID_RE.match(order_id or ""):
        raise not_found
    res = _db().table("orders").select("*").eq("id", order_id).limit(1).execute()
    order = res.data[0] if res.data else None
    if not order:
        raise not_found
    require_project_access(user.id, order["project_id"], tab="shop")
    if not order.get("invoice_number"):
        raise shop_error(404, "invoice_not_ready", "This order has no invoice yet. It gets one when it is paid or confirmed.")
    return build_invoice(order, _currency(order["project_id"]), public=False)


class InvoiceSettings(BaseModel):
    legal_name: Optional[str] = Field(default=None, max_length=400)
    gstin: Optional[str] = Field(default=None, max_length=40)
    registered_address: Optional[str] = Field(default=None, max_length=1000)
    state: Optional[str] = Field(default=None, max_length=100)


def clean_seller_fields(data: dict) -> dict:
    """Validated seller columns for the keys present in `data` (shop or
    branch). ValueError with a merchant-readable message."""
    out = {}
    limits = {"legal_name": ("Legal name", 200), "registered_address": ("Registered address", 500),
              "state": ("State", 60)}
    for key, (label, n) in limits.items():
        if key in data:
            try:
                out[key] = shop_rules.clean_text(data.get(key), n)
            except ValueError:
                raise ValueError(f"{label} must be at most {n} characters")
    if "gstin" in data:
        out["gstin"] = normalize_gstin(data.get("gstin"))
    return out


@router.get("/shop-config/{project_id}/invoice")
def get_invoice_settings(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="shop")
    res = _db().table("shop_config").select(", ".join(SELLER_FIELDS)).eq("project_id", project_id) \
        .maybe_single().execute()
    data = (res.data if res else None) or {}
    return {f: data.get(f) for f in SELLER_FIELDS}


@router.put("/shop-config/{project_id}/invoice")
def update_invoice_settings(project_id: str, body: InvoiceSettings, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="shop", min_role="admin")
    try:
        fields = clean_seller_fields(body.model_dump())
    except ValueError as e:
        raise shop_error(400, "invalid_request", str(e))
    existing = _db().table("shop_config").select("id").eq("project_id", project_id).maybe_single().execute()
    if existing and existing.data:
        _db().table("shop_config").update(fields).eq("project_id", project_id).execute()
    else:
        _db().table("shop_config").insert({"project_id": project_id, **fields}).execute()
    return {f: fields.get(f) for f in SELLER_FIELDS}
