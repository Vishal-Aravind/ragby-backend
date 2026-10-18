"""Shop customer identity: signed customer tokens, order receipts and the
shop links that carry them.

A customer token is an HMAC-signed bearer link (7 days by default) that
proves "this browser was opened from a WhatsApp chat with this phone". It is
issued ONLY from verified WhatsApp contexts (flows.py, chat.py when the
channel is whatsapp, the order confirmation messages) and never from the
website flow engine, where the phone is whatever the visitor typed.

A receipt is a short HMAC bound to one order id. It grants status read and
cancel of that one order (the Razorpay return page uses it).

Both keys are derived from SHOP_LINK_SECRET (else the Supabase service key)
with a distinct label, so neither can be confused with any other HMAC in
the app. With no secret at all this FAILS CLOSED: nothing is issued and
nothing verifies.

Env:
  SHOP_LINK_SECRET              at least 32 random bytes (recommended)
  SHOP_TOKEN_TTL_DAYS           default 7
  SHOP_REQUIRE_CUSTOMER_TOKEN   default false; true = no legacy phone orders
"""
import base64
import hashlib
import hmac
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional, Tuple
from urllib.parse import urlencode

from config import FRONTEND_URL, SUPABASE_SERVICE_ROLE_KEY

TOKEN_KEY_LABEL = b"zavo:shop-customer-token:v1"
RECEIPT_KEY_LABEL = b"zavo:shop-order-receipt:v1"

MAX_TOKEN_LEN = 512
CLOCK_SKEW_SECONDS = 300
MAX_TOKEN_LIFETIME_SECONDS = 8 * 86400

_PHONE_RE = re.compile(r"^\d{8,15}$")
_DIGITS_RE = re.compile(r"^\d+$")


def _master() -> Optional[bytes]:
    m = os.getenv("SHOP_LINK_SECRET") or SUPABASE_SERVICE_ROLE_KEY
    return m.encode() if m else None          # None = FAIL CLOSED: issue nothing, verify nothing


def _key(label: bytes) -> Optional[bytes]:    # domain separation
    m = _master()
    return hmac.new(m, label, hashlib.sha256).digest() if m else None


def token_ttl_days() -> int:
    try:
        days = int(os.getenv("SHOP_TOKEN_TTL_DAYS") or 7)
    except ValueError:
        days = 7
    # Verification refuses anything longer than 8 days, so never issue one.
    return max(1, min(days, 7))


def require_customer_token() -> bool:
    return (os.getenv("SHOP_REQUIRE_CUSTOMER_TOKEN") or "").strip().lower() in ("1", "true", "yes", "on")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def clean_phone(raw) -> Optional[str]:
    """Digits of a legacy phone param, or None when it isn't 8-15 digits."""
    if raw is None:
        return None
    text = str(raw).replace("+", "").replace(" ", "").replace("-", "").strip()
    return text if _PHONE_RE.match(text) else None


# ─────────────────────────────────────────────
# CUSTOMER TOKEN
# ─────────────────────────────────────────────

def issue_customer_token(project_id: str, phone: str, now: Optional[float] = None) -> Optional[str]:
    key = _key(TOKEN_KEY_LABEL)
    if not key or not project_id or not phone or not _PHONE_RE.match(str(phone)):
        return None
    iat = int(now if now is not None else time.time())
    exp = iat + token_ttl_days() * 86400
    payload = json.dumps({"k": "c", "p": str(project_id).lower(), "ph": str(phone), "i": iat, "e": exp},
                         separators=(",", ":"), sort_keys=True)
    payload_b64 = _b64e(payload.encode())
    sig = hmac.new(key, ("v1." + payload_b64).encode(), hashlib.sha256).digest()
    return f"v1.{payload_b64}.{_b64e(sig)}"


def _decode(token, project_id: str, now: Optional[float] = None) -> Tuple[Optional[dict], Optional[str]]:
    if not token:
        return None, "missing"
    key = _key(TOKEN_KEY_LABEL)
    if not key:
        return None, "unavailable"
    if not isinstance(token, str) or len(token) > MAX_TOKEN_LEN:
        return None, "malformed"
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v1" or not parts[1] or not parts[2]:
        return None, "malformed"
    try:
        sig = _b64d(parts[2])
    except Exception:
        return None, "malformed"
    expected = hmac.new(key, ("v1." + parts[1]).encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, sig):
        return None, "bad_signature"
    try:
        data = json.loads(_b64d(parts[1]))
    except Exception:
        return None, "malformed"
    if not isinstance(data, dict) or data.get("k") != "c":
        return None, "malformed"
    if data.get("p") != str(project_id or "").lower():
        return None, "wrong_project"
    ph, iat, exp = data.get("ph"), data.get("i"), data.get("e")
    if not isinstance(ph, str) or not _DIGITS_RE.match(ph):
        return None, "malformed"
    if not isinstance(iat, int) or not isinstance(exp, int) or isinstance(iat, bool) or isinstance(exp, bool):
        return None, "malformed"
    t = now if now is not None else time.time()
    if exp <= t:
        return None, "expired"
    if iat > t + CLOCK_SKEW_SECONDS or exp - iat > MAX_TOKEN_LIFETIME_SECONDS:
        return None, "malformed"
    return data, None


def verify_customer_token(token, project_id: str, now: Optional[float] = None) -> Tuple[Optional[str], Optional[str]]:
    """(phone, None) for a valid token, else (None, reason). Reasons:
    missing, malformed, bad_signature, expired, wrong_project, unavailable."""
    data, reason = _decode(token, project_id, now)
    return (data["ph"], None) if data else (None, reason)


def token_expiry(token, project_id: str) -> Optional[str]:
    data, _ = _decode(token, project_id)
    if not data:
        return None
    return datetime.fromtimestamp(data["e"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_identity(project_id: str, token, legacy_phone=None) -> dict:
    """{"phone", "verified", "reason"}. A valid token always wins (the body
    phone is then ignored); otherwise the legacy phone, unverified."""
    phone, reason = verify_customer_token(token, project_id)
    if phone:
        return {"phone": phone, "verified": True, "reason": None}
    return {"phone": clean_phone(legacy_phone), "verified": False, "reason": reason}


def mask_phone(phone: Optional[str]) -> Optional[str]:
    """919876543210 -> "+91 98•••••210"."""
    if not phone:
        return None
    digits = str(phone)
    if len(digits) > 10:
        cc, national = digits[:-10], digits[-10:]
    else:
        cc, national = "", digits
    keep_head, keep_tail = 2, 3
    if len(national) <= keep_head + keep_tail:
        masked = national[:1] + "•" * max(0, len(national) - 1)
    else:
        masked = national[:keep_head] + "•" * (len(national) - keep_head - keep_tail) + national[-keep_tail:]
    return f"+{cc} {masked}" if cc else masked


# ─────────────────────────────────────────────
# ORDER RECEIPT
# ─────────────────────────────────────────────

def order_receipt(order_id: str) -> Optional[str]:
    key = _key(RECEIPT_KEY_LABEL)
    if not key or not order_id:
        return None
    return _b64e(hmac.new(key, f"r1|{str(order_id).lower()}".encode(), hashlib.sha256).digest()[:16])


def verify_receipt(order_id: str, receipt) -> bool:
    if not receipt or not isinstance(receipt, str) or len(receipt) > 64:
        return False
    expected = order_receipt(order_id)
    return bool(expected) and hmac.compare_digest(expected, receipt)


# ─────────────────────────────────────────────
# LINKS
# ─────────────────────────────────────────────

def build_shop_url(project_id: str, phone: Optional[str] = None, catalog_id: Optional[str] = None,
                   order_id: Optional[str] = None, branch_id: Optional[str] = None,
                   view: Optional[str] = None) -> str:
    """The customer shop link. `t` is added only when a phone is given and
    a token can be issued; only call this with a VERIFIED phone (a WhatsApp
    sender), never a number a visitor typed."""
    params = [("catalog", catalog_id), ("phone", phone), ("order_id", order_id),
              ("branch", branch_id), ("view", view)]
    if phone:
        params.append(("t", issue_customer_token(project_id, str(phone))))
    query = urlencode([(k, str(v)) for k, v in params if v])
    base = f"{FRONTEND_URL}/shop/{project_id}"
    return f"{base}?{query}" if query else base


def build_return_url(project_id: str, order_id: str) -> str:
    return f"{FRONTEND_URL}/shop/{project_id}?" + urlencode({"paid": order_id, "r": order_receipt(order_id) or ""})
