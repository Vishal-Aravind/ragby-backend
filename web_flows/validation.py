"""Validate visitor answers (ask_input, form fields, rating). Pure, no I/O.

The widget validates too, but only this is trusted: anything stored as a
variable or sent to Leads/webhooks has passed through here.
"""
import re
from datetime import date, timedelta

from .templating import MAX_VALUE_LEN, to_decimal

# Same rule as leads._EMAIL_RE (kept in sync on purpose — leads.py imports
# the database client, this module must stay importable without it).
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")
INPUT_TYPES = {"text", "email", "phone", "number", "date"}
FIELD_TYPES = INPUT_TYPES | {"choice", "consent"}


def _parse_date_bound(value, today: date):
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text == "today":
        return today
    m = re.match(r"^today([+-]\d{1,4})$", text)
    if m:
        return today + timedelta(days=int(m.group(1)))
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def check_value(field_type: str, raw, spec: dict, today: date):
    """Returns (ok, cleaned_value, error_message)."""
    spec = spec or {}
    required = spec.get("required", True)

    if field_type == "consent":
        given = raw is True or str(raw).strip().lower() in ("true", "yes", "1", "on")
        if required and not given:
            return False, None, "Please tick this box to continue."
        return True, given, ""

    text = "" if raw is None else str(raw).strip()
    if text == "":
        if required:
            return False, None, "This is required."
        return True, "", ""
    if len(text) > MAX_VALUE_LEN:
        return False, None, f"Please keep this under {MAX_VALUE_LEN} characters."

    if field_type == "email":
        if not EMAIL_RE.match(text):
            return False, None, "Please enter a valid email address."
        return True, text.lower(), ""
    if field_type == "phone":
        digits = re.sub(r"\D", "", text)
        if not re.match(r"^\+?[\d\s\-()]+$", text) or not 7 <= len(digits) <= 15:
            return False, None, "Please enter a valid phone number."
        return True, ("+" if text.startswith("+") else "") + digits, ""
    if field_type == "number":
        num = to_decimal(text)
        if num is None or not re.match(r"^-?[\d,]*\.?\d+$", text.replace(" ", "")):
            return False, None, "Please enter a number."
        lo, hi = to_decimal(spec.get("min")), to_decimal(spec.get("max"))
        if lo is not None and num < lo:
            return False, None, f"Please enter {spec.get('min')} or more."
        if hi is not None and num > hi:
            return False, None, f"Please enter {spec.get('max')} or less."
        return True, float(num) if num != num.to_integral_value() else int(num), ""
    if field_type == "date":
        try:
            value = date.fromisoformat(text[:10])
        except ValueError:
            return False, None, "Please pick a date."
        lo, hi = _parse_date_bound(spec.get("min"), today), _parse_date_bound(spec.get("max"), today)
        if lo and value < lo:
            return False, None, f"Please pick {lo.isoformat()} or later."
        if hi and value > hi:
            return False, None, f"Please pick {hi.isoformat()} or earlier."
        return True, value.isoformat(), ""
    if field_type == "choice":
        options = [str(o).strip() for o in (spec.get("options") or []) if str(o).strip()]
        match = next((o for o in options if o.lower() == text.lower()), None)
        if match is None:
            return False, None, "Please choose one of the options."
        return True, match, ""
    # text
    return True, text, ""


def check_rating(raw, style: str):
    hi = 10 if style == "nps" else 5
    lo = 0 if style == "nps" else 1
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return False, None, "Please pick a rating."
    if not lo <= value <= hi:
        return False, None, "Please pick a rating."
    return True, value, ""
