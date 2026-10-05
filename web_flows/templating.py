"""{{variable}} templating for website flows. Pure functions, no I/O.

Merchants write "Thanks {{name}}!" or "Hi {{name|there}}" in node text. The
values come from visitors (form answers, webhook responses), so the rules
are strict:
  - one pass, never re-expanded: a visitor typing "{{email}}" as their name
    gets "{{email}}" back, not someone else's data;
  - output is plain text — the widget renders it with textContent, never as
    HTML — and capped in length;
  - inside URLs, templates are allowed only in the path/query and values are
    percent-encoded, so a value can't change which host is called.
"""
import re
from decimal import Decimal, InvalidOperation
from urllib.parse import quote, urlsplit

VAR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_TEMPLATE_RE = re.compile(r"\{\{\s*([a-z][a-z0-9_]{0,31})\s*(?:\|([^{}]{0,100}))?\}\}")

MAX_TEXT_LEN = 4000
MAX_VALUE_LEN = 1000
MAX_VARIABLES = 50

# Filled in by the engine; merchants can read them but not set them.
SYSTEM_VARS = {"page_url", "page_path", "page_title", "webhook_status"}


def is_valid_var_name(name) -> bool:
    return isinstance(name, str) and bool(VAR_NAME_RE.match(name))


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)[:MAX_VALUE_LEN]


def render_text(template, variables: dict, max_len: int = MAX_TEXT_LEN) -> str:
    """Expand {{var}} / {{var|fallback}} once. Unknown or empty variables use
    the fallback (or nothing)."""
    if not template:
        return ""
    template = str(template)

    def sub(m):
        value = _as_text((variables or {}).get(m.group(1)))
        if value == "" and m.group(2) is not None:
            return m.group(2).strip()
        return value

    out = _TEMPLATE_RE.sub(sub, template)
    return out[:max_len]


def template_vars(template) -> set:
    """Variable names referenced by a template (for editor warnings/tests)."""
    return {m.group(1) for m in _TEMPLATE_RE.finditer(str(template or ""))}


def render_url(template, variables: dict) -> str:
    """Expand a URL template. Templates may only appear after the host;
    values are percent-encoded. Raises ValueError for anything else."""
    template = (template or "").strip()
    parts = urlsplit(template)
    if "{{" in parts.scheme or "{{" in parts.netloc:
        raise ValueError("Variables can only be used in the path or query of a URL.")

    def sub(m):
        value = _as_text((variables or {}).get(m.group(1)))
        if value == "" and m.group(2) is not None:
            value = m.group(2).strip()
        return quote(value, safe="")

    return _TEMPLATE_RE.sub(sub, template)


def coerce_value(value):
    """Store answers as typed JSON: numbers stay numbers so conditions like
    budget > 50 work, everything else is a trimmed, capped string."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()[:MAX_VALUE_LEN]
    return text


def to_decimal(value):
    """Parse a value as a number for comparisons ("1,50,000" and "Rs 200"
    included). Returns None when it isn't one."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    text = re.sub(r"[^\d.\-]", "", str(value))
    if not text or text in ("-", ".", "-."):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def set_variable(variables: dict, name: str, value) -> bool:
    """Assign within the limits. Returns False when refused (bad name, system
    variable, or too many variables)."""
    if not is_valid_var_name(name) or name in SYSTEM_VARS:
        return False
    if name not in variables and len(variables) >= MAX_VARIABLES:
        return False
    variables[name] = coerce_value(value)
    return True
