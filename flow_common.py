"""Pure helpers shared by the WhatsApp flow engine (flows.py) and the website
flow engine (web_flows/). No I/O, no channel-specific behaviour."""
import re


def option_id(label: str) -> str:
    """The id a button/list option is sent with — MUST match optionId() in
    the editor's nodeRegistry.js, which saves each connection under it. They
    used to differ ("Price?" -> "price?" here, "price" there), so tapping
    such an option found no connection and nothing happened."""
    text = (label or "").strip()
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if slug:
        return slug
    h = 5381
    for b in text.encode("utf-8"):
        h = ((h * 33) ^ b) & 0xFFFFFFFF
    digits, out = "0123456789abcdefghijklmnopqrstuvwxyz", ""
    while True:
        h, r = divmod(h, 36)
        out = digits[r] + out
        if not h:
            break
    return "opt_" + out
