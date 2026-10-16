"""One way to turn whatever a merchant typed or uploaded into an exact
international phone number - or refuse it.

Why this exists: campaigns used to put "+" in front of whatever digits they
had. Meta reads "+" as "the country code comes first", so a local Indian
number like 9444032687 went out as +94 (Sri Lanka) 44032687 - a stranger,
billed. Numbers are now checked against Google's libphonenumber rules for a
known country, and anything that can't be made exact is skipped with a
reason instead of guessed.
"""
import re
from typing import Optional, Tuple

import phonenumbers
import sentry_sdk

from clients import supabase

FALLBACK_REGION = "IN"

_EXCEL_RE = re.compile(r"\d[.,]?\d*\s*e\s*\+?\s*\d+", re.I)
_STRIP_RE = re.compile(r"[\s\-().‎‏‪-‮]")


def valid_region(region) -> Optional[str]:
    region = str(region or "").strip().upper()
    return region if region in phonenumbers.SUPPORTED_REGIONS else None


def calling_code(region: str) -> int:
    return phonenumbers.country_code_for_region(region)


def _parse_valid(text: str, region: Optional[str]):
    try:
        num = phonenumbers.parse(text, region)
    except phonenumbers.NumberParseException:
        return None
    return num if phonenumbers.is_valid_number(num) else None


def normalize(raw, region: str, international_first: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """(digits with country code, None) or (None, reason a merchant can act on).

    - "+..." or "00..." is read as international.
    - Anything else is read as a number from `region` first; only if that
      isn't a valid number there is it tried as international (so numbers
      saved as 919876543210 keep working).
    - international_first flips that order, for numbers that came from
      WhatsApp itself (always country code + number): a Singapore contact
      6591234567 is also a valid-looking Indian number, and must stay +65.
    """
    region = valid_region(region) or FALLBACK_REGION
    if raw is None:
        return None, "No number"
    text = str(raw).strip()
    if not text:
        return None, "No number"
    # Excel shows long numbers as 9.19884E+11 and saves them that way in a
    # CSV - the real last digits are gone, so never try to rebuild them.
    if _EXCEL_RE.fullmatch(text.replace(" ", "")):
        return None, "Excel changed this number (9.19E+11). Format the phone column as Text and upload again"
    if re.fullmatch(r"\d+\.0+", text):        # 9150530543.0 from a number cell
        text = text.split(".")[0]
    text = _STRIP_RE.sub("", text)
    if text.startswith("00"):
        text = "+" + text[2:]
    if not re.fullmatch(r"\+?\d{4,17}", text):
        return None, "Not a phone number"

    if text.startswith("+"):
        num = _parse_valid(text, None)
        if not num:
            return None, "Not a valid number"
    else:
        local = lambda: _parse_valid(text, region)
        intl = lambda: _parse_valid("+" + text, None)
        num = (intl() or local()) if international_first else (local() or intl())
        if not num:
            return None, (f"Not a valid +{calling_code(region)} number - "
                          "if it's from another country, add its country code")
    return f"{num.country_code}{num.national_number}", None


def pretty(digits: str) -> str:
    """+91 91505 30543 - for showing the merchant exactly what will be used."""
    try:
        return phonenumbers.format_number(phonenumbers.parse("+" + str(digits), None),
                                          phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    except phonenumbers.NumberParseException:
        return "+" + str(digits)


def default_region_for_project(project_id: str) -> str:
    """The country of the business's own WhatsApp number - the same default
    Meta itself uses for a number written without a code."""
    try:
        res = supabase.table("whatsapp_integrations").select("display_phone_number") \
            .eq("project_id", project_id).limit(1).execute()
        shown = (res.data[0].get("display_phone_number") if res.data else "") or ""
        digits = re.sub(r"\D", "", shown)
        if digits:
            region = phonenumbers.region_code_for_number(phonenumbers.parse("+" + digits, None))
            if valid_region(region):
                return region
    except Exception as e:
        sentry_sdk.capture_exception(e)
    return FALLBACK_REGION
