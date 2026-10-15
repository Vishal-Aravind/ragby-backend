"""
Custom WhatsApp message templates: create one from Zavo (text or image
header, body with variables, footer, buttons), list every template in the
account, and delete one.

The rules Meta enforces are checked here first (build_components), so a
merchant sees "a variable can't be at the very start" in plain words instead
of a cryptic rejection a few minutes later. Meta still has the final say -
its own explanation is passed back when it refuses.
"""
import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import sentry_sdk

from auth import verify_token, require_project_access
from campaigns import _wa_integration
from config import META_APP_ID, SUPABASE_URL
from ratelimit import is_rate_limited
from whatsapp import http

router = APIRouter()

GRAPH = "https://graph.facebook.com/v25.0"

# English + the main Indian languages (Meta's codes).
LANGUAGES = {
    "en_US": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
}
CATEGORIES = {"MARKETING", "UTILITY"}

NAME_RE = re.compile(r"^[a-z0-9_]{1,512}$")
VAR_RE = re.compile(r"\{\{\s*(\d+)\s*\}\}")
PHONE_RE = re.compile(r"^\+[1-9]\d{6,14}$")

MAX_BODY = 1024
MAX_HEADER = 60
MAX_FOOTER = 60
MAX_BUTTON_TEXT = 25
MAX_BUTTONS = 10
MAX_URL_BUTTONS = 2
MAX_PHONE_BUTTONS = 1
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class TemplateButton(BaseModel):
    type: str = Field(max_length=20)                  # quick_reply | url | phone
    text: str = Field(default="", max_length=60)
    url: Optional[str] = Field(default=None, max_length=2000)
    phone: Optional[str] = Field(default=None, max_length=20)


class TemplateSpec(BaseModel):
    project_id: str
    name: str = Field(max_length=512)
    category: str = Field(max_length=20)
    language: str = Field(max_length=10)
    header_type: str = Field(default="none", max_length=10)    # none | text | image
    header_text: Optional[str] = Field(default=None, max_length=200)
    header_example: Optional[str] = Field(default=None, max_length=200)
    header_image_url: Optional[str] = Field(default=None, max_length=1000)
    body: str = Field(max_length=2000)
    body_examples: List[str] = Field(default_factory=list, max_length=20)
    footer: Optional[str] = Field(default=None, max_length=200)
    buttons: List[TemplateButton] = Field(default_factory=list, max_length=12)


def _vars_in(text: str) -> List[int]:
    return [int(n) for n in VAR_RE.findall(text or "")]


def _check_variables(text: str, where: str, errors: list) -> int:
    """Variables must be {{1}}, {{2}}, ... with no gaps, and Meta refuses a
    text that starts or ends with one or has two side by side."""
    nums = _vars_in(text)
    if not nums:
        return 0
    distinct = sorted(set(nums))
    if distinct != list(range(1, len(distinct) + 1)):
        errors.append(f"{where}: number the variables {{{{1}}}}, {{{{2}}}}, ... in order, without gaps.")
    stripped = text.strip()
    if VAR_RE.match(stripped):
        errors.append(f"{where}: can't start with a variable - put a word before it.")
    if re.search(r"\{\{\s*\d+\s*\}\}$", stripped):
        errors.append(f"{where}: can't end with a variable - put a word or a full stop after it.")
    if re.search(r"\}\}\s*\{\{", stripped):
        errors.append(f"{where}: two variables can't be right next to each other - put a word between them.")
    return len(distinct)


def build_components(spec: TemplateSpec, header_handle: Optional[str] = None):
    """Check a template against Meta's rules and build its components.
    Returns (errors, components). header_handle is the uploaded image's
    handle (only needed for an image header; None while just validating)."""
    errors = []
    if not NAME_RE.match(spec.name or ""):
        errors.append("Name: lowercase letters, numbers and _ only (e.g. diwali_offer).")
    if spec.category not in CATEGORIES:
        errors.append("Category: choose Marketing or Utility.")
    if spec.language not in LANGUAGES:
        errors.append("Language: choose one from the list.")

    components = []

    # Header
    if spec.header_type == "text":
        text = (spec.header_text or "").strip()
        if not text:
            errors.append("Header: type the header text, or choose no header.")
        elif len(text) > MAX_HEADER:
            errors.append(f"Header: at most {MAX_HEADER} characters.")
        n = _vars_in(text)
        header = {"type": "HEADER", "format": "TEXT", "text": text}
        if n:
            if set(n) != {1} or len(n) > 1:
                errors.append("Header: it can have only one variable, {{1}}.")
            if not (spec.header_example or "").strip():
                errors.append("Header: add an example value for {{1}}.")
            header["example"] = {"header_text": [(spec.header_example or "").strip()]}
        components.append(header)
    elif spec.header_type == "image":
        if not (spec.header_image_url or "").startswith("https://"):
            errors.append("Header: upload an image (JPG or PNG).")
        components.append({"type": "HEADER", "format": "IMAGE",
                           "example": {"header_handle": [header_handle or "pending"]}})
    elif spec.header_type != "none":
        errors.append("Header: choose none, text or image.")

    # Body
    body = (spec.body or "").strip()
    if not body:
        errors.append("Message: type the message.")
    elif len(body) > MAX_BODY:
        errors.append(f"Message: at most {MAX_BODY} characters.")
    count = _check_variables(body, "Message", errors) if body else 0
    examples = [str(e).strip() for e in (spec.body_examples or [])][:count]
    if count and (len(examples) < count or not all(examples)):
        errors.append("Message: add an example value for every variable.")
    body_comp = {"type": "BODY", "text": body}
    if count:
        body_comp["example"] = {"body_text": [examples]}
    components.append(body_comp)

    # Footer
    footer = (spec.footer or "").strip()
    if footer:
        if len(footer) > MAX_FOOTER:
            errors.append(f"Footer: at most {MAX_FOOTER} characters.")
        if _vars_in(footer):
            errors.append("Footer: variables aren't allowed in the footer.")
        components.append({"type": "FOOTER", "text": footer})

    # Buttons - quick replies first, then links/calls (Meta wants them grouped).
    quick, cta = [], []
    for i, b in enumerate(spec.buttons or []):
        label = (b.text or "").strip()
        where = f"Button {i + 1}"
        if not label:
            errors.append(f"{where}: type the button text.")
        elif len(label) > MAX_BUTTON_TEXT:
            errors.append(f"{where}: at most {MAX_BUTTON_TEXT} characters.")
        if b.type == "quick_reply":
            quick.append({"type": "QUICK_REPLY", "text": label})
        elif b.type == "url":
            url = (b.url or "").strip()
            if not url.startswith("https://") and not url.startswith("http://"):
                errors.append(f"{where}: enter a full link starting with https://")
            if "{{" in url:
                errors.append(f"{where}: links with variables aren't supported - use a fixed link.")
            cta.append({"type": "URL", "text": label, "url": url})
        elif b.type == "phone":
            phone = re.sub(r"[\s-]", "", b.phone or "")
            if not PHONE_RE.match(phone):
                errors.append(f"{where}: enter the number with country code, e.g. +919876543210.")
            cta.append({"type": "PHONE_NUMBER", "text": label, "phone_number": phone})
        else:
            errors.append(f"{where}: choose quick reply, website or call.")
    if len(quick) + len(cta) > MAX_BUTTONS:
        errors.append(f"Buttons: at most {MAX_BUTTONS}.")
    if sum(1 for b in cta if b["type"] == "URL") > MAX_URL_BUTTONS:
        errors.append(f"Buttons: at most {MAX_URL_BUTTONS} website buttons.")
    if sum(1 for b in cta if b["type"] == "PHONE_NUMBER") > MAX_PHONE_BUTTONS:
        errors.append("Buttons: only one call button.")
    if quick or cta:
        components.append({"type": "BUTTONS", "buttons": quick + cta})

    return errors, components


def _upload_header_image(image_url: str, token: str) -> str:
    """Meta needs a template's sample image uploaded to it first (resumable
    upload) and refers to it by a handle. Only images in our own storage are
    fetched, so this can't be pointed at an arbitrary address."""
    allowed = f"{(SUPABASE_URL or '').rstrip('/')}/storage/v1/object/public/flow-media/"
    if not SUPABASE_URL or not image_url.startswith(allowed):
        raise HTTPException(status_code=400, detail="Header: upload the image again.")
    if not META_APP_ID:
        raise HTTPException(status_code=503, detail="Image templates aren't available right now. Please try again later.")

    img = http.get(image_url, timeout=20)
    if not img.ok or not img.content:
        raise HTTPException(status_code=400, detail="Header: couldn't read the uploaded image. Upload it again.")
    if len(img.content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Header: the image must be 5MB or smaller.")
    ctype = (img.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Header: use a JPG or PNG image.")

    start = http.post(f"{GRAPH}/{META_APP_ID}/uploads", params={
        "file_name": "header." + ("png" if ctype == "image/png" else "jpg"),
        "file_length": len(img.content), "file_type": ctype, "access_token": token,
    })
    if not start.ok or not start.json().get("id"):
        print(f"Meta upload session failed: {start.text}")
        raise HTTPException(status_code=400, detail="Header: WhatsApp couldn't accept the image. Try another one.")
    done = http.post(f"{GRAPH}/{start.json()['id']}",
                     headers={"Authorization": f"OAuth {token}", "file_offset": "0"},
                     data=img.content)
    handle = done.json().get("h") if done.ok else None
    if not handle:
        print(f"Meta upload failed: {done.text}")
        raise HTTPException(status_code=400, detail="Header: WhatsApp couldn't accept the image. Try another one.")
    return handle


def _meta_error_text(resp) -> str:
    """Meta's explanation written for people (error_user_msg), not its raw
    error object, which can name internal ids."""
    try:
        err = resp.json().get("error") or {}
    except Exception:
        return ""
    msg = " ".join(x for x in (err.get("error_user_title"), err.get("error_user_msg")) if x)
    return msg.strip()[:400]


@router.get("/templates/meta/{project_id}")
def template_options(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="templates")
    return {"languages": LANGUAGES, "categories": sorted(CATEGORIES)}


@router.post("/templates/create")
def create_template(spec: TemplateSpec, user=Depends(verify_token)):
    require_project_access(user.id, spec.project_id, tab="templates", min_role="admin")
    if is_rate_limited(f"tpl-create:{spec.project_id}", limit=20, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Too many templates submitted recently. Please wait a little.")

    errors, _ = build_components(spec)
    if errors:
        raise HTTPException(status_code=400, detail=" ".join(errors))

    wa = _wa_integration(spec.project_id)
    if not wa["waba_id"]:
        raise HTTPException(status_code=400, detail="WhatsApp not connected")

    handle = _upload_header_image(spec.header_image_url, wa["token"]) if spec.header_type == "image" else None
    _, components = build_components(spec, header_handle=handle)

    res = http.post(
        f"{GRAPH}/{wa['waba_id']}/message_templates",
        headers={"Authorization": f"Bearer {wa['token']}", "Content-Type": "application/json"},
        json={"name": spec.name, "category": spec.category, "language": spec.language, "components": components},
    )
    if res.ok:
        data = res.json()
        return {"status": data.get("status") or "PENDING", "id": data.get("id")}

    text = _meta_error_text(res)
    raw = res.text.lower()
    print(f"Meta template create failed: {res.text}")
    if "already exists" in raw or "same name" in raw:
        raise HTTPException(status_code=400, detail="A template with this name already exists in this language. Choose another name.")
    raise HTTPException(status_code=400, detail=text or "WhatsApp didn't accept this template. Check the wording and try again.")


class DeleteTemplate(BaseModel):
    project_id: str
    name: str = Field(max_length=512)
    id: Optional[str] = Field(default=None, max_length=64)


@router.post("/templates/delete")
def delete_template(req: DeleteTemplate, user=Depends(verify_token)):
    require_project_access(user.id, req.project_id, tab="templates", min_role="admin")
    if not NAME_RE.match(req.name or ""):
        raise HTTPException(status_code=400, detail="Invalid template name")
    wa = _wa_integration(req.project_id)
    if not wa["waba_id"]:
        raise HTTPException(status_code=400, detail="WhatsApp not connected")
    params = {"name": req.name}
    if req.id and req.id.isdigit():
        params["hsm_id"] = req.id          # just this language, not every language of the name
    res = http.delete(f"{GRAPH}/{wa['waba_id']}/message_templates",
                      headers={"Authorization": f"Bearer {wa['token']}"}, params=params)
    if res.ok:
        return {"status": "deleted"}
    print(f"Meta template delete failed: {res.text}")
    sentry_sdk.capture_message("Meta template delete failed")
    raise HTTPException(status_code=400, detail=_meta_error_text(res) or "WhatsApp couldn't delete this template. Try again.")


# -------------------------------------------------
# WHATSAPP BILLING (can template messages be delivered?)
# -------------------------------------------------
# Template messages are paid: without a payment method on the WhatsApp
# account, Meta accepts them and then fails delivery with code 131042
# ("Business eligibility payment issue"). Free in-window replies keep
# working, so nothing looked wrong. Two signals:
#   - the account's primary_funding_id from Meta (set = a payment method),
#   - a 131042 failure recorded on any chat in the last 7 days.
_billing_cache: dict = {}
_BILLING_TTL = 600


@router.get("/whatsapp/billing-status/{project_id}")
def billing_status(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id)
    import time as _time
    from datetime import datetime, timedelta, timezone
    from clients import supabase

    hit = _billing_cache.get(project_id)
    if hit and _time.time() - hit[0] < _BILLING_TTL:
        return hit[1]

    try:
        wa = _wa_integration(project_id)
    except HTTPException:
        return {"status": "not_connected"}

    recent_error = False
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        rows = supabase.table("chats").select("id").eq("project_id", project_id) \
            .eq("channel", "whatsapp").gte("last_send_error_at", since) \
            .ilike("last_send_error", "%131042%").limit(1).execute().data
        recent_error = bool(rows)
    except Exception as e:
        sentry_sdk.capture_exception(e)

    funding, checked = None, False
    if wa.get("waba_id"):
        try:
            res = http.get(f"{GRAPH}/{wa['waba_id']}", params={"fields": "primary_funding_id"},
                           headers={"Authorization": f"Bearer {wa['token']}"}, timeout=8)
            if res.ok:
                checked = True
                funding = res.json().get("primary_funding_id")
            else:
                print(f"WhatsApp billing check not available: {res.status_code}")
        except Exception as e:
            sentry_sdk.capture_exception(e)

    if funding:
        result = {"status": "ok"}                 # a payment method is set (newer than any old error)
    elif recent_error:
        result = {"status": "missing", "source": "failed_message"}
    elif checked:
        result = {"status": "missing", "source": "meta"}
    else:
        result = {"status": "unknown"}
    _billing_cache[project_id] = (_time.time(), result)
    return result
