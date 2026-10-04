"""Is a pasted media link something WhatsApp can actually send?

A flow media node sends its URL to Meta, which downloads it later. When the
link is a web page (YouTube, a Google Images result) or a format WhatsApp
doesn't take (WEBP, MOV), Meta still accepts the send call and then quietly
never delivers it — the customer gets nothing and nobody sees an error.
So the link is fetched here first and its real Content-Type checked: the
editor shows the result as you paste, and send_node falls back to a text
message with the link when the check fails.
"""
import time

import requests
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth import verify_token
from sources.url_guard import safe_get

router = APIRouter()

# What the WhatsApp Cloud API accepts per media type, and its size caps.
_ALLOWED = {
    "image": ({"image/jpeg", "image/png"}, 5, "a JPG or PNG image"),
    "video": ({"video/mp4", "video/3gpp"}, 16, "an MP4 or 3GP video"),
    "audio": ({"audio/aac", "audio/amr", "audio/mpeg", "audio/mp4", "audio/ogg"}, 16, "an MP3, AAC, M4A or OGG audio file"),
    "document": (None, 100, "a document file"),
}

_cache = {}  # url+kind -> (result, checked_at)
_CACHE_SECONDS = 600


def check_media_link(url: str, kind: str) -> dict:
    """Returns {"ok": bool, "reason": str}. Never raises."""
    if kind not in _ALLOWED or not url:
        return {"ok": False, "reason": "No link."}
    # Our own uploads are already restricted to WhatsApp's types.
    if "/storage/v1/object/public/" in url:
        return {"ok": True, "reason": ""}

    key = f"{kind}|{url}"
    hit = _cache.get(key)
    if hit and time.time() - hit[1] < _CACHE_SECONDS:
        return hit[0]

    types, max_mb, wanted = _ALLOWED[kind]
    try:
        res = safe_get(requests.Session(), url, timeout=6, stream=True,
                       headers={"User-Agent": "Mozilla/5.0 (compatible; ZavoBot/1.0)"})
        status = res.status_code
        ctype = (res.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        size = int(res.headers.get("Content-Length") or 0)
        res.close()
    except ValueError as e:
        result = {"ok": False, "reason": str(e)}
    except Exception:
        result = {"ok": False, "reason": "Couldn't open this link."}
    else:
        if status != 200:
            result = {"ok": False, "reason": f"This link didn't open (error {status})."}
        elif ctype == "text/html" or (types is not None and ctype not in types) or (
            types is None and not (ctype.startswith("application/") or ctype in ("text/plain", "text/csv"))
        ):
            result = {"ok": False, "reason": f"This link isn't {wanted} WhatsApp can send — it's a web page or another format."}
        elif size and size > max_mb * 1024 * 1024:
            result = {"ok": False, "reason": f"This file is over WhatsApp's {max_mb}MB limit."}
        else:
            result = {"ok": True, "reason": ""}

    _cache[key] = (result, time.time())
    if len(_cache) > 500:
        _cache.clear()
    return result


class MediaLinkCheck(BaseModel):
    url: str
    kind: str


@router.post("/flows/check-media-link")
def check_media_link_route(body: MediaLinkCheck, user=Depends(verify_token)):
    return check_media_link(body.url.strip(), body.kind)
