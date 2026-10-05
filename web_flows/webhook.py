"""The webhook node: call the merchant's own system mid-flow.

The URL, headers and body come from the merchant, the values from visitors,
and the request is made from OUR server — so this is the riskiest node:
  - https only, and every hop (redirects included) must resolve to a public
    address (sources/url_guard) — no reaching our metadata/internal network;
  - the body is built as a dict and JSON-encoded, never string-templated, so
    a visitor's answer can't inject JSON;
  - hard timeouts, a 64KB response cap and a process-wide concurrency cap, so
    a slow or huge merchant endpoint can't tie up the 512MB server;
  - URL/headers never leave the server (the engine only returns which
    branch was taken).
"""
import json
import threading
from urllib.parse import urljoin

import requests
import sentry_sdk

from sources.url_guard import assert_public_http_url
from .templating import render_text, render_url, is_valid_var_name

ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH"}
BLOCKED_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "cookie"}
MAX_HEADERS = 10
MAX_BODY_FIELDS = 30
MAX_MAPPINGS = 20
MAX_RESPONSE_BYTES = 64 * 1024
MAX_REDIRECTS = 3
TIMEOUT = (3, 5)  # connect, read

_slots = threading.BoundedSemaphore(8)


def _json_path(data, path: str):
    """"data.items.0.tier" -> data["data"]["items"][0]["tier"]; None if absent."""
    cur = data
    for part in [p for p in str(path or "").split(".") if p != ""]:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            idx = int(part)
            cur = cur[idx] if -len(cur) <= idx < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


def build_request(content: dict, variables: dict) -> dict:
    """Validate the node and build the request. Raises ValueError with a
    merchant-readable reason (shown in the editor's test, never to visitors)."""
    method = str(content.get("method") or "POST").upper()
    if method not in ALLOWED_METHODS:
        raise ValueError("Method must be GET, POST, PUT or PATCH.")
    url = render_url(content.get("url") or "", variables)
    if not url.lower().startswith("https://"):
        raise ValueError("The webhook URL must start with https://")

    headers = {"User-Agent": "Zavo-Flows/1", "Accept": "application/json"}
    for h in (content.get("headers") or [])[:MAX_HEADERS]:
        key = str((h or {}).get("key") or "").strip()
        if not key or key.lower() in BLOCKED_HEADERS or any(c in key for c in "\r\n:"):
            continue
        value = render_text((h or {}).get("value") or "", variables, max_len=2000)
        headers[key] = value.replace("\r", "").replace("\n", "")

    body = None
    if method != "GET":
        body = {}
        if content.get("include_all_vars"):
            body.update({k: v for k, v in (variables or {}).items()})
        for f in (content.get("body") or [])[:MAX_BODY_FIELDS]:
            key = str((f or {}).get("key") or "").strip()[:100]
            if key:
                body[key] = render_text((f or {}).get("value") or "", variables, max_len=2000)
        headers["Content-Type"] = "application/json"
    return {"method": method, "url": url, "headers": headers, "body": body}


def call(content: dict, variables: dict, transport=None) -> dict:
    """Run the webhook. Returns
    {"ok": bool, "status": int|None, "reason": str, "assign": {var: value}, "preview": str}
    Never raises. `transport` lets tests replace requests.request."""
    try:
        req = build_request(content, variables)
    except ValueError as e:
        return {"ok": False, "status": None, "reason": str(e), "assign": {}, "preview": ""}

    if not _slots.acquire(timeout=1):
        return {"ok": False, "status": None, "reason": "Too many webhook calls right now.", "assign": {}, "preview": ""}
    try:
        return _do_call(req, content, transport or requests.request)
    finally:
        _slots.release()


def _do_call(req: dict, content: dict, send) -> dict:
    url = req["url"]
    status = None
    try:
        for hop in range(MAX_REDIRECTS + 1):
            assert_public_http_url(url)
            if not url.lower().startswith("https://"):
                raise ValueError("Redirected to a non-https address.")
            data = json.dumps(req["body"]) if req["body"] is not None else None
            res = send(req["method"], url, headers=req["headers"], data=data,
                       timeout=TIMEOUT, allow_redirects=False, stream=True)
            status = res.status_code
            if status in (301, 302, 303, 307, 308):
                res.close()
                if req["method"] != "GET" or hop == MAX_REDIRECTS:
                    return {"ok": False, "status": status, "reason": "The webhook redirected.", "assign": {}, "preview": ""}
                location = res.headers.get("Location")
                if not location:
                    return {"ok": False, "status": status, "reason": "Redirect without a location.", "assign": {}, "preview": ""}
                url = urljoin(url, location)
                continue
            break

        declared = int(res.headers.get("Content-Length") or 0)
        if declared > MAX_RESPONSE_BYTES:
            res.close()
            return {"ok": False, "status": status, "reason": "The response was too large.", "assign": {}, "preview": ""}
        raw = b""
        for chunk in res.iter_content(8192):
            raw += chunk
            if len(raw) > MAX_RESPONSE_BYTES:
                res.close()
                return {"ok": False, "status": status, "reason": "The response was too large.", "assign": {}, "preview": ""}
        res.close()
    except ValueError as e:  # url_guard refusals
        return {"ok": False, "status": status, "reason": str(e), "assign": {}, "preview": ""}
    except requests.Timeout:
        return {"ok": False, "status": status, "reason": "The webhook timed out.", "assign": {}, "preview": ""}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {"ok": False, "status": status, "reason": "Couldn't reach the webhook.", "assign": {}, "preview": ""}

    text = raw.decode("utf-8", errors="replace")
    ok = 200 <= status < 300
    assign = {}
    if ok:
        try:
            parsed = json.loads(text) if text.strip() else None
        except ValueError:
            parsed = None
        if parsed is not None:
            for m in (content.get("mappings") or [])[:MAX_MAPPINGS]:
                var = (m or {}).get("var")
                if not is_valid_var_name(var):
                    continue
                value = _json_path(parsed, (m or {}).get("path"))
                if value is None:
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)[:1000]
                assign[var] = value if isinstance(value, (int, float, bool)) else str(value)[:1000]
    return {"ok": ok, "status": status, "reason": "" if ok else f"The webhook answered {status}.",
            "assign": assign, "preview": text[:2000]}
