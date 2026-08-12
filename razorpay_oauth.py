# razorpay_oauth.py
#
# Razorpay Partner OAuth — replaces the old manual Key ID/Secret entry in
# shop_config with a real "Connect with Razorpay" flow, same shape as
# shopify_oauth.py: one shared Partner application, OAuth'd per-project,
# popup+postMessage completion.
#
# Two real differences from Shopify's flow, both confirmed against
# Razorpay's official Partner OAuth docs before writing this file (see the
# plan doc — nothing here is guessed):
#   1. Razorpay's `state` param is documented purely as CSRF protection —
#      there is no separate Razorpay-signed HMAC on the callback the way
#      Shopify mandates, so the single-use nonce below is the only
#      request-authenticity check needed (no _verify_oauth_hmac equivalent).
#   2. Razorpay OAuth access tokens expire and the refresh_token ROTATES on
#      every use (unlike Shopify's non-expiring offline token) — this file's
#      refresh helper mirrors appointments.py's ad-hoc
#      get_google_access_token() pattern instead, and always persists the
#      newest refresh_token returned, never reuses a stale one.
#
# The razorpay-python SDK only supports Basic Auth (key_id/key_secret) —
# it has no Bearer/OAuth mode — so authenticated calls on behalf of a
# connected merchant go through _razorpay_api_request() below using raw
# `requests`, not the SDK's Client.
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from clients import supabase
from auth import verify_token, require_project_role
from config import (
    RAZORPAY_PARTNER_CLIENT_ID, RAZORPAY_PARTNER_CLIENT_SECRET,
    RAZORPAY_PARTNER_REDIRECT_URI,
)

router = APIRouter()

AUTHORIZE_URL = "https://auth.razorpay.com/authorize"
TOKEN_URL = "https://auth.razorpay.com/token"
API_BASE = "https://api.razorpay.com/v1"

# Payment Links (create/fetch) need write access, not just read.
OAUTH_SCOPE = "read_write"

# Refresh this many seconds before actual expiry — avoids a request landing
# exactly as the token expires mid-flight.
_REFRESH_SAFETY_MARGIN_SECONDS = 120

# Short-lived CSRF nonce store for the OAuth handshake — in-memory,
# single-process, same convention as shopify_oauth.py's _oauth_states.
_oauth_states = {}
_STATE_TTL_SECONDS = 600


def _issue_state(project_id: str) -> str:
    now = time.time()
    for k, (_, exp) in list(_oauth_states.items()):
        if exp < now:
            _oauth_states.pop(k, None)
    nonce = secrets.token_urlsafe(24)
    _oauth_states[nonce] = (project_id, now + _STATE_TTL_SECONDS)
    return nonce


def _consume_state(nonce: str):
    entry = _oauth_states.pop(nonce, None)
    if not entry:
        return None
    project_id, expires_at = entry
    if expires_at < time.time():
        return None
    return project_id


def _popup_html(event: str, error: str = None) -> HTMLResponse:
    import json
    payload = {"type": "RAZORPAY_AUTH", "event": event}
    if error:
        payload["error"] = error
    return HTMLResponse(
        f"<html><body><script>"
        f"window.opener.postMessage({json.dumps(payload)}, '*');"
        f"window.close();"
        f"</script></body></html>"
    )


def _store_token_response(project_id: str, token_data: dict):
    expires_in = token_data.get("expires_in")
    expires_at = None
    if expires_in is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()

    supabase.table("razorpay_connections").upsert({
        "project_id": project_id,
        "razorpay_account_id": token_data.get("razorpay_account_id"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "token_type": token_data.get("token_type") or "Bearer",
        "expires_at": expires_at,
        "scope": token_data.get("scope") or OAUTH_SCOPE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, on_conflict="project_id").execute()


def _refresh_access_token(connection: dict) -> Optional[dict]:
    """Exchanges connection['refresh_token'] for a fresh token pair and
    persists it (refresh_token rotates — the old one is single-use).
    Returns the updated connection row, or None if the refresh itself
    failed (e.g. the merchant revoked access from their own dashboard)."""
    refresh_token = connection.get("refresh_token")
    if not refresh_token:
        return None
    try:
        res = requests.post(TOKEN_URL, data={
            "client_id": RAZORPAY_PARTNER_CLIENT_ID,
            "client_secret": RAZORPAY_PARTNER_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }, timeout=15)
        res.raise_for_status()
        token_data = res.json()
    except Exception as e:
        sentry_sdk.capture_message(f"Razorpay OAuth token refresh failed for project {connection['project_id']}: {e}")
        return None

    _store_token_response(connection["project_id"], token_data)
    updated = supabase.table("razorpay_connections").select("*").eq("project_id", connection["project_id"]).maybe_single().execute()
    return updated.data if updated else None


def get_razorpay_access_token(project_id: str) -> Optional[str]:
    """Ad-hoc refresh, mirrors appointments.py's get_google_access_token —
    no caching layer, just refresh-if-needed at point of use."""
    res = supabase.table("razorpay_connections").select("*").eq("project_id", project_id).maybe_single().execute()
    connection = res.data if res else None
    if not connection:
        return None

    expires_at = connection.get("expires_at")
    needs_refresh = True
    if expires_at:
        try:
            exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            needs_refresh = (exp_dt - datetime.now(timezone.utc)).total_seconds() < _REFRESH_SAFETY_MARGIN_SECONDS
        except Exception:
            needs_refresh = True

    if needs_refresh:
        connection = _refresh_access_token(connection)
        if not connection:
            return None

    return connection.get("access_token")


def _razorpay_api_request(method: str, path: str, project_id: str, **kwargs) -> requests.Response:
    """Centralizes OAuth-authenticated calls to Razorpay's REST API on
    behalf of a connected merchant. Retries once after a forced refresh on
    401 — a token can go stale between our own expiry check and Razorpay
    actually rejecting it (clock skew, early revocation, etc)."""
    token = get_razorpay_access_token(project_id)
    if not token:
        raise ValueError("Project has no active Razorpay connection")

    url = f"{API_BASE}{path}"
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    res = requests.request(method, url, headers=headers, timeout=15, **kwargs)

    if res.status_code == 401:
        conn_res = supabase.table("razorpay_connections").select("*").eq("project_id", project_id).maybe_single().execute()
        connection = conn_res.data if conn_res else None
        refreshed = _refresh_access_token(connection) if connection else None
        if not refreshed:
            return res
        headers["Authorization"] = f"Bearer {refreshed['access_token']}"
        res = requests.request(method, url, headers=headers, timeout=15, **kwargs)

    return res


@router.get("/razorpay/oauth/start")
def razorpay_oauth_start(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)

    if not RAZORPAY_PARTNER_CLIENT_ID or not RAZORPAY_PARTNER_CLIENT_SECRET:
        raise HTTPException(status_code=400, detail="Razorpay Partner OAuth not configured. Add RAZORPAY_PARTNER_CLIENT_ID and RAZORPAY_PARTNER_CLIENT_SECRET to env vars.")

    state = _issue_state(project_id)
    auth_url = (
        f"{AUTHORIZE_URL}?response_type=code"
        f"&client_id={RAZORPAY_PARTNER_CLIENT_ID}"
        f"&redirect_uri={RAZORPAY_PARTNER_REDIRECT_URI}"
        f"&scope={OAUTH_SCOPE}"
        f"&state={state}"
    )
    return {"auth_url": auth_url}


@router.get("/razorpay/oauth/callback")
def razorpay_oauth_callback(request: Request):
    query = dict(request.query_params)
    code = query.get("code", "")
    state = query.get("state", "")

    project_id = _consume_state(state)
    if not project_id:
        return _popup_html("ERROR", "This connection link expired or was already used — please try connecting again.")

    if not code:
        return _popup_html("ERROR", "Razorpay did not return an authorization code.")

    try:
        token_res = requests.post(TOKEN_URL, data={
            "client_id": RAZORPAY_PARTNER_CLIENT_ID,
            "client_secret": RAZORPAY_PARTNER_CLIENT_SECRET,
            "grant_type": "authorization_code",
            "redirect_uri": RAZORPAY_PARTNER_REDIRECT_URI,
            "code": code,
        }, timeout=15)
        token_res.raise_for_status()
        token_data = token_res.json()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return _popup_html("ERROR", "Could not complete the Razorpay connection. Please try again.")

    _store_token_response(project_id, token_data)
    return _popup_html("FINISH")


@router.get("/razorpay/status/{project_id}")
def razorpay_status(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("razorpay_connections").select("razorpay_account_id, connected_at").eq("project_id", project_id).maybe_single().execute()
    data = res.data if res else None
    if not data:
        return {"connected": False}
    return {"connected": True, **data}


@router.delete("/razorpay/disconnect/{project_id}")
def razorpay_disconnect(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    supabase.table("razorpay_connections").delete().eq("project_id", project_id).execute()
    return {"success": True}


def handle_authorization_revoked(razorpay_account_id: str):
    """Called from shop.py's unified webhook on an
    account.app.authorization_revoked event — the merchant disconnected
    Zavo from their own Razorpay dashboard, so our stored token is dead.
    Removing the row now (rather than waiting to discover it via a failed
    refresh) means the dashboard's Connect UI flips back to disconnected
    immediately instead of silently failing on the next payment attempt."""
    if not razorpay_account_id:
        return
    supabase.table("razorpay_connections").delete().eq("razorpay_account_id", razorpay_account_id).execute()