# shopify_oauth.py
#
# Shopify app-install (OAuth) + webhook receiver. One shared Partner-Dashboard
# app (custom distribution), OAuth'd per-merchant shop domain — the same
# shape as the Google Calendar integration in appointments.py, with one
# difference: Shopify issues long-lived OFFLINE tokens (no refresh-token
# dance needed), so tokens are stored directly, matching slack.py's
# slack_integrations pattern instead.
#
# The OAuth callback lands directly on this backend (not relayed through a
# Next.js route, matching appointments.py's Google callback) and closes its
# own popup via window.opener.postMessage — unlike appointments.py's
# google_callback, this explicitly returns HTMLResponse rather than a bare
# string. FastAPI's default response_class is JSONResponse, which would
# json.dumps() a bare string (wrapping it in quotes, Content-Type
# application/json) — the <script> tag would never execute and the popup
# would never signal completion back to its opener. Confirmed no
# response_class is set anywhere in this backend, so this is a real,
# separate bug in the existing Google flow, not just theoretical — fixed
# there too as part of this change.
import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from urllib.parse import urlencode

import requests
import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request
from ratelimit import is_rate_limited
from fastapi.responses import HTMLResponse

from clients import supabase, qdrant, embeddings
from auth import verify_token, require_project_role
from shopify_client import graphql as _graphql
from config import (
    SHOPIFY_API_KEY, SHOPIFY_API_SECRET, SHOPIFY_APP_SCOPES,
    SHOPIFY_REDIRECT_URI, BACKEND_PUBLIC_URL,
    QDRANT_COLLECTION,
)

router = APIRouter()

SHOP_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*\.myshopify\.com$")

# Short-lived CSRF nonce store for the OAuth handshake — in-memory,
# single-process, same convention as ratelimit.py. Shopify's authorize page
# is reachable by anyone who knows a shop domain, so (unlike Google/Slack's
# simple state=project_id) a forged callback must not be able to attach a
# stranger's Shopify store to the wrong project.
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
    """Pops and returns the project_id this nonce was issued for, or None if
    it's missing/expired/already used. One-time use — a callback can only
    ever complete a given /start call once."""
    entry = _oauth_states.pop(nonce, None)
    if not entry:
        return None
    project_id, expires_at = entry
    if expires_at < time.time():
        return None
    return project_id


def _verify_oauth_hmac(query_params: dict) -> bool:
    """Shopify's mandated request-authenticity check for the OAuth callback
    — separate from the CSRF nonce above, which only prevents a stolen/
    replayed callback from being *usable*, not from being sent by someone
    other than Shopify in the first place."""
    params = dict(query_params)
    supplied = params.pop("hmac", None)
    params.pop("signature", None)  # deprecated legacy param, ignore if present
    if not supplied:
        return False
    message = urlencode(sorted(params.items()))
    computed = hmac.new(SHOPIFY_API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, supplied)


def _popup_html(event: str, error: str = None) -> HTMLResponse:
    payload = {"type": "SHOPIFY_AUTH", "event": event}
    if error:
        payload["error"] = error
    return HTMLResponse(
        f"<html><body><script>"
        f"window.opener.postMessage({json.dumps(payload)}, '*');"
        f"window.close();"
        f"</script></body></html>"
    )


def register_webhooks(shop_domain: str, access_token: str):
    """Registers the per-shop webhooks this integration relies on, plus the
    three mandatory compliance topics — Piece 1 holds no customer PII so
    those are a log-and-200 no-op today, but Shopify's Protected Customer
    Data approval (needed later for Piece 2's read_orders/write_orders
    scopes) has real calendar-time lead, so registering them now starts
    that clock early instead of blocking Piece 2's start."""
    callback_url = f"{BACKEND_PUBLIC_URL}/shopify/webhooks"
    topics = [
        "PRODUCTS_CREATE", "PRODUCTS_UPDATE", "PRODUCTS_DELETE", "APP_UNINSTALLED",
        "CUSTOMERS_DATA_REQUEST", "CUSTOMERS_REDACT", "SHOP_REDACT",
        # ORDERS_PAID — Piece 3: tells us when a shopper completes checkout
        # on a cart the widget built, so we can reconcile it (see
        # shopify_webhooks' "orders/paid" branch below) and let the widget
        # confirm the purchase to the shopper on refocus.
        "ORDERS_PAID",
    ]
    mutation = """
        mutation webhookSubscriptionCreate($topic: WebhookSubscriptionTopic!, $webhookSubscription: WebhookSubscriptionInput!) {
          webhookSubscriptionCreate(topic: $topic, webhookSubscription: $webhookSubscription) {
            webhookSubscription { id }
            userErrors { field message }
          }
        }
    """
    for topic in topics:
        try:
            _graphql(shop_domain, access_token, mutation, {
                "topic": topic,
                "webhookSubscription": {"callbackUrl": callback_url, "format": "JSON"},
            })
        except Exception as e:
            # One topic failing to register shouldn't abort the rest —
            # surfaced to Sentry, not silently swallowed.
            sentry_sdk.capture_exception(e)
            print(f"Shopify webhook registration failed for topic={topic}, shop={shop_domain}: {e}")


def _ensure_data_source_and_kick_off_sync(project_id: str, shop_domain: str):
    existing = supabase.table("data_sources").select("id").eq("project_id", project_id).eq("type", "shopify").maybe_single().execute()
    if existing and existing.data:
        source_id = existing.data["id"]
    else:
        inserted = supabase.table("data_sources").insert({
            "project_id": project_id,
            "type": "shopify",
            "label": shop_domain,
            "config": {"shop_domain": shop_domain},
        }).execute()
        source_id = inserted.data[0]["id"]

    # Don't block the OAuth popup on a potentially large catalog — the
    # dashboard shows last_synced_at/last_sync_error, so a pending first
    # sync is visible rather than making the merchant wait on the popup.
    def _run():
        try:
            from sources.shopify import sync_products
            sync_products(project_id, source_id, qdrant, embeddings, QDRANT_COLLECTION)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"Shopify initial sync error for project {project_id}: {e}")

    threading.Thread(target=_run, daemon=True).start()


def _sync_shop_currency(project_id: str, shop_domain: str, access_token: str):
    """One-time convenience on first connect: if the merchant hasn't set up
    Shop settings at all yet, seed shop_config.currency_code from Shopify's
    own shop currency instead of leaving it to default to 'INR' regardless
    of what the store actually sells in. Never overwrites an existing
    shop_config row — a merchant who already configured Shop settings keeps
    whatever currency_code they set, even if it happens to equal the
    default."""
    existing = supabase.table("shop_config").select("id").eq("project_id", project_id).maybe_single().execute()
    if existing and existing.data:
        return
    data = _graphql(shop_domain, access_token, "{ shop { currencyCode } }")
    currency_code = (data.get("shop") or {}).get("currencyCode")
    if currency_code:
        supabase.table("shop_config").insert({"project_id": project_id, "currency_code": currency_code}).execute()


@router.get("/shopify/oauth/start")
def shopify_oauth_start(project_id: str, shop: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    if not SHOPIFY_API_KEY or not SHOPIFY_API_SECRET:
        raise HTTPException(status_code=400, detail="Shopify integration not configured. Add SHOPIFY_API_KEY and SHOPIFY_API_SECRET to env vars.")

    shop_domain = shop.strip().lower()
    if not SHOP_DOMAIN_RE.match(shop_domain):
        raise HTTPException(status_code=400, detail="Enter your shop domain like mystore.myshopify.com")

    state = _issue_state(project_id)
    auth_url = (
        f"https://{shop_domain}/admin/oauth/authorize"
        f"?client_id={SHOPIFY_API_KEY}"
        f"&scope={SHOPIFY_APP_SCOPES}"
        f"&redirect_uri={SHOPIFY_REDIRECT_URI}"
        f"&state={state}"
    )
    return {"auth_url": auth_url}


@router.get("/shopify/oauth/callback")
def shopify_oauth_callback(request: Request):
    query = dict(request.query_params)
    shop_domain = query.get("shop", "").strip().lower()
    code = query.get("code", "")
    state = query.get("state", "")

    if not SHOP_DOMAIN_RE.match(shop_domain):
        return _popup_html("ERROR", "Invalid shop domain")

    if not _verify_oauth_hmac(query):
        sentry_sdk.capture_message(f"Shopify OAuth callback failed HMAC verification for shop={shop_domain}")
        return _popup_html("ERROR", "Could not verify this request came from Shopify")

    project_id = _consume_state(state)
    if not project_id:
        return _popup_html("ERROR", "This connection link expired or was already used — please try connecting again.")

    try:
        token_res = requests.post(
            f"https://{shop_domain}/admin/oauth/access_token",
            json={"client_id": SHOPIFY_API_KEY, "client_secret": SHOPIFY_API_SECRET, "code": code},
            timeout=15,
        )
        token_res.raise_for_status()
        token_data = token_res.json()
        access_token = token_data["access_token"]
        scope = token_data.get("scope", "")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return _popup_html("ERROR", "Could not complete the Shopify connection. Please try again.")

    supabase.table("shopify_integrations").upsert({
        "project_id": project_id,
        "shop_domain": shop_domain,
        "access_token": access_token,
        "scope": scope,
    }, on_conflict="project_id").execute()

    try:
        register_webhooks(shop_domain, access_token)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Shopify webhook registration error for {shop_domain}: {e}")

    try:
        _sync_shop_currency(project_id, shop_domain, access_token)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Shopify currency sync error for {shop_domain}: {e}")

    try:
        from shopify_storefront import mint_storefront_token
        storefront_token = mint_storefront_token(shop_domain, access_token)
        supabase.table("shopify_integrations").update({
            "storefront_access_token": storefront_token,
        }).eq("project_id", project_id).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Shopify Storefront token creation error for {shop_domain}: {e}")

    try:
        _ensure_data_source_and_kick_off_sync(project_id, shop_domain)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Shopify initial sync kickoff error for {shop_domain}: {e}")

    return _popup_html("FINISH")


@router.get("/shopify/status/{project_id}")
def shopify_status(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    res = supabase.table("shopify_integrations").select("shop_domain, last_synced_at, last_sync_error").eq("project_id", project_id).maybe_single().execute()
    data = res.data if res else None
    if not data:
        return {"connected": False}

    source_res = supabase.table("data_sources").select("id").eq("project_id", project_id).eq("type", "shopify").maybe_single().execute()
    source_id = (source_res.data or {}).get("id") if source_res else None

    return {"connected": True, "source_id": source_id, **data}


@router.delete("/shopify/disconnect/{project_id}")
def shopify_disconnect(project_id: str, user=Depends(verify_token)):
    require_project_role(user.id, project_id)
    # Leaves products/catalogs rows in place — deleting them would break
    # historical orders.items display. The source='shopify' catalog just
    # stops receiving updates.
    supabase.table("shopify_integrations").delete().eq("project_id", project_id).execute()
    return {"success": True}


@router.post("/shopify/webhooks")
async def shopify_webhooks(request: Request):
    body_bytes = await request.body()
    signature = request.headers.get("X-Shopify-Hmac-Sha256", "")
    topic = request.headers.get("X-Shopify-Topic", "")
    shop_domain = request.headers.get("X-Shopify-Shop-Domain", "")

    computed = base64.b64encode(
        hmac.new(SHOPIFY_API_SECRET.encode(), body_bytes, hashlib.sha256).digest()
    ).decode()
    if not signature or not hmac.compare_digest(computed, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Every branch below always returns 200 even after a caught failure — a
    # non-2xx response makes Shopify retry, and after sustained failures
    # auto-disables the webhook subscription entirely, silently breaking
    # future syncs. Only a bad signature (above) is worth a non-200.
    try:
        integration_res = supabase.table("shopify_integrations").select("project_id").eq("shop_domain", shop_domain).maybe_single().execute()
        integration = integration_res.data if integration_res else None
        if not integration:
            # Shop already disconnected on our side — nothing to do.
            return {"status": "ok"}

        project_id = integration["project_id"]

        if topic in ("products/create", "products/update"):
            payload = json.loads(body_bytes)
            product_gid = f"gid://shopify/Product/{payload.get('id')}"
            source_res = supabase.table("data_sources").select("id").eq("project_id", project_id).eq("type", "shopify").maybe_single().execute()
            source_id = (source_res.data or {}).get("id") if source_res else None
            if source_id:
                from sources.shopify import sync_single_product
                sync_single_product(project_id, product_gid, qdrant, embeddings, QDRANT_COLLECTION, source_id)

        elif topic == "products/delete":
            payload = json.loads(body_bytes)
            product_gid = f"gid://shopify/Product/{payload.get('id')}"
            from sources.shopify import delete_product
            delete_product(project_id, product_gid, qdrant, QDRANT_COLLECTION)

        elif topic == "app/uninstalled":
            supabase.table("shopify_integrations").delete().eq("project_id", project_id).execute()

        elif topic == "orders/paid":
            payload = json.loads(body_bytes)
            note_attributes = payload.get("note_attributes") or []
            chat_id = next((a.get("value") for a in note_attributes if a.get("name") == "ragby_chat_id"), None)
            shopify_order_id = str(payload.get("id"))

            if chat_id:
                cart_session_res = supabase.table("shopify_cart_sessions") \
                    .select("id").eq("chat_id", chat_id).eq("status", "open") \
                    .order("created_at", desc=True).limit(1).execute()
                cart_session = (cart_session_res.data or [None])[0]
                if cart_session:
                    from datetime import datetime, timezone
                    supabase.table("shopify_cart_sessions").update({
                        "status": "completed",
                        "shopify_order_id": shopify_order_id,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", cart_session["id"]).execute()

                # Best-effort reconciliation into `orders` for unified
                # merchant reporting (same table Piece 2's WhatsApp orders
                # use). Wrapped separately from the cart_session update
                # above — the exact REST field shapes here are the
                # least-verified part of this whole integration, and a
                # failure here must never stop the widget's refocus check
                # from seeing the completed cart_session, which is the part
                # that actually matters to the shopper.
                try:
                    already = supabase.table("orders").select("id").eq("shopify_order_id", shopify_order_id).maybe_single().execute()
                    if not already or not already.data:
                        items_data = [{
                            "name": li.get("name") or li.get("title"),
                            "price": float(li.get("price") or 0),
                            "quantity": li.get("quantity") or 1,
                        } for li in (payload.get("line_items") or [])]
                        phone = ((payload.get("customer") or {}).get("phone")) or payload.get("phone") or ""
                        supabase.table("orders").insert({
                            "project_id": project_id,
                            "phone_number": phone.replace("+", "").replace(" ", ""),
                            "items": items_data,
                            "subtotal": float(payload.get("subtotal_price") or 0),
                            "gst_amount": float(payload.get("total_tax") or 0),
                            "total": float(payload.get("total_price") or 0),
                            "status": "confirmed",
                            "payment_status": "paid",
                            "delivery_type": "Shopify Checkout",
                            "shopify_order_id": shopify_order_id,
                        }).execute()
                except Exception as e:
                    sentry_sdk.capture_exception(e)
                    print(f"Shopify orders/paid → orders table reconciliation error: {e}")
            # No ragby_chat_id attribute means this order didn't originate
            # from our widget (a regular storefront sale, or the merchant's
            # own POS/admin order) — nothing for us to reconcile.

        elif topic in ("customers/data_request", "customers/redact", "shop/redact"):
            # Mandatory compliance topics — Piece 1 holds no customer PII,
            # nothing to act on.
            pass

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Shopify webhook handler error (topic={topic}, shop={shop_domain}): {e}")

    return {"status": "ok"}


@router.get("/public/shopify/cart-status/{chat_id}")
def shopify_cart_status(chat_id: str, request: Request):
    """Polled by the storefront widget when the shopper's tab regains focus
    after being sent to Shopify's checkout — no auth, matching the rest of
    the /public/* surface. Looks up the most recent cart this conversation
    built (there could be more than one if the shopper abandoned an earlier
    one) rather than assuming exactly one ever exists per chat."""
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if is_rate_limited(f"shopify-cart-status:{chat_id}:{ip}", limit=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment.")
    res = supabase.table("shopify_cart_sessions") \
        .select("status, checkout_url, shopify_order_id") \
        .eq("chat_id", chat_id) \
        .order("created_at", desc=True).limit(1).execute()
    row = (res.data or [None])[0]
    if not row:
        return {"status": "none"}
    return row
