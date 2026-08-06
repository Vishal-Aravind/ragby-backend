# shopify_storefront.py
#
# Shopify Storefront API client — cart building + checkout handoff for the
# Shopify storefront widget (Piece 3). Deliberately separate from
# shopify_client.py's Admin API graphql() helper: the Storefront API is a
# different endpoint (no /admin/ prefix) with a different auth header
# (X-Shopify-Storefront-Access-Token, not X-Shopify-Access-Token) — and it's
# the ONLY Shopify-compliant way to actually collect payment for a widget
# embedded on a merchant's live storefront (Shopify's API License §2.3.18
# prohibits an app-hosted alternative to Shopify Checkout there). This must
# never be conflated with shop.py's Razorpay-based WhatsApp checkout, which
# stays compliant only because WhatsApp isn't "on" the Shopify storefront.
import requests
import sentry_sdk

from clients import supabase
from config import SHOPIFY_API_VERSION


def _storefront_graphql(shop_domain: str, storefront_token: str, query: str, variables: dict = None) -> dict:
    res = requests.post(
        f"https://{shop_domain}/api/{SHOPIFY_API_VERSION}/graphql.json",
        json={"query": query, "variables": variables or {}},
        headers={"X-Shopify-Storefront-Access-Token": storefront_token, "Content-Type": "application/json"},
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    if data.get("errors"):
        raise ValueError(f"Shopify Storefront API error: {data['errors']}")
    return data["data"]


def _get_integration(project_id: str) -> dict:
    res = supabase.table("shopify_integrations").select("*").eq("project_id", project_id).maybe_single().execute()
    data = res.data if res else None
    if not data or not data.get("storefront_access_token"):
        raise ValueError("Checkout isn't set up for this store yet.")
    return data


_CART_CREATE_MUTATION = """
mutation cartCreate($input: CartInput!) {
  cartCreate(input: $input) {
    cart { id checkoutUrl }
    userErrors { field message }
  }
}
"""


def create_checkout_from_chat(project_id: str, chat_id: str, requested_items: list) -> dict:
    """Builds a real Shopify cart from items the AI understood in
    conversation and returns its checkout URL. Every item is re-resolved
    against the real synced catalog here — mirrors shop.py's
    create_order_from_chat, which never trusts a model-supplied price
    either. Items with no shopify_variant_id (manually-added, non-Shopify
    products) can't be checked out this way — Shopify's own checkout can
    only sell real Shopify variants, so those are called out as a distinct
    error rather than silently dropped or guessed at."""
    from shop import find_product_by_name

    integration = _get_integration(project_id)

    unmatched = []
    not_on_shopify = []
    lines = []
    item_summaries = []
    total = 0.0

    for req in requested_items:
        product = find_product_by_name(project_id, req["product_name"])
        if not product:
            unmatched.append(req["product_name"])
            continue
        if not product.get("shopify_variant_id"):
            not_on_shopify.append(product["name"])
            continue
        qty = max(1, int(req.get("quantity", 1)))
        lines.append({"merchandiseId": product["shopify_variant_id"], "quantity": qty})
        item_summaries.append(f"{product['name']} x{qty}")
        total += product["price"] * qty

    if unmatched:
        raise ValueError(f"Couldn't find these items in the menu: {', '.join(unmatched)}. Ask the customer to confirm the exact item name.")
    if not_on_shopify:
        raise ValueError(f"These items aren't part of the connected Shopify catalog, so they can't be checked out here: {', '.join(not_on_shopify)}.")
    if not lines:
        raise ValueError("No valid items to check out.")

    data = _storefront_graphql(integration["shop_domain"], integration["storefront_access_token"], _CART_CREATE_MUTATION, {
        "input": {
            "lines": lines,
            # Stamped onto the cart so the eventual order's note_attributes
            # let the orders/paid webhook match it back to this exact
            # conversation — Shopify carts/orders have no other built-in
            # link back to whatever created them.
            "attributes": [{"key": "ragby_chat_id", "value": chat_id}],
        }
    })
    result = data["cartCreate"]
    errors = result.get("userErrors") or []
    if errors:
        raise ValueError(f"Could not build the checkout: {'; '.join(e['message'] for e in errors)}")

    cart = result["cart"]

    supabase.table("shopify_cart_sessions").insert({
        "project_id": project_id,
        "chat_id": chat_id,
        "shopify_cart_id": cart["id"],
        "checkout_url": cart["checkoutUrl"],
        "status": "open",
    }).execute()

    return {
        "checkout_url": cart["checkoutUrl"],
        "items": item_summaries,
        "total": round(total, 2),
    }


def mint_storefront_token(shop_domain: str, access_token: str) -> str:
    """Called once from shopify_oauth.py right after connecting — uses the
    Admin API (the token we already have) to create the separate Storefront
    API token this module needs for cart building."""
    from shopify_client import graphql

    mutation = """
        mutation storefrontAccessTokenCreate($input: StorefrontAccessTokenInput!) {
          storefrontAccessTokenCreate(input: $input) {
            storefrontAccessToken { accessToken }
            userErrors { field message }
          }
        }
    """
    data = graphql(shop_domain, access_token, mutation, {"input": {"title": "Ragby Chat Widget"}})
    result = data["storefrontAccessTokenCreate"]
    errors = result.get("userErrors") or []
    if errors:
        raise ValueError(f"Could not create Storefront API token: {'; '.join(e['message'] for e in errors)}")
    return result["storefrontAccessToken"]["accessToken"]
