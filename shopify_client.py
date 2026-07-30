# shopify_client.py
#
# Shared Shopify Admin GraphQL caller — extracted once a third caller
# (shop.py's order write-back) needed the exact same request/cost-backoff
# logic that sources/shopify.py and shopify_oauth.py each had their own
# earlier copy of.
import time

import requests

from config import SHOPIFY_API_VERSION


def graphql(shop_domain: str, access_token: str, query: str, variables: dict = None) -> dict:
    res = requests.post(
        f"https://{shop_domain}/admin/api/{SHOPIFY_API_VERSION}/graphql.json",
        json={"query": query, "variables": variables or {}},
        headers={"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"},
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    if data.get("errors"):
        raise ValueError(f"Shopify GraphQL error: {data['errors']}")

    # Cost-based leaky-bucket backoff — sleep before the NEXT call if this
    # one nearly drained the bucket, per Shopify's documented guidance,
    # rather than waiting to get throttled and retrying blind.
    cost = (data.get("extensions") or {}).get("cost") or {}
    throttle = cost.get("throttleStatus") or {}
    available = throttle.get("currentlyAvailable")
    restore_rate = throttle.get("restoreRate") or 50
    requested = cost.get("requestedQueryCost") or 0
    if available is not None and available < requested:
        time.sleep(max(0.5, requested / max(restore_rate, 1)))

    return data["data"]
