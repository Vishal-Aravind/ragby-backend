# sources/shopify.py
#
# Shopify catalog sync — the one source type that writes to BOTH Qdrant (RAG
# product Q&A, matching gsheets/excel/website's convention) AND the
# products/catalogs SQL tables (so the existing Shop/ordering feature in
# shop.py can sell real Shopify inventory). Two decisions here are load-
# bearing for later features that build on this data, not just style choices:
#
# 1. Variant-level rows: one `products` row per Shopify VARIANT, not per
#    product. A Shopify product with size/color options has multiple
#    purchasable variants at different prices/stock — getting this wrong
#    wouldn't error, it would just produce a working-looking but wrong
#    storefront for the WhatsApp ordering flow and the future widget.
# 2. Field-ownership split: Shopify-owned fields (name, description, price,
#    image_url, sku, inventory_quantity, is_available) are overwritten every
#    sync. Merchant-owned fields (category, sort_order, gst_percent,
#    catalog_id) are set once on first insert and never touched again, so a
#    merchant's manual re-categorization survives resyncs. This is why
#    upserts here are select-then-insert/update rather than a single blind
#    upsert — a blind upsert would silently clobber merchant customization
#    on every resync.

import re
import uuid
from datetime import datetime, timezone

import sentry_sdk
from qdrant_client import models

from clients import supabase
from shopify_client import graphql as _graphql

_MAX_VARIANTS_PER_PRODUCT = 100  # v1 cap — see plan's "Explicitly NOT doing"
_HTML_TAG_RE = re.compile(r"<[^>]+>")

_SHOPIFY_OWNED_FIELDS = {
    "name", "description", "price", "image_url",
    "sku", "inventory_quantity", "is_available",
}

_PRODUCT_QUERY = """
query Products($cursor: String) {
  products(first: 50, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      title
      descriptionHtml
      featuredImage { url }
      variants(first: 100) {
        nodes {
          id
          title
          sku
          price
          inventoryQuantity
          availableForSale
          image { url }
        }
      }
    }
  }
}
"""

_PRODUCT_BY_ID_QUERY = """
query ProductById($id: ID!) {
  product(id: $id) {
    id
    title
    descriptionHtml
    featuredImage { url }
    variants(first: 100) {
      nodes {
        id
        title
        sku
        price
        inventoryQuantity
        availableForSale
        image { url }
      }
    }
  }
}
"""


def _strip_html(html: str) -> str:
    if not html:
        return ""
    return _HTML_TAG_RE.sub(" ", html).strip()


def _get_integration(project_id: str) -> dict:
    res = supabase.table("shopify_integrations").select("*").eq("project_id", project_id).maybe_single().execute()
    data = res.data if res else None
    if not data:
        raise ValueError("No connected Shopify store for this project.")
    return data


def _mark_sync_result(project_id: str, error: str = None):
    supabase.table("shopify_integrations").update({
        "last_synced_at": datetime.now(timezone.utc).isoformat(),
        "last_sync_error": error,
    }).eq("project_id", project_id).execute()


def _ensure_catalog(project_id: str) -> str:
    res = supabase.table("catalogs").select("id").eq("project_id", project_id).eq("source", "shopify").maybe_single().execute()
    existing = res.data if res else None
    if existing:
        return existing["id"]
    inserted = supabase.table("catalogs").insert({
        "project_id": project_id,
        "name": "Shopify Products",
        "description": "Synced automatically from your connected Shopify store.",
        "is_active": True,
        "source": "shopify",
    }).execute()
    return inserted.data[0]["id"]


def _variant_row(project_id: str, catalog_id: str, product: dict, variant: dict, sort_order: int) -> dict:
    product_title = product["title"]
    variant_title = variant.get("title") or ""
    # Shopify uses the literal placeholder "Default Title" for products with
    # no real variant options — don't surface that string to a customer.
    name = product_title if variant_title in ("", "Default Title") else f"{product_title} — {variant_title}"
    image_url = (variant.get("image") or {}).get("url") or (product.get("featuredImage") or {}).get("url")
    return {
        "project_id": project_id,
        "catalog_id": catalog_id,
        "name": name,
        "description": _strip_html(product.get("descriptionHtml") or ""),
        "price": float(variant.get("price") or 0),
        "image_url": image_url,
        "sku": variant.get("sku") or None,
        "inventory_quantity": variant.get("inventoryQuantity"),
        "is_available": bool(variant.get("availableForSale")),
        "shopify_product_id": product["id"],
        "shopify_variant_id": variant["id"],
        "sort_order": sort_order,
    }


def _upsert_variant_rows(project_id: str, catalog_id: str, product: dict, sort_order_start: int):
    variants = (product.get("variants") or {}).get("nodes") or []
    capped_variants = variants[:_MAX_VARIANTS_PER_PRODUCT]

    for i, variant in enumerate(capped_variants):
        row = _variant_row(project_id, catalog_id, product, variant, sort_order_start + i)
        existing = supabase.table("products").select("id").eq("project_id", project_id).eq("shopify_variant_id", variant["id"]).maybe_single().execute()
        if existing and existing.data:
            # Only ever touch Shopify-owned fields on an existing row —
            # category/sort_order/gst_percent/catalog_id are merchant-owned
            # and must survive every resync untouched.
            update_fields = {k: v for k, v in row.items() if k in _SHOPIFY_OWNED_FIELDS}
            supabase.table("products").update(update_fields).eq("id", existing.data["id"]).execute()
        else:
            supabase.table("products").insert({**row, "category": "General", "gst_percent": 0}).execute()

    # Any variant that existed before this sync but is no longer present in
    # Shopify's current variant list gets soft-disabled, never hard-deleted —
    # preserves in-flight cart references and order-history integrity
    # (orders.items is already a frozen per-order snapshot; same instinct).
    seen_variant_ids = {v["id"] for v in capped_variants}
    existing_res = supabase.table("products").select("id, shopify_variant_id").eq("project_id", project_id).eq("shopify_product_id", product["id"]).execute()
    stale_ids = [r["id"] for r in (existing_res.data or []) if r["shopify_variant_id"] not in seen_variant_ids]
    if stale_ids:
        supabase.table("products").update({"is_available": False}).in_("id", stale_ids).execute()


def _reindex_product_in_qdrant(project_id: str, source_id: str, product: dict, qdrant, embeddings, collection: str):
    qdrant.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="shopify_product_id", match=models.MatchValue(value=product["id"]))]
        )
    )

    variants = (product.get("variants") or {}).get("nodes") or []
    price_lines = ", ".join(
        f"{v.get('title') or 'Default'}: {v.get('price')}" for v in variants[:_MAX_VARIANTS_PER_PRODUCT]
    )
    text = f"{product['title']}. {_strip_html(product.get('descriptionHtml') or '')} Pricing — {price_lines}".strip()
    if not text:
        return

    vector = embeddings.embed_documents([text])[0]
    qdrant.upload_points(
        collection_name=collection,
        points=[models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "project_id": project_id,
                "source_id": source_id,
                "source_type": "shopify",
                "shopify_product_id": product["id"],
                "text": text,
            },
        )],
    )


def sync_products(project_id: str, source_id: str, qdrant, embeddings, collection: str) -> dict:
    """Full/backstop reconciliation sync — paginates the entire catalog.
    Used both for the first sync right after OAuth connect and for the
    6-hourly reconciliation job (webhooks are the fast primary path;
    Shopify's own docs note webhook delivery isn't 100% guaranteed)."""
    integration = _get_integration(project_id)
    shop_domain = integration["shop_domain"]
    access_token = integration["access_token"]

    try:
        catalog_id = _ensure_catalog(project_id)

        cursor = None
        sort_order = 0
        product_count = 0
        while True:
            data = _graphql(shop_domain, access_token, _PRODUCT_QUERY, {"cursor": cursor})
            connection = data["products"]
            for product in connection["nodes"]:
                _upsert_variant_rows(project_id, catalog_id, product, sort_order)
                _reindex_product_in_qdrant(project_id, source_id, product, qdrant, embeddings, collection)
                # Step by the full variant cap, not a small fixed amount —
                # a product can have up to _MAX_VARIANTS_PER_PRODUCT variants,
                # each consuming sort_order_start + i. A small step (e.g. 10)
                # would let a product with >10 variants collide with the
                # next product's range, scrambling the merchant-facing order.
                sort_order += _MAX_VARIANTS_PER_PRODUCT
                product_count += 1
            if not connection["pageInfo"]["hasNextPage"]:
                break
            cursor = connection["pageInfo"]["endCursor"]

        _mark_sync_result(project_id, error=None)
        return {"products_synced": product_count}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        _mark_sync_result(project_id, error=str(e))
        raise


def sync_single_product(project_id: str, shopify_product_gid: str, qdrant, embeddings, collection: str, source_id: str):
    """Efficient webhook-driven path — fetches and upserts just one product,
    without re-paginating the whole catalog."""
    integration = _get_integration(project_id)
    shop_domain = integration["shop_domain"]
    access_token = integration["access_token"]

    catalog_id = _ensure_catalog(project_id)
    data = _graphql(shop_domain, access_token, _PRODUCT_BY_ID_QUERY, {"id": shopify_product_gid})
    product = data.get("product")
    if not product:
        # Product no longer exists (a create/update webhook can race a
        # near-simultaneous delete) — treat it the same as a delete.
        delete_product(project_id, shopify_product_gid, qdrant, collection)
        return

    # New variants get appended after whatever sort_order already exists for
    # this product, so a newly-added variant doesn't jump ahead of the
    # merchant's existing manual ordering of other products.
    existing_max = supabase.table("products").select("sort_order").eq("project_id", project_id).eq("shopify_product_id", shopify_product_gid).order("sort_order", desc=True).limit(1).execute()
    sort_order_start = ((existing_max.data or [{}])[0].get("sort_order") or 0) + 10

    _upsert_variant_rows(project_id, catalog_id, product, sort_order_start)
    _reindex_product_in_qdrant(project_id, source_id, product, qdrant, embeddings, collection)


def delete_product(project_id: str, shopify_product_gid: str, qdrant, collection: str):
    """Soft-delete: mark rows unavailable rather than hard-deleting, so past
    orders' frozen item snapshots and any in-flight cart reference still
    resolve. Only the Qdrant points (pure RAG index, safe to hard-delete)
    are actually removed."""
    supabase.table("products").update({"is_available": False}).eq("project_id", project_id).eq("shopify_product_id", shopify_product_gid).execute()
    qdrant.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="shopify_product_id", match=models.MatchValue(value=shopify_product_gid))]
        )
    )
