import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from qdrant_client import models

from clients import supabase, qdrant, embeddings
from config import QDRANT_COLLECTION
from auth import verify_token, require_project_access
from ratelimit import is_rate_limited
from usage import get_plan_limits

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_CRAWL_PAGES = 100


def _clamp_max_pages(raw) -> int:
    """max_pages is caller-supplied and drives how many pages we fetch and
    embed. It was passed through unclamped, so `max_pages: 1000000` was
    accepted; a null/NaN value also crashed the crawl loop on a comparison."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 30
    return max(1, min(value, MAX_CRAWL_PAGES))
from sources.gsheets import sync_sheet
from sources.postgres import introspect_schema, validate_url
from sources.excel import sync_excel_url, sync_excel_bytes
from sources.website import sync_website
from sources.shopify import sync_products as sync_shopify_products
from sources.sheet_tables import reembed_from_tables
from sources.table_query import invalidate as invalidate_tables

router = APIRouter()


def _purge_source_points(source_id: str):
    """Delete every Qdrant point belonging to a source."""
    qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id)
            )]
        )
    )


def _redact_source(source: dict) -> dict:
    """Postgres sources store the raw connection string (with credentials)
    in config.url — never send that back to the browser once it's stored,
    only at introspect-time when the user is actively typing it in."""
    if source.get("type") == "postgres" and (source.get("config") or {}).get("url"):
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(source["config"]["url"])
        netloc = parts.hostname or ""
        if parts.port:
            netloc += f":{parts.port}"
        if parts.username:
            netloc = f"{parts.username}:***@{netloc}"
        redacted = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
        source = {**source, "config": {**source["config"], "url": redacted}}
    return source


@router.get("/sources")
def list_sources(project_id: str, user=Depends(verify_token)):
    require_project_access(user.id, project_id, tab="documents")
    res = supabase.table("data_sources") \
        .select("*") \
        .eq("project_id", project_id) \
        .execute()
    return [_redact_source(s) for s in res.data]


@router.post("/sources/add")
def add_source(data: dict, user=Depends(verify_token)):
    require_project_access(user.id, data["projectId"], tab="documents")

    project_id = data["projectId"]
    if is_rate_limited(f"source-add:{project_id}", limit=10, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="Too many sources added at once. Please wait a minute and try again.",
        )

    limits = get_plan_limits(project_id)
    existing = supabase.table("data_sources")         .select("id", count="exact")         .eq("project_id", project_id)         .execute()
    if (existing.count or 0) >= limits["sources"]:
        raise HTTPException(
            status_code=403,
            detail=f"You've reached your plan's limit of {limits['sources']} connected sources. Disconnect one, or upgrade your plan, to add more.",
        )

    res = supabase.table("data_sources").insert({
        "project_id": data["projectId"],
        "type": data["type"],
        "label": data.get("label") or data["type"],
        "config": data["config"],
        "allowed_schema": data.get("allowed_schema"),
    }).execute()
    source = res.data[0]

    skipped_tabs = []
    sync_result = {}

    # The data_sources row above is inserted BEFORE any of these sync calls
    # run. If a sync call raises (network error, crawler blocked, embedding
    # API failure, etc.) instead of cleanly returning, the row was
    # previously left behind with no actual indexed content — showing up
    # as "connected" in the UI while the bot has nothing to answer from.
    # Wrapping each in try/except so ANY failure is treated the same way:
    # clean up the orphaned row and tell the user honestly, instead of a
    # raw 500 and a silently broken "connected" source.
    try:
        if data["type"] == "gsheets":
            cfg = data["config"]
            sync_result = sync_sheet(
                cfg["sheet_id"], cfg["range"], data["projectId"],
                source["id"], qdrant, embeddings, QDRANT_COLLECTION
            )
            skipped_tabs = sync_result.get("skipped_tabs", [])

        elif data["type"] == "excel_online":
            cfg = data["config"]
            sync_result = sync_excel_url(
                cfg["url"], data["projectId"],
                source["id"], qdrant, embeddings, QDRANT_COLLECTION
            )

        elif data["type"] == "website":
            cfg = data["config"]
            sync_result = sync_website(
                url=cfg["url"],
                project_id=data["projectId"],
                source_id=source["id"],
                qdrant=qdrant,
                embeddings=embeddings,
                collection=QDRANT_COLLECTION,
                full_site=cfg.get("full_site", True),
                max_pages=_clamp_max_pages(cfg.get("max_pages")),
            )
            if sync_result["pages_indexed"] == 0:
                raise ValueError("Could not read any content from this website. It may block automated access, or only show its content with JavaScript.")

        elif data["type"] == "shopify":
            # In practice this data_sources row is usually created by the
            # OAuth callback (shopify_oauth.py) itself, not this generic
            # form — wiring it here too keeps the resync/list UI uniform
            # across every source type.
            sync_result = sync_shopify_products(
                data["projectId"], source["id"], qdrant, embeddings, QDRANT_COLLECTION
            )

        elif data["type"] == "postgres":
            # Unlike the other types, there's nothing to embed — this just
            # re-verifies the connection is real (same checks as
            # /sources/introspect) before treating the row as connected.
            # Previously this branch didn't exist at all, so a postgres
            # source skipped both the SSRF check and any connectivity
            # verification and was always reported as saved successfully.
            cfg = data["config"]
            validate_url(cfg["url"])
            introspect_schema(cfg["url"])
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"add_source sync error ({data['type']}): {e}")
        # Purge points too, not just the row — a sync that failed partway
        # may already have uploaded vectors, and once the row is gone
        # nothing can ever reach them again (no delete path takes an
        # orphaned source_id), so they'd keep answering questions forever.
        _purge_source_points(source["id"])
        supabase.table("data_sources").delete().eq("id", source["id"]).execute()
        raise HTTPException(
            status_code=400,
            detail=str(e) if isinstance(e, ValueError) else "Failed to connect this source. Please check the details and try again."
        )

    return {
        "id": source["id"],
        "skipped_tabs": skipped_tabs,
        "capped_tabs": sync_result.get("capped_tabs", []),
        "truncated": sync_result.get("truncated", False),
        "indexed_count": sync_result.get("indexed_count"),
        "total_count": sync_result.get("total_count"),
        "hidden_columns": sync_result.get("hidden_columns", []),
    }


def _require_role_for_source(user_id: str, source_id: str, min_role: str = None) -> str:
    """Verifies the caller has a role on the project that OWNS this source,
    and returns that project_id — callers should use the returned value
    rather than any project id supplied by the caller."""
    res = supabase.table("data_sources").select("project_id").eq("id", source_id).maybe_single().execute()
    source = res.data if res else None
    if not source:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user_id, source["project_id"], tab="documents", min_role=min_role)
    return source["project_id"]

@router.delete("/sources/{source_id}")
def delete_source(source_id: str, user=Depends(verify_token)):
    project_id = _require_role_for_source(user.id, source_id, min_role="admin")
    _purge_source_points(source_id)
    # source_tables rows go with it (FK cascade); drop the cached copy too so
    # the bot stops answering from it immediately.
    supabase.table("data_sources").delete().eq("id", source_id).execute()
    invalidate_tables(project_id)
    return {"status": "deleted"}


# -------------------------------------------------
# SPREADSHEET COLUMNS (which ones the bot may use)
# -------------------------------------------------
_TABLE_SOURCE_TYPES = ("gsheets", "excel_online", "excel_local")


def _table_source(user_id: str, source_id: str, min_role: str = None) -> dict:
    project_id = _require_role_for_source(user_id, source_id, min_role=min_role)
    s = supabase.table("data_sources").select("type").eq("id", source_id).single().execute().data
    if s["type"] not in _TABLE_SOURCE_TYPES:
        raise HTTPException(status_code=400, detail="Only spreadsheet sources have columns.")
    return {"project_id": project_id, "type": s["type"]}


@router.get("/sources/{source_id}/columns")
def get_source_columns(source_id: str, user=Depends(verify_token)):
    _table_source(user.id, source_id)
    res = supabase.table("source_tables") \
        .select("tab, columns, hidden_columns") \
        .eq("source_id", source_id) \
        .execute()
    return {"tabs": [
        {"tab": r["tab"], "columns": r.get("columns") or [], "hidden": r.get("hidden_columns") or []}
        for r in (res.data or [])
    ]}


@router.put("/sources/{source_id}/columns")
def set_source_columns(source_id: str, data: dict, user=Depends(verify_token)):
    # Exposing a column makes it answerable to anyone chatting with the bot,
    # so this takes the same admin role as deleting the source.
    src = _table_source(user.id, source_id, min_role="admin")

    # Re-embeds the whole source against our OpenAI key.
    if is_rate_limited(f"source-columns:{source_id}", limit=5, window_seconds=300):
        raise HTTPException(
            status_code=429,
            detail="Column settings were just changed. Please wait a few minutes before changing them again.",
        )

    requested = {t.get("tab"): t.get("hidden") for t in (data.get("tabs") or []) if isinstance(t, dict)}
    res = supabase.table("source_tables") \
        .select("tab, columns") \
        .eq("source_id", source_id) \
        .execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Reload this source first, then choose its columns.")

    for r in res.data:
        hidden = requested.get(r["tab"])
        if not isinstance(hidden, list):
            continue
        # Only real column names are stored.
        valid = [c for c in (r.get("columns") or []) if c in set(map(str, hidden))]
        supabase.table("source_tables") \
            .update({"hidden_columns": valid}) \
            .eq("source_id", source_id) \
            .eq("tab", r["tab"]) \
            .execute()

    source_type = "gsheets" if src["type"] == "gsheets" else "excel"
    try:
        # Hidden columns must leave the vector index too, not just the
        # table query.
        reembed_from_tables(source_id, src["project_id"], source_type, qdrant, embeddings, QDRANT_COLLECTION)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"set_source_columns re-embed failed (source {source_id}): {e}")
        raise HTTPException(
            status_code=502,
            detail="Saved, but the bot's search index couldn't be updated. Please press Reload on this source.",
        )
    finally:
        invalidate_tables(src["project_id"])

    return {"status": "saved"}


@router.post("/sources/sync/{source_id}")
def resync_source(source_id: str, user=Depends(verify_token)):
    _require_role_for_source(user.id, source_id)

    # The Reload button has no in-flight guard in the UI, and every press
    # re-embeds the ENTIRE source against our OpenAI key. This is the
    # cheapest place to stop a stuck-refresh loop from becoming a bill.
    if is_rate_limited(f"source-sync:{source_id}", limit=5, window_seconds=300):
        raise HTTPException(
            status_code=429,
            detail="This source was just refreshed. Please wait a few minutes before refreshing it again.",
        )

    res = supabase.table("data_sources").select("*").eq("id", source_id).single().execute()
    s = res.data

    # No pre-emptive purge here: every sync_* function already deletes this
    # source's points itself as part of its own run. Deleting here too meant
    # a resync that failed BEFORE reaching the sync call (bad config, failed
    # validation) destroyed the existing index for nothing, leaving the
    # source "connected" but genuinely empty.
    sync_result = {}
    try:
        if s["type"] == "gsheets":
            cfg = s["config"]
            sync_result = sync_sheet(
                cfg["sheet_id"], cfg["range"],
                s["project_id"], source_id,
                qdrant, embeddings, QDRANT_COLLECTION
            )
        elif s["type"] == "excel_online":
            cfg = s["config"]
            sync_result = sync_excel_url(
                cfg["url"], s["project_id"],
                source_id, qdrant, embeddings, QDRANT_COLLECTION
            )
        elif s["type"] == "website":
            cfg = s["config"]
            sync_result = sync_website(
                url=cfg["url"],
                project_id=s["project_id"],
                source_id=source_id,
                qdrant=qdrant,
                embeddings=embeddings,
                collection=QDRANT_COLLECTION,
                full_site=cfg.get("full_site", True),
                max_pages=_clamp_max_pages(cfg.get("max_pages")),
            )
            if sync_result["pages_indexed"] == 0:
                raise ValueError("Could not read any content from this website. It may block automated access, or only show its content with JavaScript.")

        elif s["type"] == "shopify":
            sync_result = sync_shopify_products(
                s["project_id"], source_id, qdrant, embeddings, QDRANT_COLLECTION
            )

        elif s["type"] == "postgres":
            # Re-verify the connection is still reachable; also refreshes
            # allowed_schema's validity implicitly (introspect_schema will
            # fail if the DB is gone/credentials rotated).
            cfg = s["config"]
            validate_url(cfg["url"])
            introspect_schema(cfg["url"])
    except Exception as e:
        sentry_sdk.capture_exception(e)
        # The old points for this source were already deleted above before
        # this ran — a failed resync used to silently report "synced"
        # anyway, leaving the source connected but genuinely empty with no
        # indication anything went wrong.
        print(f"resync_source error ({s['type']}): {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e) if isinstance(e, ValueError) else "Failed to refresh this source. Please try again."
        )

    return {
        "status": "synced",
        "truncated": sync_result.get("truncated", False),
        "indexed_count": sync_result.get("indexed_count"),
        "total_count": sync_result.get("total_count"),
        "capped_tabs": sync_result.get("capped_tabs", []),
        "hidden_columns": sync_result.get("hidden_columns", []),
    }


@router.post("/sources/introspect")
def introspect(data: dict, user=Depends(verify_token)):
    # Was the only endpoint in this file with no project check — any logged-in
    # user could make this worker open a connection to a host of their
    # choosing and read the driver's response, i.e. a network probe oracle.
    project_id = data.get("projectId") or data.get("project_id")
    if not project_id:
        raise HTTPException(status_code=400, detail="projectId is required")
    require_project_access(user.id, project_id, tab="documents")

    if is_rate_limited(f"db-introspect:{project_id}", limit=10, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="Too many connection attempts. Please wait a minute and try again.",
        )

    db_url = data.get("db_url", "")
    try:
        validate_url(db_url)
        schema = introspect_schema(db_url)
        return {"schema": schema}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        sentry_sdk.capture_exception(e)
        # Unlike the Meta/Shopify/Razorpay cases elsewhere, this one IS
        # worth showing close to verbatim — it's about the user's OWN
        # database (wrong host, bad password, firewall), not our
        # integration, so the driver's message is genuinely actionable.
        raise HTTPException(status_code=400, detail=f"Couldn't connect to that database — {str(e)}")


@router.post("/sources/upload-excel")
async def upload_excel(
    file: UploadFile = File(...),
    projectId: str = Form(...),
    label: str = Form(""),
    source_id: str = Form(""),
    user=Depends(verify_token)
):
    require_project_access(user.id, projectId, tab="documents")

    # Keyed on the verified user id, not the caller-supplied projectId —
    # naming a different project you also belong to otherwise handed you a
    # fresh bucket, making the limit trivial to sidestep.
    if is_rate_limited(f"excel-upload:{user.id}", limit=10, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="Too many uploads at once. Please wait a minute and try again.",
        )

    file_bytes = await file.read()

    # No byte cap existed anywhere on this path — not here, not in the
    # Next.js proxy — so a scripted 2GB POST was read straight into the
    # worker's memory. Mirrors the 25MB cap on the document upload route.
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="That file is too large. The limit is 25MB.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="That file is empty.")

    if source_id:
        # source_id is caller-supplied and was previously trusted on the
        # strength of the projectId check above — but that only proves the
        # caller has a role on the project THEY named, not that source_id
        # belongs to it. Without this, passing another tenant's source_id
        # wiped their vectors and overwrote their row via the service-role
        # client below.
        # Replacing a source's entire contents destroys the existing index
        # just as thoroughly as deleting it, so it takes the same admin
        # role that delete_source requires. Creating a NEW source only
        # needs the documents permission.
        owning_project_id = _require_role_for_source(user.id, source_id, min_role="admin")
        try:
            sync_result = sync_excel_bytes(file_bytes, owning_project_id, source_id, qdrant, embeddings, QDRANT_COLLECTION)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"upload_excel re-upload error (source {source_id}): {e}")
            raise HTTPException(
                status_code=400,
                detail=str(e) if isinstance(e, ValueError) else "Couldn't read that file. Please check it and try again.",
            )
        supabase.table("data_sources").update({
            "config": {"filename": file.filename},
            "label": label or file.filename,
        }).eq("id", source_id).execute()
        return {
            "id": source_id,
            "filename": file.filename,
            "truncated": sync_result.get("truncated", False),
            "indexed_count": sync_result.get("indexed_count"),
            "total_count": sync_result.get("total_count"),
            "hidden_columns": sync_result.get("hidden_columns", []),
        }

    # This endpoint creates a data_sources row just like add_source does,
    # but skipped the plan cap entirely — so uploading here instead of
    # through /sources/add was an unlimited way around it.
    limits = get_plan_limits(projectId)
    existing = supabase.table("data_sources")         .select("id", count="exact")         .eq("project_id", projectId)         .execute()
    if (existing.count or 0) >= limits["sources"]:
        raise HTTPException(
            status_code=403,
            detail=f"You've reached your plan's limit of {limits['sources']} connected sources. Disconnect one, or upgrade your plan, to add more.",
        )

    res = supabase.table("data_sources").insert({
        "project_id": projectId,
        "type": "excel_local",
        "label": label or file.filename,
        "config": {"filename": file.filename},
        "allowed_schema": None,
    }).execute()
    source = res.data[0]

    # Was completely unguarded, unlike add_source — a corrupt workbook left
    # a permanently orphaned row showing as "connected" with no content,
    # and returned a raw 500 whose body was shown to the user.
    try:
        sync_result = sync_excel_bytes(file_bytes, projectId, source["id"], qdrant, embeddings, QDRANT_COLLECTION)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"upload_excel error (project {projectId}): {e}")
        _purge_source_points(source["id"])
        supabase.table("data_sources").delete().eq("id", source["id"]).execute()
        raise HTTPException(
            status_code=400,
            detail=str(e) if isinstance(e, ValueError) else "Couldn't read that file. Please check it and try again.",
        )

    return {
        "id": source["id"],
        "filename": file.filename,
        "truncated": sync_result.get("truncated", False),
        "indexed_count": sync_result.get("indexed_count"),
        "total_count": sync_result.get("total_count"),
        "hidden_columns": sync_result.get("hidden_columns", []),
    }