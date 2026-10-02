import json
import uuid
from datetime import datetime, timezone

from memlog import mem_summary, release_memory

import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from qdrant_client import models
from starlette.concurrency import run_in_threadpool

from clients import supabase, qdrant
from config import QDRANT_COLLECTION
from auth import verify_token, require_project_access
from ratelimit import is_rate_limited
from usage import get_plan_limits, count_knowledge_items, knowledge_limit_message

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


from sources.gsheets import read_sheet
from sources.postgres import introspect_schema, validate_url
from sources.excel import read_excel, fetch_excel_from_url
from sources.sheet_tables import preview_tables
from sources.table_query import invalidate as invalidate_tables
from jobs import enqueue

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


# -------------------------------------------------
# SYNC STATE
# -------------------------------------------------
# Indexing runs as a background job (jobs.py -> job_runner.py), so these
# endpoints return as soon as the job is queued. The job records progress
# and the outcome in config (jsonb, no schema change):
#   sync_status  queued -> syncing -> done | failed
#   sync_error / sync_result
# which the dashboard polls. A status stuck at queued/syncing long past any
# possible run time means the job never finished; the UI shows that as
# incomplete and offers a Reload.
_INDEXED_TYPES = ("gsheets", "excel_online", "excel_local", "website", "shopify")
_ADDABLE_TYPES = ("gsheets", "excel_online", "website", "shopify", "postgres")


def _queued_config(config: dict) -> dict:
    return {
        **(config or {}),
        "sync_status": "queued",
        "sync_started_at": datetime.now(timezone.utc).isoformat(),
        "sync_error": None,
    }


def _is_indexing(config: dict) -> bool:
    """A job for this source is queued or running (and not abandoned)."""
    config = config or {}
    if config.get("sync_status") not in ("queued", "syncing"):
        return False
    try:
        started = datetime.fromisoformat(config.get("sync_started_at"))
    except (TypeError, ValueError):
        return False
    return (datetime.now(timezone.utc) - started).total_seconds() < 45 * 60


@router.post("/sources/add")
def add_source(data: dict, user=Depends(verify_token)):
    require_project_access(user.id, data["projectId"], tab="documents")

    project_id = data["projectId"]
    if data.get("type") not in _ADDABLE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported source type.")
    if is_rate_limited(f"source-add:{project_id}", limit=10, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="Too many sources added at once. Please wait a minute and try again.",
        )

    limits = get_plan_limits(project_id)
    if count_knowledge_items(project_id) >= limits["items"]:
        raise HTTPException(status_code=403, detail=knowledge_limit_message(limits["items"]))

    if data["type"] == "postgres":
        # Nothing to embed — this just re-verifies the connection is real
        # (same checks as /sources/introspect) before saving it, so it stays
        # synchronous. Previously this branch didn't exist at all, so a
        # postgres source skipped both the SSRF check and any connectivity
        # verification and was always reported as saved successfully.
        cfg = data["config"]
        try:
            validate_url(cfg["url"])
            introspect_schema(cfg["url"])
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise HTTPException(
                status_code=400,
                detail=str(e) if isinstance(e, ValueError) else "Failed to connect this database. Please check the details and try again.",
            )
        res = supabase.table("data_sources").insert({
            "project_id": project_id,
            "type": "postgres",
            "label": data.get("label") or "postgres",
            "config": cfg,
            "allowed_schema": data.get("allowed_schema"),
        }).execute()
        return {"id": res.data[0]["id"], "status": "done"}

    res = supabase.table("data_sources").insert({
        "project_id": project_id,
        "type": data["type"],
        "label": data.get("label") or data["type"],
        "config": _queued_config(data["config"]),
        "allowed_schema": data.get("allowed_schema"),
    }).execute()
    source = res.data[0]

    # If this sync fails (private sheet, blocked site), the job marks the
    # source failed with the reason rather than deleting it, so the
    # merchant sees why and can Reload or delete it.
    enqueue("sync_source", {"source_id": source["id"], "hidden_columns": data.get("hidden_columns")})
    return {"id": source["id"], "status": "queued"}


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
# DATABASE TABLES/COLUMNS (which ones the bot may query)
# -------------------------------------------------
# A database source is read live at question time, so changing what it may
# see needs no re-indexing: saving just updates data_sources.allowed_schema,
# which run_text_to_sql enforces on every query (see sources/postgres.py).
def _db_source(user_id: str, source_id: str, min_role: str = None) -> dict:
    project_id = _require_role_for_source(user_id, source_id, min_role=min_role)
    row = supabase.table("data_sources").select("type, config, allowed_schema").eq("id", source_id).single().execute().data
    if row["type"] != "postgres":
        raise HTTPException(status_code=400, detail="Only database sources have tables and columns.")
    return {"project_id": project_id, "config": row.get("config") or {}, "allowed_schema": row.get("allowed_schema")}


def _read_db_schema(db_url: str) -> dict:
    try:
        validate_url(db_url)
        return introspect_schema(db_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        sentry_sdk.capture_exception(e)
        # About the customer's OWN database (wrong host, rotated password,
        # firewall), so the driver's message is actionable — same as /introspect.
        raise HTTPException(status_code=400, detail=f"Couldn't connect to that database — {e}")


@router.get("/sources/{source_id}/database-columns")
def get_database_columns(source_id: str, user=Depends(verify_token)):
    src = _db_source(user.id, source_id)
    if is_rate_limited(f"db-introspect:{src['project_id']}", limit=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a minute and try again.")
    schema = _read_db_schema(src["config"].get("url", ""))

    saved = src["allowed_schema"]
    if not saved:
        # No restriction stored: everything is visible.
        allowed = {t: list(cols) for t, cols in schema.items()}
    else:
        # An empty column list has always meant "all columns of that table".
        allowed = {t: (saved[t] or list(schema[t])) for t in saved if t in schema}
    return {"schema": schema, "allowed": allowed}


@router.put("/sources/{source_id}/database-columns")
def set_database_columns(source_id: str, data: dict, user=Depends(verify_token)):
    # Exposing a column makes it answerable to anyone chatting with the bot,
    # so this takes the same admin role as deleting the source.
    src = _db_source(user.id, source_id, min_role="admin")
    if is_rate_limited(f"db-columns:{source_id}", limit=10, window_seconds=300):
        raise HTTPException(status_code=429, detail="Settings were just changed. Please wait a few minutes and try again.")

    requested = data.get("allowed")
    if not isinstance(requested, dict):
        raise HTTPException(status_code=400, detail="Invalid request.")
    schema = _read_db_schema(src["config"].get("url", ""))

    # Only tables and columns that really exist, and only non-empty picks:
    # the browser's list is never trusted as identifiers.
    allowed = {}
    for table, cols in requested.items():
        if table not in schema or not isinstance(cols, list):
            continue
        keep = [c for c in schema[table] if c in {str(x) for x in cols}]
        if keep:
            allowed[table] = keep
    if not allowed:
        raise HTTPException(status_code=400, detail="Choose at least one table with at least one column.")

    supabase.table("data_sources").update({"allowed_schema": allowed}).eq("id", source_id).execute()
    return {"status": "saved", "tables": len(allowed)}


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

    # The table query uses the new choice straight away; the search index
    # is rebuilt by a job so hidden columns leave it too.
    invalidate_tables(src["project_id"])
    _mark_queued(source_id)
    enqueue("reembed_columns", {"source_id": source_id})
    return {"status": "queued"}


def _mark_queued(source_id: str):
    row = supabase.table("data_sources").select("config").eq("id", source_id).single().execute().data
    supabase.table("data_sources").update(
        {"config": _queued_config(row.get("config"))}
    ).eq("id", source_id).execute()


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

    s = supabase.table("data_sources").select("*").eq("id", source_id).single().execute().data

    if s["type"] == "postgres":
        # Re-verify the connection is still reachable (credentials rotated,
        # DB gone). Nothing to embed, so it stays synchronous.
        cfg = s["config"]
        try:
            validate_url(cfg["url"])
            introspect_schema(cfg["url"])
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise HTTPException(
                status_code=400,
                detail=str(e) if isinstance(e, ValueError) else "Failed to refresh this source. Please try again.",
            )
        return {"status": "done"}

    if s["type"] == "excel_local":
        raise HTTPException(status_code=400, detail="Re-upload the file to refresh an uploaded Excel source.")
    if _is_indexing(s.get("config")):
        raise HTTPException(status_code=409, detail="This source is already being indexed.")

    # No pre-emptive purge: the job replaces this source's points itself,
    # and a failed job leaves the previous index untouched.
    _mark_queued(source_id)
    enqueue("sync_source", {"source_id": source_id})
    return {"status": "queued"}


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


def _json_form(raw: str, name: str):
    """Multipart can't carry nested JSON natively, so list/dict fields
    arrive as JSON strings. None when the field wasn't sent at all."""
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {name}.")


async def _read_excel_upload(file: UploadFile) -> bytes:
    file_bytes = await file.read()
    # No byte cap existed anywhere on this path — not here, not in the
    # Next.js proxy — so a scripted 2GB POST was read straight into the
    # worker's memory. Mirrors the 25MB cap on the document upload route.
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="That file is too large. The limit is 25MB.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="That file is empty.")
    return file_bytes


# -------------------------------------------------
# COLUMN PREVIEW (before connecting a sheet / Excel)
# -------------------------------------------------
# Reads the sheet/file and returns each tab's columns, with personal-data
# ones pre-unticked, so the merchant chooses what the bot may use BEFORE
# anything is indexed. Nothing is stored and nothing is embedded.
def _preview_response(book: dict) -> dict:
    response = {
        "tabs": preview_tables(book["tables"]),
        "skipped_tabs": book.get("skipped_tabs", []),
        "capped_tabs": book.get("capped_tabs", []),
        "truncated": book.get("truncated", False),
    }
    # The parsed rows are discarded right after this; give their memory
    # back before the real sync (which usually follows) starts.
    book.clear()
    release_memory()
    print(f"[mem] column preview done: {mem_summary()}")
    return response


@router.post("/sources/preview")
def preview_source(data: dict, user=Depends(verify_token)):
    project_id = data.get("projectId")
    if not project_id:
        raise HTTPException(status_code=400, detail="projectId is required")
    require_project_access(user.id, project_id, tab="documents")
    if is_rate_limited(f"source-preview:{project_id}", limit=20, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a minute and try again.")

    cfg = data.get("config") or {}
    try:
        if data.get("type") == "gsheets":
            return _preview_response(read_sheet(cfg.get("sheet_id", ""), cfg.get("range")))
        if data.get("type") == "excel_online":
            return _preview_response(read_excel(fetch_excel_from_url(cfg.get("url", "")), cfg.get("tabs")))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"preview_source error ({data.get('type')}): {e}")
        raise HTTPException(status_code=400, detail="Couldn't read that source. Please check the link and try again.")
    raise HTTPException(status_code=400, detail="Only spreadsheet sources have columns.")


@router.post("/sources/preview-excel")
async def preview_excel(
    file: UploadFile = File(...),
    projectId: str = Form(...),
    tabs: str = Form(None),
    user=Depends(verify_token),
):
    require_project_access(user.id, projectId, tab="documents")
    if is_rate_limited(f"source-preview:{projectId}", limit=20, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a minute and try again.")
    file_bytes = await _read_excel_upload(file)
    tab_list = _json_form(tabs, "tabs")
    try:
        return await run_in_threadpool(lambda: _preview_response(read_excel(file_bytes, tab_list)))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"preview_excel error (project {projectId}): {e}")
        raise HTTPException(status_code=400, detail="Couldn't read that file. Please check it and try again.")


@router.post("/sources/upload-excel")
async def upload_excel(
    file: UploadFile = File(...),
    projectId: str = Form(...),
    label: str = Form(""),
    source_id: str = Form(""),
    # JSON list of sheet names; not sent (None) = keep a re-uploaded
    # source's stored choice, [] = every sheet.
    tabs: str = Form(None),
    # JSON {sheet: [hidden columns]} from the preview popup.
    hidden_columns: str = Form(None),
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

    file_bytes = await _read_excel_upload(file)
    tab_list = _json_form(tabs, "tabs")
    hidden_override = _json_form(hidden_columns, "hidden_columns")

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
        existing_row = supabase.table("data_sources").select("config").eq("id", source_id).single().execute().data
        existing_cfg = existing_row.get("config") or {}
        if _is_indexing(existing_cfg):
            raise HTTPException(status_code=409, detail="This file is already being indexed.")
        if tab_list is None:
            # A plain re-upload keeps the sheets chosen when connecting.
            tab_list = existing_cfg.get("tabs")
        storage_path = _park_excel_upload(owning_project_id, source_id, file.filename, file_bytes)
        supabase.table("data_sources").update({
            "config": _queued_config({"filename": file.filename, "tabs": tab_list}),
            "label": label or file.filename,
        }).eq("id", source_id).execute()
        enqueue("excel_upload", {
            "source_id": source_id, "storage_path": storage_path, "hidden_columns": hidden_override,
        })
        return {"id": source_id, "filename": file.filename, "status": "queued"}

    # This endpoint creates a data_sources row just like add_source does,
    # but skipped the plan cap entirely — so uploading here instead of
    # through /sources/add was an unlimited way around it.
    limits = get_plan_limits(projectId)
    if count_knowledge_items(projectId) >= limits["items"]:
        raise HTTPException(status_code=403, detail=knowledge_limit_message(limits["items"]))

    res = supabase.table("data_sources").insert({
        "project_id": projectId,
        "type": "excel_local",
        "label": label or file.filename,
        "config": _queued_config({"filename": file.filename, "tabs": tab_list}),
        "allowed_schema": None,
    }).execute()
    source = res.data[0]

    try:
        storage_path = _park_excel_upload(projectId, source["id"], file.filename, file_bytes)
    except HTTPException:
        supabase.table("data_sources").delete().eq("id", source["id"]).execute()
        raise
    enqueue("excel_upload", {
        "source_id": source["id"], "storage_path": storage_path, "hidden_columns": hidden_override,
    })
    return {"id": source["id"], "filename": file.filename, "status": "queued"}


def _park_excel_upload(project_id: str, source_id: str, filename: str, file_bytes: bytes) -> str:
    """A queued job can't carry the file itself, so the bytes wait in
    Storage until the job reads them (and deletes them — uploaded Excel
    files were never kept). Under the project's prefix, so deleting the
    project removes any leftovers too."""
    ext = "xls" if filename.lower().endswith(".xls") else "xlsx"
    path = f"{project_id}/_source_uploads/{source_id}-{uuid.uuid4().hex}.{ext}"
    content_type = (
        "application/vnd.ms-excel" if ext == "xls"
        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    try:
        supabase.storage.from_("documents").upload(path, file_bytes, {"content-type": content_type})
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"excel upload parking failed for source {source_id}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't upload that file. Please try again.")
    return path
