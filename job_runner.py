"""The indexing jobs themselves.

Runs on the worker service (worker_main.py) or, as a fallback, in a
background thread on the main backend (jobs.py). Either way each job
records its own outcome where the dashboard can see it:

  data sources: config.sync_status  queued -> syncing -> done | failed
                config.sync_error   why it failed (shown to the merchant)
                config.sync_result  skipped/capped tabs, row counts
  files:        status  queued -> processing -> indexed | failed
                error / result      same idea

Every job catches its own errors: a job that fails for an ordinary reason
(private sheet, corrupt file) is recorded as failed, not retried. Only a
worker that dies outright (killed, timed out) gets the job re-delivered by
the queue.
"""
from datetime import datetime, timezone

import sentry_sdk

from clients import supabase, qdrant, embeddings
from config import QDRANT_COLLECTION
from memlog import mem_summary, release_memory

MAX_CRAWL_PAGES = 100
_GENERIC_SOURCE_ERROR = "Couldn't index this source. Please try again."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_max_pages(raw) -> int:
    """max_pages is caller-supplied and drives how many pages we fetch and
    embed. It was passed through unclamped, so `max_pages: 1000000` was
    accepted; a null/NaN value also crashed the crawl loop on a comparison."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 30
    return max(1, min(value, MAX_CRAWL_PAGES))


def update_source_state(source_id: str, **fields):
    """Merge sync_* fields into the source's config. Never raises: status
    bookkeeping must not be what fails a job."""
    try:
        row = supabase.table("data_sources").select("config").eq("id", source_id).maybe_single().execute()
        if not row or not row.data:
            return  # source deleted while the job was queued/running
        config = {**(row.data.get("config") or {}), **fields}
        supabase.table("data_sources").update({"config": config}).eq("id", source_id).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"source state update failed for {source_id}: {e}")


def _result_summary(sync_result: dict) -> dict:
    return {
        "skipped_tabs": sync_result.get("skipped_tabs", []),
        "capped_tabs": sync_result.get("capped_tabs", []),
        "truncated": sync_result.get("truncated", False),
        "indexed_count": sync_result.get("indexed_count"),
        "total_count": sync_result.get("total_count"),
    }


def _load_source(source_id: str):
    res = supabase.table("data_sources").select("*").eq("id", source_id).maybe_single().execute()
    return res.data if res else None


# -------------------------------------------------
# SOURCE JOBS
# -------------------------------------------------
def _sync_by_type(s: dict, hidden_override: dict = None) -> dict:
    from sources.gsheets import sync_sheet
    from sources.excel import sync_excel_url
    from sources.website import sync_website
    from sources.shopify import sync_products as sync_shopify_products

    cfg = s.get("config") or {}
    project_id, source_id = s["project_id"], s["id"]
    if s["type"] == "gsheets":
        return sync_sheet(cfg["sheet_id"], cfg.get("range"), project_id, source_id,
                          qdrant, embeddings, QDRANT_COLLECTION, hidden_override=hidden_override)
    if s["type"] == "excel_online":
        return sync_excel_url(cfg["url"], project_id, source_id, qdrant, embeddings,
                              QDRANT_COLLECTION, tabs=cfg.get("tabs"), hidden_override=hidden_override)
    if s["type"] == "website":
        result = sync_website(
            url=cfg["url"], project_id=project_id, source_id=source_id,
            qdrant=qdrant, embeddings=embeddings, collection=QDRANT_COLLECTION,
            full_site=cfg.get("full_site", True), max_pages=clamp_max_pages(cfg.get("max_pages")),
        )
        if result["pages_indexed"] == 0:
            raise ValueError("Could not read any content from this website. It may block automated access, or only show its content with JavaScript.")
        return result
    if s["type"] == "shopify":
        return sync_shopify_products(project_id, source_id, qdrant, embeddings, QDRANT_COLLECTION)
    raise ValueError(f"Sources of type {s['type']} aren't indexed.")


def _run_source_job(source_id: str, work):
    """Shared wrapper: mark syncing, run `work(source)`, record the outcome.
    A failed sync leaves the previous index in place (replace_points rolls
    back its partial batches), so a failed Reload loses nothing."""
    s = _load_source(source_id)
    if not s:
        print(f"[job] source {source_id} no longer exists, skipping")
        return
    update_source_state(source_id, sync_status="syncing", sync_started_at=_now(), sync_error=None)
    try:
        result = work(s) or {}
        update_source_state(source_id, sync_status="done", sync_finished_at=_now(),
                            sync_error=None, sync_result=_result_summary(result))
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"[job] source {source_id} ({s['type']}) failed: {e}")
        update_source_state(source_id, sync_status="failed", sync_finished_at=_now(),
                            sync_error=str(e) if isinstance(e, ValueError) else _GENERIC_SOURCE_ERROR)


def run_sync_source(payload: dict):
    """Connect or Reload a sheet / online Excel / website / Shopify source."""
    from sources.sheet_tables import parse_hidden_override
    hidden = parse_hidden_override(payload.get("hidden_columns"))
    _run_source_job(payload["source_id"], lambda s: _sync_by_type(s, hidden))


def run_excel_upload(payload: dict):
    """Index an uploaded Excel file. The browser's bytes were parked in
    Storage by the upload route (a job can't carry a file); removed once
    read, since Excel uploads were never kept."""
    from sources.excel import sync_excel_bytes
    from sources.sheet_tables import parse_hidden_override

    path = payload["storage_path"]

    def work(s):
        try:
            file_bytes = supabase.storage.from_("documents").download(path)
        except Exception:
            raise ValueError("The uploaded file couldn't be found. Please upload it again.")
        return sync_excel_bytes(
            file_bytes, s["project_id"], s["id"], qdrant, embeddings, QDRANT_COLLECTION,
            (s.get("config") or {}).get("tabs"), parse_hidden_override(payload.get("hidden_columns")),
        )

    try:
        _run_source_job(payload["source_id"], work)
    finally:
        try:
            supabase.storage.from_("documents").remove([path])
        except Exception as e:
            sentry_sdk.capture_exception(e)


def run_reembed_columns(payload: dict):
    """Rebuild a spreadsheet source's vectors after its visible columns
    changed, so hidden columns leave the search index too."""
    from sources.sheet_tables import reembed_from_tables
    from sources.table_query import invalidate

    def work(s):
        source_type = "gsheets" if s["type"] == "gsheets" else "excel"
        count = reembed_from_tables(s["id"], s["project_id"], source_type, qdrant, embeddings, QDRANT_COLLECTION)
        invalidate(s["project_id"])
        return {"indexed_count": count}

    _run_source_job(payload["source_id"], work)


# -------------------------------------------------
# DISPATCH
# -------------------------------------------------
def run_ingest_file(payload: dict):
    from ingest import process_file
    process_file(payload)


JOBS = {
    "ingest_file": run_ingest_file,
    "sync_source": run_sync_source,
    "excel_upload": run_excel_upload,
    "reembed_columns": run_reembed_columns,
}


def run_job(kind: str, payload: dict):
    handler = JOBS.get(kind)
    if not handler:
        print(f"[job] unknown job kind {kind!r}, ignoring")
        return
    print(f"[job] start {kind}: {mem_summary()}")
    try:
        handler(payload)
    except Exception as e:
        # Handlers record their own failures; this is the last-resort net.
        sentry_sdk.capture_exception(e)
        print(f"[job] {kind} crashed: {e}")
    finally:
        release_memory()
        print(f"[job] done {kind}: {mem_summary()}")
