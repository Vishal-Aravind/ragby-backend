"""
Daily clean-up of uploaded media nobody uses any more.

Everything uploaded from the dashboard (flow images/videos/audio/documents,
product images, event page images, template/campaign header images) lives
in the `flow-media` bucket, and nothing deleted it when it was removed from
a flow, product or page - files only piled up.

Mark and sweep, so a removed file always gets a grace period:
  1. Collect every flow-media path still referenced anywhere.
  2. A file not referenced is noted in storage_orphans with today's date
     (the first day it was seen unused).
  3. A noted file that's referenced again is un-noted.
  4. A file unused for GRACE_DAYS is deleted.

Fails safe: if any lookup errors, or no references are found at all while
files exist (a sign the lookup is wrong, not that nothing is used), the run
deletes nothing.
"""
import json
import re
from datetime import datetime, timedelta, timezone

import sentry_sdk

from clients import supabase

BUCKET = "flow-media"
GRACE_DAYS = 7
MIN_AGE = timedelta(hours=24)       # a fresh upload may not be saved anywhere yet
MAX_DELETE_PER_RUN = 500
PAGE = 1000

# Only files in these upload folders are ever cleaned up - the ones whose
# every use is covered by REFERENCE_SOURCES below. Paths look like
# "<user id>/<folder>/<file>". A new feature that uploads gets its own folder
# (src/app/api/storage/upload/route.js FOLDER_RULES only accepts listed
# folders), so its files are left alone until that folder is added here AND
# the place its URLs are saved is added to REFERENCE_SOURCES.
CLEANABLE_FOLDERS = {
    "message_media", "message_video", "message_audio", "message_document",   # WhatsApp flows
    "web-image", "web-video", "web-audio", "web-document",                    # website flows
    "products",                                                               # shop
    "event-pages",                                                            # registrations
    "wa-template",                                                            # templates / campaigns
}


def _cleanable(path: str) -> bool:
    parts = path.split("/")
    return len(parts) >= 3 and parts[1] in CLEANABLE_FOLDERS

# Every place a flow-media URL can be saved. Whole rows are scanned as text,
# so a URL anywhere in a JSON column (node content, page blocks, campaign
# header) counts.
REFERENCE_SOURCES = [
    ("flow_nodes", "content"),
    ("events", "banner_url, page_json"),
    ("products", "*"),
    ("catalogs", "*"),
    ("campaigns", "template_variables"),
    ("shop_config", "*"),
]

PATH_RE = re.compile(r"/storage/v1/object/public/flow-media/([^\"'\s?#)\\]+)")


def _referenced_paths() -> set:
    refs = set()
    for table, columns in REFERENCE_SOURCES:
        offset = 0
        while True:
            rows = supabase.table(table).select(columns).range(offset, offset + PAGE - 1).execute().data or []
            for row in rows:
                for m in PATH_RE.finditer(json.dumps(row, ensure_ascii=False)):
                    refs.add(m.group(1))
            if len(rows) < PAGE:
                break
            offset += PAGE
    return refs


def _parse_ts(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _list_files(prefix: str = "", depth: int = 0):
    """(path, created_at) for every file in the bucket. Folders have no id."""
    if depth > 6:
        return
    bucket = supabase.storage.from_(BUCKET)
    offset = 0
    while True:
        items = bucket.list(prefix or None, {"limit": PAGE, "offset": offset, "sortBy": {"column": "name", "order": "asc"}}) or []
        for it in items:
            name = it.get("name")
            if not name or name == ".emptyFolderPlaceholder":
                continue
            path = f"{prefix}/{name}" if prefix else name
            if it.get("id") is None:
                yield from _list_files(path, depth + 1)
            else:
                yield path, _parse_ts(it.get("created_at"))
        if len(items) < PAGE:
            break
        offset += PAGE


def plan_cleanup(files, refs: set, noted: dict, now: datetime):
    """Pure decision step (unit-tested).

    files: [(path, created_at)], refs: referenced paths,
    noted: {path: first_unused_at} from storage_orphans.
    Returns (to_note, to_unnote, to_delete)."""
    unused = {p for p, created in files
              if _cleanable(p) and p not in refs and (created is None or now - created >= MIN_AGE)}
    to_note = sorted(unused - set(noted))
    to_unnote = sorted(set(noted) - unused)
    cutoff = now - timedelta(days=GRACE_DAYS)
    to_delete = sorted(p for p in unused & set(noted) if noted[p] and noted[p] <= cutoff)[:MAX_DELETE_PER_RUN]
    return to_note, to_unnote, to_delete


def run_storage_cleanup() -> dict:
    now = datetime.now(timezone.utc)
    refs = _referenced_paths()
    files = list(_list_files())
    if files and not refs:
        print("storage cleanup: no references found at all - skipping as a precaution")
        return {"files": len(files), "references": 0, "skipped": True}

    rows = []
    offset = 0
    while True:
        page = supabase.table("storage_orphans").select("path, first_unused_at").range(offset, offset + PAGE - 1).execute().data or []
        rows.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    noted = {r["path"]: _parse_ts(r.get("first_unused_at")) for r in rows}

    to_note, to_unnote, to_delete = plan_cleanup(files, refs, noted, now)

    for i in range(0, len(to_unnote), 100):
        supabase.table("storage_orphans").delete().in_("path", to_unnote[i:i + 100]).execute()
    for i in range(0, len(to_note), 500):
        supabase.table("storage_orphans").upsert(
            [{"path": p, "first_unused_at": now.isoformat()} for p in to_note[i:i + 500]], on_conflict="path"
        ).execute()

    deleted = 0
    for i in range(0, len(to_delete), 100):
        chunk = to_delete[i:i + 100]
        try:
            supabase.storage.from_(BUCKET).remove(chunk)
            supabase.table("storage_orphans").delete().in_("path", chunk).execute()
            deleted += len(chunk)
        except Exception as e:
            sentry_sdk.capture_exception(e)
    return {"files": len(files), "references": len(refs), "noted": len(to_note),
            "back_in_use": len(to_unnote), "deleted": deleted}
