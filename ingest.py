import io
import time
from typing import Optional

import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from text_splitter import split_text
from qdrant_client import models

from clients import supabase, qdrant, embeddings
from vector_sync import replace_points
from config import QDRANT_COLLECTION, MAX_CHUNKS_PER_INGEST
from auth import verify_token, require_project_access
from ratelimit import is_rate_limited
from usage import get_plan_limits, count_knowledge_items, knowledge_limit_message
from jobs import enqueue

router = APIRouter()


def _purge_file_points(file_id: str):
    """Delete every Qdrant point belonging to a file."""
    qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="file_id",
                match=models.MatchValue(value=file_id)
            )]
        )
    )


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class IngestRequest(BaseModel):
    projectId: str
    filename: str
    filePath: str
    # The object's size as Supabase's own Storage metadata API reported it,
    # moments before this call. Re-saving a note overwrites the SAME storage
    # key, and a download immediately after that overwrite can race ahead of
    # it and return the PREVIOUS version's bytes — silently embedding stale
    # content into a freshly-created, otherwise-successful-looking Qdrant
    # point. Optional so an older frontend build that doesn't send it still
    # works, just without this protection.
    expectedBytes: Optional[int] = None
    # The storage object this upload replaces (a re-saved note, a same-named
    # re-upload). Deleted by the job only after the new one is indexed.
    oldStoragePath: Optional[str] = None


# -------------------------------------------------
# TEXT EXTRACTORS
# -------------------------------------------------
def extract_pdf(b):
    import pdfplumber
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        return [(i+1, p.extract_text() or "") for i, p in enumerate(pdf.pages) if p.extract_text()]

def extract_docx(b):
    from docx import Document
    d = Document(io.BytesIO(b))
    return [(1, "\n".join(p.text for p in d.paragraphs if p.text.strip()))]

def extract_pptx(b):
    from pptx import Presentation
    prs = Presentation(io.BytesIO(b))
    out = []
    for i, s in enumerate(prs.slides):
        txt = "\n".join(sh.text for sh in s.shapes if hasattr(sh, "text"))
        if txt.strip():
            out.append((i+1, txt))
    return out

def extract_excel(b):
    import pandas as pd
    xls = pd.ExcelFile(io.BytesIO(b))
    return [(n, xls.parse(n).astype(str).fillna("").to_csv(index=False)) for n in xls.sheet_names]

def extract_txt(b):
    return [(1, b.decode("utf-8", errors="ignore"))]


EXTRACTORS = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "ppt": extract_pptx,
    "pptx": extract_pptx,
    "xls": extract_excel,
    "xlsx": extract_excel,
    "txt": extract_txt,
}


# -------------------------------------------------
# INGEST ENDPOINT
# -------------------------------------------------
@router.post("/ingest")
def ingest(req: IngestRequest, user=Depends(verify_token)):
    """Validate an uploaded file and queue it for indexing. The heavy part
    (download, extract, embed) runs as a job — see process_file below and
    jobs.py — so this returns in well under a second."""
    require_project_access(user.id, req.projectId, tab="documents")

    # Every ingest costs real OpenAI money, and nothing here was throttled
    # or capped before — a scripted loop could re-embed indefinitely.
    if is_rate_limited(f"ingest:{req.projectId}", limit=20, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="Too many uploads at once. Please wait a minute and try again.",
        )

    # FIX: filePath is otherwise fully caller-controlled — without this
    # check, a user with a role on their OWN project could point filePath
    # at another project's storage object and have it indexed (crediting
    # req.projectId) as if it were their own document, exfiltrating another
    # tenant's file content into their own chatbot's knowledge base.
    if not req.filePath.startswith(f"{req.projectId}/"):
        raise HTTPException(status_code=403, detail="filePath does not belong to this project")
    # Same for the object the job deletes once the new one is indexed.
    if req.oldStoragePath and not req.oldStoragePath.startswith(f"{req.projectId}/"):
        raise HTTPException(status_code=403, detail="oldStoragePath does not belong to this project")

    row = supabase.table("files") \
        .select("id") \
        .eq("project_id", req.projectId) \
        .eq("filename", req.filename) \
        .execute()

    if not row.data:
        # Was returning this with HTTP 200, so the caller's .ok check passed
        # and a failed ingest looked identical to a successful one.
        raise HTTPException(status_code=404, detail="File not found")

    file_id = row.data[0]["id"]

    def _reject(status_code: int, detail: str):
        # Shared by both plan-limit checks below: clean up rather than
        # leaving a failed row and a stored object the user didn't get any
        # value from.
        supabase.table("files").delete().eq("id", file_id).execute()
        try:
            supabase.storage.from_("documents").remove([req.filePath])
        except Exception as e:
            # The files row is already deleted above, so a failure here
            # leaves the actual uploaded object orphaned in storage forever
            # — counting against the merchant's storage usage for a
            # document that was never processed and no longer appears
            # anywhere in the product. Single call per over-limit attempt,
            # not a loop.
            sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=status_code, detail=detail)

    # Count everything ELSE in the knowledge base — this file's row already
    # exists, created by the upload route before it called us.
    #
    # Skipped when this REPLACES a file that's already indexed (an edited
    # note re-saved, a same-named re-upload): that adds no item. Without
    # this, a merchant already over a lowered limit who edited a note had it
    # rejected — and _reject deletes the row, so the note was lost. "Already
    # indexed" is read from Qdrant, not from anything the caller sends.
    limits = get_plan_limits(req.projectId)
    already_indexed = qdrant.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=models.Filter(must=[
            models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))
        ]),
        exact=False,
    ).count > 0
    if not already_indexed and count_knowledge_items(req.projectId, exclude_file_id=file_id) >= limits["items"]:
        _reject(403, knowledge_limit_message(limits["items"]))

    # Real backstop behind the frontend's own pre-check and the upload-url
    # route's fast-fail — a client that skips both could still reach here.
    # expectedBytes is the browser-reported size, already plumbed through
    # for the overwrite-race fix, so this needs no extra download or lookup.
    if req.expectedBytes is not None and req.expectedBytes > limits["maxFileMB"] * 1024 * 1024:
        _reject(
            403,
            f"This file is too large for your plan (limit {limits['maxFileMB']}MB). Upgrade your plan to upload larger files.",
        )

    ext = req.filename.lower().split(".")[-1]
    if ext not in EXTRACTORS:
        _set_file_state(file_id, "failed", error="That file type isn't supported.")
        raise HTTPException(status_code=400, detail="That file type isn't supported.")

    _set_file_state(file_id, "queued")
    enqueue("ingest_file", {
        "file_id": file_id,
        "project_id": req.projectId,
        "filename": req.filename,
        "file_path": req.filePath,
        "expected_bytes": req.expectedBytes,
        "old_storage_path": req.oldStoragePath,
    })
    return {"status": "queued", "id": file_id}


def _set_file_state(file_id: str, status: str, error: str = None, result: dict = None):
    update = {"status": status, "error": error, "result": result}
    try:
        supabase.table("files").update(update).eq("id", file_id).execute()
    except Exception as e:
        # Before the migration adding error/result has run, still record the
        # status itself rather than leaving the row stuck.
        print(f"file state update with error/result failed ({e}); retrying status only")
        supabase.table("files").update({"status": status}).eq("id", file_id).execute()


# -------------------------------------------------
# THE JOB (runs on the worker, see job_runner.py)
# -------------------------------------------------
_PROCESS_ERROR = "We couldn't process that file. Please try uploading it again."


def process_file(payload: dict):
    """Download, extract, chunk and embed one uploaded file, recording the
    outcome on its files row (status + error/result)."""
    file_id = payload["file_id"]
    project_id = payload["project_id"]
    filename = payload["filename"]
    file_path = payload["file_path"]
    expected_bytes = payload.get("expected_bytes")
    old_storage_path = payload.get("old_storage_path")

    # A newer upload of the same file (a note saved twice quickly) points
    # the row at a newer storage object. This older job must then do
    # nothing, or it could finish last and index the previous version.
    row = supabase.table("files").select("storage_path").eq("id", file_id).maybe_single().execute()
    if not row or not row.data:
        print(f"[ingest] file {file_id} deleted before its job ran, skipping")
        return
    if row.data.get("storage_path") and row.data["storage_path"] != file_path:
        print(f"[ingest] file {file_id} has a newer upload, skipping stale job")
        return

    _set_file_state(file_id, "processing")
    ext = filename.lower().split(".")[-1]
    extractor = EXTRACTORS.get(ext)
    if not extractor:
        _set_file_state(file_id, "failed", error="That file type isn't supported.")
        return

    # Everything below can fail on someone else's infrastructure (Supabase
    # storage, OpenAI, Qdrant) or on a corrupt/password-protected file.
    # Without this, any of those left the row pinned at "processing" forever
    # with no reason recorded.
    try:
        b = supabase.storage.from_("documents").download(file_path)
        # A handful of retries if the download doesn't yet match the size
        # the BROWSER'S OWN File object reported before it ever uploaded
        # anything — deliberately not a re-query of Storage's own metadata,
        # which is subject to the exact same overwrite-propagation lag as
        # this download and so isn't a trustworthy reference point either.
        # Bounded backoff, ~7.5s worst case.
        if expected_bytes is not None:
            attempts = 0
            while len(b) != expected_bytes and attempts < 6:
                time.sleep(0.3 * (attempts + 1))
                b = supabase.storage.from_("documents").download(file_path)
                attempts += 1
            if len(b) != expected_bytes:
                print(
                    f"ingest: downloaded {len(b)} bytes for file {file_id}, "
                    f"expected {expected_bytes}, after {attempts} retries — proceeding anyway"
                )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"ingest download failed for file {file_id}: {e}")
        _set_file_state(file_id, "failed", error=_PROCESS_ERROR)
        return

    # A renamed .zip/.exe/whatever-to-.pdf, or any genuinely corrupt file,
    # fails HERE — inside the parsing library, not our infrastructure. That
    # is routine bad user input, not a bug to page anyone about.
    try:
        pages = extractor(b)
    except Exception as e:
        print(f"extraction failed for file {file_id} ({ext}): {e}")
        _set_file_state(
            file_id, "failed",
            error=f"This doesn't look like a valid .{ext} file. It may be corrupted, password-protected, or renamed from a different file type.",
        )
        return
    del b

    try:
        chunks, metas = [], []
        for page, text in pages:
            for c in split_text(text, chunk_size=1500, chunk_overlap=200):
                chunks.append(c)
                metas.append({
                    "project_id": project_id,
                    "file_id": file_id,
                    "filename": filename,
                    "page_number": page,
                    "source_type": "document",
                    "text": c,
                })

        if not chunks:
            _set_file_state(
                file_id, "failed",
                error="Couldn't read any text from that file. If it's a scanned PDF, it needs to contain selectable text.",
            )
            return

        # A single enormous file would otherwise become one unbounded
        # embedding bill. Index the first N chunks and stop there.
        total_found = len(chunks)
        truncated = total_found > MAX_CHUNKS_PER_INGEST
        if truncated:
            chunks = chunks[:MAX_CHUNKS_PER_INGEST]
            metas = metas[:MAX_CHUNKS_PER_INGEST]
            print(f"ingest truncated file {file_id} to {MAX_CHUNKS_PER_INGEST} chunks")

        # Re-ingesting reuses the same file_id (the row is upserted), so the
        # previous version's chunks must be replaced, not left alongside the
        # new ones.
        replace_points(qdrant, embeddings, QDRANT_COLLECTION, chunks, metas, "file_id", file_id)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"ingest failed for file {file_id}: {e}")
        _set_file_state(file_id, "failed", error=_PROCESS_ERROR)
        return

    _set_file_state(file_id, "indexed", result={
        "truncated": truncated,
        "indexed_count": len(chunks),
        "total_count": total_found,
    })

    # Only now — the new object is confirmed indexed — is the OLD physical
    # object (a prior upload or note edit under this filename) removed.
    # Never eagerly: if indexing had failed, the old object is left alone
    # rather than lost.
    if old_storage_path and old_storage_path != file_path:
        try:
            supabase.storage.from_("documents").remove([old_storage_path])
        except Exception as e:
            print(f"old storage object cleanup failed for {file_id}: {e}")


@router.delete("/document/{file_id}")
def delete_document(file_id: str, user=Depends(verify_token)):
    row = supabase.table("files").select("project_id").eq("id", file_id).maybe_single().execute()
    file_row = row.data if row else None
    if not file_row:
        raise HTTPException(status_code=404, detail="Not found")
    require_project_access(user.id, file_row["project_id"], tab="documents")

    _purge_file_points(file_id)
    supabase.table("files").delete().eq("id", file_id).execute()
    return {"status": "deleted"}