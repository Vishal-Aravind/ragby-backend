import io
import time
import uuid
from typing import Optional

import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import pdfplumber
from docx import Document
from pptx import Presentation
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from clients import supabase, qdrant, embeddings
from config import QDRANT_COLLECTION, MAX_CHUNKS_PER_INGEST
from auth import verify_token, require_project_access
from ratelimit import is_rate_limited
from usage import get_plan_limits

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


# -------------------------------------------------
# TEXT EXTRACTORS
# -------------------------------------------------
def extract_pdf(b):
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        return [(i+1, p.extract_text() or "") for i, p in enumerate(pdf.pages) if p.extract_text()]

def extract_docx(b):
    d = Document(io.BytesIO(b))
    return [(1, "\n".join(p.text for p in d.paragraphs if p.text.strip()))]

def extract_pptx(b):
    prs = Presentation(io.BytesIO(b))
    out = []
    for i, s in enumerate(prs.slides):
        txt = "\n".join(sh.text for sh in s.shapes if hasattr(sh, "text"))
        if txt.strip():
            out.append((i+1, txt))
    return out

def extract_excel(b):
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

    # Count OTHER documents in the project — this one's row already exists,
    # created by the upload route before it called us.
    limits = get_plan_limits(req.projectId)
    existing = supabase.table("files")         .select("id", count="exact")         .eq("project_id", req.projectId)         .neq("id", file_id)         .execute()
    if (existing.count or 0) >= limits["documents"]:
        _reject(
            403,
            f"You've reached your plan's limit of {limits['documents']} documents. Delete one, or upgrade your plan, to add more.",
        )

    # Real backstop behind the frontend's own pre-check and the upload-url
    # route's fast-fail — a client that skips both could still reach here.
    # expectedBytes is the browser-reported size, already plumbed through
    # for the overwrite-race fix, so this needs no extra download or lookup.
    if req.expectedBytes is not None and req.expectedBytes > limits["maxFileMB"] * 1024 * 1024:
        _reject(
            403,
            f"This file is too large for your plan (limit {limits['maxFileMB']}MB). Upgrade your plan to upload larger files.",
        )

    supabase.table("files").update({"status": "processing"}).eq("id", file_id).execute()

    ext = req.filename.lower().split(".")[-1]
    extractor = EXTRACTORS.get(ext)
    if not extractor:
        supabase.table("files").update({"status": "failed"}).eq("id", file_id).execute()
        raise HTTPException(status_code=400, detail="That file type isn't supported.")

    # Everything below can fail on someone else's infrastructure (Supabase
    # storage, OpenAI, Qdrant) or on a corrupt/password-protected file. Without
    # this, any of those left the row pinned at "processing" forever with no
    # reason recorded and an unhandled 500 to the caller.
    try:
        b = supabase.storage.from_("documents").download(req.filePath)
        # A handful of retries if the download doesn't yet match the size
        # the BROWSER'S OWN File object reported before it ever uploaded
        # anything — deliberately not a re-query of Storage's own metadata,
        # which is subject to the exact same overwrite-propagation lag as
        # this download and so isn't a trustworthy reference point either.
        # Bounded backoff, ~7.5s worst case: this is a race measured in
        # well under a second normally, not a real outage worth blocking
        # much longer for.
        if req.expectedBytes is not None:
            attempts = 0
            while len(b) != req.expectedBytes and attempts < 6:
                time.sleep(0.3 * (attempts + 1))
                b = supabase.storage.from_("documents").download(req.filePath)
                attempts += 1
            if len(b) != req.expectedBytes:
                print(
                    f"ingest: downloaded {len(b)} bytes for file {file_id}, "
                    f"expected {req.expectedBytes}, after {attempts} retries — proceeding anyway"
                )
    except HTTPException:
        raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"ingest download failed for file {file_id}: {e}")
        supabase.table("files").update({"status": "failed"}).eq("id", file_id).execute()
        raise HTTPException(
            status_code=502,
            detail="We couldn't process that file. Please try uploading it again.",
        )

    # A renamed .zip/.exe/whatever-to-.pdf, or any genuinely corrupt file,
    # fails HERE — inside the parsing library, not our infrastructure. That
    # used to fall into the same catch-all below as an OpenAI/Qdrant/Supabase
    # outage: same 502, same Sentry alert, same generic message, even though
    # this is routine bad user input and happens constantly, not a bug to
    # page anyone about.
    try:
        pages = extractor(b)
    except Exception as e:
        print(f"extraction failed for file {file_id} ({ext}): {e}")
        supabase.table("files").update({"status": "failed"}).eq("id", file_id).execute()
        raise HTTPException(
            status_code=400,
            detail=f"This doesn't look like a valid .{ext} file. It may be corrupted, password-protected, or renamed from a different file type.",
        )

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks, metas = [], []

        for page, text in pages:
            for c in splitter.split_text(text):
                chunks.append(c)
                metas.append({
                    "project_id": req.projectId,
                    "file_id": file_id,
                    "filename": req.filename,
                    "page_number": page,
                    "source_type": "document",
                    "text": c,
                })

        if not chunks:
            supabase.table("files").update({"status": "failed"}).eq("id", file_id).execute()
            raise HTTPException(
                status_code=400,
                detail="Couldn't read any text from that file. If it's a scanned PDF, it needs to contain selectable text.",
            )

        # A single enormous file would otherwise become one unbounded
        # embedding bill. Index the first N chunks and stop there.
        total_found = len(chunks)
        truncated = total_found > MAX_CHUNKS_PER_INGEST
        if truncated:
            chunks = chunks[:MAX_CHUNKS_PER_INGEST]
            metas = metas[:MAX_CHUNKS_PER_INGEST]
            print(f"ingest truncated file {file_id} to {MAX_CHUNKS_PER_INGEST} chunks")

        vectors = embeddings.embed_documents(chunks)

        # Re-ingesting reuses the same file_id (the row is upserted), so
        # without this the previous version's chunks stayed in Qdrant
        # alongside the new ones and the bot kept answering from content
        # the user believed they had replaced.
        _purge_file_points(file_id)

        qdrant.upload_points(
            collection_name=QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v,
                    payload=m
                ) for v, m in zip(vectors, metas)
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"ingest failed for file {file_id}: {e}")
        supabase.table("files").update({"status": "failed"}).eq("id", file_id).execute()
        raise HTTPException(
            status_code=502,
            detail="We couldn't process that file. Please try uploading it again.",
        )

    supabase.table("files").update({"status": "indexed"}).eq("id", file_id).execute()
    return {
        "status": "indexed",
        "chunks_indexed": len(chunks),
        "indexed_count": len(chunks),
        "total_count": total_found,
        "truncated": truncated,
    }


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