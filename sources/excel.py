# sources/excel.py

import uuid
import io
import requests
import pandas as pd
from qdrant_client import models

from config import MAX_CHUNKS_PER_INGEST
from sources.url_guard import assert_public_http_url, safe_get

MAX_EXCEL_BYTES = 25 * 1024 * 1024


def fetch_excel_from_url(url: str) -> bytes:
    """
    Fetch Excel bytes from a URL, handling OneDrive/SharePoint sharing links.
    """
    # This fetches a caller-supplied URL server-side and indexes whatever
    # comes back into their chatbot, so without this an internal address
    # could be read out through chat. Reachable via POST /sources/add with
    # type "excel_online" — there is no UI for it, so nobody would notice.
    assert_public_http_url(url)

    session = requests.Session()
    res = safe_get(session, url)
    resolved_url = res.url

    content_type = res.headers.get("Content-Type", "")
    if "html" in content_type:
        if "onedrive.live.com" in resolved_url:
            resolved_url = resolved_url.replace("redir?", "download?")
            resolved_url = resolved_url.replace("download=0", "download=1")
            if "download=" not in resolved_url:
                resolved_url += "&download=1"
            res = safe_get(session, resolved_url)
        elif "sharepoint.com" in resolved_url:
            sep = "&" if "?" in resolved_url else "?"
            res = safe_get(session, resolved_url + sep + "download=1")
        else:
            raise ValueError(
                "This link returns a webpage, not an Excel file. "
                "Please use a direct download link.\n\n"
                "In OneDrive: open the file → File → Download → copy the browser URL before it saves."
            )

    content_type = res.headers.get("Content-Type", "")
    if "html" in content_type:
        raise ValueError(
            "Could not get a direct download from this OneDrive link. "
            "Please open the file in OneDrive, click File → Download, "
            "and copy the URL from the browser address bar before the file saves."
        )

    # No size cap meant the whole body was buffered into the worker's RAM.
    if len(res.content) > MAX_EXCEL_BYTES:
        raise ValueError("That file is too large. The limit is 25MB.")

    return res.content


def excel_bytes_to_chunks(file_bytes: bytes):
    """
    Convert Excel bytes to list of (sheet_name, text) tuples.
    FIX: removed duplicate definition — keeping the better version with
    engine detection (openpyxl/xlrd) and nan filtering.
    """
    # Try openpyxl first (xlsx), fall back to xlrd (xls)
    for engine in ["openpyxl", "xlrd"]:
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
            results = []
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name).astype(str).fillna("")
                for _, row in df.iterrows():
                    text = ", ".join(
                        f"{col}: {row[col]}" for col in df.columns
                        if str(row[col]).strip() and str(row[col]) != "nan"
                    )
                    if text.strip():
                        results.append((sheet_name, text))
            return results
        except Exception:
            continue

    raise ValueError("Could not read the Excel file. Make sure it is a valid .xlsx or .xls file.")


def sync_excel_url(
    url: str,
    project_id: str,
    source_id: str,
    qdrant,
    embeddings,
    collection: str
):
    """Fetch Excel from a URL, embed and store in Qdrant."""
    file_bytes = fetch_excel_from_url(url)
    return _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection, source_label="excel_online")


def sync_excel_bytes(
    file_bytes: bytes,
    project_id: str,
    source_id: str,
    qdrant,
    embeddings,
    collection: str
):
    """Embed and store local Excel bytes in Qdrant."""
    return _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection, source_label="excel_local")


def _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection, source_label):
    # Purge happens after a successful parse (see below). Deleting first
    # meant re-uploading a corrupt or password-protected workbook wiped the
    # working index and left the source "connected" with nothing in it.
    rows = excel_bytes_to_chunks(file_bytes)

    # Previously returned success here, so an empty or header-only workbook
    # produced a green "connected" source the bot could never answer from.
    if not rows:
        raise ValueError(
            "Couldn't read any rows from that file. Check that it has a header "
            "row and at least one row of data."
        )

    chunks = [text for _, text in rows]
    metas = [
        {
            "project_id": project_id,
            "source_id": source_id,
            "source_type": "excel",
            "sheet_tab": sheet,
            "text": text,
        }
        for sheet, text in rows
    ]

    # One row is one embedding, so a 200k-row workbook was 200k embeddings
    # in a single unbounded call against our own OpenAI key.
    if len(chunks) > MAX_CHUNKS_PER_INGEST:
        print(f"excel source {source_id} truncated to {MAX_CHUNKS_PER_INGEST} rows")
        chunks = chunks[:MAX_CHUNKS_PER_INGEST]
        metas = metas[:MAX_CHUNKS_PER_INGEST]

    vectors = embeddings.embed_documents(chunks)

    # Safe to drop the old index only now that replacement content exists.
    qdrant.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id)
            )]
        )
    )

    qdrant.upload_points(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=v,
                payload=m
            ) for v, m in zip(vectors, metas)
        ]
    )

    print(f"[{source_label}] Synced {len(chunks)} rows from Excel")
    return {"chunks_indexed": len(chunks)}