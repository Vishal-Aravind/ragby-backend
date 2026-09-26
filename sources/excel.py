# sources/excel.py

import io
import requests
import pandas as pd

from config import MAX_CHUNKS_PER_INGEST
from sources.url_guard import assert_public_http_url, safe_get
from vector_sync import replace_points
from sources.sheet_tables import (
    dataframe_to_table, load_previous, resolve_hidden, row_text, save_tables,
)

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


def excel_bytes_to_tables(file_bytes: bytes):
    """
    Convert Excel bytes to a list of (sheet_name, columns, rows) tables.
    Tries openpyxl first (xlsx), falls back to xlrd (xls).
    """
    for engine in ["openpyxl", "xlrd"]:
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
            results = []
            for sheet_name in xls.sheet_names:
                columns, rows = dataframe_to_table(xls.parse(sheet_name))
                if rows:
                    results.append((str(sheet_name), columns, rows))
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
    parsed = excel_bytes_to_tables(file_bytes)

    # Previously returned success here, so an empty or header-only workbook
    # produced a green "connected" source the bot could never answer from.
    if not parsed:
        raise ValueError(
            "Couldn't read any rows from that file. Check that it has a header "
            "row and at least one row of data."
        )

    # One row is one embedding, so a 200k-row workbook was 200k embeddings
    # in a single unbounded call against our own OpenAI key. The same cap
    # bounds the stored table copy.
    total_found = sum(len(rows) for _, _, rows in parsed)
    truncated = total_found > MAX_CHUNKS_PER_INGEST
    if truncated:
        print(f"excel source {source_id} truncated to {MAX_CHUNKS_PER_INGEST} rows")

    previous = load_previous(source_id)
    tables, chunks, metas = [], [], []
    room = MAX_CHUNKS_PER_INGEST
    for sheet, columns, rows in parsed:
        if room <= 0:
            break
        rows = rows[:room]
        room -= len(rows)
        # Personal-data columns (emails, phones) start hidden: they're left
        # out of the embedded text as well as the table query.
        hidden = resolve_hidden(previous, sheet, columns, rows)
        hidden_set = set(hidden)
        tables.append({"tab": sheet, "columns": columns, "rows": rows, "hidden": hidden})
        for row in rows:
            text = row_text(columns, row, hidden_set)
            if not text:
                continue
            chunks.append(text)
            metas.append({
                "project_id": project_id,
                "source_id": source_id,
                "source_type": "excel",
                "sheet_tab": sheet,
                "text": text,
            })

    replace_points(qdrant, embeddings, collection, chunks, metas, "source_id", source_id)
    # Only after the vectors were replaced, so a failed sync leaves the
    # previous table and index consistent with each other.
    save_tables(source_id, project_id, tables)

    stored = sum(len(t["rows"]) for t in tables)
    print(f"[{source_label}] Synced {stored} rows from Excel ({len(chunks)} embedded)")
    return {
        "chunks_indexed": len(chunks),
        "indexed_count": stored,
        "total_count": total_found,
        "truncated": truncated,
        "hidden_columns": sorted({c for t in tables for c in t["hidden"]}),
    }