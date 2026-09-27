# sources/excel.py

import io
import requests
import pandas as pd

from config import MAX_CHUNKS_PER_INGEST
from sources.url_guard import assert_public_http_url, safe_get
from sources.sheet_tables import dataframe_to_table, index_tables, normalize_tab_list

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


def read_excel(file_bytes: bytes, tabs=None) -> dict:
    """Parse the workbook into capped tables, without indexing anything —
    shared by the column preview and the real sync. `tabs` empty/None reads
    every sheet; otherwise only the named ones (case-insensitive)."""
    requested = normalize_tab_list(tabs)
    xls = None
    # Try openpyxl first (xlsx), fall back to xlrd (xls).
    for engine in ["openpyxl", "xlrd"]:
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
            break
        except Exception:
            continue
    if xls is None:
        raise ValueError("Could not read the Excel file. Make sure it is a valid .xlsx or .xls file.")

    names = [str(n) for n in xls.sheet_names]
    skipped = []
    if requested is None:
        to_read = names
    else:
        by_lower = {n.strip().casefold(): n for n in names}
        to_read = []
        for t in requested:
            match = by_lower.get(t.strip().casefold())
            if match is None:
                skipped.append(t)
            elif match not in to_read:
                to_read.append(match)

    # One row is one embedding, so a 200k-row workbook was 200k embeddings
    # in a single unbounded call against our own OpenAI key. The same cap
    # bounds the stored table copy.
    tables, capped_tabs = [], []
    room, total_found, truncated = MAX_CHUNKS_PER_INGEST, 0, False
    for name in to_read:
        if room <= 0:
            capped_tabs.append(name)
            truncated = True
            continue
        columns, rows = dataframe_to_table(xls.parse(name, nrows=room + 1))
        if not rows:
            skipped.append(name)
            continue
        total_found += len(rows)
        if len(rows) > room:
            rows = rows[:room]
            truncated = True
        room -= len(rows)
        tables.append((name, columns, rows))

    # Previously returned success here, so an empty or header-only workbook
    # produced a green "connected" source the bot could never answer from.
    if not tables:
        raise ValueError(
            "Couldn't read any rows from that file. Check that it has a header "
            "row and at least one row of data, and that any sheet names you "
            "entered match the file."
        )
    return {
        "tables": tables,
        "skipped_tabs": skipped,
        "capped_tabs": capped_tabs,
        "row_count": sum(len(r) for _, _, r in tables),
        "total_found": total_found,
        "truncated": truncated,
    }


def sync_excel_url(url: str, project_id: str, source_id: str, qdrant, embeddings,
                   collection: str, tabs=None, hidden_override: dict = None):
    """Fetch Excel from a URL, embed and store in Qdrant."""
    file_bytes = fetch_excel_from_url(url)
    return _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection,
                             "excel_online", tabs, hidden_override)


def sync_excel_bytes(file_bytes: bytes, project_id: str, source_id: str, qdrant, embeddings,
                     collection: str, tabs=None, hidden_override: dict = None):
    """Embed and store local Excel bytes in Qdrant."""
    return _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection,
                             "excel_local", tabs, hidden_override)


def _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection,
                      source_label, tabs=None, hidden_override=None):
    # Purge happens after a successful parse (inside index_tables).
    # Deleting first meant re-uploading a corrupt or password-protected
    # workbook wiped the working index and left the source "connected" with
    # nothing in it.
    book = read_excel(file_bytes, tabs)
    if book["truncated"]:
        print(f"excel source {source_id} truncated to {MAX_CHUNKS_PER_INGEST} rows")

    stored, embedded = index_tables(
        book["tables"], project_id, source_id, "excel",
        qdrant, embeddings, collection, hidden_override,
    )
    print(f"[{source_label}] Synced {book['row_count']} rows from Excel ({embedded} embedded)")
    return {
        "chunks_indexed": embedded,
        "indexed_count": book["row_count"],
        # Rows past the cap are never parsed (memory), so the true total is
        # unknown once truncated; the caller then shows the generic
        # "only part was indexed" message.
        "total_count": None if book["truncated"] else book["total_found"],
        "truncated": book["truncated"],
        "skipped_tabs": book["skipped_tabs"],
        "capped_tabs": book["capped_tabs"],
        "hidden_columns": sorted({c for t in stored for c in t["hidden"]}),
    }
