# sources/gsheets.py

import re
import uuid
from urllib.parse import quote

import sentry_sdk
import pandas as pd
from qdrant_client import models

# FIX: removed unused RecursiveCharacterTextSplitter import


def get_sheet_names(sheet_id: str, range_name: str):
    if not range_name or range_name.strip().lower() in ("", "all"):
        return None, []
    requested = [r.strip() for r in range_name.split(",") if r.strip()]
    return requested, []


SHEET_ID_RE = re.compile(r"^[A-Za-z0-9-_]{20,}$")


def validate_sheet_id(sheet_id: str):
    """The sheet id goes straight into a docs.google.com URL path. The host
    is hardcoded so this isn't SSRF, but an unvalidated value reshapes the
    path (`?`, `#`, `../`) and, more commonly, is simply a mis-parsed URL
    that would 404 and produce a silently empty source."""
    if not sheet_id or not SHEET_ID_RE.match(sheet_id):
        raise ValueError(
            "That doesn't look like a Google Sheet link. Open your sheet, "
            "click Share → Anyone with the link → Viewer, then paste the URL "
            "from your browser's address bar."
        )


def fetch_tab(sheet_id: str, tab_name: str):
    # tab_name is user-supplied and goes into a query string — a name
    # containing & or # would otherwise rewrite the gviz parameters.
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={quote(tab_name, safe='')}"
    )
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Could not fetch tab '{tab_name}': {e}")
        return None


def sync_sheet(sheet_id: str, range_name: str, project_id: str, source_id: str, qdrant, embeddings, collection: str):
    validate_sheet_id(sheet_id)
    requested_tabs, _ = get_sheet_names(sheet_id, range_name)

    # There is no way to enumerate a sheet's tabs without Google credentials
    # — the old code called the v3 "worksheets feed" API, which Google shut
    # down in 2021, so it ALWAYS failed and silently fell back to reading
    # only the first tab while reporting a full sync. Rather than lie, read
    # the first tab when no tabs are named, and let the caller tell the user
    # to name tabs explicitly if they need more than one.
    tabs_to_read = requested_tabs if requested_tabs is not None else [None]

    # NOTE: the purge deliberately happens AFTER the fetch loop below, not
    # here. Deleting first meant a sheet that had since been made private
    # wiped the existing index and then "succeeded" with nothing.

    all_chunks = []
    all_metas = []
    skipped = []
    synced = []

    for tab in tabs_to_read:
        if tab is None:
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
            try:
                df = pd.read_csv(url)
                tab_label = "default"
            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"Could not fetch default tab: {e}")
                skipped.append("default")
                continue
        else:
            df = fetch_tab(sheet_id, tab)
            if df is None or df.empty:
                skipped.append(tab)
                continue
            tab_label = tab

        for _, row in df.iterrows():
            # FIX: filter out nan and empty values
            text = ", ".join(
                f"{col}: {row[col]}" for col in df.columns
                if str(row[col]).strip() and str(row[col]) != "nan"
            )
            if not text.strip():
                continue
            all_chunks.append(text)
            all_metas.append({
                "project_id": project_id,
                "source_id": source_id,
                "source_type": "gsheets",
                "sheet_tab": tab_label,
                "text": text,
            })

        synced.append(tab_label)

    # Previously this returned successfully, so a private/deleted/unreachable
    # sheet was saved as a "connected" source with nothing behind it — the
    # single most likely real-world failure, and completely invisible.
    # Raising lets add_source's existing handler clean up the orphaned row
    # and show the user a real message (same guard the website branch uses).
    if not all_chunks:
        raise ValueError(
            "Couldn't read any data from that sheet. Check that it's shared "
            "as 'Anyone with the link can view', that it isn't empty, and "
            "that any tab names you entered match exactly."
        )

    vectors = embeddings.embed_documents(all_chunks)

    # Only now that we have real data is it safe to drop the old index.
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
            ) for v, m in zip(vectors, all_metas)
        ]
    )

    print(f"Synced {len(all_chunks)} rows from tabs: {synced}, skipped: {skipped}")
    return {"synced_tabs": synced, "skipped_tabs": skipped}