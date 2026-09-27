# sources/gsheets.py

import io
import re
from urllib.parse import quote

import requests
import sentry_sdk
import pandas as pd

from config import MAX_SHEET_ROWS
from sources.sheet_tables import dataframe_to_table, index_tables, normalize_tab_list

# FIX: removed unused RecursiveCharacterTextSplitter import


def get_sheet_names(sheet_id: str, range_name):
    # The frontend sends a real list of tab names (a chip/tag input, not a
    # comma-separated field) — a tab literally named "Q1, Actuals" could
    # never be selected through a delimiter that's also a valid character in
    # the name. Comma splitting is kept only for sources connected before
    # that change (see normalize_tab_list).
    return normalize_tab_list(range_name), []


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


# The export is parsed in memory on a 512MB instance, so keep it bounded;
# a 5,000-row sheet exports at well under 2MB.
MAX_WORKBOOK_BYTES = 10 * 1024 * 1024


def fetch_workbook(sheet_id: str):
    """{tab name: DataFrame} for every tab, via the sheet's public .xlsx
    export. None if it can't be read (private sheet → Google returns an
    HTML login page, network error, oversized, unparseable)."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    try:
        with requests.get(url, timeout=30, stream=True) as res:
            if not res.ok or "spreadsheetml" not in res.headers.get("Content-Type", ""):
                return None
            body = bytearray()
            for chunk in res.iter_content(64 * 1024):
                body.extend(chunk)
                if len(body) > MAX_WORKBOOK_BYTES:
                    print(f"sheet {sheet_id} export over {MAX_WORKBOOK_BYTES} bytes, using CSV fallback")
                    return None
        xls = pd.ExcelFile(io.BytesIO(bytes(body)), engine="openpyxl")
        # Only as many rows as could ever be indexed (+1 so truncation is
        # still detected), not the whole tab.
        return {str(name): xls.parse(name, nrows=MAX_SHEET_ROWS + 1) for name in xls.sheet_names}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Could not export sheet {sheet_id} as xlsx: {e}")
        return None


def read_sheet(sheet_id: str, range_name) -> dict:
    """Fetch and parse the sheet into capped tables, without indexing
    anything — shared by the column preview and the real sync.
    Leaving tabs empty reads every tab; naming tabs reads only those."""
    validate_sheet_id(sheet_id)
    requested_tabs, _ = get_sheet_names(sheet_id, range_name)

    # The whole workbook (every tab, with real tab names) comes from the
    # public .xlsx export — no Google credentials needed. The UI always
    # promised "leave empty to index every tab" but only the first tab was
    # ever read. If the export fails, fall back to the per-tab CSV
    # endpoint, which can only read the first tab when none are named.
    workbook = fetch_workbook(sheet_id)
    if workbook is not None:
        if requested_tabs is None:
            tabs_to_read = list(workbook)
        else:
            by_lower = {name.strip().casefold(): name for name in workbook}
            tabs_to_read = [
                by_lower.get(t.strip().casefold(), t) for t in requested_tabs
            ]
    else:
        tabs_to_read = requested_tabs if requested_tabs is not None else [None]

    # NOTE: the purge deliberately happens AFTER all fetching, not before.
    # Deleting first meant a sheet that had since been made private wiped
    # the existing index and then "succeeded" with nothing.

    tables = []
    row_count = 0
    skipped = []
    synced = []
    # Tabs never even attempted because an earlier tab already hit the row
    # cap — distinct from `skipped`, which means a tab failed to fetch. The
    # old code only checked the cap at the BOTTOM of this loop, so if tab 1
    # alone exceeded it, tabs 2+ were dropped by `break` with no record of
    # them anywhere, not even here. Checking at the top instead means every
    # remaining requested tab is accounted for one way or another.
    capped_tabs = []
    truncated = False

    for tab in tabs_to_read:
        if row_count >= MAX_SHEET_ROWS:
            capped_tabs.append(tab if tab is not None else "default")
            truncated = True
            continue

        if workbook is not None:
            df = workbook.get(tab)
            if df is None or df.empty:
                skipped.append(tab)
                continue
            tab_label = tab
        elif tab is None:
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

        columns, rows = dataframe_to_table(df)

        # Truncate THIS tab's rows if it alone pushed past the ceiling,
        # rather than embedding an unbounded number of rows on our own
        # OpenAI key. Any tabs still left in tabs_to_read are caught by the
        # check at the top of the next iteration — no `break` here, so they
        # end up recorded in capped_tabs instead of silently vanishing.
        room = MAX_SHEET_ROWS - row_count
        if len(rows) > room:
            print(f"sheet {sheet_id} truncated at {MAX_SHEET_ROWS} rows")
            rows = rows[:room]
            truncated = True
        row_count += len(rows)
        tables.append((tab_label, columns, rows))
        synced.append(tab_label)

    # Previously this returned successfully, so a private/deleted/unreachable
    # sheet was saved as a "connected" source with nothing behind it — the
    # single most likely real-world failure, and completely invisible.
    # Raising lets add_source's existing handler clean up the orphaned row
    # and show the user a real message (same guard the website branch uses).
    if not row_count:
        raise ValueError(
            "Couldn't read any data from that sheet. Check that it's shared "
            "as 'Anyone with the link can view', that it isn't empty, and "
            "that any tab names you entered match exactly."
        )

    return {
        "tables": tables,
        "synced_tabs": synced,
        "skipped_tabs": skipped,
        "capped_tabs": capped_tabs,
        "row_count": row_count,
        "truncated": truncated,
    }


def sync_sheet(sheet_id: str, range_name, project_id: str, source_id: str, qdrant, embeddings,
               collection: str, hidden_override: dict = None):
    """hidden_override: {tab: [columns]} the merchant chose in the preview;
    tabs not in it get the automatic personal-data defaults."""
    sheet = read_sheet(sheet_id, range_name)
    stored, embedded = index_tables(
        sheet["tables"], project_id, source_id, "gsheets",
        qdrant, embeddings, collection, hidden_override,
    )
    print(f"Synced {embedded} rows from tabs: {sheet['synced_tabs']}, skipped: {sheet['skipped_tabs']}, capped: {sheet['capped_tabs']}")
    return {
        "synced_tabs": sheet["synced_tabs"],
        "skipped_tabs": sheet["skipped_tabs"],
        "capped_tabs": sheet["capped_tabs"],
        "indexed_count": sheet["row_count"],
        "truncated": sheet["truncated"],
        "hidden_columns": sorted({c for t in stored for c in t["hidden"]}),
    }