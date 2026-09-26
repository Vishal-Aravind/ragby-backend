# sources/gsheets.py

import re
from urllib.parse import quote

import sentry_sdk
import pandas as pd

from config import MAX_SHEET_ROWS
from vector_sync import replace_points
from sources.sheet_tables import (
    dataframe_to_table, load_previous, resolve_hidden, row_text, save_tables,
)

# FIX: removed unused RecursiveCharacterTextSplitter import


def get_sheet_names(sheet_id: str, range_name):
    if not range_name:
        return None, []
    # The frontend now sends a real list of tab names (a chip/tag input,
    # not a single comma-separated text field) — a tab literally named
    # "Q1, Actuals" could never be selected correctly through a delimiter
    # that's also a valid character in the thing being delimited. Splitting
    # on "," is kept only so a source connected before this change (whose
    # stored config still has the old comma-joined string) keeps working
    # without a migration.
    if isinstance(range_name, list):
        requested = [str(r).strip() for r in range_name if str(r).strip()]
        return requested, []
    if range_name.strip().lower() in ("", "all"):
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

    previous = load_previous(source_id)
    all_chunks = []
    all_metas = []
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

        # Personal-data columns (emails, phones) start hidden: they're left
        # out of the embedded text as well as the table query.
        hidden = resolve_hidden(previous, tab_label, columns, rows)
        hidden_set = set(hidden)
        tables.append({"tab": tab_label, "columns": columns, "rows": rows, "hidden": hidden})

        for row in rows:
            text = row_text(columns, row, hidden_set)
            if not text:
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
    if not row_count:
        raise ValueError(
            "Couldn't read any data from that sheet. Check that it's shared "
            "as 'Anyone with the link can view', that it isn't empty, and "
            "that any tab names you entered match exactly."
        )

    replace_points(qdrant, embeddings, collection, all_chunks, all_metas, "source_id", source_id)
    # Only after the vectors were replaced, so a failed sync leaves the
    # previous table and index consistent with each other.
    save_tables(source_id, project_id, tables)

    print(f"Synced {len(all_chunks)} rows from tabs: {synced}, skipped: {skipped}, capped: {capped_tabs}")
    return {
        "synced_tabs": synced,
        "skipped_tabs": skipped,
        "capped_tabs": capped_tabs,
        "indexed_count": row_count,
        "truncated": truncated,
        "hidden_columns": sorted({c for t in tables for c in t["hidden"]}),
    }