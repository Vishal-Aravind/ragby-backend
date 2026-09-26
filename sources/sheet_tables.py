"""Queryable copies of spreadsheet sources (Google Sheets, Excel).

Sheet rows used to exist only as embedded text in Qdrant — fine for fuzzy
questions, useless for "items under 200", "cheapest", or "how many", and
local Excel uploads weren't stored anywhere at all. Each sync now also saves
the parsed table (one row per tab) to `source_tables`, which the chat's
table-query path (sources/table_query.py) filters, sorts and counts.

Privacy: anyone chatting with the bot can query these tables, so columns that
look like personal data (emails, phone numbers, IDs) start HIDDEN. Hidden
columns are excluded from both the table query and the embedded row text;
the merchant can expose them from the dashboard. A merchant's choice for an
existing column survives every re-sync.
"""
import re

import sentry_sdk

from clients import supabase
from vector_sync import replace_points

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_CHARS_RE = re.compile(r"^[\d\s+()\-.]{8,20}$")
_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")
_PERSONAL_NAME_RE = re.compile(
    r"e-?mail|phone|mobile|\bcell\b|whats\s*app|contact\s*(no|num)|\btel(ephone)?\b|"
    r"\bdob\b|date of birth|birth\s*date|aadh?aa?r|\bpan\b|passport|\bssn\b",
    re.I,
)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if text.lower() in ("nan", "none", "nat"):
        return ""
    # Google Sheets exports text-forced cells (e.g. phone numbers) with a
    # leading apostrophe.
    if text.startswith("'"):
        text = text[1:]
    return text


def dataframe_to_table(df) -> tuple:
    """(columns, rows) with clean unique column names and string cells.
    Blank/unnamed headers become "Column N" so they can still be referenced;
    fully empty rows are dropped."""
    columns, seen = [], set()
    for i, raw in enumerate(df.columns):
        name = str(raw).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"Column {i + 1}"
        base, n = name, 2
        while name in seen:
            name = f"{base} ({n})"
            n += 1
        seen.add(name)
        columns.append(name)

    rows = []
    for values in df.itertuples(index=False, name=None):
        cells = [_cell(v) for v in values]
        if any(cells):
            rows.append(cells)
    return columns, rows


def _looks_like_phone(value: str) -> bool:
    # 10+ digits, not 8: an 8-9 digit value is far more often a price
    # (e.g. a property at 12000000) than a phone number, and a phone column
    # is usually caught by its name anyway.
    if not _PHONE_CHARS_RE.match(value) or _DATE_RE.match(value) or re.search(r"\.\d{1,2}$", value):
        return False
    return 10 <= len(re.sub(r"\D", "", value)) <= 15


def detect_personal_columns(columns: list, rows: list) -> set:
    """Columns whose name or values look like personal contact data."""
    hidden = set()
    for i, col in enumerate(columns):
        if _PERSONAL_NAME_RE.search(col):
            hidden.add(col)
            continue
        values = [r[i] for r in rows[:200] if i < len(r) and r[i]]
        if not values:
            continue
        emails = sum(1 for v in values if _EMAIL_RE.match(v))
        phones = sum(1 for v in values if _looks_like_phone(v))
        if emails >= len(values) / 2 or phones >= len(values) / 2:
            hidden.add(col)
    return hidden


def load_previous(source_id: str) -> dict:
    """{tab: {"columns": [...], "hidden": [...]}} from the last sync, so a
    merchant's column choices survive a re-sync. Empty if none (or if the
    table doesn't exist yet)."""
    try:
        res = supabase.table("source_tables") \
            .select("tab, columns, hidden_columns") \
            .eq("source_id", source_id) \
            .execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {}
    return {
        r["tab"]: {"columns": r.get("columns") or [], "hidden": r.get("hidden_columns") or []}
        for r in (res.data or [])
    }


def resolve_hidden(previous: dict, tab: str, columns: list, rows: list) -> list:
    """Keep the merchant's earlier choice for known columns; auto-hide only
    NEW columns that look personal."""
    detected = detect_personal_columns(columns, rows)
    prev = previous.get(tab)
    if prev is None:
        return [c for c in columns if c in detected]
    known, prev_hidden = set(prev["columns"]), set(prev["hidden"])
    return [c for c in columns if c in prev_hidden or (c not in known and c in detected)]


def row_text(columns: list, row: list, hidden: set) -> str:
    return ", ".join(
        f"{col}: {val}" for col, val in zip(columns, row) if val and col not in hidden
    )


def save_tables(source_id: str, project_id: str, tables: list):
    """Replace this source's stored tables. Non-fatal: if it fails (e.g. the
    migration hasn't run yet) the sync still succeeds and chat falls back to
    vector search for this source."""
    from sources.table_query import invalidate

    try:
        supabase.table("source_tables").delete().eq("source_id", source_id).execute()
        if tables:
            supabase.table("source_tables").insert([
                {
                    "source_id": source_id,
                    "project_id": project_id,
                    "tab": t["tab"],
                    "columns": t["columns"],
                    "rows": t["rows"],
                    "hidden_columns": t["hidden"],
                }
                for t in tables
            ]).execute()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"save_tables failed for source {source_id}: {e}")
    finally:
        invalidate(project_id)


def reembed_from_tables(source_id: str, project_id: str, source_type: str,
                        qdrant, embeddings, collection: str) -> int:
    """Rebuild this source's vectors from its stored tables — used when the
    merchant changes which columns are visible, so hidden columns leave the
    vector index too. Works for local Excel uploads, whose file isn't kept."""
    res = supabase.table("source_tables") \
        .select("tab, columns, rows, hidden_columns") \
        .eq("source_id", source_id) \
        .execute()
    chunks, metas = [], []
    for t in res.data or []:
        hidden = set(t.get("hidden_columns") or [])
        for row in t.get("rows") or []:
            text = row_text(t["columns"], row, hidden)
            if not text:
                continue
            chunks.append(text)
            metas.append({
                "project_id": project_id,
                "source_id": source_id,
                "source_type": source_type,
                "sheet_tab": t["tab"],
                "text": text,
            })
    replace_points(qdrant, embeddings, collection, chunks, metas, "source_id", source_id)
    return len(chunks)
