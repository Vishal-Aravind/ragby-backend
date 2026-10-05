"""Condition node evaluation. Pure, no I/O.

content = {"rules": [{"id": "r_ab12", "match": "all"|"any",
                      "rows": [{"var": "budget", "op": "gt", "value": "50"}]}]}

Rules are checked top to bottom and the first match wins; nothing matching
means the "else" handle (which the editor always shows). A missing variable
counts as "", and a numeric comparison where either side isn't a number is
simply false — never an error that strands the visitor.
"""
from .templating import render_text, to_decimal

OPERATORS = {
    "equals", "not_equals", "contains", "not_contains", "starts_with",
    "gt", "gte", "lt", "lte", "is_empty", "not_empty",
}
MAX_RULES = 20
MAX_ROWS = 10


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).strip().lower()


def _row_true(row: dict, variables: dict) -> bool:
    op = row.get("op")
    if op not in OPERATORS:
        return False
    actual = variables.get(row.get("var") or "")
    expected = render_text(row.get("value") or "", variables, max_len=1000)

    if op == "is_empty":
        return _text(actual) == ""
    if op == "not_empty":
        return _text(actual) != ""
    if op in ("gt", "gte", "lt", "lte"):
        a, b = to_decimal(actual), to_decimal(expected)
        if a is None or b is None:
            return False
        return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]

    a, b = _text(actual), _text(expected)
    if op == "equals":
        num_a, num_b = to_decimal(actual), to_decimal(expected)
        if a != b and num_a is not None and num_b is not None and a and b:
            return num_a == num_b  # "50" equals "50.0"
        return a == b
    if op == "not_equals":
        return not _row_true({**row, "op": "equals"}, variables)
    if op == "contains":
        return b in a
    if op == "not_contains":
        return b not in a
    if op == "starts_with":
        return a.startswith(b)
    return False


def evaluate(rules, variables: dict) -> str:
    """Return the id of the first matching rule, or "else"."""
    for rule in (rules or [])[:MAX_RULES]:
        rows = [r for r in (rule.get("rows") or [])[:MAX_ROWS] if isinstance(r, dict)]
        if not rows or not rule.get("id"):
            continue
        results = [_row_true(r, variables or {}) for r in rows]
        matched = any(results) if rule.get("match") == "any" else all(results)
        if matched:
            return rule["id"]
    return "else"
