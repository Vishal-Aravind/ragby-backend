"""A tiny in-memory stand-in for the supabase-py client, enough for the
shop tests: table().select/insert/update/delete/upsert with eq, neq, in_,
is_, not_.is_, contains, or_, order, limit, maybe_single, single; rpc()
dispatches to registered Python callables. No network, no database.
"""
import copy
import re
import uuid
from datetime import datetime, timezone


class Result:
    def __init__(self, data):
        self.data = data


def _lit(v):
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "null":
        return None
    return v


def _split_top(expr: str):
    parts, depth, cur = [], 0, ""
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)
    return parts


def _term(row, term: str) -> bool:
    term = term.strip()
    if term.startswith("and(") and term.endswith(")"):
        return all(_term(row, t) for t in _split_top(term[4:-1]))
    m = re.match(r"^([a-z_]+)\.not\.is\.(\w+)$", term)
    if m:
        return row.get(m.group(1)) is not _lit(m.group(2)) if _lit(m.group(2)) is None else row.get(m.group(1)) != _lit(m.group(2))
    m = re.match(r"^([a-z_]+)\.(eq|neq|is)\.(.+)$", term)
    if not m:
        raise ValueError(f"fake or_ can't parse {term!r}")
    col, op, val = m.group(1), m.group(2), _lit(m.group(3))
    if op == "eq":
        return row.get(col) == val
    if op == "neq":
        return row.get(col) != val
    return row.get(col) is val


class Query:
    def __init__(self, db, table):
        self.db, self.table = db, table
        self.op, self.payload = "select", None
        self.filters = []
        self._limit = None
        self._single = None
        self._negate_next = False
        self.on_conflict = None

    # ── ops ──
    def select(self, *_a, **_k):
        self.op = "select"
        return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.op, self.payload = "update", payload
        return self

    def upsert(self, payload, on_conflict=None):
        self.op, self.payload, self.on_conflict = "upsert", payload, on_conflict
        return self

    def delete(self):
        self.op = "delete"
        return self

    # ── filters ──
    @property
    def not_(self):
        self._negate_next = True
        return self

    def _add(self, f):
        if self._negate_next:
            self._negate_next = False
            inner = f
            f = lambda r: not inner(r)     # noqa: E731
        self.filters.append(f)
        return self

    def eq(self, col, val):
        return self._add(lambda r: r.get(col) == val)

    def neq(self, col, val):
        return self._add(lambda r: r.get(col) != val)

    def in_(self, col, vals):
        vals = list(vals)
        return self._add(lambda r: r.get(col) in vals)

    def is_(self, col, val):
        target = _lit(val) if isinstance(val, str) else val
        return self._add(lambda r: r.get(col) is target)

    def contains(self, col, vals):
        return self._add(lambda r: all(v in (r.get(col) or []) for v in vals))

    def or_(self, expr):
        terms = _split_top(expr)
        return self._add(lambda r: any(_term(r, t) for t in terms))

    def order(self, *_a, **_k):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def maybe_single(self):
        self._single = "maybe"
        return self

    def single(self):
        self._single = "single"
        return self

    # ── run ──
    def _match(self, rows):
        return [r for r in rows if all(f(r) for f in self.filters)]

    def execute(self):
        db = self.db
        db.calls.append((self.table, self.op, copy.deepcopy(self.payload)))
        rows = db.tables.setdefault(self.table, [])
        if self.op == "select":
            out = [copy.deepcopy(r) for r in self._match(rows)]
            if self._limit is not None:
                out = out[: self._limit]
            if self._single:
                if not out:
                    return None if self._single == "maybe" else Result(None)
                return Result(out[0])
            return Result(out)
        if self.op == "insert":
            items = self.payload if isinstance(self.payload, list) else [self.payload]
            created = []
            for item in items:
                row = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
                       **db.defaults.get(self.table, {}), **copy.deepcopy(item)}
                for cols in db.unique.get(self.table, []):
                    if all(row.get(c) is not None for c in cols) and any(
                            all(r.get(c) == row.get(c) for c in cols) for r in rows):
                        raise Exception('{"code":"23505","message":"duplicate key value violates unique constraint"}')
                rows.append(row)
                created.append(copy.deepcopy(row))
            return Result(created)
        if self.op == "update":
            hit = self._match(rows)
            for r in hit:
                r.update(copy.deepcopy(self.payload))
            return Result([copy.deepcopy(r) for r in hit])
        if self.op == "delete":
            hit = self._match(rows)
            for r in hit:
                rows.remove(r)
            return Result([copy.deepcopy(r) for r in hit])
        if self.op == "upsert":
            keys = (self.on_conflict or "id").split(",")
            row = copy.deepcopy(self.payload)
            for r in rows:
                if all(r.get(k) == row.get(k) for k in keys):
                    r.update(row)
                    return Result([copy.deepcopy(r)])
            rows.append(row)
            return Result([copy.deepcopy(row)])
        raise AssertionError(self.op)


class RpcCall:
    def __init__(self, db, name, params):
        self.db, self.name, self.params = db, name, params

    def execute(self):
        self.db.rpc_calls.append((self.name, copy.deepcopy(self.params)))
        fn = self.db.rpcs.get(self.name)
        if fn is None:
            raise Exception(f"rpc {self.name} not registered")
        return Result(fn(self.params))


class FakeSupabase:
    def __init__(self, tables=None, rpcs=None, unique=None, defaults=None):
        self.tables = tables or {}
        self.rpcs = rpcs or {}
        self.unique = unique or {}
        self.defaults = defaults or {}
        self.calls = []
        self.rpc_calls = []

    def table(self, name):
        return Query(self, name)

    def rpc(self, name, params):
        return RpcCall(self, name, params)
