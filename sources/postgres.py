# sources/postgres.py
# Handles both PostgreSQL and MySQL via SQLAlchemy

import ipaddress
import re
import socket
from urllib.parse import urlsplit

import sentry_sdk
import sqlalchemy
from functools import lru_cache


SUPPORTED_PREFIXES = (
    "postgresql://",
    "postgres://",
    "mysql://",
    "mysql+pymysql://",
)


def normalize_url(db_url: str) -> str:
    """
    Normalize URL so SQLAlchemy uses the right driver.
    - postgres:// → postgresql://  (SQLAlchemy 2.x dropped bare postgres://)
    - mysql://    → mysql+pymysql:// (needs pymysql driver)
    """
    if db_url.startswith("postgres://"):
        return db_url.replace("postgres://", "postgresql://", 1)
    if db_url.startswith("mysql://"):
        return db_url.replace("mysql://", "mysql+pymysql://", 1)
    return db_url


def validate_url(db_url: str):
    """Raise ValueError if URL is not a supported database type, or if it
    resolves to a private/internal/loopback address.

    FIX: this previously only checked the scheme prefix — any authenticated
    user could point this at an internal host (localhost, RFC1918 ranges,
    cloud metadata endpoints, etc.) and get the full table/column schema of
    whatever database lives there. This is a practical mitigation (resolves
    the hostname once and checks the IP), not a complete defense against
    DNS-rebinding-style TOCTOU attacks — reasonable given the trust level
    here (any signed-up user can already point this at a real internet
    database), not a fully isolated execution context."""
    if not any(db_url.startswith(p) for p in SUPPORTED_PREFIXES):
        raise ValueError(
            "Unsupported database. Supported: PostgreSQL (postgresql://) and MySQL (mysql:// or mysql+pymysql://)"
        )

    host = urlsplit(db_url).hostname
    if not host:
        raise ValueError("Could not parse a host from this connection string.")

    try:
        addrinfo = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ValueError("Could not resolve this database host.")

    for family, _, _, _, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("This host is not reachable — internal/private network addresses aren't allowed.")


def _connect_args(db_url: str) -> dict:
    """Per-dialect connect-time timeout so a slow/unreachable customer DB
    can't hang the request indefinitely."""
    if "mysql" in db_url:
        return {"connect_timeout": 5, "read_timeout": 10}
    return {"connect_timeout": 5, "options": "-c statement_timeout=10000"}


def get_schema(db_url: str, allowed_schema: dict | None = None) -> str:
    """
    Introspect the database and return a schema string for the LLM.
    Filters to allowed_schema if provided.
    """
    db_url = normalize_url(db_url)
    validate_url(db_url)
    engine = sqlalchemy.create_engine(db_url, connect_args=_connect_args(db_url))
    try:
        insp = sqlalchemy.inspect(engine)
        lines = []
        for table in insp.get_table_names():
            if allowed_schema and table not in allowed_schema:
                continue
            cols = insp.get_columns(table)
            allowed_cols = allowed_schema.get(table) if allowed_schema else None
            col_str = ", ".join(
                f"{c['name']} {c['type']}"
                for c in cols
                if not allowed_cols or c["name"] in allowed_cols
            )
            lines.append(f"Table {table}: ({col_str})")
        return "\n".join(lines)
    finally:
        engine.dispose()


def introspect_schema(db_url: str) -> dict:
    """
    Returns full schema as { table: [col, ...] } for frontend checkbox picker.
    """
    db_url = normalize_url(db_url)
    validate_url(db_url)
    engine = sqlalchemy.create_engine(db_url, connect_args=_connect_args(db_url))
    try:
        insp = sqlalchemy.inspect(engine)
        schema = {}
        for table in insp.get_table_names():
            cols = insp.get_columns(table)
            schema[table] = [c["name"] for c in cols]
        return schema
    finally:
        engine.dispose()


def _only_allowed_tables(sql: str, allowed_schema: dict) -> bool:
    """Conservative check that every table named after FROM/JOIN is one of
    the allowed tables. Not a real SQL parser — just guards against the LLM
    (via prompt injection or drift) reaching past the tables it was shown,
    since the prompt instruction alone isn't a hard boundary."""
    referenced = re.findall(r'(?:FROM|JOIN)\s+["`]?([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
    allowed_lower = {t.lower() for t in allowed_schema.keys()}
    return all(t.lower() in allowed_lower for t in referenced)


def run_text_to_sql(
    question: str,
    db_url: str,
    openai_client,
    allowed_schema: dict | None = None
) -> str:
    db_url = normalize_url(db_url)
    validate_url(db_url)
    schema = get_schema(db_url, allowed_schema)

    # Tell LLM which dialect to use
    dialect = "MySQL" if "mysql" in db_url else "PostgreSQL"

    sql_prompt = f"""You are a {dialect} SQL expert. Given this schema:
{schema}

STRICT RULES:
- NEVER use SELECT * — always list column names explicitly
- Only use columns that appear in the schema above
- Only write SELECT queries, never INSERT/UPDATE/DELETE

Write a single safe read-only SELECT query to answer: "{question}"
Use {dialect} syntax only.
Return ONLY the SQL query, nothing else."""

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0,
        max_tokens=300,
    )
    sql = resp.choices[0].message.content.strip()

    # Strip markdown code fences if LLM wraps output
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.startswith("sql") or sql.startswith("mysql") or sql.startswith("postgresql"):
            sql = sql.split("\n", 1)[1]
        sql = sql.strip()

    # Hard safety: only a single SELECT allowed, no stacked statements.
    stripped = sql.strip().rstrip(";")
    if not stripped.upper().startswith("SELECT"):
        return "Query blocked: only SELECT statements are allowed."
    if ";" in stripped:
        return "Query blocked: multiple statements are not allowed."

    # allowed_schema is what the picker UI/customer actually consented to
    # exposing — enforce it here too, not just via the prompt instruction,
    # since the LLM's output isn't a trusted boundary on its own.
    if allowed_schema and not _only_allowed_tables(stripped, allowed_schema):
        return "Query blocked: references a table outside the allowed schema."

    if "LIMIT" not in stripped.upper():
        stripped = f"{stripped} LIMIT 200"

    engine = sqlalchemy.create_engine(db_url, connect_args=_connect_args(db_url))
    try:
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(stripped))
            rows = result.fetchmany(200)
            if not rows:
                return "Query returned no results."
            cols = list(result.keys())
            lines = [", ".join(cols)]
            for row in rows:
                lines.append(", ".join(str(v) for v in row))
            return "\n".join(lines)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"run_text_to_sql query failed: {e}")
        return "I couldn't get that information from the database right now."
    finally:
        engine.dispose()