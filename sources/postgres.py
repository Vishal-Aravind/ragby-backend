# sources/postgres.py
# Handles both PostgreSQL and MySQL via SQLAlchemy

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
    """Raise ValueError if URL is not a supported database type."""
    if not any(db_url.startswith(p) for p in SUPPORTED_PREFIXES):
        raise ValueError(
            "Unsupported database. Supported: PostgreSQL (postgresql://) and MySQL (mysql:// or mysql+pymysql://)"
        )


def get_schema(db_url: str, allowed_schema: dict | None = None) -> str:
    """
    Introspect the database and return a schema string for the LLM.
    Filters to allowed_schema if provided.
    """
    db_url = normalize_url(db_url)
    engine = sqlalchemy.create_engine(db_url)
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
    engine = sqlalchemy.create_engine(db_url)
    try:
        insp = sqlalchemy.inspect(engine)
        schema = {}
        for table in insp.get_table_names():
            cols = insp.get_columns(table)
            schema[table] = [c["name"] for c in cols]
        return schema
    finally:
        engine.dispose()


def run_text_to_sql(
    question: str,
    db_url: str,
    openai_client,
    allowed_schema: dict | None = None
) -> str:
    db_url = normalize_url(db_url)
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

    # Hard safety: only SELECT allowed
    if not sql.upper().strip().startswith("SELECT"):
        return "Query blocked: only SELECT statements are allowed."

    engine = sqlalchemy.create_engine(db_url)
    try:
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(sql))
            rows = result.fetchmany(50)
            if not rows:
                return "Query returned no results."
            cols = list(result.keys())
            lines = [", ".join(cols)]
            for row in rows:
                lines.append(", ".join(str(v) for v in row))
            return "\n".join(lines)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return f"Query failed: {str(e)}"
    finally:
        engine.dispose()