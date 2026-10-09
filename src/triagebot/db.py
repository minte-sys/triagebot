"""SQLite connection and schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS issues_fts USING fts5(
    title, body, repo UNINDEXED, number UNINDEXED, state UNINDEXED, labels UNINDEXED,
    tokenize = 'porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    heading, text, repo UNINDEXED, path UNINDEXED,
    tokenize = 'porter unicode61'
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database, creating the file and tables if needed."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
    except sqlite3.OperationalError as exc:
        if "fts5" in str(exc):
            raise RuntimeError(
                f"This Python's SQLite ({sqlite3.sqlite_version}) was built without FTS5"
            ) from exc
        raise
    return conn
