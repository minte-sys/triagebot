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
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    tool_mode TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    stop_reason TEXT NOT NULL DEFAULT '',
    steps INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    decision_json TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS trace_events (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    args_json TEXT,
    result TEXT NOT NULL DEFAULT '',
    is_error INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    repo TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    final_payload_json TEXT,
    reject_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    audit INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS executed_actions (
    id INTEGER PRIMARY KEY,
    proposal_id INTEGER NOT NULL REFERENCES proposals(id),
    dry_run INTEGER NOT NULL,
    description TEXT NOT NULL,
    github_ref_json TEXT,
    executed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    action_type TEXT NOT NULL,
    proposal_id INTEGER REFERENCES proposals(id),
    outcome TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS action_modes (
    repo TEXT NOT NULL,
    action_type TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'propose',
    since_decision_id INTEGER NOT NULL DEFAULT 0,
    actions_since_audit INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (repo, action_type)
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    proposal_id INTEGER NOT NULL REFERENCES proposals(id),
    issue_number INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    proposed_json TEXT NOT NULL,
    corrected_json TEXT,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY,
    case_number INTEGER NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    tool_mode TEXT NOT NULL,
    run_id INTEGER REFERENCES runs(id),
    status TEXT NOT NULL,
    decision_json TEXT,
    steps INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    seconds REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (case_number, model, prompt_version, tool_mode)
);
CREATE TABLE IF NOT EXISTS embeddings (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vector TEXT NOT NULL,
    PRIMARY KEY (model, text_hash)
);
"""

# Columns added after a table was first created. CREATE TABLE IF NOT EXISTS leaves
# existing tables alone, so older databases get them here.
ADDED_COLUMNS = {
    "proposals": [("audit", "INTEGER NOT NULL DEFAULT 0")],
}


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
    _add_missing_columns(conn)
    return conn


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
    conn.commit()
