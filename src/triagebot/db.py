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
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS executed_actions (
    id INTEGER PRIMARY KEY,
    proposal_id INTEGER NOT NULL REFERENCES proposals(id),
    dry_run INTEGER NOT NULL,
    description TEXT NOT NULL,
    github_ref_json TEXT,
    executed_at TEXT NOT NULL
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
