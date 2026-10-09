"""Write runs and their step-by-step events to SQLite."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Recorder:
    """Every event is committed immediately, so the trace survives a crash mid-run."""

    def __init__(
        self, conn: sqlite3.Connection, echo: Callable[[str], None] | None = None
    ) -> None:
        self.conn = conn
        self.echo = echo
        self._seq: dict[int, int] = {}

    def start_run(
        self, repo: str, issue_number: int, model: str, prompt_version: str, tool_mode: str
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO runs (repo, issue_number, model, prompt_version, tool_mode, started_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (repo, issue_number, model, prompt_version, tool_mode, _now()),
            )
        run_id = cur.lastrowid
        self._seq[run_id] = 0
        return run_id

    def event(
        self,
        run_id: int,
        kind: str,
        name: str,
        args: Any = None,
        result: str = "",
        is_error: bool = False,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
    ) -> None:
        self._seq[run_id] = self._seq.get(run_id, 0) + 1
        args_json = None if args is None else json.dumps(args, ensure_ascii=False)
        with self.conn:
            self.conn.execute(
                "INSERT INTO trace_events (run_id, seq, kind, name, args_json, result, is_error,"
                " tokens_in, tokens_out, latency_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, self._seq[run_id], kind, name, args_json, result, int(is_error),
                 tokens_in, tokens_out, latency_ms, _now()),
            )
        if self.echo:
            self.echo(format_event(self._seq[run_id], kind, name, args_json, result, is_error,
                                   tokens_in, tokens_out, latency_ms))

    def end_run(
        self,
        run_id: int,
        status: str,
        stop_reason: str,
        steps: int,
        tokens_in: int,
        tokens_out: int,
        decision: dict | None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status = ?, stop_reason = ?, steps = ?, tokens_in = ?,"
                " tokens_out = ?, decision_json = ?, ended_at = ? WHERE id = ?",
                (status, stop_reason, steps, tokens_in, tokens_out,
                 None if decision is None else json.dumps(decision, ensure_ascii=False),
                 _now(), run_id),
            )


def _short(text: str | None, limit: int = 160) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def format_event(
    seq: int,
    kind: str,
    name: str,
    args_json: str | None,
    result: str,
    is_error: bool,
    tokens_in: int | None,
    tokens_out: int | None,
    latency_ms: int | None,
) -> str:
    if kind == "llm":
        head = f"{seq:>3} llm   {latency_ms} ms, tokens in/out {tokens_in}/{tokens_out}"
        return f"{head}\n      {_short(result)}"
    marker = "ERROR " if is_error else ""
    return f"{seq:>3} {kind:<5} {name}({_short(args_json, 120)})\n      {marker}{_short(result)}"
