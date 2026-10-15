"""Maintainer corrections (edits, rejections, undos), fed back into future prompts."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from triagebot.autonomy.proposals import Proposal

MAX_IN_PROMPT = 5
MAX_PAYLOAD_CHARS = 300


class CorrectionStore:
    def __init__(self, conn: sqlite3.Connection, repo: str) -> None:
        self.conn = conn
        self.repo = repo

    def add(self, proposal: Proposal, corrected: dict[str, Any] | None, reason: str = "") -> None:
        """corrected is the maintainer's version, or None if they rejected or undid it."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO corrections (repo, proposal_id, issue_number, action_type,"
                " proposed_json, corrected_json, reason, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self.repo, proposal.id, proposal.issue_number, proposal.action_type,
                 json.dumps(proposal.payload),
                 json.dumps(corrected) if corrected is not None else None,
                 reason, datetime.now(UTC).isoformat(timespec="seconds")),
            )

    def recent(self, limit: int = MAX_IN_PROMPT) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM corrections WHERE repo = ? ORDER BY id DESC LIMIT ?",
            (self.repo, limit),
        ).fetchall()


def _short(payload_json: str) -> str:
    text = json.dumps(json.loads(payload_json), ensure_ascii=False)
    return text if len(text) <= MAX_PAYLOAD_CHARS else text[:MAX_PAYLOAD_CHARS] + "..."


def format_corrections(rows: list[sqlite3.Row]) -> str:
    """One line per correction, oldest first. Empty string if there are none."""
    lines = []
    for row in reversed(rows):
        if row["corrected_json"]:
            verdict = f"the maintainer changed it to {_short(row['corrected_json'])}"
        else:
            verdict = "the maintainer rejected it"
        reason = f" Reason: {row['reason']}" if row["reason"] else ""
        lines.append(f"- #{row['issue_number']}, {row['action_type']}: you proposed "
                     f"{_short(row['proposed_json'])}; {verdict}.{reason}")
    return "\n".join(lines)
