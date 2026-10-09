"""Turn an agent decision into one proposal per action type, and store them."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from triagebot.agent.decision import Decision

ACTION_TYPES = ("labels", "comment", "duplicate", "request_info")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Proposal:
    id: int
    run_id: int
    repo: str
    issue_number: int
    action_type: str
    payload: dict[str, Any]
    rationale: str
    status: str
    final_payload: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Proposal:
        final = row["final_payload_json"]
        return cls(row["id"], row["run_id"], row["repo"], row["issue_number"], row["action_type"],
                   json.loads(row["payload_json"]), row["rationale"], row["status"],
                   json.loads(final) if final else None)


def payloads_from_decision(decision: Decision, current_labels: list[str]) -> list[tuple[str, dict]]:
    """Split a decision into (action_type, payload) pairs. Empty parts become nothing."""
    out: list[tuple[str, dict]] = []
    new_labels = [lb for lb in decision.labels if lb not in current_labels]
    if new_labels:
        out.append(("labels", {"labels": new_labels}))
    if decision.duplicate_of:
        out.append(("duplicate", {"duplicate_of": decision.duplicate_of}))
    if decision.missing_info:
        out.append(("request_info", {"questions": decision.missing_info}))
    if decision.comment.strip():
        out.append(("comment", {"body": decision.comment.strip()}))
    return out


class ProposalStore:
    def __init__(self, conn: sqlite3.Connection, repo: str) -> None:
        self.conn = conn
        self.repo = repo

    def create_from_decision(
        self, run_id: int, issue_number: int, decision: Decision, current_labels: list[str]
    ) -> list[Proposal]:
        """Store new proposals. Older pending proposals for the same issue are superseded."""
        with self.conn:
            self.conn.execute(
                "UPDATE proposals SET status = 'superseded', decided_at = ?"
                " WHERE repo = ? AND issue_number = ? AND status = 'pending'",
                (_now(), self.repo, issue_number),
            )
            ids = []
            for action_type, payload in payloads_from_decision(decision, current_labels):
                cur = self.conn.execute(
                    "INSERT INTO proposals (run_id, repo, issue_number, action_type, payload_json,"
                    " rationale, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, self.repo, issue_number, action_type, json.dumps(payload),
                     decision.reasoning, _now()),
                )
                ids.append(cur.lastrowid)
        return [self.get(i) for i in ids]

    def get(self, proposal_id: int) -> Proposal | None:
        row = self.conn.execute(
            "SELECT * FROM proposals WHERE id = ? AND repo = ?", (proposal_id, self.repo)
        ).fetchone()
        return Proposal.from_row(row) if row else None

    def pending(self) -> list[Proposal]:
        rows = self.conn.execute(
            "SELECT * FROM proposals WHERE repo = ? AND status = 'pending'"
            " ORDER BY issue_number, id",
            (self.repo,),
        ).fetchall()
        return [Proposal.from_row(r) for r in rows]

    def decide(
        self, proposal_id: int, status: str, final_payload: dict | None = None, reason: str = ""
    ) -> None:
        if status not in ("approved", "edited", "rejected", "failed"):
            raise ValueError(f"Unknown status {status!r}")
        with self.conn:
            self.conn.execute(
                "UPDATE proposals SET status = ?, final_payload_json = ?, reject_reason = ?,"
                " decided_at = ? WHERE id = ? AND status = 'pending'",
                (status, json.dumps(final_payload) if final_payload is not None else None,
                 reason, _now(), proposal_id),
            )
