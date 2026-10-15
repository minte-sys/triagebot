"""Trust ledger: every human verdict per action type, and the mode each type has earned.

The decisions table is append-only and is the source of truth. action_modes only caches
the current mode, so every promotion and demotion can be explained from the history.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from triagebot.autonomy.proposals import ACTION_TYPES
from triagebot.config import TrustConfig

OUTCOMES = ("approved", "edited", "rejected", "undone")
SOURCES = ("review", "audit", "undo")


@dataclass
class TrustStatus:
    action_type: str
    mode: str
    decisions: int  # human decisions since the last mode change
    approved: int  # approved unchanged, within the window
    in_window: int
    audits: int
    audits_passed: int


class Ledger:
    def __init__(self, conn: sqlite3.Connection, repo: str, policy: TrustConfig) -> None:
        self.conn = conn
        self.repo = repo
        self.policy = policy

    def mode(self, action_type: str) -> str:
        return self._state(action_type)["mode"]

    def record(
        self, action_type: str, proposal_id: int | None, outcome: str, source: str, note: str = ""
    ) -> str | None:
        """Append one verdict and apply the trust rules. Returns a message if the mode changed."""
        if outcome not in OUTCOMES or source not in SOURCES:
            raise ValueError(f"Unknown outcome or source: {outcome!r}, {source!r}")
        self._append(action_type, proposal_id, outcome, source, note)

        mode = self.mode(action_type)
        if outcome == "undone" and mode == "auto":
            return self._change(action_type, "propose", "an action was undone")
        if mode == "propose":
            reason = self._promotion_reason(action_type)
            if reason:
                return self._change(action_type, "auto", reason)
        elif source == "audit":
            reason = self._demotion_reason(action_type)
            if reason:
                return self._change(action_type, "propose", reason)
        return None

    def demote(self, action_type: str, note: str = "demoted by hand") -> str | None:
        if self.mode(action_type) == "propose":
            return None
        return self._change(action_type, "propose", note)

    def take_audit_turn(self, action_type: str) -> bool:
        """Count one auto action. True means this one should be held for an audit."""
        count = self._state(action_type)["actions_since_audit"] + 1
        audit = count >= self.policy.audit_every
        with self.conn:
            self.conn.execute(
                "UPDATE action_modes SET actions_since_audit = ?"
                " WHERE repo = ? AND action_type = ?",
                (0 if audit else count, self.repo, action_type),
            )
        return audit

    def status(self) -> list[TrustStatus]:
        rows = []
        for action_type in ACTION_TYPES:
            reviews = self._outcomes(action_type, ("review", "undo"))
            window = reviews[: self.policy.window]
            audits = self._outcomes(action_type, ("audit",))[: self.policy.audit_window]
            rows.append(TrustStatus(
                action_type, self.mode(action_type), len(reviews),
                window.count("approved"), len(window),
                len(audits), audits.count("approved"),
            ))
        return rows

    def history(self, action_type: str, limit: int = 10) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM decisions WHERE repo = ? AND action_type = ? ORDER BY id DESC LIMIT ?",
            (self.repo, action_type, limit),
        ).fetchall()

    def _promotion_reason(self, action_type: str) -> str | None:
        p = self.policy
        reviews = self._outcomes(action_type, ("review", "undo"))
        window = reviews[: p.window]
        approved = window.count("approved")
        if len(reviews) >= p.min_decisions and approved * 100 >= p.min_approval_pct * len(window):
            return f"{approved} of the last {len(window)} approved unchanged"
        return None

    def _demotion_reason(self, action_type: str) -> str | None:
        p = self.policy
        audits = self._outcomes(action_type, ("audit",))[: p.audit_window]
        passed = audits.count("approved")
        if len(audits) >= p.min_audits and passed * 100 < p.min_audit_accuracy_pct * len(audits):
            return f"only {passed} of the last {len(audits)} audits passed"
        return None

    def _outcomes(self, action_type: str, sources: tuple[str, ...]) -> list[str]:
        """Human verdicts since the last mode change, newest first."""
        since = self._state(action_type)["since_decision_id"]
        marks = ", ".join("?" * len(sources))
        rows = self.conn.execute(
            f"SELECT outcome FROM decisions WHERE repo = ? AND action_type = ? AND id > ?"
            f" AND source IN ({marks}) ORDER BY id DESC",
            (self.repo, action_type, since, *sources),
        ).fetchall()
        return [r["outcome"] for r in rows]

    def _change(self, action_type: str, mode: str, reason: str) -> str:
        """Switch mode and restart the windows, so trust is re-earned from fresh decisions."""
        outcome = "promoted" if mode == "auto" else "demoted"
        last_id = self._append(action_type, None, outcome, "system", reason)
        with self.conn:
            self.conn.execute(
                "UPDATE action_modes SET mode = ?, since_decision_id = ?, actions_since_audit = 0"
                " WHERE repo = ? AND action_type = ?",
                (mode, last_id, self.repo, action_type),
            )
        return f"{action_type} {outcome} to {mode} mode: {reason}"

    def _append(
        self, action_type: str, proposal_id: int | None, outcome: str, source: str, note: str
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO decisions (repo, action_type, proposal_id, outcome, source, note,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.repo, action_type, proposal_id, outcome, source, note,
                 datetime.now(UTC).isoformat(timespec="seconds")),
            )
        return cur.lastrowid

    def _state(self, action_type: str) -> sqlite3.Row:
        if action_type not in ACTION_TYPES:
            raise ValueError(f"Unknown action type {action_type!r}")
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO action_modes (repo, action_type) VALUES (?, ?)",
                (self.repo, action_type),
            )
        return self.conn.execute(
            "SELECT * FROM action_modes WHERE repo = ? AND action_type = ?",
            (self.repo, action_type),
        ).fetchone()
