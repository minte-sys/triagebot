"""The only code that writes to GitHub. It re-checks every proposal before acting."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from triagebot.autonomy.proposals import Proposal
from triagebot.github.client import GitHubClient, GitHubError

MAX_COMMENT_CHARS = 2000
MAX_LABELS = 5


class ExecutionRefused(Exception):
    """The proposal broke a safety rule, so nothing was written."""


@dataclass
class ExecutionResult:
    dry_run: bool
    description: str
    github_ref: dict[str, Any] = field(default_factory=dict)


def marker(proposal_id: int) -> str:
    """Hidden tag in every comment, so the same proposal is never posted twice."""
    return f"<!-- triagebot:proposal:{proposal_id} -->"


class Executor:
    def __init__(
        self,
        gh: GitHubClient,
        conn: sqlite3.Connection,
        repo: str,
        repo_labels: set[str],
        dry_run: bool = True,
    ) -> None:
        self.gh = gh
        self.conn = conn
        self.repo = repo
        self.repo_labels = repo_labels
        self.dry_run = dry_run

    def execute(self, proposal: Proposal, payload: dict[str, Any]) -> ExecutionResult:
        """Validate and run one approved proposal. payload is the final (possibly edited) one."""
        self._check(proposal, payload)
        handler = getattr(self, f"_do_{proposal.action_type}")
        result = handler(proposal, payload)
        with self.conn:
            self.conn.execute(
                "INSERT INTO executed_actions (proposal_id, dry_run, description, github_ref_json,"
                " executed_at) VALUES (?, ?, ?, ?, ?)",
                (proposal.id, int(result.dry_run), result.description,
                 json.dumps(result.github_ref), datetime.now(UTC).isoformat(timespec="seconds")),
            )
        return result

    def _check(self, proposal: Proposal, payload: dict[str, Any]) -> None:
        if proposal.repo != self.repo:
            raise ExecutionRefused(f"Proposal is for {proposal.repo}, not {self.repo}")
        done = self.conn.execute(
            "SELECT 1 FROM executed_actions WHERE proposal_id = ? AND dry_run = 0", (proposal.id,)
        ).fetchone()
        if done:
            raise ExecutionRefused(f"Proposal {proposal.id} was already executed")

        kind = proposal.action_type
        if kind == "labels":
            labels = payload.get("labels") or []
            if not labels or len(labels) > MAX_LABELS:
                raise ExecutionRefused(f"Expected 1 to {MAX_LABELS} labels, got {len(labels)}")
            unknown = [lb for lb in labels if lb not in self.repo_labels]
            if unknown:
                raise ExecutionRefused(f"Labels do not exist in the repository: {unknown}")
        elif kind == "duplicate":
            target = payload.get("duplicate_of")
            if not isinstance(target, int) or target < 1 or target == proposal.issue_number:
                raise ExecutionRefused(f"Invalid duplicate target: {target!r}")
            try:
                original = self.gh.get_issue(self.repo, target, max_comments=0)
            except GitHubError as exc:
                raise ExecutionRefused(f"Duplicate target #{target} not found: {exc}") from exc
            if original.is_pull_request:
                raise ExecutionRefused(f"#{target} is a pull request, not an issue")
        elif kind in ("comment", "request_info"):
            body = self._comment_body(proposal, payload)
            if not body.strip() or len(body) > MAX_COMMENT_CHARS:
                raise ExecutionRefused(f"Comment must be 1 to {MAX_COMMENT_CHARS} characters")
        else:
            raise ExecutionRefused(f"Unknown action type {kind!r}")

    def _comment_body(self, proposal: Proposal, payload: dict[str, Any]) -> str:
        if proposal.action_type == "request_info":
            questions = "\n".join(f"- {q}" for q in payload.get("questions", []) if q.strip())
            if not questions:
                return ""
            intro = "Thanks for the report! To look into this we need a bit more information:"
            return f"{intro}\n\n{questions}"
        return str(payload.get("body", "")).strip()

    def _post_once(self, proposal: Proposal, body: str) -> int:
        """Post a comment unless one with this proposal's marker already exists."""
        tag = marker(proposal.id)
        for existing in self.gh.list_comments(self.repo, proposal.issue_number):
            if tag in existing.body:
                return existing.id
        return self.gh.create_comment(self.repo, proposal.issue_number, f"{body}\n\n{tag}").id

    def _do_labels(self, proposal: Proposal, payload: dict[str, Any]) -> ExecutionResult:
        labels = payload["labels"]
        description = f"add labels {labels} to #{proposal.issue_number}"
        if self.dry_run:
            return ExecutionResult(True, description)
        self.gh.add_labels(self.repo, proposal.issue_number, labels)
        return ExecutionResult(False, description, {"labels_added": labels})

    def _do_duplicate(self, proposal: Proposal, payload: dict[str, Any]) -> ExecutionResult:
        target = payload["duplicate_of"]
        label = ["duplicate"] if "duplicate" in self.repo_labels else []
        description = (f"mark #{proposal.issue_number} as a duplicate of #{target}"
                       + (" and add the duplicate label" if label else ""))
        if self.dry_run:
            return ExecutionResult(True, description)
        comment_id = self._post_once(proposal, f"Duplicate of #{target}")
        if label:
            self.gh.add_labels(self.repo, proposal.issue_number, label)
        return ExecutionResult(False, description,
                               {"comment_id": comment_id, "labels_added": label})

    def _do_comment(self, proposal: Proposal, payload: dict[str, Any]) -> ExecutionResult:
        body = self._comment_body(proposal, payload)
        description = f"comment on #{proposal.issue_number}: {body[:80]}"
        if self.dry_run:
            return ExecutionResult(True, description)
        return ExecutionResult(False, description, {"comment_id": self._post_once(proposal, body)})

    _do_request_info = _do_comment
