"""Reverse exactly what the executor wrote for one proposal."""

from __future__ import annotations

import json
from dataclasses import dataclass

from triagebot.autonomy.corrections import CorrectionStore
from triagebot.autonomy.ledger import Ledger
from triagebot.autonomy.proposals import Proposal, ProposalStore
from triagebot.github.client import GitHubClient, GitHubError


class UndoError(Exception):
    """There is nothing to undo, or it was undone already."""


@dataclass
class UndoResult:
    dry_run: bool
    steps: list[str]
    mode_change: str | None = None


def undo(
    proposal: Proposal,
    gh: GitHubClient,
    store: ProposalStore,
    ledger: Ledger,
    corrections: CorrectionStore,
    dry_run: bool = True,
    reason: str = "",
) -> UndoResult:
    if proposal.status == "undone":
        raise UndoError(f"Proposal {proposal.id} was already undone")
    row = store.conn.execute(
        "SELECT github_ref_json FROM executed_actions WHERE proposal_id = ? AND dry_run = 0",
        (proposal.id,),
    ).fetchone()
    if row is None:
        raise UndoError(f"Proposal {proposal.id} was never written to GitHub, so there is "
                        "nothing to undo")

    ref = json.loads(row["github_ref_json"] or "{}")
    number = proposal.issue_number
    steps = [f"remove label {lb!r} from #{number}" for lb in ref.get("labels_added", [])]
    if ref.get("comment_id"):
        steps.append(f"delete comment {ref['comment_id']} on #{number}")
    if dry_run:
        return UndoResult(True, steps)

    for label in ref.get("labels_added", []):
        _ignore_missing(lambda lb=label: gh.remove_label(store.repo, number, lb))
    if ref.get("comment_id"):
        _ignore_missing(lambda: gh.delete_comment(store.repo, ref["comment_id"]))

    store.mark_undone(proposal.id)
    corrections.add(proposal, None, reason or "undone by a maintainer")
    change = ledger.record(proposal.action_type, proposal.id, "undone", "undo", reason)
    return UndoResult(False, steps, change)


def _ignore_missing(call) -> None:
    """A label or comment someone already removed by hand counts as undone."""
    try:
        call()
    except GitHubError as exc:
        if exc.status_code != 404:
            raise
