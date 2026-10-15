"""Decide for each new proposal: wait for review, run it now, or hold it for an audit."""

from __future__ import annotations

from dataclasses import dataclass

from triagebot.autonomy.executor import ExecutionRefused, Executor
from triagebot.autonomy.ledger import Ledger
from triagebot.autonomy.proposals import Proposal, ProposalStore
from triagebot.github.client import GitHubError


@dataclass
class Routed:
    proposal: Proposal
    route: str  # "review", "audit", "auto" or "failed"
    message: str


class Gate:
    def __init__(
        self, store: ProposalStore, ledger: Ledger, executor: Executor, auto_enabled: bool
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.executor = executor
        self.auto_enabled = auto_enabled

    def route(self, proposals: list[Proposal]) -> list[Routed]:
        return [self._route_one(p) for p in proposals]

    def _route_one(self, proposal: Proposal) -> Routed:
        if not self.auto_enabled or self.ledger.mode(proposal.action_type) != "auto":
            return Routed(proposal, "review", "waiting for review")
        if self.ledger.take_audit_turn(proposal.action_type):
            self.store.hold_for_audit(proposal.id)
            return Routed(proposal, "audit", "held for an audit review")
        try:
            result = self.executor.execute(proposal, proposal.payload)
        except ExecutionRefused as exc:
            self.store.decide(proposal.id, "failed", reason=str(exc))
            return Routed(proposal, "failed", f"refused: {exc}")
        except GitHubError as exc:
            return Routed(proposal, "review", f"GitHub error, waiting for review: {exc}")
        self.store.decide(proposal.id, "auto")
        verb = "would" if result.dry_run else "done:"
        return Routed(proposal, "auto", f"auto, {verb} {result.description}")
