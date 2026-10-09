"""Interactive review of pending proposals: approve, edit, reject or skip each one."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from triagebot.autonomy.executor import ExecutionRefused, Executor
from triagebot.autonomy.proposals import Proposal, ProposalStore
from triagebot.github.client import GitHubError

Ask = Callable[[str], str]
Say = Callable[[str], None]


def describe(proposal: Proposal, payload: dict[str, Any]) -> str:
    kind = proposal.action_type
    if kind == "labels":
        return f"add labels: {', '.join(payload['labels'])}"
    if kind == "duplicate":
        return f"mark as duplicate of #{payload['duplicate_of']}"
    if kind == "request_info":
        return "ask the author:\n" + "\n".join(f"      - {q}" for q in payload["questions"])
    return f"post comment:\n      {payload['body']}"


def edit_payload(proposal: Proposal, ask: Ask) -> dict[str, Any] | None:
    """Ask for a replacement payload. Returns None if the input can't be used."""
    kind = proposal.action_type
    if kind == "labels":
        labels = [lb.strip() for lb in ask("  Labels, comma-separated: ").split(",") if lb.strip()]
        return {"labels": labels} if labels else None
    if kind == "duplicate":
        answer = ask("  Duplicate of issue number: ").strip().lstrip("#")
        return {"duplicate_of": int(answer)} if answer.isdigit() else None
    if kind == "request_info":
        raw = ask("  Questions, separated by | : ")
        questions = [q.strip() for q in raw.split("|") if q.strip()]
        return {"questions": questions} if questions else None
    body = ask("  New comment (one line): ").strip()
    return {"body": body} if body else None


def review(store: ProposalStore, executor: Executor, ask: Ask = input, say: Say = print) -> Counter:
    pending = store.pending()
    counts: Counter = Counter()
    if not pending:
        say("Nothing to review.")
        return counts
    mode = "DRY RUN: nothing will be written to GitHub" if executor.dry_run else "LIVE"
    say(f"{len(pending)} pending proposals ({mode})")

    for n, proposal in enumerate(pending, 1):
        say(f"\n[{n}/{len(pending)}] proposal {proposal.id}, issue #{proposal.issue_number}, "
            f"{proposal.action_type}")
        say(f"    {describe(proposal, proposal.payload)}")
        say(f"    why: {proposal.rationale}")
        choice = ask("  (a)pprove, (e)dit, (r)eject, (s)kip, (q)uit? ").strip().lower()[:1]

        if choice == "q":
            break
        if choice == "r":
            store.decide(proposal.id, "rejected", reason=ask("  Reason (optional): ").strip())
            counts["rejected"] += 1
            continue
        if choice not in ("a", "e"):
            counts["skipped"] += 1
            continue

        payload, status = proposal.payload, "approved"
        if choice == "e":
            edited = edit_payload(proposal, ask)
            if edited is None:
                say("  Could not use that input; skipped.")
                counts["skipped"] += 1
                continue
            if edited != proposal.payload:
                payload, status = edited, "edited"

        try:
            result = executor.execute(proposal, payload)
        except ExecutionRefused as exc:
            store.decide(proposal.id, "failed", payload, reason=str(exc))
            say(f"  Refused: {exc}")
            counts["failed"] += 1
            continue
        except GitHubError as exc:
            say(f"  GitHub error, left pending: {exc}")
            counts["skipped"] += 1
            continue
        store.decide(proposal.id, status, payload if status == "edited" else None)
        say(f"  {'Would' if result.dry_run else 'Done:'} {result.description}")
        counts[status] += 1

    say("\n" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) if counts else "")
    return counts
