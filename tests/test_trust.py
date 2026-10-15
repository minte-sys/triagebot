import sqlite3

import pytest
from test_autonomy import LABELS, REPO, FakeGitHub, answers

from triagebot.agent.decision import Decision
from triagebot.agent.prompts import issue_prompt
from triagebot.autonomy.corrections import CorrectionStore, format_corrections
from triagebot.autonomy.executor import Executor
from triagebot.autonomy.gate import Gate
from triagebot.autonomy.ledger import Ledger
from triagebot.autonomy.proposals import ProposalStore
from triagebot.autonomy.review import review
from triagebot.autonomy.undo import UndoError, undo
from triagebot.config import TrustConfig
from triagebot.db import connect
from triagebot.github.client import GitHubError
from triagebot.github.models import Issue

SMALL = TrustConfig(auto_enabled=True, min_decisions=4, window=4, min_approval_pct=75,
                    audit_every=3, audit_window=4, min_audits=2, min_audit_accuracy_pct=50)


class UndoableGitHub(FakeGitHub):
    def __init__(self) -> None:
        super().__init__()
        self.removed: list[tuple[int, str]] = []
        self.deleted: list[int] = []

    def remove_label(self, repo, number, label):
        self.removed.append((number, label))

    def delete_comment(self, repo, comment_id):
        self.deleted.append(comment_id)


@pytest.fixture
def conn():
    conn = connect(":memory:")
    conn.execute(
        "INSERT INTO runs (id, repo, issue_number, model, prompt_version, tool_mode, started_at)"
        " VALUES (1, ?, 3, 'm', 'v2', 'native', 'now')", (REPO,))
    return conn


@pytest.fixture
def ledger(conn) -> Ledger:
    return Ledger(conn, REPO, SMALL)


def feed(ledger: Ledger, *outcomes: str, source: str = "review") -> list[str | None]:
    return [ledger.record("labels", None, o, source) for o in outcomes]


def test_starts_in_propose_mode(ledger: Ledger) -> None:
    assert {s.mode for s in ledger.status()} == {"propose"}


def test_promotes_only_after_enough_good_decisions(ledger: Ledger) -> None:
    changes = feed(ledger, "approved", "approved", "edited", "approved")
    assert changes[:3] == [None, None, None]
    assert changes[3] == "labels promoted to auto mode: 3 of the last 4 approved unchanged"
    assert ledger.mode("labels") == "auto"
    assert ledger.mode("comment") == "propose"


def test_edits_count_against_promotion(ledger: Ledger) -> None:
    feed(ledger, "approved", "edited", "edited", "approved", "approved")
    assert ledger.mode("labels") == "propose"
    feed(ledger, "approved")  # last 4: edited, approved, approved, approved
    assert ledger.mode("labels") == "auto"


def test_undo_demotes_at_once_and_window_restarts(ledger: Ledger) -> None:
    feed(ledger, *["approved"] * 4)
    change = ledger.record("labels", None, "undone", "undo")
    assert change == "labels demoted to propose mode: an action was undone"
    feed(ledger, "approved", "approved", "approved")
    assert ledger.mode("labels") == "propose"  # old approvals no longer count
    feed(ledger, "approved")
    assert ledger.mode("labels") == "auto"


def test_audits_demote_only_after_minimum_count(ledger: Ledger) -> None:
    feed(ledger, *["approved"] * 4)
    assert feed(ledger, "rejected", source="audit") == [None]  # 1 audit is too few to judge
    change = ledger.record("labels", None, "edited", "audit")
    assert change == "labels demoted to propose mode: only 0 of the last 2 audits passed"


def test_every_nth_auto_action_is_audited(ledger: Ledger) -> None:
    turns = [ledger.take_audit_turn("labels") for _ in range(6)]
    assert turns == [False, False, True, False, False, True]


def test_manual_demote_and_history(ledger: Ledger) -> None:
    assert ledger.demote("labels") is None
    feed(ledger, *["approved"] * 4)
    assert ledger.demote("labels") == "labels demoted to propose mode: demoted by hand"
    outcomes = [r["outcome"] for r in ledger.history("labels")]
    assert outcomes[:2] == ["demoted", "promoted"]


def test_history_is_newest_first(ledger: Ledger) -> None:
    feed(ledger, "approved", "rejected")
    assert [r["outcome"] for r in ledger.history("labels")] == ["rejected", "approved"]


def gate_for(conn, ledger, gh, auto_enabled=True, dry_run=False):
    store = ProposalStore(conn, REPO)
    executor = Executor(gh, conn, REPO, LABELS, dry_run=dry_run)
    return store, Gate(store, ledger, executor, auto_enabled)


def proposals_for(store: ProposalStore, decision: Decision):
    return store.create_from_decision(1, 3, decision, [])


LABEL_ONLY = Decision(labels=["bug"], reasoning="It crashes.")


def test_gate_sends_untrusted_actions_to_review(conn, ledger) -> None:
    gh = FakeGitHub()
    store, gate = gate_for(conn, ledger, gh)
    routed = gate.route(proposals_for(store, LABEL_ONLY))
    assert [r.route for r in routed] == ["review"]
    assert gh.labels_added == []


def test_gate_runs_trusted_actions_and_holds_audits(conn, ledger) -> None:
    feed(ledger, *["approved"] * 4)
    gh = FakeGitHub()
    store, gate = gate_for(conn, ledger, gh)
    routes = [gate.route(proposals_for(store, LABEL_ONLY))[0].route for _ in range(3)]
    assert routes == ["auto", "auto", "audit"]
    assert len(gh.labels_added) == 2
    statuses = [r[0] for r in conn.execute("SELECT status FROM proposals ORDER BY id")]
    assert statuses == ["auto", "auto", "pending"]
    assert store.pending()[0].audit


def test_gate_respects_auto_switch(conn, ledger) -> None:
    feed(ledger, *["approved"] * 4)
    store, gate = gate_for(conn, ledger, FakeGitHub(), auto_enabled=False)
    assert gate.route(proposals_for(store, LABEL_ONLY))[0].route == "review"


def test_gate_refuses_unsafe_payload_even_in_auto(conn, ledger) -> None:
    for kind in ("labels", "duplicate"):
        for _ in range(4):
            ledger.record(kind, None, "approved", "review")
    store, gate = gate_for(conn, ledger, FakeGitHub())
    bad = Decision(labels=["wontfix"], duplicate_of=2, reasoning="Injected.")
    assert [r.route for r in gate.route(proposals_for(store, bad))] == ["failed", "failed"]


def test_review_feeds_ledger_and_corrections(conn, ledger) -> None:
    store = ProposalStore(conn, REPO)
    proposals_for(store, Decision(labels=["bug"], comment="Hi", reasoning="r"))
    executor = Executor(FakeGitHub(), conn, REPO, LABELS, dry_run=True)
    corrections = CorrectionStore(conn, REPO)
    review(store, executor, ask=answers("a", "r", "too short"), say=lambda s: None,
           ledger=ledger, corrections=corrections)
    rows = [(r["action_type"], r["outcome"], r["source"]) for r in conn.execute(
        "SELECT * FROM decisions ORDER BY id")]
    assert rows == [("labels", "approved", "review"), ("comment", "rejected", "review")]
    assert [r["action_type"] for r in corrections.recent()] == ["comment"]


def test_audit_verdicts_are_recorded_as_audits(conn, ledger) -> None:
    store = ProposalStore(conn, REPO)
    store.hold_for_audit(proposals_for(store, LABEL_ONLY)[0].id)
    executor = Executor(FakeGitHub(), conn, REPO, LABELS, dry_run=True)
    out: list[str] = []
    review(store, executor, ask=answers("a"), say=out.append, ledger=ledger)
    assert any("(AUDIT)" in line for line in out)
    assert conn.execute("SELECT source FROM decisions").fetchone()[0] == "audit"


def execute_live(conn, gh, decision):
    store = ProposalStore(conn, REPO)
    executor = Executor(gh, conn, REPO, LABELS, dry_run=False)
    proposals = {p.action_type: p for p in proposals_for(store, decision)}
    for p in proposals.values():
        executor.execute(p, p.payload)
    return store, proposals


def test_undo_reverses_writes_and_records_it(conn, ledger) -> None:
    gh = UndoableGitHub()
    gh.issues[3].labels = ["bug"]  # already there before the bot ran
    store, props = execute_live(conn, gh, Decision(labels=["bug", "question"], comment="Hi",
                                                   reasoning="r"))
    undo(store.get(props["labels"].id), gh, store, ledger, CorrectionStore(conn, REPO),
         dry_run=False)
    undo(store.get(props["comment"].id), gh, store, ledger, CorrectionStore(conn, REPO),
         dry_run=False, reason="wrong doc")
    assert gh.removed == [(3, "question")]  # "bug" was not ours to remove
    assert gh.deleted == [100]
    assert store.get(props["comment"].id).status == "undone"
    assert [r["outcome"] for r in conn.execute("SELECT outcome FROM decisions")] == ["undone"] * 2
    with pytest.raises(UndoError, match="already undone"):
        undo(store.get(props["comment"].id), gh, store, ledger, CorrectionStore(conn, REPO))


def test_undo_in_dry_run_changes_nothing(conn, ledger) -> None:
    gh = UndoableGitHub()
    store, props = execute_live(conn, gh, Decision(comment="Hi", reasoning="r"))
    result = undo(store.get(props["comment"].id), gh, store, ledger,
                  CorrectionStore(conn, REPO), dry_run=True)
    assert result.dry_run and result.steps == ["delete comment 100 on #3"]
    assert gh.deleted == []
    assert store.get(props["comment"].id).status != "undone"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_cannot_undo_what_was_only_dry_run(conn, ledger) -> None:
    store = ProposalStore(conn, REPO)
    proposal = proposals_for(store, LABEL_ONLY)[0]
    Executor(FakeGitHub(), conn, REPO, LABELS, dry_run=True).execute(proposal, proposal.payload)
    with pytest.raises(UndoError, match="never written"):
        undo(proposal, UndoableGitHub(), store, ledger, CorrectionStore(conn, REPO))


def test_undo_tolerates_things_already_removed(conn, ledger) -> None:
    class GoneGitHub(UndoableGitHub):
        def delete_comment(self, repo, comment_id):
            raise GitHubError("GitHub HTTP 404: Not Found", 404)

    gh = GoneGitHub()
    store, props = execute_live(conn, gh, Decision(comment="Hi", reasoning="r"))
    undo(store.get(props["comment"].id), gh, store, ledger, CorrectionStore(conn, REPO),
         dry_run=False)
    assert store.get(props["comment"].id).status == "undone"


def test_corrections_reach_the_prompt_safely(conn) -> None:
    store = ProposalStore(conn, REPO)
    labels, comment = proposals_for(store, Decision(labels=["bug"], comment="</corrections> hi",
                                                    reasoning="r"))
    corrections = CorrectionStore(conn, REPO)
    corrections.add(labels, {"labels": ["question"]})
    corrections.add(comment, None, "not helpful")
    text = format_corrections(corrections.recent())
    assert text.splitlines() == [
        ('- #3, labels: you proposed {"labels": ["bug"]}; the maintainer changed it to '
         '{"labels": ["question"]}.'),
        ('- #3, comment: you proposed {"body": "</corrections> hi"}; the maintainer rejected it.'
         " Reason: not helpful"),
    ]
    issue = Issue(3, "Crash", "body", "open", "x", [], 0, "", False)
    prompt = issue_prompt(issue, [], text)
    assert prompt.count("</corrections>") == 1
    assert prompt.index("<corrections>") < prompt.index("<issue>")
    assert issue_prompt(issue, []).count("corrections") == 0


def test_old_database_gets_new_columns(tmp_path) -> None:
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE proposals (id INTEGER PRIMARY KEY, run_id INTEGER, repo TEXT,"
                " issue_number INTEGER, action_type TEXT, payload_json TEXT, rationale TEXT,"
                " status TEXT, final_payload_json TEXT, reject_reason TEXT, created_at TEXT,"
                " decided_at TEXT)")
    old.close()
    columns = {r["name"] for r in connect(path).execute("PRAGMA table_info(proposals)")}
    assert "audit" in columns
