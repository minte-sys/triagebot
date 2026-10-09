import itertools

import pytest

from triagebot.agent.decision import Decision
from triagebot.autonomy.executor import ExecutionRefused, Executor, marker
from triagebot.autonomy.proposals import ProposalStore, payloads_from_decision
from triagebot.autonomy.review import review
from triagebot.db import connect
from triagebot.github.client import GitHubError
from triagebot.github.models import Comment, Issue

REPO = "me/sandbox"
LABELS = {"bug", "duplicate", "question"}


def issue(number: int, pr: bool = False) -> Issue:
    return Issue(number, f"Issue {number}", "", "open", "x", [], 0, "", pr)


class FakeGitHub:
    """Records write calls instead of sending them."""

    def __init__(self) -> None:
        self.issues = {1: issue(1), 2: issue(2, pr=True), 3: issue(3)}
        self.comments: dict[int, list[Comment]] = {}
        self.labels_added: list[tuple[int, list[str]]] = []
        self.ids = itertools.count(100)

    def get_issue(self, repo, number, max_comments=10):
        if number not in self.issues:
            raise GitHubError("GitHub HTTP 404: Not Found", 404)
        return self.issues[number]

    def list_comments(self, repo, number):
        return self.comments.get(number, [])

    def create_comment(self, repo, number, body):
        comment = Comment("triagebot", body, next(self.ids))
        self.comments.setdefault(number, []).append(comment)
        return comment

    def add_labels(self, repo, number, labels):
        self.labels_added.append((number, labels))
        return labels


DECISION = Decision(
    labels=["bug", "duplicate"], duplicate_of=1, missing_info=["Which version?"],
    comment="Looks like #1.", reasoning="Same crash as #1.",
)


@pytest.fixture
def conn():
    connect_ = connect(":memory:")
    connect_.execute(
        "INSERT INTO runs (id, repo, issue_number, model, prompt_version, tool_mode, started_at)"
        " VALUES (1, ?, 3, 'm', 'v1', 'native', 'now')", (REPO,))
    return connect_


@pytest.fixture
def store(conn) -> ProposalStore:
    return ProposalStore(conn, REPO)


def make(store: ProposalStore, decision: Decision = DECISION, current=()):
    return {p.action_type: p for p in store.create_from_decision(1, 3, decision, list(current))}


def test_decision_splits_into_one_proposal_per_action() -> None:
    kinds = [k for k, _ in payloads_from_decision(DECISION, ["bug"])]
    assert kinds == ["labels", "duplicate", "request_info", "comment"]
    assert payloads_from_decision(DECISION, ["bug"])[0][1] == {"labels": ["duplicate"]}


def test_empty_decision_proposes_nothing() -> None:
    assert payloads_from_decision(Decision(reasoning="Nothing to do."), []) == []


def test_new_run_supersedes_old_pending_proposals(store: ProposalStore) -> None:
    make(store)
    make(store)
    assert len(store.pending()) == 4
    statuses = [r[0] for r in store.conn.execute("SELECT status FROM proposals ORDER BY id")]
    assert statuses == ["superseded"] * 4 + ["pending"] * 4


def test_dry_run_writes_nothing(store: ProposalStore) -> None:
    gh = FakeGitHub()
    executor = Executor(gh, store.conn, REPO, LABELS, dry_run=True)
    for p in make(store).values():
        result = executor.execute(p, p.payload)
        assert result.dry_run
    assert gh.comments == {} and gh.labels_added == []


def test_live_execution(store: ProposalStore) -> None:
    gh = FakeGitHub()
    executor = Executor(gh, store.conn, REPO, LABELS, dry_run=False)
    props = make(store)
    executor.execute(props["labels"], props["labels"].payload)
    executor.execute(props["duplicate"], props["duplicate"].payload)
    executor.execute(props["request_info"], props["request_info"].payload)
    assert gh.labels_added == [(3, ["bug", "duplicate"]), (3, ["duplicate"])]
    bodies = [c.body for c in gh.comments[3]]
    assert bodies[0].startswith("Duplicate of #1")
    assert "- Which version?" in bodies[1]
    assert all("triagebot:proposal" in b for b in bodies)


def test_same_proposal_never_executes_twice(store: ProposalStore) -> None:
    gh = FakeGitHub()
    executor = Executor(gh, store.conn, REPO, LABELS, dry_run=False)
    comment = make(store)["comment"]
    executor.execute(comment, comment.payload)
    with pytest.raises(ExecutionRefused, match="already executed"):
        executor.execute(comment, comment.payload)
    assert len(gh.comments[3]) == 1


def test_existing_marker_comment_is_reused(store: ProposalStore) -> None:
    gh = FakeGitHub()
    comment = make(store)["comment"]
    gh.comments[3] = [Comment("triagebot", f"Looks like #1.\n\n{marker(comment.id)}", 55)]
    result = Executor(gh, store.conn, REPO, LABELS, dry_run=False).execute(comment, comment.payload)
    assert result.github_ref == {"comment_id": 55}
    assert len(gh.comments[3]) == 1


@pytest.mark.parametrize(
    ("kind", "payload", "error"),
    [
        ("labels", {"labels": ["wontfix"]}, "do not exist"),
        ("labels", {"labels": []}, "Expected 1 to"),
        ("duplicate", {"duplicate_of": 3}, "Invalid duplicate target"),
        ("duplicate", {"duplicate_of": 404}, "not found"),
        ("duplicate", {"duplicate_of": 2}, "pull request"),
        ("comment", {"body": "x" * 2001}, "1 to 2000"),
    ],
)
def test_executor_refuses_unsafe_payloads(store, kind, payload, error) -> None:
    gh = FakeGitHub()
    proposal = make(store)[kind]
    with pytest.raises(ExecutionRefused, match=error):
        Executor(gh, store.conn, REPO, LABELS, dry_run=False).execute(proposal, payload)
    assert gh.comments == {} and gh.labels_added == []


def answers(*replies: str):
    it = iter(replies)
    return lambda prompt: next(it)


def test_review_approve_edit_reject_skip(store: ProposalStore) -> None:
    make(store)  # labels, duplicate, request_info, comment
    gh = FakeGitHub()
    executor = Executor(gh, store.conn, REPO, LABELS, dry_run=False)
    out: list[str] = []
    counts = review(store, executor,
                    ask=answers("a", "e", "2", "r", "not needed", "s"), say=out.append)
    assert counts == {"approved": 1, "failed": 1, "rejected": 1, "skipped": 1}
    rows = dict(store.conn.execute("SELECT action_type, status FROM proposals"))
    assert rows == {"labels": "approved", "duplicate": "failed",
                    "request_info": "rejected", "comment": "pending"}
    assert any("Refused" in line and "pull request" in line for line in out)


def test_review_edit_is_recorded(store: ProposalStore) -> None:
    make(store, Decision(labels=["bug"], reasoning="It crashes."))
    executor = Executor(FakeGitHub(), store.conn, REPO, LABELS, dry_run=True)
    review(store, executor, ask=answers("e", "bug, question"), say=lambda s: None)
    row = store.conn.execute("SELECT status, final_payload_json FROM proposals").fetchone()
    assert row[0] == "edited" and "question" in row[1]
