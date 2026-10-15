from functools import partial
from pathlib import Path

import pytest
from test_agent import FakeLLM, call
from test_trust import UndoableGitHub

from triagebot import cli
from triagebot.autonomy.review import review
from triagebot.github.models import Label

CONFIG = """
[llm.openai_compat]
base_url = "https://example.test"
model = "fake-model"

[github]
repo = "me/sandbox"
allowed_repos = ["me/sandbox"]

[search]
db_path = "test.db"

[safety]
dry_run = false

[trust]
auto_enabled = true
min_decisions = 1
window = 1
audit_every = 3
"""


class CliGitHub(UndoableGitHub):
    def list_labels(self, repo):
        return [Label("bug", ""), Label("duplicate", ""), Label("question", "")]


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gh = CliGitHub()
    monkeypatch.setattr(cli, "make_github", lambda cfg: gh)

    def configure() -> list[str]:
        path = tmp_path / "config.toml"
        path.write_text(CONFIG)
        return ["--config", str(path)]

    return gh, configure


def triage(monkeypatch, args, labels) -> FakeLLM:
    llm = FakeLLM(call("finish", {"labels": labels, "reasoning": "It crashes."}))
    monkeypatch.setattr(cli, "make_client", lambda cfg: llm)
    assert cli.main([*args, "run", "3"]) == 0
    return llm


def reply_with(monkeypatch, *replies: str) -> None:
    it = iter(replies)
    monkeypatch.setattr(cli, "review", partial(review, ask=lambda prompt: next(it)))


def test_earned_trust_end_to_end(setup, monkeypatch, capsys) -> None:
    gh, configure = setup
    args = configure()

    triage(monkeypatch, args, ["bug"])
    assert "waiting for review" in capsys.readouterr().out

    reply_with(monkeypatch, "e", "question")
    cli.main([*args, "review"])
    assert "promoted" not in capsys.readouterr().out  # an edit doesn't earn trust

    llm = triage(monkeypatch, args, ["bug"])
    assert "Including 1 past corrections" in capsys.readouterr().out
    assert "<corrections>" in llm.calls[0][0][1].content

    reply_with(monkeypatch, "a")
    cli.main([*args, "review"])
    assert "labels promoted to auto mode" in capsys.readouterr().out

    triage(monkeypatch, args, ["question"])
    out = capsys.readouterr().out
    assert "auto, done: add labels ['question']" in out

    cli.main([*args, "trust"])
    assert "labels        auto" in capsys.readouterr().out

    proposal_id = int(out.split("Proposal ")[1].split(" ")[0])
    assert cli.main([*args, "undo", str(proposal_id)]) == 0
    out = capsys.readouterr().out
    assert "Done: remove label 'question' from #3" in out
    assert "labels demoted to propose mode: an action was undone" in out
    assert gh.removed == [(3, "question")]


def test_undo_unknown_proposal(setup, capsys) -> None:
    _, configure = setup
    assert cli.main([*configure(), "undo", "42"]) == 1
    assert "No proposal 42" in capsys.readouterr().out
