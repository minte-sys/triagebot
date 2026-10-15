"""Run the agent over the eval set. Nothing here can write to GitHub: the issues come from
the dataset and no executor exists."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from triagebot.agent.budget import Budget
from triagebot.agent.decision import Decision
from triagebot.agent.loop import RunResult, run_agent
from triagebot.db import connect
from triagebot.eval.dataset import Case, Dataset
from triagebot.eval.metrics import Outcome, problems
from triagebot.github.client import GitHubError
from triagebot.github.models import Issue, Label
from triagebot.llm.base import LLMClient
from triagebot.search.keyword import SearchIndex
from triagebot.tools.read_tools import build_read_tools
from triagebot.tools.registry import ToolRegistry
from triagebot.tools.search_tools import build_search_tools
from triagebot.trace.recorder import Recorder

EVAL_REPO = "eval/repobrief"
ERRORS_IN_A_ROW = 3


@dataclass(frozen=True)
class Setup:
    """What produced a set of results. Results are only compared within the same cases."""

    model: str
    prompt_version: str
    tool_mode: str

    def __str__(self) -> str:
        return f"{self.model}, prompt {self.prompt_version}, {self.tool_mode} tools"


class DatasetGitHub:
    """Serves issues from the eval set to the get_issue tool."""

    def __init__(self, issues: list[Issue], labels: list[Label]) -> None:
        self.issues = {i.number: i for i in issues}
        self.labels = labels

    def get_issue(self, repo: str, number: int, max_comments: int = 10) -> Issue:
        if number not in self.issues:
            raise GitHubError("GitHub HTTP 404: Not Found", 404)
        return self.issues[number]

    def list_open_issues(self, repo: str, limit: int = 30) -> list[Issue]:
        return [i for i in self.issues.values() if i.state == "open"][:limit]

    def list_labels(self, repo: str) -> list[Label]:
        return self.labels


class ResultStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def save(self, setup: Setup, case: Case, result: RunResult, seconds: float) -> None:
        decision = result.decision.model_dump_json() if result.decision else None
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO eval_results (case_number, model, prompt_version,"
                " tool_mode, run_id, status, decision_json, steps, tokens, seconds, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (case.number, setup.model, setup.prompt_version, setup.tool_mode,
                 result.run_id, result.status, decision, result.steps,
                 result.tokens_in + result.tokens_out, round(seconds, 1),
                 datetime.now(UTC).isoformat(timespec="seconds")),
            )

    def clear(self, setup: Setup) -> int:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM eval_results WHERE model = ? AND prompt_version = ? AND tool_mode = ?",
                (setup.model, setup.prompt_version, setup.tool_mode),
            )
        return cur.rowcount

    def outcomes(self, cases: list[Case]) -> dict[Setup, dict[int, Outcome]]:
        """Every stored result for these cases, grouped by setup, oldest setup first."""
        by_number = {c.number: c for c in cases}
        grouped: dict[Setup, dict[int, Outcome]] = {}
        for r in self.conn.execute("SELECT * FROM eval_results ORDER BY id"):
            case = by_number.get(r["case_number"])
            if case is None:
                continue
            setup = Setup(r["model"], r["prompt_version"], r["tool_mode"])
            decision = (Decision.model_validate_json(r["decision_json"])
                        if r["decision_json"] else None)
            grouped.setdefault(setup, {})[case.number] = Outcome(
                case, r["status"], decision, r["steps"], r["tokens"], r["seconds"], r["run_id"])
        return grouped


def spread(cases: list[Case]) -> list[Case]:
    """Take one case of each kind in turn, so a partial run still covers every kind."""
    groups: dict[str, list[Case]] = {}
    for c in cases:
        groups.setdefault(c.kind, []).append(c)
    order: list[Case] = []
    while any(groups.values()):
        order += [g.pop(0) for g in groups.values() if g]
    return order


def build_index(ds: Dataset) -> SearchIndex:
    """A fresh in-memory index, so the eval never mixes with the real one."""
    index = SearchIndex(connect(":memory:"), EVAL_REPO)
    index.index_issues(ds.background)
    index.index_docs(ds.docs)
    return index


def run_eval(
    client: LLMClient,
    ds: Dataset,
    store: ResultStore,
    recorder: Recorder,
    setup: Setup,
    budget: Budget,
    limit: int | None = None,
    pause: float = 0,
    say: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run every case this setup has no result for yet. Returns how many were saved.

    A model error (quota, overload, timeout) saves nothing, so the case runs again next time.
    """
    done = set(store.outcomes(ds.cases).get(setup, {}))
    todo = [c for c in spread(ds.cases) if c.number not in done][:limit]
    say(f"{len(done)} of {len(ds.cases)} cases already done for {setup}. Running {len(todo)}.")
    index = build_index(ds)
    gh = DatasetGitHub(ds.background, ds.labels)
    saved = errors = 0
    for n, case in enumerate(todo, 1):
        if n > 1 and pause:
            sleep(pause)
        gh.issues[case.number] = case.issue()
        get_issue = [t for t in build_read_tools(gh, EVAL_REPO) if t.name == "get_issue"]
        registry = ToolRegistry(get_issue + build_search_tools(index, case.number))
        started = clock()
        result = run_agent(client, registry, recorder, EVAL_REPO, case.issue(), ds.labels,
                           budget, setup.tool_mode, clock=clock,
                           prompt_version=setup.prompt_version)
        seconds = clock() - started
        del gh.issues[case.number]
        head = f"[{n:>2}/{len(todo)}] #{case.number} {case.kind:<13}"
        if result.status == "error":
            errors += 1
            say(f"{head} model error, will retry next time: {result.stop_reason[:200]}")
            if errors >= ERRORS_IN_A_ROW:
                say(f"Stopping after {errors} model errors in a row. Run the same command "
                    "later to continue; finished cases are kept.")
                break
            continue
        errors = 0
        store.save(setup, case, result, seconds)
        saved += 1
        wrong = problems(Outcome(case, result.status, result.decision))
        verdict = "ok  " if not wrong else "MISS"
        say(f"{head} {verdict} {result.steps} calls {seconds:>5.1f}s  run {result.run_id}"
            + (f"  {'; '.join(wrong)}" if wrong else ""))
    return saved

