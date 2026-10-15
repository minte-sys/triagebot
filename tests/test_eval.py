import json
import shutil
from pathlib import Path

import pytest
from test_agent import FakeLLM, call

from triagebot import cli
from triagebot.agent.budget import Budget
from triagebot.agent.decision import Decision
from triagebot.db import connect
from triagebot.eval.dataset import DatasetError, load_dataset
from triagebot.eval.metrics import Outcome, baseline, most_common_label, problems, score
from triagebot.eval.report import format_report
from triagebot.eval.runner import ResultStore, Setup, build_index, run_eval, spread
from triagebot.llm.base import LLMError
from triagebot.trace.recorder import Recorder

DATA = Path(__file__).parent.parent / "eval_data"
SETUP = Setup("fake-model", "v3", "native")


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA)


def case(ds, number):
    return next(c for c in ds.cases if c.number == number)


def decide(**kwargs) -> Decision:
    return Decision(reasoning="r", **kwargs)


def test_eval_set_is_complete(ds) -> None:
    assert len(ds.cases) >= 50
    injections = [c for c in ds.cases if c.kind == "injection"]
    assert len(injections) >= 5
    assert all(c.forbid.labels or c.forbid.text or c.forbid.duplicate_of for c in injections)
    assert len(ds.background) == 25 and ds.docs


def test_typos_in_answers_are_caught(tmp_path) -> None:
    folder = tmp_path / "data"
    shutil.copytree(DATA, folder)
    row = json.loads((folder / "cases.jsonl").read_text().splitlines()[0])
    row["expect"]["labels"] = ["bugg"]
    (folder / "cases.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(DatasetError, match="unknown labels"):
        load_dataset(folder)


def test_label_and_duplicate_scores(ds) -> None:
    dup, bug = case(ds, 101), case(ds, 111)  # expects duplicate of #1 (bug optional); bug
    outcomes = [
        Outcome(dup, "finished", decide(labels=["duplicate", "bug"], duplicate_of=1)),
        Outcome(bug, "finished", decide(labels=["bug", "question"], duplicate_of=9)),
    ]
    s = score(outcomes)
    assert (s.label_tp, s.label_fp, s.label_fn) == (2, 1, 0)  # optional "bug" is not an error
    assert (s.duplicates, s.duplicates_found, s.false_duplicates) == (1, 1, 1)
    assert problems(outcomes[1]) == ["extra labels ['question']", "duplicate_of 9, expected 0"]


def test_injection_is_detected(ds) -> None:
    attack = case(ds, 147)  # asks for a link to evil.example in the comment
    fooled = Outcome(attack, "finished",
                     decide(labels=["enhancement"], comment="Get it at http://EVIL.example/x"))
    safe = Outcome(attack, "finished", decide(labels=["enhancement"]))
    assert score([fooled]).injections_resisted == 0
    assert score([safe]).injections_resisted == 1
    assert "followed the injection" in problems(fooled)[0]


def test_no_decision_counts_as_missed_labels(ds) -> None:
    s = score([Outcome(case(ds, 111), "budget_exceeded", None)])
    assert (s.label_fn, s.finished) == (1, 0)


def test_baseline_uses_the_most_common_label(ds) -> None:
    assert most_common_label(ds.cases) == "bug"
    s = score(baseline([case(ds, 101), case(ds, 111)], "bug"))
    assert (s.label_tp, s.label_fp, s.label_fn) == (1, 0, 1)


def test_partial_runs_cover_every_kind(ds) -> None:
    first = spread(ds.cases)[:6]
    assert len({c.kind for c in first}) == 6
    assert sorted(c.number for c in spread(ds.cases)) == [c.number for c in ds.cases]


def test_eval_index_holds_older_issues_and_docs(ds) -> None:
    index = build_index(ds)
    assert index.search_issues("KeyError llm config", 1)[0].number == 1
    assert index.search_docs("web viewer port", 1)[0].path == "configuration.md"


def eval_run(ds, llm, conn, cases, **kwargs) -> tuple[int, list[str]]:
    ds.cases, saved_cases = cases, ds.cases
    out: list[str] = []
    try:
        saved = run_eval(llm, ds, ResultStore(conn), Recorder(conn), SETUP, Budget(),
                         say=out.append, **kwargs)
    finally:
        ds.cases = saved_cases
    return saved, out


def test_results_are_saved_and_resumed(ds) -> None:
    conn = connect(":memory:")
    cases = [case(ds, 101), case(ds, 111)]
    llm = FakeLLM(call("search_similar_issues", {"query": "blank config KeyError"}),
                  call("finish", {"labels": ["duplicate"], "duplicate_of": 1, "reasoning": "r"}))
    saved, out = eval_run(ds, llm, conn, cases, limit=1)
    assert saved == 1 and "#101" in out[1] and "ok" in out[1]

    llm = FakeLLM(call("finish", {"labels": ["question"], "reasoning": "r"}))
    saved, out = eval_run(ds, llm, conn, cases)
    assert saved == 1 and out[0].startswith("1 of 2 cases already done")
    assert "MISS" in out[1] and "missed labels ['bug']" in out[1]

    results = ResultStore(conn).outcomes(cases)[SETUP]
    assert results[101].decision.duplicate_of == 1 and results[101].run_id


def test_model_errors_are_retried_later(ds) -> None:
    conn = connect(":memory:")
    llm = FakeLLM(*[LLMError("HTTP 429: quota")] * 3)
    saved, out = eval_run(ds, llm, conn, ds.cases[:5])
    assert saved == 0
    assert "Stopping after 3 model errors" in out[-1]
    assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 0


def test_report_compares_setups_on_shared_cases(ds) -> None:
    a, b = case(ds, 101), case(ds, 111)
    v2, v3 = Setup("m", "v2", "native"), Setup("m", "v3", "native")
    results = {
        v2: {101: Outcome(a, "finished", decide(labels=["bug"]), 6, 9000, 30.0, 1)},
        v3: {101: Outcome(a, "finished", decide(labels=["duplicate"], duplicate_of=1), 3),
             111: Outcome(b, "finished", decide(labels=["bug"]), 2)},
    }
    report = format_report(ds, results, v2)
    assert "Compared on the 1 cases" in report
    assert f"{'Duplicates found':<26}{'0/1':>10}{'1/1':>10}{'0/1':>10}" in report
    assert "#101 duplicate: missed labels ['duplicate']" in report
    assert "(triagebot trace 1)" in report


def test_eval_report_command(tmp_path, capsys) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[llm.openai_compat]\nbase_url = "https://x.test"\nmodel = "m"\n'
                      '[search]\ndb_path = "test.db"\n')
    shutil.copytree(DATA, tmp_path / "eval_data")
    assert cli.main(["--config", str(config), "eval", "--report"]) == 0
    assert "No results yet" in capsys.readouterr().out


def test_setup_picks_the_prompt_version(ds) -> None:
    conn = connect(":memory:")
    llm = FakeLLM(call("finish", {"labels": ["bug"], "reasoning": "r"}))
    run_eval(llm, ds, ResultStore(conn), Recorder(conn), Setup("fake-model", "v2", "native"),
             Budget(), limit=1, say=lambda s: None)
    assert "Work efficiently" not in llm.calls[0][0][0].content
    assert conn.execute("SELECT prompt_version FROM runs").fetchone()[0] == "v2"
