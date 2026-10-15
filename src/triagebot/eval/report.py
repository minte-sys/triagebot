"""Turn stored eval results into a comparison table."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from string import ascii_uppercase

from triagebot.eval.dataset import Dataset
from triagebot.eval.metrics import (
    Outcome,
    Scores,
    baseline,
    most_common_label,
    problems,
    score,
)
from triagebot.eval.runner import Setup

MAX_FAILURES = 20


def _frac(a: int, b: int) -> str:
    return f"{a}/{b}" if b else "-"


def _avg(total: float, n: int, digits: int = 1) -> str:
    return f"{total / n:,.{digits}f}" if n else "-"


ROWS: list[tuple[str, Callable[[Scores, bool], str]]] = [
    ("Label precision", lambda s, b: f"{s.precision:.2f}"),
    ("Label recall", lambda s, b: f"{s.recall:.2f}"),
    ("Label F1", lambda s, b: f"{s.f1:.2f}"),
    ("Duplicates found", lambda s, b: _frac(s.duplicates_found, s.duplicates)),
    ("False duplicates", lambda s, b: str(s.false_duplicates)),
    ("Asked for info correctly", lambda s, b: _frac(s.info_right, s.info_scored)),
    ("Pointed to the right doc", lambda s, b: _frac(s.docs_cited, s.docs)),
    ("Injections resisted", lambda s, b: _frac(s.injections_resisted, s.injections)),
    ("Finished", lambda s, b: _frac(s.finished, s.cases)),
    ("Avg model calls", lambda s, b: "-" if b else _avg(s.steps, s.cases)),
    ("Avg tokens", lambda s, b: "-" if b else _avg(s.tokens, s.cases, 0)),
    ("Avg seconds", lambda s, b: "-" if b else _avg(s.seconds, s.cases)),
]


def format_report(ds: Dataset, results: dict[Setup, dict[int, Outcome]],
                  focus: Setup | None = None) -> str:
    kinds = Counter(c.kind for c in ds.cases)
    lines = [f"Eval set: {len(ds.cases)} cases ("
             + ", ".join(f"{n} {k}" for k, n in kinds.most_common()) + ")"]
    results = {setup: r for setup, r in results.items() if r}
    if not results:
        lines.append("No results yet. Start with: triagebot eval --limit 5")
        return "\n".join(lines)

    setups = list(results)
    common = set.intersection(*(set(r) for r in results.values()))
    cases = [c for c in ds.cases if c.number in common]
    lines.append(f"\nCompared on the {len(cases)} cases every setup below has finished:")
    names = ascii_uppercase[: len(setups)]
    for name, setup in zip(names, setups, strict=False):
        lines.append(f"  {name} = {setup} ({len(results[setup])} of {len(ds.cases)} done)")

    columns = [(name, score([results[s][c.number] for c in cases]), False)
               for name, s in zip(names, setups, strict=False)]
    common_label = most_common_label(ds.cases)
    columns.append(("baseline", score(baseline(cases, common_label)), True))
    lines.append("\n" + f"{'':<26}" + "".join(f"{name:>10}" for name, _, _ in columns))
    for row, show in ROWS:
        lines.append(f"{row:<26}" + "".join(f"{show(s, b):>10}" for _, s, b in columns))
    lines.append(f"baseline = always {common_label!r}, never a duplicate, no questions, no comment.")

    focus = focus if focus in results else setups[-1]
    failures = [o for c in ds.cases if (o := results[focus].get(c.number)) and problems(o)]
    lines.append(f"\nCases to look at for {focus}: {len(failures)}")
    for o in failures[:MAX_FAILURES]:
        trace = f"  (triagebot trace {o.run_id})" if o.run_id else ""
        lines.append(f"  #{o.case.number} {o.case.kind}: {'; '.join(problems(o))}{trace}")
    if len(failures) > MAX_FAILURES:
        lines.append(f"  ... and {len(failures) - MAX_FAILURES} more")
    return "\n".join(lines)
