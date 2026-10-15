"""Compare duplicate search methods on the eval set: keyword (BM25), embeddings, and both
combined. Each query asks: where in the results does the right older issue appear?"""

from __future__ import annotations

from dataclasses import dataclass

from triagebot.eval.dataset import Dataset
from triagebot.eval.runner import build_index
from triagebot.search.embeddings import Embedder, cosine

METHODS = ("BM25", "Embeddings", "Hybrid")
TOP = 10
RRF_K = 60


@dataclass
class Query:
    name: str
    text: str
    target: int


@dataclass
class Ranking:
    query: Query
    ranks: dict[str, int | None]  # method -> position of the right issue, None if not in top 10


def rrf(*rankings: list[int]) -> list[int]:
    """Reciprocal rank fusion: an issue near the top of either list ends up near the top."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, number in enumerate(ranking, 1):
            scores[number] = scores.get(number, 0.0) + 1 / (RRF_K + position)
    return sorted(scores, key=lambda n: -scores[n])


def _rank(target: int, numbers: list[int]) -> int | None:
    return numbers.index(target) + 1 if target in numbers[:TOP] else None


def query_groups(ds: Dataset) -> dict[str, list[Query]]:
    return {
        "Duplicate issues from the eval set": [
            Query(f"#{c.number}", f"{c.title}\n{c.body}", c.expect.duplicate_of)
            for c in ds.cases if c.expect.duplicate_of
        ],
        "Paraphrases that share few words": [
            Query(f"q{n}", q.query, q.duplicate_of) for n, q in enumerate(ds.search_queries, 1)
        ],
    }


def compare_search(ds: Dataset, embedder: Embedder) -> dict[str, list[Ranking]]:
    groups = query_groups(ds)
    queries = [q for group in groups.values() for q in group]
    index = build_index(ds)
    older = ds.background
    vectors = embedder.embed([f"{i.title}\n{i.body}" for i in older] + [q.text for q in queries])
    older_vectors = vectors[: len(older)]
    query_vectors = dict(zip((id(q) for q in queries), vectors[len(older):], strict=True))

    def rank(q: Query) -> Ranking:
        bm25 = [h.number for h in index.search_issues(q.text, TOP)]
        qv = query_vectors[id(q)]
        by_similarity = sorted(zip(older, older_vectors, strict=True),
                               key=lambda pair: -cosine(qv, pair[1]))
        embedded = [issue.number for issue, _ in by_similarity[:TOP]]
        found = {"BM25": bm25, "Embeddings": embedded, "Hybrid": rrf(bm25, embedded)}
        return Ranking(q, {m: _rank(q.target, found[m]) for m in METHODS})

    return {name: [rank(q) for q in group] for name, group in groups.items()}


def _summary(rows: list[Ranking]) -> list[str]:
    n = len(rows)
    lines = [f"{'':<12}{'1st':>8}{'top 3':>8}{'MRR':>8}"]
    for m in METHODS:
        ranks = [r.ranks[m] for r in rows]
        first = sum(rank == 1 for rank in ranks)
        top3 = sum(rank is not None and rank <= 3 for rank in ranks)
        mrr = sum(1 / rank for rank in ranks if rank) / n if n else 0.0
        lines.append(f"{m:<12}{f'{first}/{n}':>8}{f'{top3}/{n}':>8}{mrr:>8.2f}")
    return lines


def format_search_report(groups: dict[str, list[Ranking]], details: bool = False) -> str:
    lines = []
    for name, rows in groups.items():
        lines += [f"{name} ({len(rows)}): where does the right older issue appear?",
                  *_summary(rows), ""]
    lines.append("1st = ranked first. MRR = average of 1/position: 1.00 means always first.")
    if details:
        lines.append(f"\n{'query':<7}{'older':<7}" + "".join(f"{m:>11}" for m in METHODS)
                     + "   text")
        for rows in groups.values():
            for r in rows:
                ranks = "".join(f"{r.ranks[m] or '-':>11}" for m in METHODS)
                first_line = r.query.text.splitlines()[0][:45]
                lines.append(f"{r.query.name:<7}#{r.query.target:<6}{ranks}   {first_line}")
        lines.append("Numbers are positions in the results; - means not in the top 10.")
    return "\n".join(lines)
