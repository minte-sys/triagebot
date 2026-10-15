"""Find similar issues by meaning (embeddings), falling back to keywords (BM25)."""

from __future__ import annotations

from triagebot.llm.base import LLMError
from triagebot.search.embeddings import Embedder, cosine
from triagebot.search.keyword import IssueHit, SearchIndex

SNIPPET_CHARS = 150


def issue_text(title: str, body: str) -> str:
    return f"{title}\n{body}"


class SemanticIssueSearch:
    """Ranks indexed issues by cosine similarity to the query.

    The issue vectors come from the embedding cache, so a search usually costs one request:
    the query itself. If the embedding service fails, BM25 answers instead.
    """

    def __init__(self, index: SearchIndex, embedder: Embedder) -> None:
        self.index = index
        self.embedder = embedder
        self.last_error = ""

    def search(self, text: str, limit: int = 5, exclude: int | None = None,
               state: str = "all") -> tuple[list[IssueHit], str]:
        """Returns the hits and the method that produced them."""
        rows = [r for r in self.index.all_issues()
                if int(r["number"]) != exclude and state in ("all", r["state"])]
        if not rows:
            return [], "embeddings"
        try:
            vectors = self.embedder.embed(
                [issue_text(r["title"], r["body"]) for r in rows] + [text])
        except LLMError as exc:
            self.last_error = str(exc)
            return self.index.search_issues(text, limit, exclude, state), "keywords (fallback)"
        query = vectors[-1]
        scored = sorted(zip(rows, vectors[:-1], strict=True),
                        key=lambda pair: -cosine(query, pair[1]))[:limit]
        return [
            IssueHit(number=int(r["number"]), title=r["title"], state=r["state"],
                     labels=[lb for lb in r["labels"].split(",") if lb],
                     score=round(cosine(query, v), 3), snippet=r["body"][:SNIPPET_CHARS])
            for r, v in scored
        ], "embeddings"
