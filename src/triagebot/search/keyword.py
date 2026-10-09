"""BM25 keyword search over issues and docs, using SQLite FTS5."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from triagebot.github.models import Issue
from triagebot.search.docs import DocChunk

WORD = re.compile(r"\w+")
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "so",
    "that", "the", "this", "to", "was", "we", "what", "when", "where", "which", "with", "you",
    "your"
})
MAX_TERMS = 32
TITLE_WEIGHT = 2.0
HEADING_WEIGHT = 2.0


@dataclass
class IssueHit:
    number: int
    title: str
    state: str
    labels: list[str]
    score: float
    snippet: str


@dataclass
class DocHit:
    path: str
    heading: str
    text: str
    score: float


def to_match_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: each word quoted, joined with OR.

    Quoting means characters like : - " ( * in issue text can never be read as FTS5 syntax.
    """
    terms: list[str] = []
    for word in WORD.findall(text.lower()):
        if len(word) > 1 and word not in STOPWORDS and word not in terms:
            terms.append(word)
    return " OR ".join(f'"{t}"' for t in terms[:MAX_TERMS])


class SearchIndex:
    """Search index for one repository."""

    def __init__(self, conn: sqlite3.Connection, repo: str) -> None:
        self.conn = conn
        self.repo = repo

    def index_issues(self, issues: list[Issue]) -> int:
        with self.conn:
            for issue in issues:
                self.conn.execute(
                    "DELETE FROM issues_fts WHERE repo = ? AND number = ?", (self.repo, issue.number)
                )
                self.conn.execute(
                    "INSERT INTO issues_fts (title, body, repo, number, state, labels) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (issue.title, issue.body, self.repo, issue.number, issue.state,
                     ",".join(issue.labels)),
                )
        return len(issues)

    def index_docs(self, chunks: list[DocChunk]) -> int:
        """Replace all docs for this repo."""
        with self.conn:
            self.conn.execute("DELETE FROM docs_fts WHERE repo = ?", (self.repo,))
            self.conn.executemany(
                "INSERT INTO docs_fts (heading, text, repo, path) VALUES (?, ?, ?, ?)",
                [(c.heading, c.text, self.repo, c.path) for c in chunks],
            )
        return len(chunks)

    def issue_count(self) -> int:
        row = self.conn.execute(
            "SELECT count(*) FROM issues_fts WHERE repo = ?", (self.repo,)
        ).fetchone()
        return row[0]

    def doc_count(self) -> int:
        row = self.conn.execute(
            "SELECT count(*) FROM docs_fts WHERE repo = ?", (self.repo,)
        ).fetchone()
        return row[0]

    def search_issues(self, text: str, limit: int = 5, exclude: int | None = None) -> list[IssueHit]:
        query = to_match_query(text)
        if not query:
            return []
        rows = self.conn.execute(
            "SELECT number, title, state, labels, bm25(issues_fts, ?, 1.0) AS bm, "
            "snippet(issues_fts, -1, '[', ']', '...', 12) AS snip "
            "FROM issues_fts WHERE issues_fts MATCH ? AND repo = ? ORDER BY bm LIMIT ?",
            (TITLE_WEIGHT, query, self.repo, limit + 1),
        ).fetchall()
        hits = [
            IssueHit(
                number=int(r["number"]),
                title=r["title"],
                state=r["state"],
                labels=[lb for lb in r["labels"].split(",") if lb],
                score=round(-r["bm"], 3),
                snippet=r["snip"],
            )
            for r in rows
            if int(r["number"]) != exclude
        ]
        return hits[:limit]

    def search_docs(self, text: str, limit: int = 3) -> list[DocHit]:
        query = to_match_query(text)
        if not query:
            return []
        rows = self.conn.execute(
            "SELECT path, heading, text, bm25(docs_fts, ?, 1.0) AS bm "
            "FROM docs_fts WHERE docs_fts MATCH ? AND repo = ? ORDER BY bm LIMIT ?",
            (HEADING_WEIGHT, query, self.repo, limit),
        ).fetchall()
        return [DocHit(r["path"], r["heading"], r["text"], round(-r["bm"], 3)) for r in rows]
