"""Search tools over the local BM25 index."""

from __future__ import annotations

from dataclasses import asdict

from pydantic import BaseModel, Field

from triagebot.search.keyword import SearchIndex
from triagebot.tools.registry import Tool, ToolError

MAX_DOC_CHARS = 800


class SearchIssuesArgs(BaseModel):
    query: str = Field(
        min_length=3, max_length=500, description="Words describing the problem, e.g. the issue title"
    )
    limit: int = Field(default=5, ge=1, le=10, description="How many results to return")


class SearchDocsArgs(BaseModel):
    query: str = Field(min_length=3, max_length=500, description="What to look up in the docs")
    limit: int = Field(default=3, ge=1, le=10, description="How many sections to return")


def build_search_tools(index: SearchIndex, exclude_issue: int | None = None) -> list[Tool]:
    """exclude_issue keeps the issue being triaged out of its own duplicate search."""

    def search_similar_issues(args: SearchIssuesArgs) -> list[dict]:
        if index.issue_count() == 0:
            raise ToolError("The issue index is empty. Run `triagebot index` first.")
        return [asdict(h) for h in index.search_issues(args.query, args.limit, exclude_issue)]

    def search_docs(args: SearchDocsArgs) -> list[dict]:
        if index.doc_count() == 0:
            raise ToolError("No docs are indexed for this repository.")
        hits = index.search_docs(args.query, args.limit)
        for h in hits:
            if len(h.text) > MAX_DOC_CHARS:
                h.text = h.text[:MAX_DOC_CHARS] + " [...]"
        return [asdict(h) for h in hits]

    return [
        Tool("search_similar_issues",
             "Keyword search (BM25) over past open and closed issues. Higher score = more similar. "
             "Use it to find possible duplicates.",
             SearchIssuesArgs, search_similar_issues),
        Tool("search_docs",
             "Keyword search (BM25) over the repository's documentation, one result per section.",
             SearchDocsArgs, search_docs),
    ]
