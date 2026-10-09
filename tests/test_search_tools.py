import json

from test_search import ISSUES

from triagebot.db import connect
from triagebot.search.keyword import SearchIndex
from triagebot.tools.registry import ToolRegistry
from triagebot.tools.search_tools import build_search_tools

REPO = "me/sandbox"


def test_empty_index_is_an_error_result() -> None:
    registry = ToolRegistry(build_search_tools(SearchIndex(connect(":memory:"), REPO)))
    result = registry.execute("search_similar_issues", {"query": "config crash"})
    assert result.is_error
    assert "triagebot index" in result.content


def test_search_similar_issues_excludes_bound_issue() -> None:
    index = SearchIndex(connect(":memory:"), REPO)
    index.index_issues(ISSUES)
    registry = ToolRegistry(build_search_tools(index, exclude_issue=3))
    result = registry.execute("search_similar_issues", {"query": "blank config crash", "limit": 2})
    hits = json.loads(result.content)
    assert hits[0]["number"] == 1
    assert set(hits[0]) == {"number", "title", "state", "labels", "score", "snippet"}
    assert all(h["number"] != 3 for h in hits)


def test_query_too_short_is_rejected() -> None:
    registry = ToolRegistry(build_search_tools(SearchIndex(connect(":memory:"), REPO)))
    result = registry.execute("search_docs", {"query": "a"})
    assert result.is_error and "query" in result.content
