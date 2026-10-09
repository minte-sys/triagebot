import pytest

from triagebot.db import connect
from triagebot.github.models import Issue
from triagebot.search.docs import chunk_markdown
from triagebot.search.keyword import SearchIndex, to_match_query

REPO = "me/sandbox"


def issue(number: int, title: str, body: str, state: str = "open", labels=()) -> Issue:
    return Issue(number, title, body, state, "someone", list(labels), 0, "", False)


ISSUES = [
    issue(1, "App crashes when config file is empty",
          "I created an empty config.toml and ran the app. It crashed with KeyError: 'llm'.",
          labels=["bug"]),
    issue(2, "How do I install this on Windows?", "The README only shows Linux commands.",
          labels=["question"]),
    issue(3, "Crash on startup with blank config", "Empty config file -> KeyError on launch."),
    issue(4, "Dark mode for the web page", "Would be nice to have.", state="closed"),
    issue(5, "Typo in README", "Second paragraph says teh instead of the.", state="closed"),
    issue(6, "Support Python 3.13", "Tests fail on the newest Python release."),
    issue(7, "Add a --verbose flag", "Printing each request would help debugging."),
    issue(8, "Rate limit errors from GitHub", "Got HTTP 403 after many requests."),
]

DOCS = """# Installation

## Windows
Install Python 3.11, then run pip install in PowerShell.

```bash
# not a heading
pip install -e .
```

## Linux
Use your package manager to install Python.
"""


@pytest.fixture
def index() -> SearchIndex:
    idx = SearchIndex(connect(":memory:"), REPO)
    idx.index_issues(ISSUES)
    return idx


def test_duplicate_ranks_first(index: SearchIndex) -> None:
    hits = index.search_issues("Crash on startup with blank config. Empty config file -> KeyError")
    assert [h.number for h in hits[:2]] == [3, 1]


def test_exclude_removes_the_issue_being_triaged(index: SearchIndex) -> None:
    hits = index.search_issues("Crash on startup with blank config", exclude=3)
    assert hits[0].number == 1
    assert 3 not in [h.number for h in hits]


def test_stemming_matches_word_forms(index: SearchIndex) -> None:
    assert [h.number for h in index.search_issues("installing")] == [2]


def test_terms_in_half_the_corpus_score_zero() -> None:
    small = SearchIndex(connect(":memory:"), REPO)
    small.index_issues(ISSUES[:3])
    assert all(h.score == 0 for h in small.search_issues("config crash"))


def test_scores_are_positive_and_sorted(index: SearchIndex) -> None:
    scores = [h.score for h in index.search_issues("config crash")]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_reindexing_does_not_duplicate(index: SearchIndex) -> None:
    index.index_issues(ISSUES)
    assert index.issue_count() == len(ISSUES)


def test_other_repos_are_invisible(index: SearchIndex) -> None:
    other = SearchIndex(index.conn, "someone/else")
    assert other.search_issues("config crash") == []


@pytest.mark.parametrize("text", ['KeyError: "llm" (x) AND OR NOT *', "a the of", "!!!", ""])
def test_any_text_is_a_safe_query(index: SearchIndex, text: str) -> None:
    index.search_issues(text)


def test_match_query_quotes_terms_and_drops_stopwords() -> None:
    assert to_match_query('The app: "crash" -> NOT ok') == '"app" OR "crash" OR "ok"'


def test_markdown_is_chunked_by_heading_outside_code_fences() -> None:
    chunks = chunk_markdown("install.md", DOCS)
    assert [c.heading for c in chunks] == ["Windows", "Linux"]
    assert "# not a heading" in chunks[0].text


def test_search_docs(index: SearchIndex) -> None:
    index.index_docs(chunk_markdown("install.md", DOCS))
    hits = index.search_docs("install on windows powershell")
    assert hits[0].heading == "Windows"
    assert hits[0].path == "install.md"
