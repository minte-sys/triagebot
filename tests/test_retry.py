import json

import pytest
from test_agent import FakeLLM, call
from test_cli import CliGitHub
from test_llm_clients import fake_http
from test_search import ISSUES

from triagebot import cli
from triagebot.db import connect
from triagebot.llm.base import LLMError, Message
from triagebot.llm.openai_compat import OpenAICompatClient
from triagebot.llm.retry import RetryingClient, RetryPolicy, http_error, with_retries
from triagebot.search.keyword import SearchIndex
from triagebot.search.semantic import SemanticIssueSearch, issue_text
from triagebot.tools.registry import ToolRegistry
from triagebot.tools.search_tools import build_search_tools

QUOTA_PER_DAY = '{"error": {"code": 429, "details": [{"quotaId": "RequestsPerDayPerModel"}]}}'


def test_error_classification() -> None:
    assert http_error(503, "overloaded").retryable
    assert not http_error(400, "bad request").retryable
    assert not http_error(400, "bad request").model_unavailable
    slow_down = http_error(429, '{"retryDelay": "17s"}')
    assert slow_down.retryable and slow_down.retry_after == 17
    assert http_error(429, "{}", {"retry-after": "5"}).retry_after == 5


def test_daily_quota_is_not_retried_but_allows_fallback() -> None:
    err = http_error(429, QUOTA_PER_DAY)
    assert not err.retryable and err.model_unavailable
    assert "daily quota" in str(err)


def test_client_raises_classified_errors() -> None:
    client = OpenAICompatClient("https://x.test", "K", "m", http=fake_http(503, {}, []))
    with pytest.raises(LLMError) as err:
        client.chat([Message("user", "hi")])
    assert err.value.retryable and err.value.status_code == 503


class Flaky:
    def __init__(self, *errors: LLMError) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def retry(call):
    waits: list[float] = []
    return with_retries(call, RetryPolicy(), sleep=waits.append, rand=lambda: 0.5), waits


def test_backoff_doubles_until_success() -> None:
    result, waits = retry(Flaky(http_error(503, ""), http_error(503, ""), http_error(503, "")))
    assert result == "ok"
    assert waits == [2, 4, 8]


def test_gives_up_after_max_retries() -> None:
    flaky = Flaky(*[http_error(503, "")] * 4)
    with pytest.raises(LLMError):
        retry(flaky)
    assert flaky.calls == 4


def test_does_not_retry_what_waiting_cannot_fix() -> None:
    for error in (http_error(400, "bad"), http_error(429, QUOTA_PER_DAY),
                  http_error(429, '{"retryDelay": "3600s"}')):
        flaky = Flaky(error)
        with pytest.raises(LLMError):
            retry(flaky)
        assert flaky.calls == 1


def test_provider_delay_is_respected() -> None:
    _, waits = retry(Flaky(http_error(429, '{"retryDelay": "7s"}')))
    assert waits == [7]


def test_retrying_client_wraps_chat() -> None:
    inner = FakeLLM(http_error(503, ""), call("finish", {"reasoning": "r"}))
    client = RetryingClient(inner, RetryPolicy(), sleep=lambda s: None)
    assert client.model == "fake-model"
    assert client.chat([Message("user", "hi")]).tool_calls[0].name == "finish"


class WordEmbedder:
    """A vector per text: which of a few topic words it mentions."""

    model = "words"
    TOPICS = ("config", "windows", "install", "crash")

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise http_error(503, "down")
        return [[float(t.lower().count(w)) + 0.01 for w in self.TOPICS] for t in texts]


def semantic_tools(embedder):
    index = SearchIndex(connect(":memory:"), "me/sandbox")
    index.index_issues(ISSUES)
    search = SemanticIssueSearch(index, embedder)
    return search, ToolRegistry(build_search_tools(index, 3, search))


def test_semantic_search_ranks_by_meaning() -> None:
    _, registry = semantic_tools(WordEmbedder())
    hits = json.loads(registry.execute("search_similar_issues", {"query": "config crash"}).content)
    assert hits[0]["number"] == 1 and hits[0]["matched_by"] == "embeddings"
    assert all(h["number"] != 3 for h in hits)


def test_semantic_search_falls_back_to_keywords() -> None:
    search, registry = semantic_tools(WordEmbedder(fail=True))
    hits = json.loads(registry.execute("search_similar_issues", {"query": "config crash"}).content)
    assert hits[0]["number"] == 1 and hits[0]["matched_by"] == "keywords (fallback)"
    assert "503" in search.last_error


def test_issue_text_matches_eval_format() -> None:
    assert issue_text("T", "B") == "T\nB"


FALLBACK_CONFIG = """
[llm.openai_compat]
base_url = "https://example.test"
model = "main-model"
fallback_models = ["backup-model"]

[github]
repo = "me/sandbox"
allowed_repos = ["me/sandbox"]

[search]
db_path = "test.db"
"""


def test_run_falls_back_to_the_next_model(tmp_path, monkeypatch, capsys) -> None:
    config = tmp_path / "config.toml"
    config.write_text(FALLBACK_CONFIG)
    monkeypatch.setattr(cli, "make_github", lambda cfg: CliGitHub())
    fakes = {"main-model": FakeLLM(http_error(429, QUOTA_PER_DAY)),
             "backup-model": FakeLLM(call("finish", {"labels": ["bug"], "reasoning": "r"}))}

    def make(llm_cfg):
        fake = fakes[llm_cfg.model]
        fake.model = llm_cfg.model
        return fake

    monkeypatch.setattr(cli, "make_client", make)
    assert cli.main(["--config", str(config), "run", "3"]) == 0
    out = capsys.readouterr().out
    assert "main-model is unavailable. Starting again with the fallback model backup-model" in out
    assert "Proposal 1 (labels)" in out
