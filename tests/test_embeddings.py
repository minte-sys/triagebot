import json
import shutil
from pathlib import Path

import pytest
from test_llm_clients import fake_http

from triagebot.config import load_config
from triagebot.db import connect
from triagebot.eval.dataset import DatasetError, load_dataset
from triagebot.eval.search_eval import compare_search, format_search_report, rrf
from triagebot.llm.base import LLMError
from triagebot.search.embeddings import (
    CachedEmbedder,
    OllamaEmbedder,
    OpenAICompatEmbedder,
    cosine,
)

DATA = Path(__file__).parent.parent / "eval_data"


def test_openai_compat_embeddings_come_back_in_input_order() -> None:
    seen = []
    payload = {"data": [{"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]}]}
    embedder = OpenAICompatEmbedder("https://x.test", "KEY", "emb",
                                    http=fake_http(200, payload, seen))
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert seen[0].url.path == "/embeddings"
    assert json.loads(seen[0].content) == {"model": "emb", "input": ["a", "b"]}


def test_ollama_embeddings() -> None:
    seen = []
    embedder = OllamaEmbedder("http://local", "nomic",
                              http=fake_http(200, {"embeddings": [[0.5, 0.5]]}, seen))
    assert embedder.embed(["a"]) == [[0.5, 0.5]]
    assert seen[0].url.path == "/api/embed"


def test_embedding_errors_keep_the_status() -> None:
    embedder = OpenAICompatEmbedder("https://x.test", "KEY", "emb",
                                    http=fake_http(429, {"error": "quota"}, []))
    with pytest.raises(LLMError) as err:
        embedder.embed(["a"])
    assert err.value.status_code == 429


def test_cosine() -> None:
    assert cosine([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 3]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


class CountingEmbedder:
    model = "count"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, texts):
        self.batches.append(texts)
        return [[float(len(t)), 1.0] for t in texts]


def test_cache_pays_for_each_text_once() -> None:
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, connect(":memory:"))
    texts = [f"text {n}" for n in range(60)]
    first = cached.embed(texts)
    assert [len(b) for b in inner.batches] == [50, 10]
    assert cached.embed([texts[3], "new", texts[3]]) == [first[3], [3.0, 1.0], first[3]]
    assert inner.batches[-1] == ["new"]
    assert (cached.new, cached.cached) == (61, 2)


def test_rrf_rewards_issues_both_methods_rank_high() -> None:
    assert rrf([1, 2, 3], [3, 1, 4])[:2] == [1, 3]


class OracleEmbedder:
    """Puts each query right on top of the issue it describes."""

    model = "oracle"

    def __init__(self, ds) -> None:
        self.targets = {f"{i.title}\n{i.body}": i.number for i in ds.background}
        for c in ds.cases:
            self.targets[f"{c.title}\n{c.body}"] = c.expect.duplicate_of or 99
        for q in ds.search_queries:
            self.targets[q.query] = q.duplicate_of

    def embed(self, texts):
        return [[1.0 if n == self.targets[t] else 0.0 for n in range(100)] for t in texts]


def test_compare_search_reports_each_method() -> None:
    ds = load_dataset(DATA)
    groups = compare_search(ds, OracleEmbedder(ds))
    duplicates, paraphrases = groups.values()
    assert len(duplicates) == 11 and len(paraphrases) == len(ds.search_queries)
    assert all(r.ranks["Embeddings"] == 1 for rows in groups.values() for r in rows)
    report = format_search_report(groups, details=True)
    assert f"{'Embeddings':<12}{'20/20':>8}{'20/20':>8}{1:>8.2f}" in report
    assert "q1     #1" in report


def test_search_queries_must_point_at_older_issues(tmp_path) -> None:
    folder = tmp_path / "data"
    shutil.copytree(DATA, folder)
    (folder / "search_queries.jsonl").write_text('{"query": "x", "duplicate_of": 77}\n')
    with pytest.raises(DatasetError, match="#77"):
        load_dataset(folder)


def test_embedding_config_follows_the_backend(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[llm.openai_compat]\nbase_url = "https://g.test/"\nmodel = "m"\n'
                    'api_key_env = "K"\n[llm.ollama]\nbase_url = "http://local"\nmodel = "q"\n'
                    '[embeddings.ollama]\nmodel = "mxbai"\n')
    cfg = load_config(path)
    assert (cfg.embeddings.model, cfg.embeddings.base_url) == ("gemini-embedding-001",
                                                               "https://g.test")
    local = load_config(path, backend="ollama").embeddings
    assert (local.backend, local.model) == ("ollama", "mxbai")
