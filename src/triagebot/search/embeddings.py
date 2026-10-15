"""Embeddings: turn text into vectors so that texts with similar meaning end up close
together, even when they share no words."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from typing import Protocol

import httpx

from triagebot.config import EmbeddingConfig, require_secret
from triagebot.llm.base import LLMError

BATCH = 50


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatEmbedder:
    """POST /embeddings, as offered by OpenAI-compatible APIs such as Gemini's."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 60,
                 http: httpx.Client | None = None) -> None:
        self.model = model
        self.base_url = base_url
        self.http = http or httpx.Client(timeout=timeout_s)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post(self.http, f"{self.base_url}/embeddings",
                     {"model": self.model, "input": texts}, self.headers)
        try:
           rows = data["data"]
           if all("index" in row for row in rows):
                rows = sorted(rows, key=lambda row: row["index"])
           return [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected embeddings response: {str(data)[:300]}") from exc


class OllamaEmbedder:
    """POST /api/embed on a local Ollama."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 120,
                 http: httpx.Client | None = None) -> None:
        self.model = model
        self.base_url = base_url
        self.http = http or httpx.Client(timeout=timeout_s)

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post(self.http, f"{self.base_url}/api/embed",
                     {"model": self.model, "input": texts}, {})
        if not isinstance(data.get("embeddings"), list):
            raise LLMError(f"Unexpected embeddings response: {str(data)[:300]}")
        return data["embeddings"]


def _post(http: httpx.Client, url: str, body: dict, headers: dict) -> dict:
    try:
        resp = http.post(url, json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise LLMError(f"Embedding request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
    return resp.json()


def make_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.backend == "ollama":
        return OllamaEmbedder(cfg.base_url, cfg.model, cfg.timeout_s)
    api_key = require_secret(cfg.api_key, cfg.api_key_env)
    return OpenAICompatEmbedder(cfg.base_url, api_key, cfg.model, cfg.timeout_s)


def cosine(a: list[float], b: list[float]) -> float:
    """1 means same direction (same meaning), 0 means unrelated."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class CachedEmbedder:
    """Stores every vector in SQLite, so the same text is never paid for twice."""

    def __init__(self, embedder: Embedder, conn: sqlite3.Connection) -> None:
        self.embedder = embedder
        self.model = embedder.model
        self.conn = conn
        self.new = 0
        self.cached = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        found = {
            r["text_hash"]: json.loads(r["vector"])
            for r in self.conn.execute(
                f"SELECT text_hash, vector FROM embeddings WHERE model = ? AND text_hash IN"
                f" ({', '.join('?' * len(keys))})", (self.model, *keys))
        } if keys else {}
        missing = list(dict.fromkeys(
            (k, t) for k, t in zip(keys, texts, strict=True) if k not in found))
        for start in range(0, len(missing), BATCH):
            batch = missing[start:start + BATCH]
            vectors = self.embedder.embed([t for _, t in batch])
            if len(vectors) != len(batch):
                raise LLMError(f"Asked for {len(batch)} embeddings, got {len(vectors)}")
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (model, text_hash, vector) VALUES (?, ?, ?)",
                    [(self.model, k, json.dumps(v)) for (k, _), v in zip(batch, vectors,
                                                                          strict=True)],
                )
            found.update({k: v for (k, _), v in zip(batch, vectors, strict=True)})
        self.new += len(missing)
        self.cached += len(texts) - len(missing)
        return [found[k] for k in keys]
