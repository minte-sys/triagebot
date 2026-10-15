"""Configuration loading."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKENDS = ("openai_compat", "ollama")
TOOL_MODES = ("native", "json")


class ConfigError(Exception):
    """Invalid or missing configuration."""


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model: str
    base_url: str
    api_key_env: str | None
    api_key: str | None
    temperature: float
    timeout_s: float
    fallback_models: tuple[str, ...] = ()
    max_retries: int = 3
    retry_max_wait_s: float = 60


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str
    model: str
    base_url: str
    api_key_env: str | None
    api_key: str | None
    timeout_s: float


EMBEDDING_MODELS = {"openai_compat": "gemini-embedding-001", "ollama": "nomic-embed-text"}


@dataclass(frozen=True)
class GitHubConfig:
    repo: str  # owner/name
    allowed_repos: tuple[str, ...]
    base_url: str
    token_env: str
    token: str | None


ISSUE_SEARCH = ("bm25", "embeddings")


@dataclass(frozen=True)
class SearchConfig:
    db_path: Path
    docs_dir: Path | None
    issue_search: str = "bm25"


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int
    max_seconds: float
    max_tokens: int
    tool_mode: str
    prompt_version: str


@dataclass(frozen=True)
class SafetyConfig:
    dry_run: bool


@dataclass(frozen=True)
class TrustConfig:
    auto_enabled: bool = False
    min_decisions: int = 20
    window: int = 20
    min_approval_pct: int = 90
    audit_every: int = 5
    audit_window: int = 10
    min_audits: int = 5
    min_audit_accuracy_pct: int = 80


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    github: GitHubConfig
    search: SearchConfig
    agent: AgentConfig
    safety: SafetyConfig
    trust: TrustConfig = TrustConfig()
    embeddings: EmbeddingConfig | None = None


def load_config(path: str | Path = "config.toml", backend: str | None = None) -> Config:
    """Load config.toml and .env. Secrets are validated where they are used."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path.resolve()}")

    load_dotenv()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return Config(
        llm=_load_llm(data.get("llm", {}), backend),
        github=_load_github(data),
        search=_load_search(data.get("search", {}), path.parent),
        agent=_load_agent(data.get("agent", {})),
        safety=SafetyConfig(dry_run=bool(data.get("safety", {}).get("dry_run", True))),
        trust=_load_trust(data.get("trust", {})),
        embeddings=_load_embeddings(data, backend),
    )


def _load_llm(llm: dict, backend: str | None) -> LLMConfig:
    chosen = backend or llm.get("backend", "openai_compat")
    if chosen not in BACKENDS:
        raise ConfigError(f"Unknown backend {chosen!r}. Use one of: {', '.join(BACKENDS)}")
    section = llm.get(chosen, {})
    key_env = section.get("api_key_env")
    return LLMConfig(
        backend=chosen,
        model=section["model"],
        base_url=section["base_url"].rstrip("/"),
        api_key_env=key_env,
        api_key=os.environ.get(key_env) if key_env else None,
        temperature=float(llm.get("temperature", 0.0)),
        timeout_s=float(llm.get("timeout_s", 60)),
        fallback_models=tuple(section.get("fallback_models", [])),
        max_retries=int(llm.get("max_retries", 3)),
        retry_max_wait_s=float(llm.get("retry_max_wait_s", 60)),
    )


def _load_embeddings(data: dict, backend: str | None) -> EmbeddingConfig | None:
    """Embeddings reuse the URL and key of the [llm.<backend>] section."""
    section = data.get("embeddings", {})
    chosen = backend or section.get("backend", "openai_compat")
    if chosen not in BACKENDS:
        raise ConfigError(f"Unknown embeddings backend {chosen!r}. Use one of: {', '.join(BACKENDS)}")
    llm = data.get("llm", {})
    connection = llm.get(chosen, {})
    if "base_url" not in connection:
        return None
    key_env = connection.get("api_key_env")
    return EmbeddingConfig(
        backend=chosen,
        model=section.get(chosen, {}).get("model", EMBEDDING_MODELS[chosen]),
        base_url=connection["base_url"].rstrip("/"),
        api_key_env=key_env,
        api_key=os.environ.get(key_env) if key_env else None,
        timeout_s=float(llm.get("timeout_s", 60)),
    )


def _load_github(data: dict) -> GitHubConfig:
    gh = data.get("github", {})
    repo = gh.get("repo", "")
    allowed = tuple(gh.get("allowed_repos", []))
    if repo and repo not in allowed:
        raise ConfigError(f"github.repo {repo!r} is not in github.allowed_repos")
    token_env = gh.get("token_env", "GITHUB_TOKEN")
    return GitHubConfig(
        repo=repo,
        allowed_repos=allowed,
        base_url=gh.get("base_url", "https://api.github.com").rstrip("/"),
        token_env=token_env,
        token=os.environ.get(token_env) or None,
    )


def _load_search(search: dict, base: Path) -> SearchConfig:
    """Relative paths are resolved against the folder containing config.toml."""
    docs_dir = search.get("docs_dir")
    issue_search = search.get("issue_search", "bm25")
    if issue_search not in ISSUE_SEARCH:
        raise ConfigError(
            f"Unknown search.issue_search {issue_search!r}. Use one of: {', '.join(ISSUE_SEARCH)}"
        )
    return SearchConfig(
        db_path=base / search.get("db_path", "data/triagebot.db"),
        docs_dir=base / docs_dir if docs_dir else None,
        issue_search=issue_search,
    )


def _load_agent(agent: dict) -> AgentConfig:
    from triagebot.agent.prompts import PROMPT_VERSION, PROMPTS  # avoids an import cycle

    mode = agent.get("tool_mode", "native")
    if mode not in TOOL_MODES:
        raise ConfigError(f"Unknown agent.tool_mode {mode!r}. Use one of: {', '.join(TOOL_MODES)}")
    version = agent.get("prompt_version", PROMPT_VERSION)
    if version not in PROMPTS:
        raise ConfigError(
            f"Unknown agent.prompt_version {version!r}. Use one of: {', '.join(PROMPTS)}"
        )
    return AgentConfig(
        max_steps=int(agent.get("max_steps", 8)),
        max_seconds=float(agent.get("max_seconds", 180)),
        max_tokens=int(agent.get("max_tokens", 40_000)),
        tool_mode=mode,
        prompt_version=version,
    )


def _load_trust(trust: dict) -> TrustConfig:
    defaults = TrustConfig()
    values = {}
    for name in ("min_decisions", "window", "min_approval_pct", "audit_every", "audit_window",
                 "min_audits", "min_audit_accuracy_pct"):
        value = trust.get(name, getattr(defaults, name))
        if not isinstance(value, int) or value < 1:
            raise ConfigError(f"trust.{name} must be a whole number of at least 1, got {value!r}")
        if name.endswith("_pct") and value > 100:
            raise ConfigError(f"trust.{name} is a percentage and must be at most 100")
        values[name] = value
    return TrustConfig(auto_enabled=bool(trust.get("auto_enabled", False)), **values)


def require_secret(value: str | None, env_name: str | None) -> str:
    """Return the secret or raise ConfigError naming the missing variable."""
    if not value:
        raise ConfigError(f"{env_name} is not set. Add it to your .env file (see .env.example).")
    return value
