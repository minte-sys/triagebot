"""Configuration loading."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKENDS = ("openai_compat", "ollama")


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


@dataclass(frozen=True)
class GitHubConfig:
    repo: str  # owner/name
    allowed_repos: tuple[str, ...]
    base_url: str
    token_env: str
    token: str | None


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    github: GitHubConfig


def load_config(path: str | Path = "config.toml", backend: str | None = None) -> Config:
    """Load config.toml and .env. Secrets are validated where they are used."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path.resolve()}")

    load_dotenv()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return Config(llm=_load_llm(data.get("llm", {}), backend), github=_load_github(data))


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


def require_secret(value: str | None, env_name: str | None) -> str:
    """Return the secret or raise ConfigError naming the missing variable."""
    if not value:
        raise ConfigError(f"{env_name} is not set. Add it to your .env file (see .env.example).")
    return value
