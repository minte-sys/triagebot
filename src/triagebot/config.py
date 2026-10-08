"""Load settings from config.toml and secrets from .env, in one place."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKENDS = ("openai_compat", "ollama")


class ConfigError(Exception):
    """The config file or environment is missing something we need."""


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model: str
    base_url: str
    api_key: str | None
    temperature: float
    timeout_s: float


@dataclass(frozen=True)
class Config:
    llm: LLMConfig


def load_config(path: str | Path = "config.toml", backend: str | None = None) -> Config:
    """Read config.toml. `backend` overrides the file, e.g. from a CLI flag."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path.resolve()}")

    load_dotenv()  # copies values from .env into os.environ (does not overwrite real env vars)
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    llm = data.get("llm", {})
    chosen = backend or llm.get("backend", "openai_compat")
    if chosen not in BACKENDS:
        raise ConfigError(f"Unknown backend {chosen!r}. Use one of: {', '.join(BACKENDS)}")

    section = llm.get(chosen, {})
    api_key = None
    if key_env := section.get("api_key_env"):
        api_key = os.environ.get(key_env)
        if not api_key:
            raise ConfigError(f"{key_env} is not set. Copy .env.example to .env and fill it in.")

    return Config(
        llm=LLMConfig(
            backend=chosen,
            model=section["model"],
            base_url=section["base_url"].rstrip("/"),
            api_key=api_key,
            temperature=float(llm.get("temperature", 0.0)),
            timeout_s=float(llm.get("timeout_s", 60)),
        )
    )
