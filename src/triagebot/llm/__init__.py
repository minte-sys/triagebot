"""LLM backends."""

from triagebot.config import LLMConfig, require_secret
from triagebot.llm.base import LLMClient
from triagebot.llm.ollama import OllamaClient
from triagebot.llm.openai_compat import OpenAICompatClient


def make_client(cfg: LLMConfig) -> LLMClient:
    if cfg.backend == "ollama":
        return OllamaClient(cfg.base_url, cfg.model, cfg.temperature, cfg.timeout_s)
    api_key = require_secret(cfg.api_key, cfg.api_key_env)
    return OpenAICompatClient(cfg.base_url, api_key, cfg.model, cfg.temperature, cfg.timeout_s)
