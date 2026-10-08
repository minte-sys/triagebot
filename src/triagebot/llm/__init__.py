"""Model layer: pick a backend from config and return an LLMClient."""

from triagebot.config import LLMConfig
from triagebot.llm.base import LLMClient
from triagebot.llm.ollama import OllamaClient
from triagebot.llm.openai_compat import OpenAICompatClient


def make_client(cfg: LLMConfig) -> LLMClient:
    if cfg.backend == "ollama":
        return OllamaClient(cfg.base_url, cfg.model, cfg.temperature, cfg.timeout_s)
    assert cfg.api_key is not None  # load_config already checked this
    return OpenAICompatClient(cfg.base_url, cfg.api_key, cfg.model, cfg.temperature, cfg.timeout_s)
