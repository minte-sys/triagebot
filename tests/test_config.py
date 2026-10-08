from pathlib import Path

import pytest

from triagebot.config import ConfigError, load_config

TOML = """
[llm]
backend = "openai_compat"

[llm.openai_compat]
base_url = "https://example.test/v1/"
model = "test-model"
api_key_env = "TEST_KEY"

[llm.ollama]
base_url = "http://localhost:11434"
model = "qwen2.5:3b"
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(TOML)
    return path


def test_reads_key_from_environment(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    cfg = load_config(config_file)
    assert cfg.llm.api_key == "secret"
    assert cfg.llm.base_url == "https://example.test/v1"  # trailing slash removed


def test_missing_key_is_a_clear_error(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_KEY", raising=False)
    monkeypatch.chdir(config_file.parent)  # so a real .env elsewhere isn't picked up
    with pytest.raises(ConfigError, match="TEST_KEY"):
        load_config(config_file)


def test_backend_override_needs_no_key(config_file: Path) -> None:
    cfg = load_config(config_file, backend="ollama")
    assert cfg.llm.model == "qwen2.5:3b"
    assert cfg.llm.api_key is None


def test_unknown_backend_rejected(config_file: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown backend"):
        load_config(config_file, backend="gpt9000")
