from pathlib import Path

import pytest

from triagebot.config import ConfigError, load_config
from triagebot.llm import make_client

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

[github]
repo = "me/sandbox"
allowed_repos = ["me/sandbox"]
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
    assert cfg.llm.base_url == "https://example.test/v1"


def test_missing_key_is_a_clear_error(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_KEY", raising=False)
    monkeypatch.chdir(config_file.parent)
    cfg = load_config(config_file)
    with pytest.raises(ConfigError, match="TEST_KEY"):
        make_client(cfg.llm)


def test_backend_override_needs_no_key(config_file: Path) -> None:
    cfg = load_config(config_file, backend="ollama")
    assert cfg.llm.model == "qwen2.5:3b"
    assert cfg.llm.api_key is None


def test_unknown_backend_rejected(config_file: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown backend"):
        load_config(config_file, backend="gpt9000")


def test_repo_must_be_in_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(TOML.replace('allowed_repos = ["me/sandbox"]', 'allowed_repos = ["me/other"]'))
    with pytest.raises(ConfigError, match="allowed_repos"):
        load_config(path)


def test_search_paths_are_relative_to_config(config_file: Path) -> None:
    config_file.write_text(TOML + '\n[search]\ndocs_dir = "../sandbox/docs"\n')
    cfg = load_config(config_file)
    assert cfg.search.db_path == config_file.parent / "data/triagebot.db"
    assert cfg.search.docs_dir == config_file.parent / "../sandbox/docs"


def test_agent_defaults_and_tool_mode_validation(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert (cfg.agent.max_steps, cfg.agent.tool_mode) == (8, "native")
    config_file.write_text(TOML + '\n[agent]\ntool_mode = "telepathy"\n')
    with pytest.raises(ConfigError, match="tool_mode"):
        load_config(config_file)
