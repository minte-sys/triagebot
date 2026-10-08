"""GitHub access."""

from triagebot.config import ConfigError, GitHubConfig, require_secret
from triagebot.github.client import GitHubClient


def make_github(cfg: GitHubConfig) -> GitHubClient:
    if not cfg.repo:
        raise ConfigError("github.repo is not set in config.toml")
    token = require_secret(cfg.token, cfg.token_env)
    return GitHubClient(token, cfg.allowed_repos, cfg.base_url)
