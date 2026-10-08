import json

import pytest
from conftest import REPO

from triagebot.github.client import GitHubClient
from triagebot.tools.read_tools import build_read_tools
from triagebot.tools.registry import ToolRegistry


@pytest.fixture
def registry(gh: GitHubClient) -> ToolRegistry:
    return ToolRegistry(build_read_tools(gh, REPO))


def test_get_issue_returns_compact_json(registry: ToolRegistry) -> None:
    result = registry.execute("get_issue", {"issue_number": 3})
    assert not result.is_error
    data = json.loads(result.content)
    assert data["number"] == 3
    assert data["comments"] == [{"author": "maintainer", "body": "Can you share the full traceback?"}]


@pytest.mark.parametrize("bad", [{"issue_number": 0}, {"issue_number": "abc"}, {}])
def test_invalid_arguments_become_error_results(registry: ToolRegistry, bad: dict) -> None:
    result = registry.execute("get_issue", bad)
    assert result.is_error
    assert "issue_number" in result.content


def test_unknown_tool_is_an_error_result(registry: ToolRegistry) -> None:
    result = registry.execute("delete_repo", {})
    assert result.is_error
    assert "get_issue" in result.content


def test_github_failure_becomes_error_result(registry: ToolRegistry) -> None:
    result = registry.execute("get_issue", {"issue_number": 404})
    assert result.is_error and "404" in result.content


def test_pull_request_is_refused(registry: ToolRegistry) -> None:
    data = json.loads(registry.execute("get_issue", {"issue_number": 2}).content)
    assert "pull request" in data["error"]


def test_list_labels(registry: ToolRegistry) -> None:
    labels = json.loads(registry.execute("list_labels", {}).content)
    assert {"name": "question", "description": ""} in labels


def test_schemas_are_clean_and_constrained(registry: ToolRegistry) -> None:
    spec = {s.name: s for s in registry.specs()}["get_issue"]
    assert "title" not in json.dumps(spec.parameters)
    assert spec.parameters["properties"]["issue_number"]["minimum"] == 1
    assert spec.parameters["required"] == ["issue_number"]
