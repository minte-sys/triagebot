"""Tool definitions, argument validation and execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from triagebot.github.client import GitHubError
from triagebot.llm.base import ToolSpec


class ToolError(Exception):
    """A tool could not run; the message is shown to the model."""


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[Any], Any]
    kind: Literal["read", "write"] = "read"

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, _clean_schema(self.args_model.model_json_schema()))


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self.tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool call. Errors are returned as results, not raised."""
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool {name!r}. Available: {', '.join(self.tools)}", True)
        try:
            validated = tool.args_model.model_validate(args)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return ToolResult(f"Invalid arguments for {name}: {problems}", True)
        try:
            output = tool.func(validated)
        except (GitHubError, ToolError) as exc:
            return ToolResult(f"{name} failed: {exc}", True)
        return ToolResult(json.dumps(output, ensure_ascii=False))


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove pydantic "title" keys, which some providers reject."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema
