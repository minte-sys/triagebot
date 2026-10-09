"""Tool calling through plain JSON text, for models whose native tool calling is unreliable."""

from __future__ import annotations

import json
import re

from triagebot.llm.base import ToolCall, ToolSpec

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def json_instructions(specs: list[ToolSpec]) -> str:
    lines = [
        "To use a tool, reply with exactly one JSON object and nothing else, in this form:",
        '{"tool": "<tool name>", "args": {<arguments>}}',
        "",
        "Tools:",
    ]
    for spec in specs:
        lines.append(f"- {spec.name}: {spec.description}")
        lines.append(f"  arguments schema: {json.dumps(spec.parameters)}")
    return "\n".join(lines)


def parse_tool_call(text: str) -> ToolCall | None:
    """Find a {"tool": ..., "args": ...} object in model text, also inside ``` fences.

    {"name": ..., "arguments": ...} is accepted too, because small models often write that.
    """
    for candidate in [*FENCE.findall(text), text]:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            obj = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool", obj.get("name"))
        args = obj.get("args", obj.get("arguments", {}))
        if isinstance(name, str) and isinstance(args, dict):
            return ToolCall("json_call", name, args)
    return None
