"""Backend for OpenAI-compatible Chat Completions APIs."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from triagebot.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout_s: float = 60,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.http = http or httpx.Client(timeout=timeout_s)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = [_tool_to_wire(t) for t in tools]

        start = time.perf_counter()
        try:
            resp = self.http.post(
                f"{self.base_url}/chat/completions", json=body, headers=self.headers
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Request failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)

        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
        return _parse_response(resp.json(), latency_ms)


def _tool_to_wire(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _message_to_wire(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    if m.provider_message is not None:
        # Gemini rejects tool calls that lose their thought signature.
        return m.provider_message
    wire: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        wire["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in m.tool_calls
        ]
    return wire


def _parse_response(data: dict[str, Any], latency_ms: int) -> LLMResponse:
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from exc

    calls = []
    for i, c in enumerate(msg.get("tool_calls") or []):
        fn = c["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raise LLMError(f"Tool {fn['name']} got invalid JSON arguments: {exc}") from exc
        calls.append(ToolCall(id=c.get("id") or f"call_{i}", name=fn["name"], arguments=args))

    usage = data.get("usage") or {}
    return LLMResponse(
        text=msg.get("content") or "",
        tool_calls=calls,
        usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        latency_ms=latency_ms,
        raw=data,
        provider_message=msg,
    )
