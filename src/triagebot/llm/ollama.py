"""Backend for a local model served by Ollama's native /api/chat endpoint."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from triagebot.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout_s: float = 120,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.http = http or httpx.Client(timeout=timeout_s)

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in messages],
            "stream": False,  # one JSON reply instead of a stream of chunks
            "options": {"temperature": self.temperature},
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        start = time.perf_counter()
        try:
            resp = self.http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}. Is it running?") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Request failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)

        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
        return _parse_response(resp.json(), latency_ms)


def _message_to_wire(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "content": m.content, "tool_name": m.name}
    wire: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        # Ollama sends arguments as a JSON *object*, unlike the OpenAI format.
        wire["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
        ]
    return wire


def _parse_response(data: dict[str, Any], latency_ms: int) -> LLMResponse:
    msg = data.get("message")
    if not isinstance(msg, dict):
        raise LLMError(f"Unexpected response shape: {str(data)[:300]}")

    calls = []
    for i, c in enumerate(msg.get("tool_calls") or []):
        fn = c["function"]
        args = fn.get("arguments") or {}
        if isinstance(args, str):  # be lenient: some models return a string anyway
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise LLMError(f"Tool {fn['name']} got invalid JSON arguments: {exc}") from exc
        # Ollama doesn't give calls an id, so we make one up to match results to calls.
        calls.append(ToolCall(id=f"call_{i}", name=fn["name"], arguments=args))

    return LLMResponse(
        text=msg.get("content") or "",
        tool_calls=calls,
        usage=Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0)),
        latency_ms=latency_ms,
        raw=data,
    )
