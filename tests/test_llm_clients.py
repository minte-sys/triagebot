"""Backend tests with a fake HTTP server, so they run offline and cost nothing."""

import json

import httpx
import pytest

from triagebot.llm.base import LLMError, Message, ToolCall, ToolSpec
from triagebot.llm.ollama import OllamaClient
from triagebot.llm.openai_compat import OpenAICompatClient

TOOL = ToolSpec("multiply", "Multiply", {"type": "object", "properties": {}})


def fake_http(status: int, payload: dict, seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compat_parses_tool_call() -> None:
    seen: list[httpx.Request] = []
    payload = {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "abc", "type": "function",
             "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'}}]}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 7},
    }
    client = OpenAICompatClient("https://x.test", "KEY", "m", http=fake_http(200, payload, seen))

    resp = client.chat([Message("user", "hi")], tools=[TOOL])

    assert resp.tool_calls == [ToolCall("abc", "multiply", {"a": 2, "b": 3})]
    assert resp.text == ""
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (50, 7)
    assert seen[0].headers["Authorization"] == "Bearer KEY"
    assert seen[0].url.path == "/chat/completions"


def test_openai_compat_sends_tool_results_in_openai_shape() -> None:
    seen: list[httpx.Request] = []
    payload = {"choices": [{"message": {"content": "6"}}]}
    client = OpenAICompatClient("https://x.test", "KEY", "m", http=fake_http(200, payload, seen))
    history = [
        Message("assistant", "", tool_calls=[ToolCall("abc", "multiply", {"a": 2, "b": 3})]),
        Message("tool", "6", tool_call_id="abc", name="multiply"),
    ]

    client.chat(history)

    sent = json.loads(seen[0].content)["messages"]
    assert sent[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 2, "b": 3}'
    assert sent[1] == {"role": "tool", "tool_call_id": "abc", "content": "6"}


def test_openai_compat_echoes_provider_extras_back() -> None:
    """Gemini adds extra fields (thought signatures) that must come back unchanged."""
    seen: list[httpx.Request] = []
    assistant = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "abc", "type": "function", "extra": {"signature": "SIG123"},
         "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'}}]}
    payload = {"choices": [{"message": assistant}]}
    client = OpenAICompatClient("https://x.test", "KEY", "m", http=fake_http(200, payload, seen))

    first = client.chat([Message("user", "hi")], tools=[TOOL])
    client.chat([Message("user", "hi"), first.to_message()])

    sent = json.loads(seen[1].content)["messages"]
    assert sent[1] == assistant  # byte-for-byte what the provider gave us


def test_openai_compat_raises_on_rate_limit() -> None:
    client = OpenAICompatClient("https://x.test", "KEY", "m", http=fake_http(429, {}, []))
    with pytest.raises(LLMError) as err:
        client.chat([Message("user", "hi")])
    assert err.value.status_code == 429


def test_ollama_parses_tool_call_with_object_arguments() -> None:
    seen: list[httpx.Request] = []
    payload = {
        "message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "multiply", "arguments": {"a": 2, "b": 3}}}]},
        "prompt_eval_count": 40,
        "eval_count": 9,
    }
    client = OllamaClient("http://ollama.test", "qwen2.5:3b", http=fake_http(200, payload, seen))

    resp = client.chat([Message("user", "hi")], tools=[TOOL])

    assert resp.tool_calls[0].name == "multiply"
    assert resp.tool_calls[0].arguments == {"a": 2, "b": 3}
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (40, 9)
    assert json.loads(seen[0].content)["stream"] is False
