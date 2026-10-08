"""The one interface every model backend implements.

The rest of TriageBot only ever imports from this file, so swapping
Gemini for Ollama never touches agent code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # set on assistant messages
    tool_call_id: str | None = None  # set on tool messages: which call this answers
    name: str | None = None  # set on tool messages: which tool produced it
    # The provider's own copy of an assistant message. Backends send it back unchanged,
    # so provider-specific extras (like Gemini's thought signatures) survive the round trip.
    provider_message: dict[str, Any] | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # a JSON Schema object describing the arguments


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    latency_ms: int
    raw: dict[str, Any]  # the untouched provider response, kept for traces
    provider_message: dict[str, Any] | None = None  # the assistant message exactly as sent

    def to_message(self) -> Message:
        """The assistant turn to append to the conversation history."""
        return Message(
            "assistant", self.text, tool_calls=self.tool_calls,
            provider_message=self.provider_message,
        )


class LLMError(Exception):
    """Any failure talking to a model: HTTP error, timeout, or unparseable reply."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code  # lets Stage 7 retry on 429 and 5xx only


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        """Send the conversation, get back text and/or tool calls."""
        ...
