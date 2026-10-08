"""Provider-neutral LLM types and the LLMClient protocol."""

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
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    # Original provider message, resent verbatim to preserve provider-specific fields.
    provider_message: dict[str, Any] | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


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
    raw: dict[str, Any]
    provider_message: dict[str, Any] | None = None

    def to_message(self) -> Message:
        """Convert to an assistant message for the conversation history."""
        return Message(
            "assistant", self.text, tool_calls=self.tool_calls,
            provider_message=self.provider_message,
        )


class LLMError(Exception):
    """HTTP error, timeout, or unparseable response from a model."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        ...
