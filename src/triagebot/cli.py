"""Command-line entry point. More commands arrive in later stages."""

from __future__ import annotations

import argparse
import sys

from triagebot.config import BACKENDS, ConfigError, load_config
from triagebot.llm import make_client
from triagebot.llm.base import LLMError, Message, ToolSpec

# A toy tool, only for proving the tool-calling round trip works.
MULTIPLY = ToolSpec(
    name="multiply",
    description="Multiply two integers and return the product.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)


def cmd_ping(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, backend=args.backend)
    client = make_client(cfg.llm)
    print(f"Backend: {cfg.llm.backend}   Model: {cfg.llm.model}")

    messages = [
        Message("system", "You are a careful assistant. Use tools for arithmetic."),
        Message("user", "What is 1234 times 5678? Use the multiply tool."),
    ]

    # Round 1: the model should ask us to run a tool instead of answering.
    first = client.chat(messages, tools=[MULTIPLY])
    print(f"[1] {first.latency_ms} ms, tokens in/out {first.usage.input_tokens}/"
          f"{first.usage.output_tokens}")
    if not first.tool_calls:
        print(f"Model answered without calling the tool: {first.text!r}")
        return 1

    call = first.tool_calls[0]
    print(f"    model called {call.name}({call.arguments})")
    result = int(call.arguments["a"]) * int(call.arguments["b"])  # *we* run the tool

    # Round 2: give the result back and let the model write the final answer.
    messages.append(first.to_message())
    messages.append(Message("tool", str(result), tool_call_id=call.id, name=call.name))
    second = client.chat(messages, tools=[MULTIPLY])
    print(f"[2] {second.latency_ms} ms, tokens in/out {second.usage.input_tokens}/"
          f"{second.usage.output_tokens}")
    print(f"    answer: {second.text.strip()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triagebot")
    parser.add_argument("--config", default="config.toml", help="path to config.toml")
    parser.add_argument("--backend", choices=BACKENDS, help="override the backend in config")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ping", help="make one tool-calling round trip to the model").set_defaults(
        func=cmd_ping
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
