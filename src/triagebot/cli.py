"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from triagebot.config import BACKENDS, Config, ConfigError, load_config
from triagebot.db import connect
from triagebot.github import make_github
from triagebot.github.client import GitHubError
from triagebot.llm import make_client
from triagebot.llm.base import LLMError, Message, ToolSpec
from triagebot.search.docs import load_docs
from triagebot.search.keyword import SearchIndex
from triagebot.tools.read_tools import build_read_tools
from triagebot.tools.registry import ToolRegistry
from triagebot.tools.search_tools import build_search_tools

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

    first = client.chat(messages, tools=[MULTIPLY])
    print(f"[1] {first.latency_ms} ms, tokens in/out {first.usage.input_tokens}/"
          f"{first.usage.output_tokens}")
    if not first.tool_calls:
        print(f"Model answered without calling the tool: {first.text!r}")
        return 1

    call = first.tool_calls[0]
    print(f"    model called {call.name}({call.arguments})")
    result = int(call.arguments["a"]) * int(call.arguments["b"])

    messages.append(first.to_message())
    messages.append(Message("tool", str(result), tool_call_id=call.id, name=call.name))
    second = client.chat(messages, tools=[MULTIPLY])
    print(f"[2] {second.latency_ms} ms, tokens in/out {second.usage.input_tokens}/"
          f"{second.usage.output_tokens}")
    print(f"    answer: {second.text.strip()}")
    return 0


def _index(cfg: Config) -> SearchIndex:
    return SearchIndex(connect(cfg.search.db_path), cfg.github.repo)


def _registry(args: argparse.Namespace) -> ToolRegistry:
    cfg = load_config(args.config, backend=args.backend)
    tools = build_read_tools(make_github(cfg.github), cfg.github.repo)
    return ToolRegistry(tools + build_search_tools(_index(cfg)))


def cmd_index(args: argparse.Namespace) -> int:
    """Fetch all issues and read the docs folder into the search index."""
    cfg = load_config(args.config, backend=args.backend)
    index = _index(cfg)
    issues = make_github(cfg.github).list_issues(cfg.github.repo, state="all")
    print(f"Indexed {index.index_issues(issues)} issues from {cfg.github.repo}")

    docs_dir = cfg.search.docs_dir
    if docs_dir is None:
        print("No search.docs_dir in config; skipping docs")
    elif not docs_dir.is_dir():
        print(f"Docs folder not found: {docs_dir.resolve()}; skipping docs")
    else:
        print(f"Indexed {index.index_docs(load_docs(docs_dir))} doc sections from {docs_dir}")
    print(f"Database: {cfg.search.db_path}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """Print each tool with the schema sent to the model."""
    for spec in _registry(args).specs():
        print(f"{spec.name}: {spec.description}")
        print(f"    args schema: {json.dumps(spec.parameters)}")
    return 0


def cmd_tool(args: argparse.Namespace) -> int:
    """Run a single tool with key=value arguments."""
    tool_args = {}
    for pair in args.args:
        key, _, value = pair.partition("=")
        try:
            tool_args[key] = json.loads(value)
        except json.JSONDecodeError:
            tool_args[key] = value
    result = _registry(args).execute(args.name, tool_args)
    if result.is_error:
        print(f"TOOL ERROR: {result.content}")
        return 1
    print(json.dumps(json.loads(result.content), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triagebot")
    parser.add_argument("--config", default="config.toml", help="path to config.toml")
    parser.add_argument("--backend", choices=BACKENDS, help="override the backend in config")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ping", help="make one tool-calling round trip to the model").set_defaults(
        func=cmd_ping
    )
    sub.add_parser("tools", help="list the tools and their argument schemas").set_defaults(
        func=cmd_tools
    )
    sub.add_parser("index", help="build the search index from GitHub issues and docs").set_defaults(
        func=cmd_index
    )
    tool = sub.add_parser("tool", help="run one tool, e.g. triagebot tool get_issue issue_number=1")
    tool.add_argument("name")
    tool.add_argument("args", nargs="*", help="key=value pairs")
    tool.set_defaults(func=cmd_tool)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, LLMError, GitHubError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
