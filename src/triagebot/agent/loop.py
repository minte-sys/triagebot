"""The agent loop: ask the model, run the tools it asks for, repeat until finish or budget."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from triagebot.agent.budget import Budget
from triagebot.agent.decision import FINISH, Decision, validate_decision
from triagebot.agent.json_mode import json_instructions, parse_tool_call
from triagebot.agent.prompts import PROMPT_VERSION, issue_prompt, system_prompt
from triagebot.github.models import Issue, Label
from triagebot.llm.base import LLMClient, LLMError, Message, ToolCall
from triagebot.tools.registry import ToolRegistry, ToolResult
from triagebot.trace.recorder import Recorder

ToolMode = Literal["native", "json"]
NUDGE = "Use a tool, or call finish with your decision if you are done."


@dataclass
class RunResult:
    run_id: int
    status: str  # finished | budget_exceeded | error
    stop_reason: str
    decision: Decision | None
    steps: int
    tokens_in: int
    tokens_out: int


def run_agent(
    client: LLMClient,
    registry: ToolRegistry,
    recorder: Recorder,
    repo: str,
    issue: Issue,
    labels: list[Label],
    budget: Budget,
    tool_mode: ToolMode = "native",
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    specs = [*registry.specs(), FINISH]
    system = system_prompt(repo)
    if tool_mode == "json":
        system += "\n\n" + json_instructions(specs)
    messages = [Message("system", system), Message("user", issue_prompt(issue, labels))]
    repo_labels = {lb.name for lb in labels}

    run_id = recorder.start_run(repo, issue.number, client.model, PROMPT_VERSION, tool_mode)
    started = clock()
    steps = tokens_in = tokens_out = 0

    def end(status: str, reason: str, decision: Decision | None = None) -> RunResult:
        recorder.end_run(run_id, status, reason, steps, tokens_in, tokens_out,
                         decision.model_dump() if decision else None)
        return RunResult(run_id, status, reason, decision, steps, tokens_in, tokens_out)

    while True:
        reason = budget.exceeded(steps, clock() - started, tokens_in + tokens_out)
        if reason:
            return end("budget_exceeded", reason)

        try:
            resp = client.chat(messages, tools=specs if tool_mode == "native" else None)
        except LLMError as exc:
            recorder.event(run_id, "error", "llm", result=str(exc), is_error=True)
            return end("error", str(exc))
        steps += 1
        tokens_in += resp.usage.input_tokens
        tokens_out += resp.usage.output_tokens

        calls = resp.tool_calls if tool_mode == "native" else []
        if not calls:
            parsed = parse_tool_call(resp.text)
            calls = [parsed] if parsed else []
        summary = ", ".join(f"{c.name}({json.dumps(c.arguments)})" for c in calls)
        recorder.event(run_id, "llm", client.model, result=f"calls {summary}" if calls
                       else f"text: {resp.text}", tokens_in=resp.usage.input_tokens,
                       tokens_out=resp.usage.output_tokens, latency_ms=resp.latency_ms)

        native = tool_mode == "native" and bool(resp.tool_calls)
        messages.append(resp.to_message() if native else Message("assistant", resp.text))
        if not calls:
            messages.append(Message("user", NUDGE))
            continue

        for call in calls:
            if call.name == FINISH.name:
                decision, error = validate_decision(call.arguments, issue.number, repo_labels)
                recorder.event(run_id, "tool", call.name, call.arguments,
                               result=error or "accepted", is_error=bool(error))
                if decision:
                    return end("finished", "", decision)
                result = ToolResult(error, is_error=True)
            else:
                result = registry.execute(call.name, call.arguments)
                recorder.event(run_id, "tool", call.name, call.arguments, result=result.content,
                               is_error=result.is_error)
            messages.append(_result_message(call, result, native))


def _result_message(call: ToolCall, result: ToolResult, native: bool) -> Message:
    content = f"Error: {result.content}" if result.is_error else result.content
    if native:
        return Message("tool", content, tool_call_id=call.id, name=call.name)
    return Message("user", f"Result of {call.name}:\n{content}")
