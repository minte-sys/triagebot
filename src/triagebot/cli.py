"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from triagebot.agent.budget import Budget
from triagebot.agent.loop import run_agent
from triagebot.agent.prompts import PROMPTS
from triagebot.autonomy.corrections import CorrectionStore, format_corrections
from triagebot.autonomy.executor import Executor
from triagebot.autonomy.gate import Gate
from triagebot.autonomy.ledger import Ledger
from triagebot.autonomy.proposals import ACTION_TYPES, ProposalStore
from triagebot.autonomy.review import describe, review
from triagebot.autonomy.undo import UndoError, undo
from triagebot.config import BACKENDS, TOOL_MODES, Config, ConfigError, load_config
from triagebot.db import connect
from triagebot.eval.dataset import DatasetError, load_dataset
from triagebot.eval.report import format_report
from triagebot.eval.runner import ResultStore, Setup, run_eval
from triagebot.eval.search_eval import compare_search, format_search_report
from triagebot.github import make_github
from triagebot.github.client import GitHubError
from triagebot.llm import make_client
from triagebot.llm.base import LLMClient, LLMError, Message, ToolSpec
from triagebot.llm.retry import RetryingClient, RetryPolicy
from triagebot.search.docs import load_docs
from triagebot.search.embeddings import CachedEmbedder, RetryingEmbedder, make_embedder
from triagebot.search.keyword import SearchIndex
from triagebot.search.semantic import SemanticIssueSearch, issue_text
from triagebot.tools.read_tools import build_read_tools
from triagebot.tools.registry import ToolRegistry
from triagebot.tools.search_tools import build_search_tools
from triagebot.trace.recorder import Recorder
from triagebot.trace.render import list_runs, render_run

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


def _policy(cfg: Config) -> RetryPolicy:
    return RetryPolicy(cfg.llm.max_retries, max_wait_s=cfg.llm.retry_max_wait_s)


def _client(cfg: Config, model: str | None = None) -> LLMClient:
    llm = replace(cfg.llm, model=model) if model else cfg.llm
    return RetryingClient(make_client(llm), _policy(cfg), say=print)


def _embedder(cfg: Config, conn) -> CachedEmbedder:
    if cfg.embeddings is None:
        raise ConfigError("No base_url for the embeddings backend in the [llm] section")
    return CachedEmbedder(RetryingEmbedder(make_embedder(cfg.embeddings), _policy(cfg), print),
                          conn)


def _semantic(cfg: Config, index: SearchIndex) -> SemanticIssueSearch | None:
    if cfg.search.issue_search != "embeddings":
        return None
    return SemanticIssueSearch(index, _embedder(cfg, index.conn))


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
    semantic = _semantic(cfg, index)
    if semantic:
        embedder = semantic.embedder
        try:
            embedder.embed([issue_text(i.title, i.body) for i in issues])
            print(f"Embedded the issues with {embedder.model}: {embedder.new} new, "
                  f"{embedder.cached} from cache")
        except LLMError as exc:
            print(f"Could not embed the issues ({exc}). Search falls back to keywords for now.")

    docs_dir = cfg.search.docs_dir
    if docs_dir is None:
        print("No search.docs_dir in config; skipping docs")
    elif not docs_dir.is_dir():
        print(f"Docs folder not found: {docs_dir.resolve()}; skipping docs")
    else:
        print(f"Indexed {index.index_docs(load_docs(docs_dir))} doc sections from {docs_dir}")
    print(f"Database: {cfg.search.db_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Triage one issue and store its decision as proposals. Nothing is posted here."""
    cfg = load_config(args.config, backend=args.backend)
    repo = cfg.github.repo
    gh = make_github(cfg.github)
    issue = gh.get_issue(repo, args.issue)
    if issue.is_pull_request:
        print(f"#{issue.number} is a pull request, not an issue")
        return 1

    conn = connect(cfg.search.db_path)
    index = SearchIndex(conn, repo)
    if index.issue_count() == 0:
        print("Warning: the search index is empty. Run `triagebot index` first.")
    get_issue = [t for t in build_read_tools(gh, repo) if t.name == "get_issue"]
    semantic = _semantic(cfg, index)
    registry = ToolRegistry(get_issue + build_search_tools(index, issue.number, semantic))
    budget = Budget(cfg.agent.max_steps, cfg.agent.max_seconds, cfg.agent.max_tokens)
    mode = args.tool_mode or cfg.agent.tool_mode
    labels = gh.list_labels(repo)
    past = CorrectionStore(conn, repo).recent()

    if past:
        print(f"Including {len(past)} past corrections in the prompt")
    models = [cfg.llm.model, *cfg.llm.fallback_models]
    for n, model in enumerate(models):
        print(f"Triaging #{issue.number} \"{issue.title}\" with {model} ({mode} tool calls)")
        result = run_agent(_client(cfg, model), registry, Recorder(conn, echo=print), repo,
                           issue, labels, budget, mode, corrections=format_corrections(past),
                           prompt_version=cfg.agent.prompt_version)
        print(f"\nRun {result.run_id}: {result.status}"
              + (f" ({' '.join(result.stop_reason.split())[:200]})" if result.stop_reason
                 else "")
              + f", {result.steps} model calls, tokens in/out "
              f"{result.tokens_in}/{result.tokens_out}")
        print(f"Full trace: triagebot trace {result.run_id}")
        if not (result.model_unavailable and n + 1 < len(models)):
            break
        print(f"\n{model} is unavailable. Starting again with the fallback model {models[n + 1]}.")
    if semantic and semantic.last_error:
        print(f"Note: embeddings failed, so issue search used keywords ({semantic.last_error[:100]})")
    if not result.decision:
        return 1

    store = ProposalStore(conn, repo)
    proposals = store.create_from_decision(result.run_id, issue.number, result.decision,
                                           issue.labels)
    print(f"\nReasoning: {result.decision.reasoning}")
    if not proposals:
        print("No actions proposed.")
        return 0
    executor = Executor(gh, conn, repo, {lb.name for lb in labels}, dry_run=cfg.safety.dry_run)
    gate = Gate(store, Ledger(conn, repo, cfg.trust), executor, cfg.trust.auto_enabled)
    routed = gate.route(proposals)
    for r in routed:
        p = r.proposal
        print(f"Proposal {p.id} ({p.action_type}): {describe(p, p.payload)}")
        print(f"      -> {r.message}")
    if any(r.route in ("review", "audit") for r in routed):
        print("Review them with: triagebot review")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Approve, edit or reject pending proposals. Writes to GitHub only if dry_run is off."""
    cfg = load_config(args.config, backend=args.backend)
    repo = cfg.github.repo
    gh = make_github(cfg.github)
    conn = connect(cfg.search.db_path)
    labels = {lb.name for lb in gh.list_labels(repo)}
    executor = Executor(gh, conn, repo, labels, dry_run=cfg.safety.dry_run)
    try:
        review(ProposalStore(conn, repo), executor, ledger=Ledger(conn, repo, cfg.trust),
               corrections=CorrectionStore(conn, repo))
    except (EOFError, KeyboardInterrupt):
        print("\nStopped. Decisions made so far are saved.")
    return 0


def cmd_undo(args: argparse.Namespace) -> int:
    """Reverse what an executed proposal wrote to GitHub, and count it against trust."""
    cfg = load_config(args.config, backend=args.backend)
    repo = cfg.github.repo
    conn = connect(cfg.search.db_path)
    store = ProposalStore(conn, repo)
    proposal = store.get(args.proposal_id)
    if proposal is None:
        print(f"No proposal {args.proposal_id} for {repo}")
        return 1
    try:
        result = undo(proposal, make_github(cfg.github), store, Ledger(conn, repo, cfg.trust),
                      CorrectionStore(conn, repo), cfg.safety.dry_run, args.reason)
    except UndoError as exc:
        print(exc)
        return 1
    for step in result.steps:
        print(f"{'Would' if result.dry_run else 'Done:'} {step}")
    if result.dry_run:
        print("Dry run: nothing changed. Set dry_run = false in config.toml to undo for real.")
    if result.mode_change:
        print(f"Trust: {result.mode_change}")
    return 0


def cmd_trust(args: argparse.Namespace) -> int:
    """Show what each action type has earned, or demote one by hand."""
    cfg = load_config(args.config, backend=args.backend)
    ledger = Ledger(connect(cfg.search.db_path), cfg.github.repo, cfg.trust)
    if args.demote:
        change = ledger.demote(args.demote)
        print(f"Trust: {change}" if change else f"{args.demote} is already in propose mode")
        return 0
    if args.history:
        for row in ledger.history(args.history):
            print(f"{row['created_at']}  {row['outcome']:<9} {row['source']:<7} {row['note']}")
        return 0

    t = cfg.trust
    print(f"Promote after {t.min_decisions} decisions with {t.min_approval_pct}% of the last "
          f"{t.window} approved unchanged.")
    print(f"In auto mode, hold 1 in every {t.audit_every} actions for an audit; demote below "
          f"{t.min_audit_accuracy_pct}% of the last {t.audit_window} audits, or on any undo.")
    if not t.auto_enabled:
        print("Auto mode is off in config ([trust] auto_enabled = false), so everything is "
              "reviewed.")
    print(f"\n{'action':<14}{'mode':<9}{'decisions':>10}  {'approved':>13}  {'audits passed':>13}")
    for s in ledger.status():
        approved = (f"{s.approved}/{s.in_window} ({100 * s.approved // s.in_window}%)"
                    if s.in_window else "-")
        audits = f"{s.audits_passed}/{s.audits}" if s.audits else "-"
        print(f"{s.action_type:<14}{s.mode:<9}{s.decisions:>10}  {approved:>13}  {audits:>13}")
    print("\nCounts start again from zero whenever an action type changes mode.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Triage the eval set and score the decisions. Nothing is written to GitHub."""
    cfg = load_config(args.config, backend=args.backend)
    ds = load_dataset(Path(args.config).parent / args.data)
    conn = connect(cfg.search.db_path)
    store = ResultStore(conn)
    setup = Setup(cfg.llm.model, args.prompt or cfg.agent.prompt_version,
                  args.tool_mode or cfg.agent.tool_mode)
    if args.restart:
        print(f"Deleted {store.clear(setup)} earlier results for {setup}")
    if not args.report:
        budget = Budget(cfg.agent.max_steps, cfg.agent.max_seconds, cfg.agent.max_tokens)
        run_eval(_client(cfg), ds, store, Recorder(conn), setup, budget,
                 args.limit, args.pause)
        print()
    print(format_report(ds, store.outcomes(ds.cases), setup))
    return 0


def cmd_eval_search(args: argparse.Namespace) -> int:
    """Compare BM25, embeddings and both combined at finding the eval set's duplicates."""
    cfg = load_config(args.config, backend=args.backend)
    ds = load_dataset(Path(args.config).parent / args.data)
    embedder = _embedder(cfg, connect(cfg.search.db_path))
    groups = compare_search(ds, embedder)
    print(f"Embeddings from {embedder.model}: {embedder.new} new, {embedder.cached} from cache\n")
    print(format_search_report(groups, args.details))
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    """Show one run step by step, or list recent runs."""
    cfg = load_config(args.config, backend=args.backend)
    conn = connect(cfg.search.db_path)
    print(render_run(conn, args.run_id) if args.run_id else list_runs(conn))
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
    run = sub.add_parser("run", help="triage one issue and create proposals (posts nothing)")
    run.add_argument("issue", type=int, help="issue number")
    run.add_argument("--tool-mode", choices=TOOL_MODES, help="override agent.tool_mode")
    run.set_defaults(func=cmd_run)
    sub.add_parser("review", help="approve, edit or reject pending proposals").set_defaults(
        func=cmd_review
    )
    undo_cmd = sub.add_parser("undo", help="reverse what an executed proposal wrote to GitHub")
    undo_cmd.add_argument("proposal_id", type=int)
    undo_cmd.add_argument("--reason", default="", help="why it was wrong; the bot learns from it")
    undo_cmd.set_defaults(func=cmd_undo)
    trust = sub.add_parser("trust", help="show the trust each action type has earned")
    trust.add_argument("--demote", choices=ACTION_TYPES, help="send an action type back to review")
    trust.add_argument("--history", choices=ACTION_TYPES, help="show recent decisions")
    trust.set_defaults(func=cmd_trust)
    ev = sub.add_parser("eval", help="triage the eval set and score it (writes nothing)")
    ev.add_argument("--limit", type=int, help="run at most this many new cases")
    ev.add_argument("--prompt", choices=list(PROMPTS), help="override agent.prompt_version")
    ev.add_argument("--tool-mode", choices=TOOL_MODES, help="override agent.tool_mode")
    ev.add_argument("--pause", type=float, default=0, help="seconds to wait between cases")
    ev.add_argument("--report", action="store_true", help="only print the scores, run nothing")
    ev.add_argument("--restart", action="store_true",
                    help="delete this setup's results first and start again")
    ev.add_argument("--data", default="eval_data", help="eval data folder, next to config.toml")
    ev.set_defaults(func=cmd_eval)
    es = sub.add_parser("eval-search", help="compare BM25 and embeddings at finding duplicates")
    es.add_argument("--details", action="store_true", help="show the result for every query")
    es.add_argument("--data", default="eval_data", help="eval data folder, next to config.toml")
    es.set_defaults(func=cmd_eval_search)
    trace = sub.add_parser("trace", help="show a run step by step, or list recent runs")
    trace.add_argument("run_id", type=int, nargs="?")
    trace.set_defaults(func=cmd_trace)
    tool = sub.add_parser("tool", help="run one tool, e.g. triagebot tool get_issue issue_number=1")
    tool.add_argument("name")
    tool.add_argument("args", nargs="*", help="key=value pairs")
    tool.set_defaults(func=cmd_tool)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, LLMError, GitHubError, DatasetError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
