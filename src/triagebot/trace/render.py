"""Read runs back from SQLite and format them for the terminal."""

from __future__ import annotations

import json
import sqlite3

from triagebot.trace.recorder import format_event


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> str:
    rows = conn.execute(
        "SELECT id, issue_number, model, status, steps, tokens_in, tokens_out, started_at"
        " FROM runs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return "No runs yet. Try: triagebot run <issue number>"
    lines = [f"{'run':>4}  {'issue':>5}  {'status':<16} {'steps':>5}  {'tokens':>11}  started"]
    for r in rows:
        tokens = f"{r['tokens_in']}/{r['tokens_out']}"
        lines.append(
            f"{r['id']:>4}  #{r['issue_number']:<4}  {r['status']:<16} {r['steps']:>5}"
            f"  {tokens:>11}  {r['started_at']}  {r['model']}"
        )
    return "\n".join(lines)


def render_run(conn: sqlite3.Connection, run_id: int) -> str:
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        return f"No run with id {run_id}"
    lines = [
        f"Run {run['id']}: issue #{run['issue_number']} in {run['repo']}",
        f"model {run['model']}, prompt {run['prompt_version']}, tool mode {run['tool_mode']}",
        f"status {run['status']}"
        + (f" ({run['stop_reason']})" if run["stop_reason"] else "")
        + f", {run['steps']} model calls, tokens in/out {run['tokens_in']}/{run['tokens_out']}",
        "",
    ]
    events = conn.execute(
        "SELECT * FROM trace_events WHERE run_id = ? ORDER BY seq", (run_id,)
    ).fetchall()
    for e in events:
        lines.append(format_event(e["seq"], e["kind"], e["name"], e["args_json"], e["result"],
                                  bool(e["is_error"]), e["tokens_in"], e["tokens_out"],
                                  e["latency_ms"]))
    if run["decision_json"]:
        lines += ["", "Decision:", json.dumps(json.loads(run["decision_json"]), indent=2)]
    return "\n".join(lines)
