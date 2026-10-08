"""Read-only tools. The repository is bound at construction, never taken from the model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from triagebot.github.client import GitHubClient
from triagebot.tools.registry import Tool

MAX_BODY_CHARS = 3000


class ListOpenIssuesArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=100, description="How many issues to return")


class GetIssueArgs(BaseModel):
    issue_number: int = Field(ge=1, description="Issue number")


class NoArgs(BaseModel):
    pass


def _truncate(text: str, limit: int = MAX_BODY_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[... truncated {len(text) - limit} chars]"


def build_read_tools(gh: GitHubClient, repo: str) -> list[Tool]:
    def list_open_issues(args: ListOpenIssuesArgs) -> list[dict]:
        return [
            {"number": i.number, "title": i.title, "labels": i.labels, "comments": i.comment_count}
            for i in gh.list_open_issues(repo, args.limit)
        ]

    def get_issue(args: GetIssueArgs) -> dict:
        issue = gh.get_issue(repo, args.issue_number)
        if issue.is_pull_request:
            return {"error": f"#{issue.number} is a pull request, not an issue"}
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "author": issue.author,
            "labels": issue.labels,
            "body": _truncate(issue.body),
            "comments": [{"author": c.author, "body": _truncate(c.body, 1000)} for c in issue.comments],
        }

    def list_labels(args: NoArgs) -> list[dict]:
        return [{"name": lb.name, "description": lb.description} for lb in gh.list_labels(repo)]

    return [
        Tool("list_open_issues", "List open issues in the repository (newest first).",
             ListOpenIssuesArgs, list_open_issues),
        Tool("get_issue", "Get one issue's title, body, labels and recent comments.",
             GetIssueArgs, get_issue),
        Tool("list_labels", "List every label that exists in the repository, with descriptions.",
             NoArgs, list_labels),
    ]
