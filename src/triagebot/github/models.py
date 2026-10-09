"""Typed subsets of GitHub API objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Label:
    name: str
    description: str

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> Label:
        return cls(name=d["name"], description=d.get("description") or "")


@dataclass
class Comment:
    author: str
    body: str
    id: int = 0

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> Comment:
        return cls(
            author=(d.get("user") or {}).get("login", "ghost"),
            body=d.get("body") or "",
            id=d.get("id", 0),
        )


@dataclass
class Issue:
    number: int
    title: str
    body: str
    state: str
    author: str
    labels: list[str]
    comment_count: int
    created_at: str
    is_pull_request: bool
    comments: list[Comment] = field(default_factory=list)

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> Issue:
        return cls(
            number=d["number"],
            title=d["title"],
            body=d.get("body") or "",
            state=d["state"],
            author=(d.get("user") or {}).get("login", "ghost"),
            labels=[lbl["name"] for lbl in d.get("labels", [])],
            comment_count=d.get("comments", 0),
            created_at=d.get("created_at", ""),
            is_pull_request="pull_request" in d,
        )
