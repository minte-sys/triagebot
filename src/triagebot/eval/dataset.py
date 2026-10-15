"""The eval set: test issues with the answers a maintainer would give, plus the older
issues and docs the agent can search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from triagebot.github.models import Comment, Issue, Label
from triagebot.search.docs import DocChunk, load_docs


class DatasetError(Exception):
    """The eval data is missing or inconsistent."""


class Expected(BaseModel):
    labels: list[str] = Field(default_factory=list)
    # Defensible either way: neither rewarded nor counted as wrong
    optional_labels: list[str] = Field(default_factory=list)
    duplicate_of: int = 0
    # None: a maintainer could reasonably go either way, so it isn't scored
    needs_info: bool | None = False
    # A doc the comment should point to
    doc: str | None = None


class Forbidden(BaseModel):
    """What a prompt injection tries to make the agent do."""

    labels: list[str] = Field(default_factory=list)
    duplicate_of: list[int] = Field(default_factory=list)
    text: list[str] = Field(default_factory=list)


class CaseComment(BaseModel):
    author: str
    body: str


class Case(BaseModel):
    number: int
    kind: str
    title: str
    body: str
    comments: list[CaseComment] = Field(default_factory=list)
    expect: Expected
    forbid: Forbidden = Field(default_factory=Forbidden)

    def issue(self) -> Issue:
        comments = [Comment(c.author, c.body) for c in self.comments]
        return Issue(self.number, self.title, self.body, "open", f"user{self.number}", [],
                     len(comments), "", False, comments)


class SearchQuery(BaseModel):
    """A user's description of an older issue in their own words, for the search comparison."""

    query: str
    duplicate_of: int


@dataclass
class Dataset:
    cases: list[Case]
    background: list[Issue]
    labels: list[Label]
    docs: list[DocChunk]
    search_queries: list[SearchQuery]


def load_dataset(folder: Path) -> Dataset:
    if not folder.is_dir():
        raise DatasetError(f"Eval data folder not found: {folder.resolve()}")
    try:
        cases = [Case.model_validate(row) for row in _jsonl(folder / "cases.jsonl")]
        background = [
            Issue(r["number"], r["title"], r["body"], r["state"], "maintainer", r["labels"],
                  0, "", False)
            for r in _jsonl(folder / "background.jsonl")
        ]
        labels = [Label(d["name"], d["description"])
                  for d in json.loads((folder / "labels.json").read_text(encoding="utf-8"))]
        queries = [SearchQuery.model_validate(r) for r in _jsonl(folder / "search_queries.jsonl")]
    except (OSError, KeyError, json.JSONDecodeError, ValidationError) as exc:
        raise DatasetError(f"Could not read the eval data in {folder}: {exc}") from exc
    dataset = Dataset(cases, background, labels, load_docs(folder / "docs"), queries)
    _check(dataset, folder)
    return dataset


def _jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _check(ds: Dataset, folder: Path) -> None:
    """Catch typos in the answers before they quietly lower a score."""
    names = {lb.name for lb in ds.labels}
    older = {i.number for i in ds.background}
    seen: set[int] = set()
    for case in ds.cases:
        where = f"case #{case.number}"
        if case.number in seen or case.number in older:
            raise DatasetError(f"{where}: the number is used twice")
        seen.add(case.number)
        e, f = case.expect, case.forbid
        unknown = set(e.labels + e.optional_labels + f.labels) - names
        if unknown:
            raise DatasetError(f"{where}: unknown labels {sorted(unknown)}")
        if e.duplicate_of and e.duplicate_of not in older:
            raise DatasetError(f"{where}: duplicate_of #{e.duplicate_of} is not an older issue")
        if e.doc and not (folder / "docs" / e.doc).is_file():
            raise DatasetError(f"{where}: doc {e.doc} not found")
    for q in ds.search_queries:
        if q.duplicate_of not in older:
            raise DatasetError(f"search query {q.query!r}: #{q.duplicate_of} is not an older issue")
