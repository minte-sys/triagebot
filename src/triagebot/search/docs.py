"""Split markdown files into sections for indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


@dataclass
class DocChunk:
    path: str
    heading: str
    text: str


def chunk_markdown(path: str, text: str) -> list[DocChunk]:
    """One chunk per heading (levels 1-3). Lines inside code fences are never headings."""
    chunks: list[DocChunk] = []
    heading = Path(path).stem
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            chunks.append(DocChunk(path, heading, body))

    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            flush()
            heading, lines = match.group(2), []
        else:
            lines.append(line)
    flush()
    return chunks


def load_docs(docs_dir: Path) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for file in sorted(docs_dir.rglob("*.md")):
        rel = file.relative_to(docs_dir).as_posix()
        chunks.extend(chunk_markdown(rel, file.read_text(encoding="utf-8")))
    return chunks
