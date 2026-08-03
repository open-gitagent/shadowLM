"""Seeds — what the user brings to the synthesizer.

Three ways in, one shape out. Describe the task in plain English, point at a
document to ground on, or hand over a few real episodes to amplify; `resolve_seed`
normalizes all three into a `Seed` the generator reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..data import Dataset
from ..rl import Trajectory

# Documents we can read as-is. Anything else is the user's to convert — silently
# extracting text from a PDF would be a whole dependency and a lossy guess.
_TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".rst")
_MAX_PATHLIKE = 4096  # beyond this a string is content, not a filename


@dataclass
class Seed:
    """Normalized synthesis input."""

    kind: str  # "task" | "document" | "episodes"
    task: str | None = None
    chunks: list[str] = field(default_factory=list)      # document passages
    exemplars: list[Trajectory] = field(default_factory=list)

    def context(self) -> str:
        """The task description every generation prompt is written against."""
        if self.task:
            return self.task
        if self.kind == "document":
            return ("Answer questions about the source material accurately and "
                    "concisely, using only what it states.")
        return "Perform the task demonstrated by the example conversations."


def chunk_text(text: str, *, target_chars: int = 2000,
               overlap_chars: int = 200) -> list[str]:
    """Split prose into ~`target_chars` passages on paragraph boundaries.

    Each chunk carries the tail of the previous one so a fact spanning the seam
    is still stated whole somewhere.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail}\n\n{para}" if tail else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def resolve_seed(*, task=None, document=None, episodes=None) -> Seed:
    """Normalize the caller's starting point into a `Seed`."""
    if document is not None:
        chunks = chunk_text(_document_text(document))
        if not chunks:
            raise ValueError("document= is empty — nothing to ground on")
        return Seed(kind="document", task=task, chunks=chunks)
    if episodes is not None:
        exemplars = _as_trajectories(episodes)
        if not exemplars:
            raise ValueError(
                "episodes= held no usable conversations (each needs a "
                "'messages' list)")
        return Seed(kind="episodes", task=task, exemplars=exemplars)
    if task:
        return Seed(kind="task", task=task)
    raise ValueError(
        "synthesize needs a seed: task='what the model should learn', "
        "document='notes.md', or episodes=[...]")


def _document_text(document) -> str:
    """The document's text — read from a path, or taken as the content itself."""
    if isinstance(document, (str, Path)) and len(str(document)) < _MAX_PATHLIKE:
        path = Path(document)
        if path.exists():
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                raise ValueError(
                    f"unsupported document type {path.suffix!r} — convert it to "
                    f".txt or .md first (supported: {', '.join(_TEXT_SUFFIXES)})")
            return path.read_text()
    return str(document)


def _as_trajectories(episodes) -> list[Trajectory]:
    """Episodes from anywhere — trajectories, a Dataset, rows, or a file."""
    if isinstance(episodes, (str, Path)):
        from .. import traces  # noqa: PLC0415 — avoids an import cycle

        path = Path(episodes)
        if path.suffix.lower() in (".jsonl", ".ndjson", ".csv", ".parquet"):
            return _rows_to_trajectories(Dataset.load(path).as_chat().rows)
        return traces.from_otlp(path)
    if isinstance(episodes, Dataset):
        return _rows_to_trajectories(episodes.as_chat().rows)
    items = list(episodes)
    if items and isinstance(items[0], Trajectory):
        return items
    return _rows_to_trajectories(items)


def _rows_to_trajectories(rows) -> list[Trajectory]:
    return [Trajectory(messages=r["messages"], tools=r.get("tools"))
            for r in rows if r.get("messages")]
