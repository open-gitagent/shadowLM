"""Datasets — the first stage of the pipeline.

A `Dataset` is just rows plus a detected *format*. The backends know how to turn a
formatted dataset into training text (applying chat templates, instruction
prompts, etc). Loading from local files is pure-stdlib; `from_hf` lazy-imports the
`datasets` library so the core SDK stays dependency-free.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Recognised dataset shapes, in priority order of detection.
CHAT = "chat"  # rows like {"messages": [{"role": ..., "content": ...}, ...]}
INSTRUCTION = "instruction"  # alpaca-style {"instruction", "input", "output"}
TEXT = "text"  # rows like {"text": "..."}
RAW = "raw"  # anything else — caller must map it

_ALPACA = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def _detect_format(rows: list[dict]) -> str:
    if not rows:
        return RAW
    keys = set(rows[0])
    if "messages" in keys:
        return CHAT
    if "instruction" in keys and "output" in keys:
        return INSTRUCTION
    if "text" in keys:
        return TEXT
    return RAW


@dataclass
class Dataset:
    """An in-memory training dataset with a detected format.

    Build one with `from_jsonl` / `from_csv` / `from_hf` / `from_list`, optionally
    normalise it with `as_chat()`, and hand it to `model.finetune(...)`.
    """

    rows: list[dict] = field(default_factory=list)
    format: str = RAW
    name: str | None = None
    source: str | None = None

    # ---- constructors -----------------------------------------------------
    @classmethod
    def from_list(cls, rows: list[dict], *, name: str | None = None) -> "Dataset":
        rows = list(rows)
        return cls(rows=rows, format=_detect_format(rows), name=name, source="list")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return cls(rows=rows, format=_detect_format(rows), name=path.stem, source=str(path))

    @classmethod
    def from_json(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("data", [])
        return cls(rows=rows, format=_detect_format(rows), name=path.stem, source=str(path))

    @classmethod
    def from_csv(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        return cls(rows=rows, format=_detect_format(rows), name=path.stem, source=str(path))

    @classmethod
    def from_hf(cls, repo: str, *, split: str = "train", token: str | None = None) -> "Dataset":
        try:
            from datasets import load_dataset  # noqa: PLC0415  (lazy, optional dep)
        except ImportError as e:  # pragma: no cover - exercised only with [torch] extra
            raise ImportError(
                "Dataset.from_hf needs the 'datasets' library: pip install shadowlm[torch]"
            ) from e
        ds = load_dataset(repo, split=split, token=token)
        rows = [dict(r) for r in ds]
        return cls(rows=rows, format=_detect_format(rows), name=repo, source=f"hf:{repo}")

    # ---- transforms -------------------------------------------------------
    def as_chat(self) -> "Dataset":
        """Normalise to chat format: every row becomes {"messages": [...]}.

        Instruction/text rows are lifted into a single user/assistant exchange so
        downstream chat-template logic has one shape to handle.
        """
        if self.format == CHAT:
            return self
        out: list[dict] = []
        for r in self.rows:
            if self.format == INSTRUCTION:
                user = r.get("instruction", "")
                if r.get("input"):
                    user = f"{user}\n\n{r['input']}"
                out.append({"messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": r.get("output", "")},
                ]})
            elif self.format == TEXT:
                out.append({"messages": [{"role": "assistant", "content": r.get("text", "")}]})
            else:
                raise ValueError(f"cannot convert {self.format!r} rows to chat automatically")
        return Dataset(rows=out, format=CHAT, name=self.name, source=self.source)

    def to_texts(self) -> list[str]:
        """Render rows to plain training strings (no tokenizer / chat template).

        Used as a fallback for the raw format; the mlx and torch backends prefer the
        tokenizer's chat template instead.
        """
        if self.format == TEXT:
            return [r["text"] for r in self.rows]
        if self.format == INSTRUCTION:
            return [_ALPACA.format(
                instruction=r.get("instruction", ""),
                input=r.get("input", ""),
                output=r.get("output", ""),
            ) for r in self.rows]
        if self.format == CHAT:
            texts = []
            for r in self.rows:
                turns = [f"{m['role']}: {m['content']}" for m in r.get("messages", [])]
                texts.append("\n".join(turns))
            return texts
        raise ValueError(f"don't know how to render {self.format!r} rows to text")

    def split(self, test_size: float | int = 0.1, *, seed: int = 0,
              shuffle: bool = True) -> tuple["Dataset", "Dataset"]:
        """Split into `(train, eval)` datasets.

        test_size is a fraction in (0, 1) or an absolute row count. Shuffles with
        `seed` by default so the split is held-out but reproducible.
        """
        import random as _random  # noqa: PLC0415

        rows = list(self.rows)
        if shuffle:
            _random.Random(seed).shuffle(rows)
        n = len(rows)
        k = test_size if isinstance(test_size, int) else int(round(n * test_size))
        k = max(1, min(k, n - 1)) if n > 1 else 0

        def _make(rs: list[dict], suffix: str) -> "Dataset":
            name = f"{self.name}[{suffix}]" if self.name else None
            return Dataset(rows=rs, format=self.format, name=name, source=self.source)

        return _make(rows[k:], "train"), _make(rows[:k], "eval")

    # ---- dunder niceties --------------------------------------------------
    def head(self, n: int = 5) -> list[dict]:
        return self.rows[:n]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]

    def __repr__(self) -> str:
        name = self.name or "dataset"
        return f"Dataset({name!r}, format={self.format!r}, rows={len(self.rows)})"
