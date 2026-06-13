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
CHAT = "chat"  # ChatML-style: {"messages": [{"role": ..., "content": ...}, ...]}
SHAREGPT = "sharegpt"  # {"conversations": [{"from": "human"|"gpt", "value": ...}]}
PREFERENCE = "preference"  # DPO pairs: {"prompt", "chosen", "rejected"[, "system"]}
INSTRUCTION = "instruction"  # alpaca-style {"instruction", "input", "output"}
TEXT = "text"  # rows like {"text": "..."} (raw text / CPT)
RAW = "raw"  # anything else — caller must map it

FORMATS = (CHAT, SHAREGPT, PREFERENCE, INSTRUCTION, TEXT, RAW)

_ALPACA = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

# ShareGPT speaker tags → chat roles.
_SHAREGPT_ROLES = {"human": "user", "gpt": "assistant", "system": "system",
                   "user": "user", "assistant": "assistant", "tool": "tool"}


# Instruction/QA datasets name their prompt and response columns many ways
# (alpaca instruction/output, gsm8k question/answer, prompt/response, …).
_PROMPT_KEYS = ("instruction", "question", "prompt", "query", "input")
_RESPONSE_KEYS = ("output", "answer", "response", "completion", "target")


def _instruction_cols(keys) -> tuple[str, str] | None:
    """The (prompt, response) column names for an instruction-style row, if any."""
    prompt = next((k for k in _PROMPT_KEYS if k in keys), None)
    response = next((k for k in _RESPONSE_KEYS if k in keys), None)
    return (prompt, response) if prompt and response else None


def _detect_format(rows: list[dict]) -> str:
    if not rows:
        return RAW
    keys = set(rows[0])
    if "messages" in keys:
        return CHAT
    if "conversations" in keys:
        return SHAREGPT
    if "chosen" in keys and "rejected" in keys:
        return PREFERENCE
    if _instruction_cols(keys):
        return INSTRUCTION
    if "text" in keys:
        return TEXT
    return RAW


def _resolve_format(rows: list[dict], override: str | None) -> str:
    """Detected format, or the caller's explicit target format."""
    if override is None or override == "auto":
        return _detect_format(rows)
    if override not in FORMATS:
        raise ValueError(f"unknown format {override!r} (expected one of {', '.join(FORMATS)})")
    return override


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
    # Every constructor accepts format= ("auto" default) to override detection —
    # the "Target Format" knob: chat | sharegpt | instruction | text | raw.
    @classmethod
    def from_list(cls, rows: list[dict], *, name: str | None = None,
                  format: str | None = None) -> "Dataset":
        rows = list(rows)
        return cls(rows=rows, format=_resolve_format(rows, format), name=name, source="list")

    @classmethod
    def from_jsonl(cls, path: str | Path, *, format: str | None = None) -> "Dataset":
        path = Path(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return cls(rows=rows, format=_resolve_format(rows, format), name=path.stem, source=str(path))

    @classmethod
    def from_json(cls, path: str | Path, *, format: str | None = None) -> "Dataset":
        path = Path(path)
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("data", [])
        return cls(rows=rows, format=_resolve_format(rows, format), name=path.stem, source=str(path))

    @classmethod
    def from_csv(cls, path: str | Path, *, format: str | None = None) -> "Dataset":
        path = Path(path)
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        return cls(rows=rows, format=_resolve_format(rows, format), name=path.stem, source=str(path))

    @classmethod
    def from_parquet(cls, path: str | Path, *, format: str | None = None) -> "Dataset":
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415  (lazy, optional dep)
        except ImportError as e:
            raise ImportError(
                "Dataset.from_parquet needs 'pyarrow': pip install pyarrow"
            ) from e
        path = Path(path)
        rows = pq.read_table(path).to_pylist()
        return cls(rows=rows, format=_resolve_format(rows, format), name=path.stem, source=str(path))

    @classmethod
    def load(cls, path: str | Path, *, format: str | None = None) -> "Dataset":
        """Load any supported file, dispatched on extension.

        Supported: .jsonl/.ndjson, .json, .csv, .parquet.
        """
        suffix = Path(path).suffix.lower()
        loaders = {
            ".jsonl": cls.from_jsonl, ".ndjson": cls.from_jsonl,
            ".json": cls.from_json, ".csv": cls.from_csv, ".parquet": cls.from_parquet,
        }
        if suffix not in loaders:
            raise ValueError(
                f"unsupported dataset file {suffix!r} (supported: {', '.join(loaders)})"
            )
        return loaders[suffix](path, format=format)

    @classmethod
    def from_hf(cls, repo: str, *, subset: str | None = None, split: str = "train",
                token: str | None = None, format: str | None = None) -> "Dataset":
        """Load from the HuggingFace Hub. `subset` is the dataset config name."""
        try:
            from datasets import load_dataset  # noqa: PLC0415  (lazy, optional dep)
        except ImportError as e:  # pragma: no cover - exercised only with [torch] extra
            raise ImportError(
                "Dataset.from_hf needs the 'datasets' library: pip install shadowlm[torch]"
            ) from e
        ds = load_dataset(repo, subset, split=split, token=token)
        rows = [dict(r) for r in ds]
        return cls(rows=rows, format=_resolve_format(rows, format), name=repo, source=f"hf:{repo}")

    @staticmethod
    def hf_info(repo: str, *, subset: str | None = None,
                token: str | None = None) -> dict:
        """A hub dataset's configs (subsets) and the splits of one — for pickers.

        Returns {configs, subset, splits}; `subset` is the resolved config the
        splits belong to (the requested one, or the first available).
        """
        try:
            from datasets import (  # noqa: PLC0415
                get_dataset_config_names, get_dataset_split_names)
        except ImportError as e:
            raise ImportError(
                "Dataset.hf_info needs the 'datasets' library: pip install shadowlm[torch]"
            ) from e
        configs = get_dataset_config_names(repo, token=token)
        chosen = subset if subset in configs else (configs[0] if configs else None)
        splits: list[str] = []
        if chosen is not None:
            try:
                splits = list(get_dataset_split_names(repo, chosen, token=token))
            except Exception:  # noqa: BLE001 — some datasets need a full load; leave blank
                splits = []
        return {"configs": configs, "subset": chosen, "splits": splits}

    @classmethod
    def hf_preview(cls, repo: str, *, subset: str | None = None, split: str = "train",
                   token: str | None = None, limit: int = 8) -> dict:
        """Stream a few rows from a hub dataset — no full download.

        Returns {format, columns, total, preview}. `total` is None when streaming
        can't count cheaply (the common case).
        """
        try:
            from datasets import load_dataset  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Dataset.hf_preview needs the 'datasets' library: pip install shadowlm[torch]"
            ) from e
        from itertools import islice  # noqa: PLC0415

        stream = load_dataset(repo, subset or None, split=split, streaming=True, token=token)
        rows = [dict(r) for r in islice(stream, limit)]
        if not rows:
            raise ValueError(f"{repo} ({subset or 'default'}/{split}) returned no rows")
        return {"format": _detect_format(rows), "columns": list(rows[0].keys()),
                "total": None, "preview": rows}

    # ---- transforms -------------------------------------------------------
    def as_chat(self) -> "Dataset":
        """Normalise to chat format: every row becomes {"messages": [...]}.

        ShareGPT conversations are re-tagged (human→user, gpt→assistant);
        instruction/text rows are lifted into a single user/assistant exchange —
        so downstream chat-template logic has one shape to handle.
        """
        if self.format == CHAT:
            return self
        out: list[dict] = []
        for r in self.rows:
            if self.format == SHAREGPT:
                row = {"messages": [
                    {"role": _SHAREGPT_ROLES.get(t.get("from", ""), "user"),
                     "content": t.get("value", "")}
                    for t in r.get("conversations", [])
                ]}
                if "tools" in r:  # keep tool schemas for function-calling data
                    row["tools"] = r["tools"]
                out.append(row)
            elif self.format == INSTRUCTION:
                cols = _instruction_cols(r) or ("instruction", "output")
                user = str(r.get(cols[0], ""))
                # alpaca-style extra context column, when distinct from the prompt
                if cols[0] != "input" and r.get("input"):
                    user = f"{user}\n\n{r['input']}"
                out.append({"messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": str(r.get(cols[1], ""))},
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
            texts = []
            for r in self.rows:
                cols = _instruction_cols(r) or ("instruction", "output")
                texts.append(_ALPACA.format(
                    instruction=str(r.get(cols[0], "")),
                    input=r.get("input", "") if cols[0] != "input" else "",
                    output=str(r.get(cols[1], "")),
                ))
            return texts
        if self.format == SHAREGPT:
            return self.as_chat().to_texts()
        if self.format == CHAT:
            texts = []
            for r in self.rows:
                turns = []
                for m in r.get("messages", []):
                    body = m.get("content") or ""
                    if m.get("tool_calls"):  # assistant turns may be tool calls
                        calls = json.dumps(m["tool_calls"])
                        body = f"{body} {calls}".strip()
                    turns.append(f"{m['role']}: {body}")
                texts.append("\n".join(turns))
            return texts
        raise ValueError(f"don't know how to render {self.format!r} rows to text")

    def as_text(self) -> "Dataset":
        """Force raw-text format: every row becomes {"text": ...}.

        The explicit "Raw Text" target format — useful when auto-detection picks
        chat/instruction but you want plain next-token training (e.g. CPT).
        """
        if self.format == TEXT:
            return self
        rows = [{"text": t} for t in self.to_texts()]
        return Dataset(rows=rows, format=TEXT, name=self.name, source=self.source)

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

    @property
    def columns(self) -> list[str]:
        return list(self.rows[0].keys()) if self.rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.rows)

    def __getitem__(self, i: int | slice) -> "dict | Dataset":
        # Slicing returns a Dataset — the row-range selection ("train on rows
        # 0–99") is just ds[0:100].
        if isinstance(i, slice):
            label = f"{self.name}[{i.start or 0}:{i.stop if i.stop is not None else len(self.rows)}]"
            return Dataset(rows=self.rows[i], format=self.format,
                           name=label if self.name else None, source=self.source)
        return self.rows[i]

    def __repr__(self) -> str:
        name = self.name or "dataset"
        return f"Dataset({name!r}, format={self.format!r}, rows={len(self.rows)})"
