"""Evaluation — score a model on a task, not just its training loss.

`finetune` reports next-token *loss*; this reports task *quality*. Point a loaded
model at a dataset, pick a metric, and get one number plus a per-row breakdown:

    res = slm.evaluate(model, "qa.jsonl")                 # contains-match
    res = slm.evaluate(model, ds, metric="exact")          # exact-match
    res = slm.evaluate(model, ds, judge=judge)             # LLM-as-judge
    res = slm.evaluate(model, ds, metric=my_score_fn)      # custom scorer
    print(res.score, res.sparkline())

This is the front half of an "eval gate" — the same capture/judge primitives,
turned toward *measuring* a model instead of training one. Pure ShadowLM: it
only needs a loaded model's `.chat()`. The built-in scorers are reused from APO
(`apo._contains_score`, `apo._judge_one`) so eval and prompt-optimization agree
on what a good answer is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data import Dataset


def _exact_score(output: str, expected: str) -> float:
    """1.0 when the output equals the expected answer (case/space-insensitive)."""
    o = " ".join(str(output).lower().split())
    e = " ".join(str(expected).lower().split())
    return 1.0 if e and o == e else 0.0


@dataclass
class EvalResult:
    """The outcome of an `evaluate` run: an aggregate score plus per-row detail."""

    metric: str
    score: float  # mean of `scores`
    scores: list[float] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)  # [{input, output, expected, score}]
    n: int = 0

    def sparkline(self) -> str:
        """A tiny unicode bar of per-row scores — handy in a REPL or log line."""
        if not self.scores:
            return ""
        bars = "▁▂▃▄▅▆▇█"
        lo, hi = min(self.scores), max(self.scores)
        rng = (hi - lo) or 1.0
        return "".join(bars[min(7, int((s - lo) / rng * 7))] for s in self.scores)

    def worst(self, k: int = 5) -> list[dict]:
        """The k lowest-scoring examples (for eyeballing where the model fails)."""
        return sorted(self.examples, key=lambda e: e["score"])[:k]

    def to_dict(self) -> dict:
        return {"metric": self.metric, "score": self.score, "n": self.n,
                "scores": self.scores, "examples": self.examples}

    def __repr__(self) -> str:
        return f"EvalResult(metric={self.metric!r}, score={self.score:.4f}, n={self.n})"


def _row_io(row: dict, fmt: str) -> tuple[str, str]:
    """Pull (input_prompt, expected_answer) out of a row, by dataset format.

    expected may be "" when the dataset carries no reference answer (e.g. judge
    scoring on prompts alone).
    """
    from .data import CHAT, PREFERENCE  # noqa: PLC0415

    if fmt == CHAT or "messages" in row:
        msgs = row.get("messages", [])
        prompt = next((m.get("content") or "" for m in msgs
                       if m.get("role") == "user"), "")
        expected = next((m.get("content") or "" for m in reversed(msgs)
                         if m.get("role") == "assistant"), "")
        return str(prompt), str(expected)
    if fmt == PREFERENCE or ("chosen" in row and "prompt" in row):
        return str(row.get("prompt", "")), str(row.get("chosen", ""))
    # instruction / QA / raw dict — auto-detect the prompt & answer columns
    from .apo import _cols  # noqa: PLC0415

    pcol, acol = _cols(row)
    if not pcol:
        from .apo import _PROMPT_KEYS  # noqa: PLC0415

        raise ValueError(
            f"no prompt column found in row (looked for {_PROMPT_KEYS}); "
            "pass chat-format rows or a dataset with a prompt/question column")
    prompt = str(row[pcol])
    # alpaca-style extra context column, when distinct from the prompt
    if pcol != "input" and row.get("input"):
        prompt = f"{prompt}\n\n{row['input']}"
    return prompt, str(row.get(acol, "")) if acol else ""


def _resolve_scorer(metric, judge):
    """Map the metric arg to a scorer `(output, expected, prompt) -> float`."""
    if callable(metric):
        return metric, getattr(metric, "__name__", "custom")
    from .apo import _contains_score, _judge_one  # noqa: PLC0415

    if metric == "contains":
        return (lambda out, exp, q: _contains_score(out, exp)), "contains"
    if metric == "exact":
        return (lambda out, exp, q: _exact_score(out, exp)), "exact"
    if metric == "judge":
        if judge is None:
            raise ValueError("metric='judge' needs a judge model: evaluate(..., judge=model)")
        return (lambda out, exp, q: _judge_one(judge, q, out, exp)), "judge"
    raise ValueError(
        f"unknown metric {metric!r} (expected 'contains', 'exact', 'judge', or a callable)")


def evaluate(
    model,
    data: Dataset | list[dict] | str,
    *,
    metric="contains",
    judge=None,
    system: str | None = None,
    sample: int | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    verbose: bool = True,
) -> EvalResult:
    """Score `model` on `data`, returning an `EvalResult`.

    model: a loaded shadowlm Model (answers each row via `.chat`).
    data: a Dataset, rows, or a path to a dataset file (jsonl/json/csv/parquet).
    metric: "contains" (default — expected answer appears in the output), "exact"
        (normalized equality), "judge" (LLM-as-judge, needs `judge=`), or a custom
        callable `(output, expected, prompt) -> float in [0, 1]`.
    judge: a Model that scores answers 0–1. Passing it defaults `metric` to "judge".
    system: optional system prompt prepended to every query.
    sample: evaluate only the first N rows.
    temperature: generation temperature — 0.0 (default) for deterministic scoring.
    """
    if isinstance(data, str):
        data = Dataset.load(data)
    fmt = data.format if isinstance(data, Dataset) else None
    rows = list(data.rows if isinstance(data, Dataset) else data)
    if sample:
        rows = rows[:sample]
    if not rows:
        raise ValueError("evaluate needs at least one row")
    if judge is not None and metric == "contains":
        metric = "judge"  # passing a judge implies judge scoring
    if fmt is None:
        from .data import _detect_format  # noqa: PLC0415

        fmt = _detect_format(rows)
    scorer, metric_name = _resolve_scorer(metric, judge)

    scores: list[float] = []
    examples: list[dict] = []
    for r in rows:
        prompt, expected = _row_io(r, fmt)
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        out = str(model.chat(msgs, temperature=temperature, max_new_tokens=max_new_tokens))
        s = max(0.0, min(1.0, float(scorer(out, expected, prompt))))
        scores.append(s)
        examples.append({"input": prompt, "output": out, "expected": expected, "score": s})

    score = sum(scores) / len(scores)
    if verbose:
        print(f"[eval] {metric_name} · {score:.3f} over {len(scores)} rows", flush=True)
    return EvalResult(metric=metric_name, score=score, scores=scores,
                      examples=examples, n=len(scores))
