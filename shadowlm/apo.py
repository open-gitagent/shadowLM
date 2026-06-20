"""Automatic Prompt Optimization (APO) — improve a system prompt, not weights.

Sometimes the cheapest win isn't fine-tuning at all: a better system prompt.
APO is the no-GPU sibling of `finetune` — same `capture → judge` front end, but
the optimized *resource* is a prompt. It scores the current prompt on your data,
asks an optimizer model to propose better ones from the failures, keeps the best,
and repeats. Returns an `APORun` with the winning prompt and a score curve.

    best = slm.optimize_prompt(model, data, rounds=4)   # exact/contains scoring
    best = slm.optimize_prompt(model, data, judge=judge) # LLM-as-judge scoring
    print(best.best_prompt, best.improvement)

Pure ShadowLM: it only needs a loaded model's `.chat()` — no extra deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data import Dataset

_PROMPT_KEYS = ("instruction", "question", "prompt", "query", "input")
_ANSWER_KEYS = ("output", "answer", "response", "completion", "label")

_OPTIMIZER_META = """You are optimizing the SYSTEM PROMPT for a task so a model answers it better.

CURRENT SYSTEM PROMPT:
{current}

EXAMPLES IT GOT WRONG (input → model output vs. expected):
{failures}

Write ONE improved system prompt that would fix these. Be specific and concise.
Reply with ONLY the new system prompt — no preamble, no quotes."""


@dataclass
class APORun:
    """The outcome of an APO run: the best prompt and how the score moved."""

    base_prompt: str
    base_score: float
    best_prompt: str
    best_score: float
    history: list[dict] = field(default_factory=list)  # [{round, score, prompt}]

    @property
    def improvement(self) -> float:
        return self.best_score - self.base_score

    def sparkline(self) -> str:
        scores = [self.base_score] + [h["score"] for h in self.history]
        bars = "▁▂▃▄▅▆▇█"
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        return "".join(bars[min(7, int((s - lo) / rng * 7))] for s in scores)

    def to_dict(self) -> dict:
        return {"base_prompt": self.base_prompt, "base_score": self.base_score,
                "best_prompt": self.best_prompt, "best_score": self.best_score,
                "improvement": self.improvement, "history": self.history}


def _cols(row: dict) -> tuple[str, str | None]:
    p = next((k for k in _PROMPT_KEYS if row.get(k)), None)
    a = next((k for k in _ANSWER_KEYS if row.get(k)), None)
    return p, a


def _norm(text: str) -> str:
    """Whitespace-collapsed, lowercased text for tolerant string matching."""
    return " ".join(str(text).lower().split())


def _contains_score(output: str, expected: str) -> float:
    """Default scorer: 1.0 if the expected answer appears in the output."""
    o, e = _norm(output), _norm(expected)
    return 1.0 if e and e in o else 0.0


def optimize_prompt(
    model,
    data: Dataset | list[dict],
    *,
    judge=None,
    score_fn=None,
    seed_prompt: str = "",
    rounds: int = 4,
    candidates: int = 4,
    sample: int | None = None,
    eval_holdout: float = 0.3,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    optimizer=None,
    verbose: bool = True,
) -> APORun:
    """Optimize a system prompt against `data`, returning the best one found.

    model: a loaded shadowlm Model (answers the task via `.chat`).
    data: Dataset or rows with a prompt column (+ an answer column for scoring).
    judge: optional Model that scores answers 0–1 (LLM-as-judge). If omitted and
        no `score_fn` is given, scoring is exact/contains-match on the answer.
    score_fn: optional `(output, expected, prompt) -> float in [0,1]`.
    optimizer: Model that proposes improved prompts (defaults to `model`). A
        stronger optimizer writes better prompts — pass a larger model here.
    eval_holdout: fraction held out for *selecting* the best prompt. APO proposes
        from the train split's failures but scores/keeps prompts on the held-out
        split, so the winner generalizes instead of overfitting the examples it
        was tuned on. Ignored when there are too few rows to split.
    rounds × candidates bounds the work: each round evaluates `candidates` new
    prompts and keeps the best-so-far (by held-out score).
    """
    rows = list(data.rows if isinstance(data, Dataset) else data)
    if sample:
        rows = rows[:sample]
    if not rows:
        raise ValueError("optimize_prompt needs at least one example row")
    pcol, acol = _cols(rows[0])
    if not pcol:
        raise ValueError(f"no prompt column found (looked for {_PROMPT_KEYS})")
    if score_fn is None and judge is None and not acol:
        raise ValueError(
            "no way to score: provide rows with an answer column, a judge model, "
            "or a score_fn(output, expected, prompt) -> float")
    optimizer = optimizer or model

    # honest selection: propose from `train` failures, select on held-out `eval`
    k = int(round(len(rows) * eval_holdout))
    if 1 <= k <= len(rows) - 1:
        eval_rows, train_rows = rows[:k], rows[k:]
    else:
        eval_rows = train_rows = rows  # too few to split — reuse (overfit risk)

    def answer(prompt: str, q: str) -> str:
        msgs = ([{"role": "system", "content": prompt}] if prompt else []) + \
               [{"role": "user", "content": q}]
        return str(model.chat(msgs, temperature=0.0, max_new_tokens=max_new_tokens))

    def score_one(out: str, exp: str, q: str) -> float:
        if score_fn is not None:
            return float(score_fn(out, exp, q))
        if judge is not None:
            return _judge_one(judge, q, out, exp)
        return _contains_score(out, exp)

    def evaluate(prompt: str, rows_) -> float:
        total = 0.0
        for r in rows_:
            out = answer(prompt, str(r[pcol]))
            total += score_one(out, str(r.get(acol, "")) if acol else "", str(r[pcol]))
        return total / len(rows_)

    def log(msg: str) -> None:
        if verbose:
            print(f"[apo] {msg}", flush=True)

    base_score = evaluate(seed_prompt, eval_rows)
    best_prompt, best_score = seed_prompt, base_score
    history: list[dict] = []
    split_note = "" if eval_rows is train_rows else f" ({len(train_rows)} train / {len(eval_rows)} eval)"
    log(f"base score {base_score:.3f}{split_note}")

    for rnd in range(1, rounds + 1):
        failures = _failures(train_rows, pcol, acol, best_prompt, answer, score_one)
        proposals = _propose(optimizer, best_prompt, failures, candidates,
                             temperature, max_new_tokens)
        round_best_p, round_best_s = best_prompt, best_score
        for cand in proposals:
            s = evaluate(cand, eval_rows)
            if s > round_best_s:
                round_best_p, round_best_s = cand, s
        best_prompt, best_score = round_best_p, round_best_s
        history.append({"round": rnd, "score": best_score, "prompt": best_prompt})
        log(f"round {rnd}/{rounds} · best {best_score:.3f}")

    log(f"done · {base_score:.3f} → {best_score:.3f} (+{best_score - base_score:.3f})")
    return APORun(seed_prompt, base_score, best_prompt, best_score, history)


def _failures(rows, pcol, acol, prompt, answer, score_one, k: int = 3) -> str:
    """The k lowest-scoring examples, rendered for the optimizer prompt."""
    scored = []
    for r in rows:
        q = str(r[pcol])
        exp = str(r.get(acol, "")) if acol else ""
        out = answer(prompt, q)
        scored.append((score_one(out, exp, q), q, out, exp))
    scored.sort(key=lambda x: x[0])
    blocks = []
    for _, q, out, exp in scored[:k]:
        block = f"- input: {q}\n  got: {out[:200]}"
        if exp:
            block += f"\n  expected: {exp[:200]}"
        blocks.append(block)
    return "\n".join(blocks) or "(no failing examples)"


def _clean(text: str) -> str:
    """Strip the wrapper junk weak optimizers add — code fences, surrounding
    quotes, and a leading 'System prompt:'/'Prompt:' label."""
    t = text.strip().strip("`").strip()
    low = t.lower()
    for pre in ("system prompt:", "new system prompt:", "prompt:", "improved prompt:"):
        if low.startswith(pre):
            t = t[len(pre):].strip()
            break
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


def _propose(optimizer, current, failures, k, temperature, max_new_tokens) -> list[str]:
    meta = _OPTIMIZER_META.format(current=current or "(none)", failures=failures)
    out, seen = [], set()
    for _ in range(k):
        text = _clean(str(optimizer.chat([{"role": "user", "content": meta}],
                                         temperature=temperature,
                                         max_new_tokens=max_new_tokens)))
        if len(text) >= 8 and text not in seen:  # drop empties, dupes, fragments
            seen.add(text)
            out.append(text)
    return out


# The shared single-answer judge: a short rubric + a tolerant number parse, used
# by both APO and `evaluate` so they agree on what a good answer is. (The RL judge
# in rl.py is a *group-relative* ranker — it can't score a lone eval row — so eval
# reuses this scorer, not judge_group.)
_JUDGE_RUBRIC = (
    "Reward correctness first, then helpfulness, then concision. "
    "Penalize factual errors and ignored instructions."
)


def _parse_judge_score(raw: str) -> float:
    """Tolerantly pull a 0–1 score out of a judge's reply.

    Small judges phrase scores many ways — a bare decimal ("0.7"), a ratio
    ("7/10"), or an integer rating ("8" → 0.8). Handle all three, then clamp.
    """
    import re  # noqa: PLC0415

    s = str(raw)
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)  # "7/10"
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return max(0.0, min(1.0, num / den)) if den else 0.0
    m = re.search(r"\d+\.\d+", s)  # a decimal like "0.7"
    if m:
        return max(0.0, min(1.0, float(m.group())))
    m = re.search(r"\d+", s)  # a bare integer — assume an x/10 rating above 1
    if m:
        v = float(m.group())
        return max(0.0, min(1.0, v if v <= 1 else v / 10.0))
    return 0.0


def _judge_one(judge, question: str, output: str, expected: str) -> float:
    prompt = (
        "Score how well the ANSWER responds to the INPUT from 0.0 to 1.0.\n"
        f"{_JUDGE_RUBRIC}\n"
        f"INPUT: {question}\nANSWER: {output}\n"
        + (f"REFERENCE: {expected}\n" if expected else "")
        + 'Reply with ONLY a number like 0.7.'
    )
    raw = str(judge.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_new_tokens=8))
    return _parse_judge_score(raw)
