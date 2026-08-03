"""Data synthesis — the fourth inlet.

`capture()` records a live agent, `traces` reads one that already ran, and
`Dataset.from_*` loads what you already have. All three need the traffic to
exist. This makes training data that doesn't: describe the task in plain
English, point at a document, or hand over a few real episodes, and a teacher
model writes the rest — emitted in the exact shape the method you name accepts.

    import shadowlm as slm

    run = slm.synthesize(
        task="Triage billing emails: classify urgency, draft a reply, and "
             "escalate refunds over $200.",
        teacher=slm.synth.frontier("gpt-4o"),
        n=200, method="lora")
    print(run.report.summary())
    model.finetune(run.dataset, method="lora")

The teacher is anything that answers `.chat()` — a frontier model, a local
`slm.load(...)` model, or the student itself. Naming a `method=` picks the
output shape from that method's spec, so `method="dpo"` yields preference pairs
and `method="more_plus"` yields query-diverse paraphrase units, with no
format bookkeeping on your side.
"""

from __future__ import annotations

import math
import random
import time

from .. import methods
from ..apo import _judge_one
from . import emit
from .generate import (FLAWS, STYLES, conversation, paraphrases, plan_leaves,
                       preference)
from .quality import Dedup
from .run import SynthReport, SynthRun
from .seeds import Seed, chunk_text, resolve_seed
from .teacher import OpenAIChatTeacher, as_teacher, CountingTeacher, frontier

__all__ = [
    "synthesize", "SynthRun", "SynthReport", "Seed", "FORMATS", "resolve_output",
    "frontier", "as_teacher", "OpenAIChatTeacher", "chunk_text", "emit",
]

FORMATS = ("chat", "text", "preference", "grpo", "groups", "otlp")
_MORE_ADAPTERS = (methods.ADAPTER_MORE, methods.ADAPTER_MORE_PLUS)
_MAX_ROUNDS = 3  # a round that keeps nothing ends the run; this caps the rest


def synthesize(
    *,
    teacher,
    task: str | None = None,
    document=None,
    episodes=None,
    n: int = 100,
    method: str | None = None,
    format: str | None = None,
    tools: list[dict] | None = None,
    student=None,
    judge=None,
    min_score: float | None = 0.6,
    dedup_threshold: float = 0.7,
    per_scenario: int = 4,
    seed: int = 3407,
    verbose: bool = True,
    on_progress=None,
) -> SynthRun:
    """Generate `n` training rows and return a `SynthRun`.

    teacher: who writes the data — a loaded Model, or `frontier("gpt-4o")`.
    task / document / episodes: the seed. Exactly one is required (`task` may
        also accompany the other two as extra context).
    method: the training method the data is for — its spec picks the output
        shape. Override with `format=` ("chat", "text", "preference", "grpo",
        "groups", "otlp"); "otlp" emits OpenTelemetry GenAI spans.
    tools: OpenAI-style tool schemas, to synthesize tool-calling episodes.
    student: a Model whose answers become the `rejected` side of preference
        pairs — DPO then targets exactly the teacher/student gap.
    judge / min_score: quality gate; the judge defaults to the teacher, and for
        document seeds it scores answers against the source passage, so the same
        call doubles as a grounding check. `min_score=None` disables it.
    dedup_threshold: reject a row whose question overlaps an accepted one by at
        least this much (token Jaccard).
    per_scenario: rows generated per scenario — attempts per group for "groups",
        and paraphrases per fact for the MoRE methods.
    """
    if per_scenario < 1:
        raise ValueError("per_scenario must be at least 1")
    started = time.time()
    fmt, mode = resolve_output(method, format)
    source = resolve_seed(task=task, document=document, episodes=episodes)
    teacher = CountingTeacher(as_teacher(teacher))
    scorer = CountingTeacher(as_teacher(judge)) if judge is not None else teacher
    student = as_teacher(student) if student is not None else None
    rng = random.Random(seed)
    report = SynthReport(requested=n)
    dedup = Dedup(dedup_threshold)
    if source.exemplars:
        # real episodes are for evaluating, not for handing back as "synthetic"
        dedup.seed(t.first_user_content() for t in source.exemplars)

    # Trajectory-GRPO learns from the spread between good and bad attempts, so
    # filtering the weak ones out would throw away exactly the signal it needs —
    # score every row, gate on nothing.
    gate = None if fmt == "groups" else min_score
    kept: list = []
    rejected: list = []
    covered: list[str] = []
    if verbose:
        print(f"[synth] {source.kind} seed · target {n} rows · format {fmt} · "
              f"teacher {teacher.name}", flush=True)

    for round_no in range(1, _MAX_ROUNDS + 1):
        if len(kept) >= n:
            break
        leaves = plan_leaves(source, teacher, rng=rng, avoid=covered,
                             count=math.ceil((n - len(kept)) / per_scenario))
        if not leaves:
            break
        report.scenarios += len(leaves)
        covered.extend(leaf.scenario for leaf in leaves)
        outcomes = _generate(leaves, source, teacher, mode=mode, tools=tools,
                             student=student, per_scenario=per_scenario)
        _score(outcomes, scorer, enabled=min_score is not None)
        before = len(kept)
        for outcome in outcomes:
            _absorb(outcome, report=report, dedup=dedup, min_score=gate,
                    kept=kept, rejected=rejected)
        if verbose:
            print(f"[synth] round {round_no} · {len(kept)}/{n} rows kept",
                  flush=True)
        if on_progress:
            on_progress(len(kept), n)
        if len(kept) == before:
            break  # a whole round survived nothing — stop spending teacher calls

    if mode != "paraphrases" and len(kept) > n:
        # a parallel round finishes every job it started, so it can overshoot;
        # paraphrase units are left whole because MoRE+ groups by fixed size
        report.surplus = len(kept) - n
        kept = kept[:n]
    if not kept:
        gated = (f", {report.rejected_judge} below min_score={min_score}"
                 if gate is not None else "")
        raise RuntimeError(
            f"synthesis produced nothing usable — {report.rejected_validation} "
            f"invalid, {report.rejected_dedup} duplicate{gated}. Loosen the gate "
            "(min_score=, dedup_threshold=) or check what the teacher is emitting.")

    report.kept = len(kept)
    scored = [t.reward for t in kept if t.reward]
    report.mean_score = sum(scored) / len(scored) if scored else None
    report.teacher_calls = teacher.calls + (0 if scorer is teacher else scorer.calls)
    report.duration_s = time.time() - started
    if mode == "paraphrases":
        report.note = f"train with more_plus_group_size={per_scenario}"

    run = _emit(fmt, kept, report, rejected, seed=seed)
    if verbose:
        print(report.summary(), flush=True)
    return run


def resolve_output(method: str | None, fmt: str | None) -> tuple[str, str]:
    """(output format, generation mode) for the method you plan to train with.

    Dispatches on the method's *spec*, never its name — so a method registered
    tomorrow that reuses an existing trainer gets the right data shape for free.
    """
    if fmt is not None and fmt not in FORMATS:
        raise ValueError(
            f"unknown format {fmt!r} (expected one of {', '.join(FORMATS)})")
    mode = "conversation"
    if method is not None:
        spec = methods.get(method)  # raises, listing the registered methods
        if spec.trainer == "dpo":
            fmt, mode = fmt or "preference", "preference"
        elif spec.adapter in _MORE_ADAPTERS:
            fmt, mode = fmt or "chat", "paraphrases"
        elif spec.trainer == "grpo":
            fmt = fmt or "groups"
        elif spec.raw_text:
            fmt = fmt or "text"
    if fmt == "preference":
        mode = "preference"
    return fmt or "chat", mode


def _generate(leaves, source, teacher, *, mode, tools, student, per_scenario):
    """One job per row wanted, run at the teacher's parallelism."""
    jobs = []
    for i, leaf in enumerate(leaves):
        if mode == "paraphrases":
            jobs.append(lambda leaf=leaf, i=i: paraphrases(
                leaf, source, teacher, k=per_scenario,
                style=STYLES[i % len(STYLES)]))
            continue
        for j in range(per_scenario):
            index = i * per_scenario + j
            style = STYLES[index % len(STYLES)]
            if mode == "preference":
                jobs.append(lambda leaf=leaf, style=style, index=index: preference(
                    leaf, source, teacher, student=student, style=style,
                    flaw=FLAWS[index % len(FLAWS)]))
            else:
                jobs.append(lambda leaf=leaf, style=style: conversation(
                    leaf, source, teacher, tools=tools, style=style))
    return _run_jobs(jobs, workers=teacher.parallelism)


def _run_jobs(jobs, *, workers: int) -> list:
    """Run jobs, preserving submission order — order groups MoRE+ units."""
    if workers <= 1 or len(jobs) <= 1:
        return [job() for job in jobs]
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: job(), jobs))


def _score(outcomes, judge, *, enabled: bool) -> None:
    """Judge one row per outcome; a unit's rows share an answer, so one score
    speaks for all of them (and a conversation outcome is a single row anyway)."""
    if not enabled:
        return
    heads = [o.trajectories[0] for o in outcomes if o.trajectories]
    _run_jobs([lambda t=t: _judge(t, judge) for t in heads],
              workers=judge.parallelism)
    for outcome in outcomes:
        for traj in outcome.trajectories[1:]:
            traj.reward = outcome.trajectories[0].reward


def _judge(traj, judge) -> None:
    """Score an episode 0–1. When the row is grounded in a source passage that
    passage is the reference, so this doubles as the hallucination check."""
    traj.reward = _judge_one(judge, traj.first_user_content(),
                             traj.final_content(),
                             traj.metadata.get("grounding") or "")
    traj.metrics["judge_score"] = traj.reward


def _absorb(outcome, *, report, dedup, min_score, kept, rejected) -> None:
    """Take or drop one outcome whole, counting it either way."""
    report.generated += outcome.attempted
    report.rejected_validation += outcome.invalid
    report.repaired += outcome.repaired
    rows = outcome.trajectories
    if not rows:
        return
    if not dedup.accept(_unit_text(rows), key=outcome.key):
        report.rejected_dedup += len(rows)
        _mark(rows, "duplicate", rejected)
    elif min_score is not None and rows[0].reward < min_score:
        report.rejected_judge += len(rows)
        _mark(rows, f"judge score {rows[0].reward:.2f} < {min_score}", rejected)
    else:
        kept.extend(rows)


def _unit_text(rows) -> str:
    return " ".join(m.get("content") or "" for t in rows for m in t.messages)


def _mark(rows, reason: str, rejected: list) -> None:
    for traj in rows:
        traj.metadata["reject_reason"] = reason
    rejected.extend(rows)


def _emit(fmt: str, kept: list, report, rejected: list, *, seed: int) -> SynthRun:
    run = SynthRun(format=fmt, report=report, trajectories=kept, rejected=rejected)
    if fmt == "groups":
        run.groups = emit.to_groups(kept)
    elif fmt == "otlp":
        run.spans = emit.to_otlp(kept, seed=seed)
    else:
        run.dataset = {"chat": emit.to_chat, "text": emit.to_text,
                       "preference": emit.to_preference,
                       "grpo": emit.to_grpo_prompts}[fmt](kept)
    return run
