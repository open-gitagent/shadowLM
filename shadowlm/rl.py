"""Agent RL primitives: trajectories, groups, and judge-scored rewards.

The shape of agent reinforcement learning in shadowLM:

  1. Roll out your agent — each episode becomes a `Trajectory` (the message
     history, optional tool schemas, and a scalar `reward`).
  2. Batch attempts at the *same* task into a `TrajectoryGroup` — rewards are
     compared within the group, so only relative quality matters.
  3. Score rewards by hand, or let a judge model rank the group for you
     (`judge_group`) — no hand-written reward function needed.
  4. Train: `group.to_preference_rows()` turns judged groups into preference
     pairs for `finetune(method="dpo")` today; trajectory-native GRPO lands
     with the GPU worker.

    judge = slm.load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    group = slm.TrajectoryGroup(rollout(model, task) for _ in range(6))
    group = slm.judge_group(group, judge)
    run = model.finetune(group.to_preference_rows(), method="dpo")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_RUBRIC = (
    "Score each attempt below on how well it accomplishes the task implied by "
    "the conversation. Reward correctness first, then helpfulness, then "
    "concision. Penalize factual errors, ignored instructions, and filler. "
    "Scores are floats from 0 (worst) to 1 (best); use the full range to "
    "separate better attempts from worse ones."
)


@dataclass
class Trajectory:
    """One episode of an agent: its messages, optional tools, and a reward."""

    messages: list[dict] = field(default_factory=list)
    tools: list[dict] | None = None
    reward: float = 0.0
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def final_content(self) -> str:
        """The last assistant message's text — the episode's answer."""
        for m in reversed(self.messages):
            if m.get("role") == "assistant":
                return m.get("content") or ""
        return ""

    def first_user_content(self) -> str:
        for m in self.messages:
            if m.get("role") == "user":
                return m.get("content") or ""
        return ""


@dataclass
class TrajectoryGroup:
    """Several attempts at the same task; rewards are relative within the group."""

    trajectories: list[Trajectory] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def __init__(self, trajectories: Iterable[Trajectory] = (), *,
                 metrics: dict | None = None, logs: list[str] | None = None) -> None:
        self.trajectories = list(trajectories)
        self.metrics = metrics or {}
        self.logs = logs or []

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)

    def best(self) -> Trajectory:
        return max(self.trajectories, key=lambda t: t.reward)

    def worst(self) -> Trajectory:
        return min(self.trajectories, key=lambda t: t.reward)

    def to_preference_rows(self) -> list[dict]:
        """Best-vs-worst preference pairs for `finetune(method="dpo")`.

        Requires scored trajectories (rewards not all equal) sharing a prompt.
        """
        if len(self.trajectories) < 2:
            raise ValueError("need at least 2 trajectories to form a preference pair")
        ranked = sorted(self.trajectories, key=lambda t: t.reward, reverse=True)
        if ranked[0].reward == ranked[-1].reward:
            raise ValueError(
                "all trajectories have equal rewards — score them first "
                "(judge_group or set trajectory.reward)"
            )
        best, worst = ranked[0], ranked[-1]
        return [{
            "prompt": best.first_user_content(),
            "chosen": best.final_content(),
            "rejected": worst.final_content(),
        }]


def _render_attempts(group: TrajectoryGroup) -> str:
    blocks = []
    for i, t in enumerate(group, 1):
        turns = "\n".join(
            f"{m.get('role')}: {m.get('content') or json.dumps(m.get('tool_calls', ''))}"
            for m in t.messages
        )
        blocks.append(f"### Attempt {i}\n{turns}")
    return "\n\n".join(blocks)


def judge_group(group: TrajectoryGroup, judge, *, rubric: str = DEFAULT_RUBRIC,
                temperature: float = 0.0) -> TrajectoryGroup:
    """Score a group's trajectories with a judge model (LLM-as-judge rewards).

    `judge` is any loaded `shadowlm` Model. Sets each trajectory's `reward` to
    the judge's 0–1 score; the previous reward is kept in
    `metrics["independent_reward"]`.
    """
    prompt = (
        f"{rubric}\n\n{_render_attempts(group)}\n\n"
        f"Reply with ONLY a JSON array of {len(group)} scores in attempt order, "
        f'like [{{"attempt": 1, "score": 0.7}}, ...].'
    )
    raw = str(judge.chat([{"role": "user", "content": prompt}],
                         temperature=temperature, max_new_tokens=64 + 24 * len(group)))
    scores = _parse_scores(raw, len(group))
    if len(set(scores)) == 1:
        # Small local judges often emit one flat score; they rank far more
        # reliably than they calibrate — fall back to best/worst ranking.
        scores = _rank_scores(group, judge, temperature)
    for t, score in zip(group.trajectories, scores):
        t.metrics["independent_reward"] = t.reward
        t.metrics["judge_score"] = score
        t.reward = score
    group.logs.append(f"judged {len(group)} attempts: {scores}")
    return group


def _rank_scores(group: TrajectoryGroup, judge, temperature: float) -> list[float]:
    prompt = (
        f"Below are {len(group)} attempts at the same task.\n\n"
        f"{_render_attempts(group)}\n\n"
        'Which attempt is BEST and which is WORST? Reply with ONLY JSON like '
        '{"best": 1, "worst": 2}.'
    )
    raw = str(judge.chat([{"role": "user", "content": prompt}],
                         temperature=temperature, max_new_tokens=32))
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"judge gave no ranking; raw output: {raw[:200]!r}")
    obj = json.JSONDecoder().raw_decode(raw[start:])[0]
    best, worst = int(obj["best"]) - 1, int(obj["worst"]) - 1
    n = len(group)
    if not (0 <= best < n and 0 <= worst < n) or best == worst:
        raise ValueError(f"judge ranking out of range: {obj}")
    return [1.0 if i == best else 0.0 if i == worst else 0.5 for i in range(n)]


def _parse_scores(text: str, n: int) -> list[float]:
    """Pull n scores out of judge output, tolerating imperfect JSON."""
    start = text.find("[")
    if start >= 0:
        try:
            arr = json.JSONDecoder().raw_decode(text[start:])[0]
            if isinstance(arr, list):
                vals = [float(x.get("score", 0.0)) if isinstance(x, dict) else float(x)
                        for x in arr]
                if len(vals) >= n:
                    return [max(0.0, min(1.0, v)) for v in vals[:n]]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    floats = [float(x) for x in re.findall(r"\b0?\.\d+|\b[01]\.0?\b", text)]
    if len(floats) >= n:
        return [max(0.0, min(1.0, v)) for v in floats[:n]]
    raise ValueError(f"judge did not return {n} scores; raw output: {text[:200]!r}")
