"""Emitters — one synthesized episode, rendered into whatever the consumer needs.

The generator produces `Trajectory` objects and nothing else; these turn them
into the exact shape a training method (or somebody else's OTel pipeline)
accepts. "Inject ready" is not a claim we make about the JSON — it means the
code that eats the output accepts it, which is what the emit tests assert by
feeding every shape back into its real reader.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import traces
from ..data import Dataset
from ..rl import Trajectory, TrajectoryGroup

SCORE_KEY = "shadowlm.synth.score"
_NAME = "synth"
# Fixed epoch: span order is all the reader uses, and a wall-clock read would
# make the same seed produce a different file every run.
_EPOCH_NS = 1_700_000_000_000_000_000
_ONE_SECOND_NS = 1_000_000_000


def to_chat(trajectories: list[Trajectory]) -> Dataset:
    """Chat rows for the supervised methods (lora, qlora, dora, full, …)."""
    return traces.to_dataset(trajectories, name=_NAME)


def to_text(trajectories: list[Trajectory]) -> Dataset:
    """Raw domain text for continued pretraining.

    The assistant prose only: CPT trains on the material itself, not on a
    transcript of somebody being asked about it.
    """
    rows = []
    for traj in trajectories:
        text = "\n\n".join(m["content"] for m in traj.messages
                           if m.get("role") == "assistant" and m.get("content"))
        if text:
            rows.append({"text": text})
    if not rows:
        raise ValueError("no assistant prose to train on")
    return Dataset.from_list(rows, name=_NAME, format="text")


def to_preference(trajectories: list[Trajectory]) -> Dataset:
    """`{prompt, chosen, rejected}` rows — the shape trl's DPOTrainer demands.

    The rejected side rides on the trajectory's metadata, so a preference sample
    is still a valid chat episode (its chosen answer) and can be emitted as one.
    """
    rows = []
    for traj in trajectories:
        prompt, chosen = traj.first_user_content(), traj.final_content()
        rejected = traj.metadata.get("rejected")
        if prompt and chosen and rejected and chosen != rejected:
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not rows:
        raise ValueError(
            "no usable preference pairs — every candidate was missing a side or "
            "scored its rejected answer as good as the chosen one")
    return Dataset.from_list(rows, name=_NAME, format="preference")


def to_grpo_prompts(trajectories: list[Trajectory]) -> Dataset:
    """`{prompt, answer}` rows for reward-function GRPO (`reward_fns=[...]`)."""
    rows = [{"prompt": t.first_user_content(), "answer": t.final_content()}
            for t in trajectories if t.first_user_content()]
    if not rows:
        raise ValueError("no rows carried an opening user turn to prompt with")
    return Dataset.from_list(rows, name=_NAME, format="instruction")


def to_groups(trajectories: list[Trajectory]) -> list[TrajectoryGroup]:
    """Attempts at the same scenario, bucketed for trajectory-GRPO.

    A group needs two members and some spread in reward or it carries no signal;
    `rl.weighted_rows` would skip those silently, so drop them here where the
    report can say how many went.
    """
    buckets: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        buckets.setdefault(traj.metadata.get("taxonomy_path", ""), []).append(traj)
    groups = [TrajectoryGroup(ts) for ts in buckets.values()
              if len(ts) >= 2 and len({t.reward for t in ts}) > 1]
    if not groups:
        raise ValueError(
            "no scored groups — trajectory-GRPO needs several attempts per "
            "scenario with differing judge scores (raise attempts=, or leave "
            "min_score enabled so the judge actually runs)")
    return groups


def to_otlp(trajectories: list[Trajectory], *, path: str | Path | None = None,
            model: str = "shadowlm-synth", seed: int = 0) -> dict:
    """Trajectories → an OTLP/JSON export of OpenTelemetry GenAI spans.

    One span per assistant turn, each span's input extending the previous span's
    input plus output — the shape a real agent loop emits. That is what lets
    `traces.from_otlp` fold them back into exactly these episodes, which is the
    round-trip the OTLP test asserts.
    """
    rng = random.Random(seed)
    spans = []
    for traj in trajectories:
        trace_id = f"{rng.getrandbits(128):032x}"
        timestamp = _EPOCH_NS
        for i, message in enumerate(traj.messages):
            if message.get("role") != "assistant":
                continue
            attributes = {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model,
                "gen_ai.conversation.id": trace_id,
                "gen_ai.input.messages": json.dumps(
                    [_to_parts(m) for m in traj.messages[:i]]),
                "gen_ai.output.messages": json.dumps([_to_parts(message)]),
            }
            if traj.tools:
                attributes["gen_ai.request.tools"] = json.dumps(traj.tools)
            spans.append({
                "traceId": trace_id,
                "spanId": f"{rng.getrandbits(64):016x}",
                "name": "chat",
                "startTimeUnixNano": str(timestamp),
                "attributes": [_kv(k, v) for k, v in attributes.items()]
                + [_kv(SCORE_KEY, float(traj.reward))],
            })
            timestamp += _ONE_SECOND_NS
    payload = {"resourceSpans": [{
        "resource": {"attributes": [_kv("service.name", "shadowlm-synth")]},
        "scopeSpans": [{"scope": {"name": "shadowlm.synth"}, "spans": spans}],
    }]}
    if path is not None:
        Path(path).write_text(json.dumps(payload, indent=2))
    return payload


def _to_parts(message: dict) -> dict:
    """An OpenAI-wire message → the OTel GenAI `{role, parts}` form.

    The exact inverse of `traces._spec_message` — read the two together before
    changing either, because their agreement *is* the round-trip guarantee.
    """
    if message.get("role") == "tool":
        return {"role": "tool", "parts": [{
            "type": "tool_call_response",
            "id": message.get("tool_call_id"),
            "response": message.get("content") or "",
        }]}
    parts = []
    if message.get("content"):
        parts.append({"type": "text", "content": message["content"]})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        parts.append({"type": "tool_call", "id": call.get("id"),
                      "name": fn.get("name", ""), "arguments": fn.get("arguments")})
    return {"role": message.get("role", "assistant"), "parts": parts}


def _kv(key: str, value) -> dict:
    """One OTLP attribute in `AnyValue` encoding."""
    encoded = ({"doubleValue": value} if isinstance(value, float)
               else {"stringValue": str(value)})
    return {"key": key, "value": encoded}
