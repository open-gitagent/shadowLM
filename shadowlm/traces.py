"""Trace ingestion — turn production agent traces into a training set.

The shadowing loop without a proxy. A frontier model already runs your agents
in production; every framework, whatever its internals, emits the same thing —
OpenTelemetry **GenAI** spans (`gen_ai.*`) for each model call. Point this at
those spans and it reconstructs the conversations the frontier model produced
and hands you a `Dataset` you can finetune a small open model on, then swap in:

    import shadowlm as slm
    from shadowlm import traces

    ds = traces.to_dataset(traces.from_otlp("export.json"))   # spans → chat data
    model = slm.load("Qwen/Qwen2.5-3B-Instruct")
    model.finetune(ds, method="lora")                         # owns the task

Because the frontier model generated the assistant turns in those spans,
supervised training on them is behavioral cloning — the open model learns to be
a drop-in for that task. Attach the eval scores you already log as rewards
(`reward_key=`) and the same traces feed RL (`group(...)` → GRPO/DPO).

This is the offline sibling of `slm.capture()`: capture records a live agent
against a proxy; this reads traces an agent already emitted. Both end at the
same place — a `Trajectory` (messages + tools + reward), which is rows for
`finetune`.

Spans can come from anywhere that speaks OTLP GenAI: an OTLP export file
(`from_otlp`), or already-parsed span dicts (`from_spans`). Observability
platforms (Langfuse, Arize/Phoenix, …) all export to this format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .data import Dataset
from .rl import Trajectory, TrajectoryGroup

# OpenTelemetry GenAI semantic-convention attribute keys.
# Spec form (the standard): whole message lists as JSON strings in the
# `{role, parts:[...]}` schema. This is what spec-compliant instrumentors emit.
_INPUT = "gen_ai.input.messages"
_OUTPUT = "gen_ai.output.messages"
_SYSTEM = "gen_ai.system_instructions"
_CONVERSATION = "gen_ai.conversation.id"
_REQ_MODEL = "gen_ai.request.model"
_RESP_MODEL = "gen_ai.response.model"
# Indexed form (older OpenLLMetry/Traceloop convention) — kept as a fallback.
_PROMPT = "gen_ai.prompt"          # gen_ai.prompt.{i}.{role,content,tool_calls...}
_COMPLETION = "gen_ai.completion"  # gen_ai.completion.{i}.{role,content,tool_calls...}
# Event-form GenAI: one log event per message on the span — also a fallback.
_EVENT_ROLES = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}


# ---- attribute normalization -----------------------------------------------
def _otlp_value(v: Any) -> Any:
    """Decode an OTLP `AnyValue` ({'stringValue': ...} etc.) to a Python value."""
    if not isinstance(v, dict):
        return v
    for k in ("stringValue", "boolValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "arrayValue" in v:
        return [_otlp_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _otlp_value(kv["value"]) for kv in v["kvlistValue"].get("values", [])}
    return v


def _flatten(attrs: Any, prefix: str = "") -> dict[str, Any]:
    """Any attribute container → a flat dot-keyed dict.

    Accepts the three shapes spans arrive in: OTLP's `[{key, value}]` list,
    a nested dict, or an already-flat dot-keyed dict.
    """
    out: dict[str, Any] = {}
    if isinstance(attrs, list):  # OTLP key/value list
        for kv in attrs:
            out[kv["key"]] = _otlp_value(kv.get("value"))
        return out
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict) and not any(
                t in v for t in ("stringValue", "intValue", "doubleValue", "boolValue")
            ):
                out.update(_flatten(v, key))
            else:
                out[key] = _otlp_value(v) if isinstance(v, dict) else v
        return out
    return out


def _indices(attrs: dict, base: str) -> list[int]:
    """The sorted integer indices present under `base.{i}...`."""
    seen = set()
    for k in attrs:
        if k.startswith(base + "."):
            head = k[len(base) + 1:].split(".", 1)[0]
            if head.isdigit():
                seen.add(int(head))
    return sorted(seen)


def _as_json_str(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v or {})


def _tool_calls(attrs: dict, base: str) -> list[dict]:
    """OpenAI-wire tool calls from `base.{j}.{id,name,arguments}` (tolerant of
    the `.function.` nesting some exporters add)."""
    calls = []
    for j in _indices(attrs, base):
        p = f"{base}.{j}"
        name = attrs.get(f"{p}.name") or attrs.get(f"{p}.function.name")
        if not name:
            continue
        args = attrs.get(f"{p}.arguments")
        if args is None:
            args = attrs.get(f"{p}.function.arguments")
        calls.append({
            "id": attrs.get(f"{p}.id") or f"call_{j}",
            "type": "function",
            "function": {"name": name, "arguments": _as_json_str(args)},
        })
    return calls


def _message(attrs: dict, p: str) -> dict | None:
    """One message from the attributes under index prefix `p`."""
    role = attrs.get(f"{p}.role")
    content = attrs.get(f"{p}.content")
    tcs = _tool_calls(attrs, f"{p}.tool_calls")
    if role is None and content is None and not tcs:
        return None
    msg: dict[str, Any] = {"role": role or "assistant"}
    if content is not None:
        msg["content"] = content
    if tcs:
        msg["tool_calls"] = tcs
        msg.setdefault("content", None)  # OpenAI shape: content present, may be null
    tcid = attrs.get(f"{p}.tool_call_id")
    if tcid:
        msg["tool_call_id"] = tcid
    return msg


def _messages(attrs: dict, base: str) -> list[dict]:
    out = []
    for i in _indices(attrs, base):
        m = _message(attrs, f"{base}.{i}")
        if m is not None:
            out.append(m)
    return out


# ---- spec form: gen_ai.{input,output}.messages as {role, parts} JSON --------
def _as_list(val: Any) -> list:
    """A `gen_ai.*.messages` attribute → a list (it's a JSON string on a span,
    or already-decoded structured records in an event body)."""
    if val is None:
        return []
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return []
    return val if isinstance(val, list) else [val]


def _spec_message(msg: dict) -> dict | None:
    """One OTel `{role, parts}` message → an OpenAI-wire message.

    Parts: text → content; tool_call → assistant tool_calls; tool_call_response
    → a tool-role result; thinking is dropped (not behavior-policy output).
    """
    if not isinstance(msg, dict):
        return None
    role = msg.get("role", "assistant")
    text, tool_calls, tool_call_id, response = [], [], None, None
    for part in msg.get("parts", []):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            if part.get("content") is not None:
                text.append(str(part["content"]))
        elif kind == "tool_call":
            tool_calls.append({
                "id": part.get("id") or f"call_{len(tool_calls)}",
                "type": "function",
                "function": {"name": part.get("name", ""),
                             "arguments": _as_json_str(part.get("arguments"))},
            })
        elif kind == "tool_call_response":
            tool_call_id = part.get("id")
            response = part.get("response")
    if role == "tool" or response is not None:
        content = response if isinstance(response, str) else (
            json.dumps(response) if response is not None else "")
        out = {"role": "tool", "content": content}
        if tool_call_id:
            out["tool_call_id"] = tool_call_id
        return out
    out: dict[str, Any] = {"role": role}
    if text:
        out["content"] = "".join(text)
    if tool_calls:
        out["tool_calls"] = tool_calls
        out.setdefault("content", None)
    if "content" not in out and not tool_calls:
        return None
    return out


def _spec_messages(val: Any) -> list[dict]:
    return [m for m in (_spec_message(x) for x in _as_list(val)) if m is not None]


# ---- OpenInference form: llm.{input,output}_messages.{i}.message.* -----------
# Used by Arize/Phoenix and OpenInference-instrumented agents. Tool calls nest
# under `.message.tool_calls.{j}.tool_call.function.{name,arguments}`.
def _oi_messages(attrs: dict, base: str) -> list[dict]:
    out = []
    for i in _indices(attrs, base):
        p = f"{base}.{i}.message"
        role = attrs.get(f"{p}.role")
        content = attrs.get(f"{p}.content")
        tcs = []
        tcbase = f"{p}.tool_calls"
        for j in _indices(attrs, tcbase):
            tc = f"{tcbase}.{j}.tool_call"
            name = attrs.get(f"{tc}.function.name") or attrs.get(f"{tc}.name")
            if not name:
                continue
            args = attrs.get(f"{tc}.function.arguments")
            if args is None:
                args = attrs.get(f"{tc}.arguments")
            tcs.append({"id": attrs.get(f"{tc}.id") or f"call_{j}", "type": "function",
                        "function": {"name": name, "arguments": _as_json_str(args)}})
        if role is None and content is None and not tcs:
            continue
        msg: dict[str, Any] = {"role": role or "assistant"}
        if content is not None:
            msg["content"] = content
        if tcs:
            msg["tool_calls"] = tcs
            msg.setdefault("content", None)
        tcid = attrs.get(f"{p}.tool_call_id")
        if tcid:
            msg["tool_call_id"] = tcid
        out.append(msg)
    return out


# ---- OpenAI-wire form: a [{role, content, tool_calls}] list (Langfuse blobs) --
def _wire_messages(val: Any) -> list[dict]:
    out = []
    for m in _as_list(val):
        if not isinstance(m, dict) or m.get("role") is None:
            continue
        msg: dict[str, Any] = {"role": m["role"]}
        content = m.get("content")
        if isinstance(content, list):  # content blocks → joined text
            content = "".join(
                str(b.get("text") or b.get("content") or "")
                for b in content if isinstance(b, dict)) or None
        if content is not None:
            msg["content"] = content
        tcs = m.get("tool_calls")
        if tcs:
            wired = []
            for k, tc in enumerate(tcs):
                fn = tc.get("function") or {}
                wired.append({"id": tc.get("id") or f"call_{k}", "type": "function",
                              "function": {"name": fn.get("name") or tc.get("name", ""),
                                           "arguments": _as_json_str(
                                               fn.get("arguments") if fn else tc.get("arguments"))}})
            msg["tool_calls"] = wired
            msg.setdefault("content", None)
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def _extract(attrs: dict, kind: str) -> list[dict]:
    """Prompt or response messages, trying every supported convention in turn:
    OTel GenAI spec → OpenInference indexed → GenAI indexed → OpenAI-wire blob."""
    if kind == "input":
        spec, oi, idx = attrs.get(_INPUT), "llm.input_messages", _PROMPT
        blob = attrs.get("llm.input_messages") or attrs.get("input.value")
    else:
        spec, oi, idx = attrs.get(_OUTPUT), "llm.output_messages", _COMPLETION
        blob = attrs.get("llm.output_messages") or attrs.get("output.value")
    return (_spec_messages(spec) or _oi_messages(attrs, oi)
            or _messages(attrs, idx) or _wire_messages(blob))


def _messages_from_events(events: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Event-form GenAI: prompt messages from `gen_ai.*.message`, the reply from
    `gen_ai.choice`."""
    prompt, completion = [], []
    for ev in events or []:
        name = ev.get("name")
        a = _flatten(ev.get("attributes", {}))
        if name in _EVENT_ROLES:
            m = _message({**a, "x.role": _EVENT_ROLES[name]}, "x") or {}
            m["role"] = _EVENT_ROLES[name]
            prompt.append(m)
        elif name == "gen_ai.choice":
            m = _message({**a, "x.role": a.get("role", "assistant")}, "x")
            if m:
                completion.append(m)
    return prompt, completion


# ---- span → call -----------------------------------------------------------
class _Call:
    __slots__ = ("trace", "ts", "messages", "response", "tools", "model")

    def __init__(self, trace, ts, messages, response, tools, model):
        self.trace, self.ts = trace, ts
        self.messages, self.response = messages, response
        self.tools, self.model = tools, model


def _span_call(span: dict) -> _Call | None:
    """An LLM/chat span → a (prompt, response) call, or None if it isn't one.

    Tries the spec `{role, parts}` form first, then the indexed-attribute form,
    then per-message log events — so a span from any GenAI instrumentor parses.
    """
    attrs = _flatten(span.get("attributes", {}))
    prompt = _extract(attrs, "input")
    completion = _extract(attrs, "output")
    if not prompt and not completion:  # fallback: log events
        prompt, completion = _messages_from_events(span.get("events", []))
    if not prompt and not completion:
        return None
    # system_instructions live apart from the message list — prepend them when
    # the prompt didn't already open with a system turn.
    sysmsgs = _system_text(attrs.get(_SYSTEM))
    if sysmsgs and not (prompt and prompt[0].get("role") == "system"):
        prompt = sysmsgs + prompt
    model = attrs.get(_RESP_MODEL) or attrs.get(_REQ_MODEL) or attrs.get("llm.model_name")
    trace = (attrs.get(_CONVERSATION) or span.get("trace_id") or span.get("traceId")
             or span.get("span_id") or "")
    ts = span.get("start_time") or span.get("startTimeUnixNano") or 0
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        ts = 0.0
    return _Call(trace, ts, prompt, completion[-1] if completion else None, None, model)


def _system_text(val: Any) -> list[dict]:
    """`gen_ai.system_instructions` is `[{type:text, content}]` — flatten to a
    single system message."""
    text = "".join(
        str(p.get("content", "")) for p in _as_list(val)
        if isinstance(p, dict) and p.get("type") == "text"
    )
    return [{"role": "system", "content": text}] if text else []


def _is_prefix(shorter: list[dict], longer: list[dict]) -> bool:
    return len(shorter) <= len(longer) and all(a == b for a, b in zip(shorter, longer))


def _reconstruct(calls: list[_Call], *, per_request: bool) -> list[list[dict]]:
    """Calls of one trace → conversation(s). Prefix-merge folds the agent loop
    (each turn re-sends the growing history) back into one conversation; branches
    or restarts split into separate ones. `per_request` keeps every call whole."""
    calls = sorted(calls, key=lambda c: c.ts)
    if per_request:
        return [c.messages + ([c.response] if c.response else []) for c in calls]
    buckets: list[list[dict]] = []
    for c in calls:
        convo = c.messages + ([c.response] if c.response else [])
        for b in reversed(buckets):
            if _is_prefix(b, c.messages):
                b[:] = convo
                break
        else:
            buckets.append(convo)
    return buckets


# ---- public API ------------------------------------------------------------
def from_spans(
    spans: Iterable[dict],
    *,
    builder: str = "conversation",
    reward_key: str | None = None,
) -> list[Trajectory]:
    """OTel GenAI spans → reconstructed `Trajectory` episodes.

    `spans` are span dicts with `attributes` (OTLP list, nested, or flat dot-keyed)
    and, for grouping, a `trace_id`. `builder` is "conversation" (default — fold
    each trace's agent loop into one multi-turn episode) or "per_request" (one
    episode per model call). `reward_key`, if given, reads a per-trace scalar
    from that span attribute and sets it as the trajectory `reward`.
    """
    if builder not in ("conversation", "per_request"):
        raise ValueError(f"unknown builder {builder!r} (conversation | per_request)")

    by_trace: dict[str, list[_Call]] = {}
    rewards: dict[str, float] = {}
    for span in spans:
        call = _span_call(span)
        if call is None:
            continue
        by_trace.setdefault(call.trace, []).append(call)
        if reward_key is not None:
            attrs = _flatten(span.get("attributes", {}))
            if reward_key in attrs:
                try:
                    rewards[call.trace] = float(attrs[reward_key])
                except (TypeError, ValueError):
                    pass

    out: list[Trajectory] = []
    for trace, calls in by_trace.items():
        model = next((c.model for c in calls if c.model), None)
        for convo in _reconstruct(calls, per_request=(builder == "per_request")):
            if not convo:
                continue
            out.append(Trajectory(
                messages=convo,
                reward=rewards.get(trace, 0.0),
                metadata={"trace_id": trace, "model": model, "source": "otel"},
            ))
    return out


def from_otlp(
    source: str | Path | dict | list,
    *,
    builder: str = "conversation",
    reward_key: str | None = None,
) -> list[Trajectory]:
    """Load spans from an OTLP/JSON export (file path, JSON string, or already
    decoded payload) and reconstruct trajectories.

    Accepts the OTLP `resourceSpans` envelope or a bare list of span dicts.
    `builder` and `reward_key` are forwarded to `from_spans`.
    """
    return from_spans(_spans_from_otlp(source), builder=builder, reward_key=reward_key)


def _spans_from_otlp(source: str | Path | dict | list, **kw) -> list[dict]:
    if isinstance(source, (str, Path)):
        text = Path(source).read_text() if Path(str(source)).exists() else str(source)
        try:
            source = json.loads(text)
        except json.JSONDecodeError:
            # JSONL: one span (or OTLP envelope) per line — the shape Arize and
            # OpenInference exporters write.
            source = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    if isinstance(source, list):
        if source and "resourceSpans" in source[0]:
            return [sp for env in source for sp in _spans_from_otlp(env)]
        return source
    spans: list[dict] = []
    for rs in source.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", rs.get("instrumentationLibrarySpans", [])):
            for sp in ss.get("spans", []):
                spans.append({
                    "trace_id": sp.get("traceId") or sp.get("trace_id"),
                    "span_id": sp.get("spanId") or sp.get("span_id"),
                    "parent_id": sp.get("parentSpanId") or sp.get("parent_id"),
                    "name": sp.get("name"),
                    "start_time": sp.get("startTimeUnixNano") or sp.get("start_time"),
                    "attributes": sp.get("attributes", []),
                    "events": sp.get("events", []),
                })
    return spans


def to_dataset(
    source: Iterable[dict] | Iterable[Trajectory] | str | Path | dict,
    *,
    builder: str = "conversation",
    reward_key: str | None = None,
    min_reward: float | None = None,
    name: str | None = None,
) -> Dataset:
    """Traces → a chat `Dataset` ready for `model.finetune(..., method="lora")`.

    `source` may be trajectories (from `from_spans`/`from_otlp`), raw span dicts,
    or an OTLP file/payload — whatever you have. `min_reward` keeps only
    trajectories scoring at least that (use with `reward_key` to distil only the
    runs your evals liked).
    """
    trajectories = _as_trajectories(source, builder=builder, reward_key=reward_key)
    if min_reward is not None:
        trajectories = [t for t in trajectories if t.reward >= min_reward]
    rows = []
    for t in trajectories:
        if not any(m.get("role") == "assistant" for m in t.messages):
            continue  # nothing for the model to learn to produce
        row: dict[str, Any] = {"messages": t.messages}
        if t.tools:
            row["tools"] = t.tools
        rows.append(row)
    if not rows:
        raise ValueError(
            "no trainable conversations in these traces — check the spans carry "
            "gen_ai.prompt/gen_ai.completion attributes (and reward_key/min_reward "
            "aren't filtering everything out)"
        )
    return Dataset.from_list(rows, name=name or "traces", format="chat")


def group(
    trajectories: Iterable[Trajectory],
    *,
    by: str | Callable[[Trajectory], str] = "task",
) -> list[TrajectoryGroup]:
    """Bucket trajectories that attempt the *same* task into `TrajectoryGroup`s
    for RL (`finetune(method="grpo")`) or preference pairs (`to_preference_rows`).

    `by` is "task" (group by the opening user message), a metadata key name, or a
    callable returning the grouping key. Only groups with ≥2 attempts are kept.
    """
    if callable(by):
        keyfn = by
    elif by == "task":
        keyfn = lambda t: t.first_user_content()  # noqa: E731
    else:
        keyfn = lambda t: str(t.metadata.get(by, ""))  # noqa: E731
    buckets: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        buckets.setdefault(keyfn(t), []).append(t)
    return [TrajectoryGroup(ts) for ts in buckets.values() if len(ts) >= 2]


def _as_trajectories(source, *, builder, reward_key) -> list[Trajectory]:
    items = list(source) if not isinstance(source, (str, Path, dict)) else None
    if items is not None and items and isinstance(items[0], Trajectory):
        return items
    if items is not None and (not items or isinstance(items[0], dict)):
        return from_spans(items, builder=builder, reward_key=reward_key)
    return from_spans(_spans_from_otlp(source), builder=builder, reward_key=reward_key)
