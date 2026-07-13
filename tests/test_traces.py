"""Trace ingestion: OTel GenAI spans → trajectories → chat dataset / RL groups."""

import json

import shadowlm as slm
from shadowlm import traces


def _llm_span(trace_id, ts, prompt, completion, *, model="gpt-4o", reward=None):
    """A chat span in OTel GenAI indexed-attribute form."""
    attrs = {"gen_ai.response.model": model}
    for i, m in enumerate(prompt):
        _put_message(attrs, f"gen_ai.prompt.{i}", m)
    for i, m in enumerate(completion):
        _put_message(attrs, f"gen_ai.completion.{i}", m)
    if reward is not None:
        attrs["eval.score"] = reward
    return {"trace_id": trace_id, "start_time": ts, "name": "chat", "attributes": attrs}


def _put_message(attrs, p, m):
    attrs[f"{p}.role"] = m["role"]
    if "content" in m and m["content"] is not None:
        attrs[f"{p}.content"] = m["content"]
    if "tool_call_id" in m:
        attrs[f"{p}.tool_call_id"] = m["tool_call_id"]
    for j, tc in enumerate(m.get("tool_calls", [])):
        attrs[f"{p}.tool_calls.{j}.id"] = tc["id"]
        attrs[f"{p}.tool_calls.{j}.name"] = tc["function"]["name"]
        attrs[f"{p}.tool_calls.{j}.arguments"] = tc["function"]["arguments"]


# A two-turn agent run: ask → tool call → tool result → final answer. The agent
# loop re-sends the growing history, so each span's prompt extends the last.
def _agent_run(trace_id):
    sys = {"role": "system", "content": "You are a weather agent."}
    user = {"role": "user", "content": "Weather in Paris?"}
    tool_call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_0", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]}
    tool_res = {"role": "tool", "tool_call_id": "call_0", "content": "18C, sunny"}
    final = {"role": "assistant", "content": "It's 18°C and sunny in Paris."}
    return [
        _llm_span(trace_id, 1.0, [sys, user], [tool_call]),
        _llm_span(trace_id, 2.0, [sys, user, tool_call, tool_res], [final], reward=1.0),
    ]


def test_conversation_builder_merges_agent_loop():
    trajs = traces.from_spans(_agent_run("t1"), reward_key="eval.score")
    assert len(trajs) == 1
    msgs = trajs[0].messages
    # full conversation reconstructed from the two overlapping spans
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant"]
    assert msgs[-1]["content"] == "It's 18°C and sunny in Paris."
    assert trajs[0].reward == 1.0
    assert trajs[0].metadata["model"] == "gpt-4o"


def test_tool_calls_survive_in_openai_wire_shape():
    traj = traces.from_spans(_agent_run("t1"))[0]
    call = traj.messages[2]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert traj.messages[3]["tool_call_id"] == "call_0"


def test_per_request_builder_keeps_each_call():
    trajs = traces.from_spans(_agent_run("t1"), builder="per_request")
    assert len(trajs) == 2  # one trajectory per model call


def test_separate_traces_stay_separate():
    spans = _agent_run("t1") + _agent_run("t2")
    assert len(traces.from_spans(spans)) == 2


def test_to_dataset_is_chat_format():
    ds = traces.to_dataset(_agent_run("t1"))
    assert ds.format == "chat"
    assert len(ds.rows) == 1
    assert "messages" in ds.rows[0]


def test_min_reward_filters():
    spans = _agent_run("t1") + _agent_run("t2")
    # only the reward-bearing completion spans carry eval.score=1.0 → both pass;
    # raise the bar above any score → nothing trainable
    import pytest
    with pytest.raises(ValueError):
        traces.to_dataset(spans, reward_key="eval.score", min_reward=2.0)


def test_group_buckets_same_task():
    trajs = traces.from_spans(_agent_run("t1") + _agent_run("t2"))
    groups = traces.group(trajs, by="task")  # same opening user message
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_otlp_envelope_parses():
    # OTLP/JSON shape: resourceSpans → scopeSpans → spans, attributes as kv list
    span = _agent_run("t1")[1]
    kv_attrs = [{"key": k, "value": {"stringValue": str(v)} if not isinstance(v, (int, float))
                 else {"doubleValue": float(v)}} for k, v in span["attributes"].items()]
    payload = {"resourceSpans": [{"scopeSpans": [{"spans": [{
        "traceId": "t1", "name": "chat", "startTimeUnixNano": "2",
        "attributes": kv_attrs}]}]}]}
    trajs = traces.from_otlp(payload)
    assert len(trajs) == 1
    assert trajs[0].messages[-1]["role"] == "assistant"


def test_event_form_genai_spans():
    # newer GenAI spec: messages as span log events instead of indexed attributes
    span = {"trace_id": "e1", "start_time": 1.0, "name": "chat", "attributes": {},
            "events": [
                {"name": "gen_ai.user.message", "attributes": {"content": "hi"}},
                {"name": "gen_ai.choice", "attributes": {"role": "assistant", "content": "hello"}},
            ]}
    trajs = traces.from_spans([span])
    assert len(trajs) == 1
    assert [m["role"] for m in trajs[0].messages] == ["user", "assistant"]


def test_exported_on_package():
    assert slm.traces is traces


# ---- spec form: gen_ai.input/output.messages as {role, parts} JSON strings ---
def _spec_span(trace_id, ts, inp, out, *, score=None):
    attrs = {
        "gen_ai.response.model": "gpt-4o",
        "gen_ai.input.messages": json.dumps(inp),
        "gen_ai.output.messages": json.dumps(out),
    }
    if score is not None:
        attrs["eval.score"] = score
    return {"trace_id": trace_id, "start_time": ts, "name": "chat", "attributes": attrs}


def _spec_run(trace_id):
    sys = {"role": "system", "parts": [{"type": "text", "content": "be helpful"}]}
    user = {"role": "user", "parts": [{"type": "text", "content": "weather in Paris?"}]}
    call = {"role": "assistant", "finish_reason": "tool_calls", "parts": [
        {"type": "tool_call", "id": "c1", "name": "get_weather",
         "arguments": '{"city":"Paris"}'}]}
    result = {"role": "tool", "parts": [
        {"type": "tool_call_response", "id": "c1", "response": "18C sunny"}]}
    final = {"role": "assistant", "finish_reason": "stop", "parts": [
        {"type": "text", "content": "It's 18C and sunny."}]}
    return [
        _spec_span(trace_id, 1.0, [sys, user], [call]),
        _spec_span(trace_id, 2.0, [sys, user, call, result], [final], score=1.0),
    ]


def test_spec_form_parses_parts_and_tool_calls():
    traj = traces.from_spans(_spec_run("s1"), reward_key="eval.score")[0]
    assert [m["role"] for m in traj.messages] == \
        ["system", "user", "assistant", "tool", "assistant"]
    tc = traj.messages[2]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
    assert traj.messages[3] == {"role": "tool", "content": "18C sunny", "tool_call_id": "c1"}
    assert traj.messages[-1]["content"] == "It's 18C and sunny."
    assert traj.reward == 1.0


def test_system_instructions_prepended_when_absent():
    span = {"trace_id": "si", "start_time": 1.0, "attributes": {
        "gen_ai.system_instructions": json.dumps([{"type": "text", "content": "be terse"}]),
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]),
        "gen_ai.output.messages": json.dumps(
            [{"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}]),
    }}
    traj = traces.from_spans([span])[0]
    assert traj.messages[0] == {"role": "system", "content": "be terse"}
    assert [m["role"] for m in traj.messages] == ["system", "user", "assistant"]


def test_conversation_id_groups_across_traces():
    # two separate traces sharing a gen_ai.conversation.id are one episode
    a = _spec_span("ta", 1.0,
                   [{"role": "user", "parts": [{"type": "text", "content": "q1"}]}],
                   [{"role": "assistant", "parts": [{"type": "text", "content": "a1"}]}])
    b = _spec_span("tb", 2.0,
                   [{"role": "user", "parts": [{"type": "text", "content": "q1"}]},
                    {"role": "assistant", "parts": [{"type": "text", "content": "a1"}]},
                    {"role": "user", "parts": [{"type": "text", "content": "q2"}]}],
                   [{"role": "assistant", "parts": [{"type": "text", "content": "a2"}]}])
    a["attributes"]["gen_ai.conversation.id"] = "conv-1"
    b["attributes"]["gen_ai.conversation.id"] = "conv-1"
    trajs = traces.from_spans([a, b])
    assert len(trajs) == 1
    assert [m["role"] for m in trajs[0].messages] == \
        ["user", "assistant", "user", "assistant"]


# ---- OpenInference form: llm.input_messages.{i}.message.* (Arize/Phoenix/HALO)
def test_openinference_indexed_form_with_tool_calls():
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.model_name": "claude-sonnet-4-5",
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": "be helpful",
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.content": "search the web",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.id": "tc1",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "web_search",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"q":"x"}',
    }
    traj = traces.from_spans([{"trace_id": "oi1", "start_time": 1.0, "attributes": attrs}])[0]
    assert [m["role"] for m in traj.messages] == ["system", "user", "assistant"]
    assert traj.metadata["model"] == "claude-sonnet-4-5"
    tc = traj.messages[2]["tool_calls"][0]
    assert tc["id"] == "tc1" and tc["function"]["name"] == "web_search"


# ---- OpenAI-wire blob form: llm.input_messages = [{role, content}] (Langfuse) -
def test_openai_wire_blob_form():
    attrs = {
        "llm.input_messages": json.dumps([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [
                {"id": "w1", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "w1", "content": "ok"},
        ]),
        "llm.output_messages": json.dumps([{"role": "assistant", "content": "done"}]),
    }
    traj = traces.from_spans([{"trace_id": "lf1", "start_time": 1.0, "attributes": attrs}])[0]
    assert [m["role"] for m in traj.messages] == ["user", "assistant", "tool", "assistant"]
    assert traj.messages[1]["tool_calls"][0]["function"]["name"] == "f"
    assert traj.messages[2]["tool_call_id"] == "w1"


def test_jsonl_one_span_per_line(tmp_path):
    # HALO/Arize style: a span per line, no resourceSpans envelope
    p = tmp_path / "traces.jsonl"
    spans = _spec_run("j1")
    p.write_text("\n".join(json.dumps(s) for s in spans))
    trajs = traces.from_otlp(str(p), reward_key="eval.score")
    assert len(trajs) == 1
    assert trajs[0].messages[-1]["content"] == "It's 18C and sunny."


def test_sample_export_file_end_to_end():
    eps = traces.from_otlp("examples/data/agent_traces.otlp.json", reward_key="eval.score")
    assert len(eps) == 2
    assert {round(t.reward, 1) for t in eps} == {1.0, 0.9}
    for t in eps:
        assert [m["role"] for m in t.messages] == \
            ["system", "user", "assistant", "tool", "assistant"]
    ds = traces.to_dataset(eps, min_reward=0.5)
    assert ds.format == "chat" and len(ds.rows) == 2
