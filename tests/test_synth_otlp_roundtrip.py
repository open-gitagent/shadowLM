"""The OTLP round trip: what we write, our own reader reads back unchanged.

This is the load-bearing test of the whole synthesizer. `emit.to_otlp` writes
OpenTelemetry GenAI spans and `traces.from_otlp` reconstructs episodes from
them; if those two ever disagree, "inject ready" is a claim rather than a fact.
Tool calls and multi-span agent loops are included because that is exactly where
a naive emitter loses information.
"""

import json

from shadowlm import traces
from shadowlm.rl import Trajectory
from shadowlm.synth import emit

_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


def _agent_episode(reward=1.0):
    """ask → tool call → tool result → answer: two assistant turns, so two spans."""
    return Trajectory(
        messages=[
            {"role": "system", "content": "You are a weather agent."},
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_0", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": "18C, sunny"},
            {"role": "assistant", "content": "It's 18°C and sunny in Paris."},
        ],
        tools=[_TOOL], reward=reward)


def _roundtrip(trajectories, **kwargs):
    payload = emit.to_otlp(trajectories, **kwargs)
    return traces.from_otlp(payload, reward_key=emit.SCORE_KEY)


def test_plain_episode_survives_intact():
    original = Trajectory(messages=[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"}], reward=0.8)
    back = _roundtrip([original])
    assert len(back) == 1
    assert back[0].messages == original.messages
    assert back[0].reward == 0.8


def test_agent_loop_folds_back_into_one_episode():
    original = _agent_episode()
    back = _roundtrip([original])
    assert len(back) == 1, "the two spans should prefix-merge into one episode"
    assert back[0].messages == original.messages


def test_tool_calls_and_schemas_survive():
    back = _roundtrip([_agent_episode()])[0]
    call = back.messages[2]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert back.messages[3]["tool_call_id"] == "call_0"
    assert back.tools == [_TOOL]


def test_separate_episodes_stay_separate():
    back = _roundtrip([_agent_episode(), _agent_episode()])
    assert len(back) == 2


def test_rewards_ride_along():
    back = _roundtrip([_agent_episode(reward=0.25)])
    assert back[0].reward == 0.25


def test_written_file_reloads(tmp_path):
    path = tmp_path / "spans.json"
    emit.to_otlp([_agent_episode()], path=path)
    back = traces.from_otlp(path, reward_key=emit.SCORE_KEY)
    assert back[0].messages[-1]["content"] == "It's 18°C and sunny in Paris."


def test_output_is_deterministic_for_a_seed():
    once = emit.to_otlp([_agent_episode()], seed=7)
    twice = emit.to_otlp([_agent_episode()], seed=7)
    assert once == twice  # ids and timestamps included — no wall clock anywhere
    assert emit.to_otlp([_agent_episode()], seed=8) != once


def test_payload_is_a_real_otlp_envelope():
    payload = emit.to_otlp([_agent_episode()])
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 2  # one per assistant turn
    attrs = {a["key"]: a["value"] for a in spans[0]["attributes"]}
    assert "stringValue" in attrs["gen_ai.input.messages"]
    assert attrs["gen_ai.operation.name"]["stringValue"] == "chat"
    assert len(spans[0]["traceId"]) == 32 and len(spans[0]["spanId"]) == 16
