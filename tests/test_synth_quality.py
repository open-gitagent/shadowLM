"""Validators and dedup — the gate everything synthesized passes through."""

from shadowlm.synth.quality import Dedup, first_json_array, validate

_TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


def _call(name="search", args='{"q": "x"}', cid="call_1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


def test_a_clean_conversation_passes():
    assert validate([{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "hello"}]) == []


def test_must_end_on_an_assistant_turn():
    problems = validate([{"role": "assistant", "content": "hi"},
                         {"role": "user", "content": "and you?"}])
    assert any("end with an assistant" in p for p in problems)


def test_empty_final_turn_is_caught():
    assert validate([{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": ""}])


def test_system_turn_must_come_first_and_once():
    assert validate([{"role": "user", "content": "hi"},
                     {"role": "system", "content": "be nice"},
                     {"role": "assistant", "content": "ok"}])


def test_unknown_role_is_caught():
    assert validate([{"role": "narrator", "content": "meanwhile"},
                     {"role": "assistant", "content": "ok"}])


def test_placeholders_and_boilerplate_are_caught():
    assert validate([{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "Dear [NAME], hello"}])
    assert validate([{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "As an AI language model I"}])


def test_wellformed_tool_exchange_passes():
    assert validate([
        {"role": "user", "content": "search x"},
        {"role": "assistant", "content": None, "tool_calls": [_call()]},
        {"role": "tool", "tool_call_id": "call_1", "content": "found"},
        {"role": "assistant", "content": "here it is"}], tools=_TOOLS) == []


def test_unparseable_tool_arguments_are_caught():
    problems = validate([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [_call(args="{oops")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "r"},
        {"role": "assistant", "content": "done"}], tools=_TOOLS)
    assert any("unparseable arguments" in p for p in problems)


def test_undeclared_tool_is_caught():
    problems = validate([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [_call(name="rm_rf")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "r"},
        {"role": "assistant", "content": "done"}], tools=_TOOLS)
    assert any("undeclared tool" in p for p in problems)


def test_unanswered_tool_call_is_caught():
    problems = validate([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [_call()]},
        {"role": "assistant", "content": "done"}], tools=_TOOLS)
    assert any("never answered" in p for p in problems)


def test_orphan_tool_result_is_caught():
    problems = validate([
        {"role": "user", "content": "x"},
        {"role": "tool", "tool_call_id": "ghost", "content": "r"},
        {"role": "assistant", "content": "done"}])
    assert any("unknown call" in p for p in problems)


def test_dedup_rejects_exact_and_near_repeats():
    dedup = Dedup(threshold=0.7)
    assert dedup.accept("how much does the plan cost")
    assert not dedup.accept("how much does the plan cost")          # exact
    assert not dedup.accept("how much does the plan cost really")   # near
    assert dedup.accept("where is the office located")              # different


def test_dedup_judges_near_repeats_on_the_query_side():
    """Same question, different answer, is still a duplicate question."""
    dedup = Dedup(threshold=0.7)
    assert dedup.accept("q: what is the refund window", key="what is the refund window")
    assert not dedup.accept("q: what is the refund window — other answer",
                            key="what is the refund window")


def test_dedup_can_be_preseeded_with_real_data():
    dedup = Dedup()
    dedup.seed(["the exact question a real user asked"])
    assert not dedup.accept("the exact question a real user asked")


def test_first_json_array_digs_it_out_of_prose():
    assert first_json_array('Sure! ["a", "b"] hope that helps') == ["a", "b"]
    assert first_json_array("no array here") is None
    assert first_json_array('broken [1, 2 then [3, 4]') == [3, 4]
