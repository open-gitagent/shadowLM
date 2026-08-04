"""The synthesis pipeline end to end, against a scripted teacher.

No network, no model: `FakeTeacher` recognises which prompt it was handed and
answers in kind, so every branch — taxonomy, instances, repair, dedup, the judge
gate — is exercised deterministically.
"""

import json

import pytest

import shadowlm as slm
from shadowlm.synth import _judged_input, synthesize
from shadowlm.synth.generate import _parse_conversation, _wire_tool_calls
from shadowlm.synth.quality import validate


class FakeTeacher:
    """Answers by recognising the prompt it got.

    `junk_replies` makes the first N conversation calls return unparseable text:
    1 exercises the corrective retry, 2 exhausts it and forces a rejection.
    """

    name = "fake"
    parallelism = 1

    def __init__(self, *, scenarios=8, scores=("0.9",), junk_replies=0):
        self.scenarios = scenarios
        self.scores = list(scores)
        self.junk_replies = junk_replies
        self.prompts = []
        self._conversations = 0
        self._scored = 0

    def chat(self, messages, **_):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "0.0 to 1.0" in prompt:
            score = self.scores[self._scored % len(self.scores)]
            self._scored += 1
            return score
        if '"scenario"' in prompt:  # taxonomy, or episode-pattern variations
            return json.dumps([{"scenario": f"scenario {i}", "difficulty": "easy",
                                "angle": f"angle {i}"} for i in range(self.scenarios)])
        if "factual claims" in prompt:
            return json.dumps(["the sky is blue", "water boils at 100C"])
        if "questions that should all retrieve" in prompt:
            fact = prompt.split("FACT: ")[1].split("\n")[0]
            return json.dumps({
                "questions": [f"alpha bravo {fact}", f"charlie delta {fact}",
                              f"echo foxtrot {fact}", f"golf hotel {fact}"],
                "answer": f"the answer to {fact}"})
        if "writing test data" in prompt:  # the user's turn for a preference pair
            self._conversations += 1
            return f"user question {self._conversations}"
        if "FLAWED" in prompt:
            return "a flawed answer"
        if "Answer this as well as you possibly can" in prompt:
            return "the good answer"
        if "training conversation" in prompt:
            self._conversations += 1
            if self._conversations <= self.junk_replies:
                return "sorry, prose instead of JSON"
            return json.dumps({"messages": [
                {"role": "user", "content": f"question {self._conversations}"},
                {"role": "assistant", "content": f"answer {self._conversations}"}]})
        raise AssertionError(f"FakeTeacher got an unexpected prompt:\n{prompt[:300]}")


def test_task_seed_produces_chat_rows():
    run = synthesize(task="triage billing email", teacher=FakeTeacher(),
                     n=8, method="lora", verbose=False)
    assert run.format == "chat"
    assert run.dataset.format == "chat"
    assert len(run.dataset.rows) == 8
    assert all(r["messages"][-1]["role"] == "assistant" for r in run.dataset.rows)


def test_the_funnel_reconciles():
    """Every generated row is accounted for — nothing silently disappears."""
    run = synthesize(task="t", teacher=FakeTeacher(scores=("0.9", "0.1")),
                     n=6, method="lora", verbose=False)
    r = run.report
    assert r.balanced, r.summary()
    assert r.generated == (r.kept + r.rejected_validation + r.rejected_dedup
                           + r.rejected_judge + r.rejected_flat + r.surplus)
    assert r.rejected_judge > 0  # the 0.1 scores were gated out
    assert r.teacher_calls > 0


def test_rows_carry_provenance():
    run = synthesize(task="t", teacher=FakeTeacher(), n=4, verbose=False)
    meta = run.trajectories[0].metadata
    assert meta["source"] == "synth"
    assert meta["teacher"] == "fake"
    assert meta["taxonomy_path"].startswith("scenario")
    assert meta["style"] and "judge_score" in run.trajectories[0].metrics


def test_same_seed_gives_the_same_rows():
    kwargs = dict(task="t", n=6, method="lora", verbose=False)
    first = synthesize(teacher=FakeTeacher(), **kwargs)
    second = synthesize(teacher=FakeTeacher(), **kwargs)
    assert first.dataset.rows == second.dataset.rows


def test_one_bad_reply_is_repaired_two_is_rejected():
    repaired = synthesize(task="t", teacher=FakeTeacher(junk_replies=1), n=4,
                          verbose=False)
    assert repaired.report.repaired == 1
    assert repaired.report.rejected_validation == 0

    rejected = synthesize(task="t", teacher=FakeTeacher(junk_replies=2), n=4,
                          verbose=False)
    assert rejected.report.rejected_validation == 1
    assert rejected.report.balanced


def test_duplicates_are_rejected_not_shipped():
    """A teacher that repeats itself gets caught rather than padding the set."""
    class Repeater(FakeTeacher):
        def chat(self, messages, **_):
            if "training conversation" in messages[-1]["content"]:
                return json.dumps({"messages": [
                    {"role": "user", "content": "the very same question"},
                    {"role": "assistant", "content": "the very same answer"}]})
            return super().chat(messages)

    run = synthesize(task="t", teacher=Repeater(), n=8, verbose=False)
    assert len(run.dataset.rows) == 1
    assert run.report.rejected_dedup > 0
    assert run.report.balanced


def test_document_seed_grounds_on_the_passage(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("The sky is blue.\n\nWater boils at 100C at sea level.")
    teacher = FakeTeacher()
    run = synthesize(document=doc, teacher=teacher, n=2, verbose=False)
    # the passage reaches both the generator (as source material to stay inside)
    # and the judge (as the reference its answer is scored against)
    assert "SOURCE MATERIAL" in "".join(teacher.prompts)
    assert "REFERENCE: The sky is blue" in "".join(teacher.prompts)
    assert run.trajectories[0].metadata["grounding"].startswith("The sky is blue")


def test_document_rejects_formats_it_cannot_read(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="convert it to"):
        synthesize(document=pdf, teacher=FakeTeacher(), n=2, verbose=False)


def test_episodes_seed_never_clones_the_real_data():
    real = slm.Trajectory(messages=[
        {"role": "user", "content": "question 1"},
        {"role": "assistant", "content": "real answer"}])

    class Cloner(FakeTeacher):
        def chat(self, messages, **_):
            if "training conversation" in messages[-1]["content"]:
                return json.dumps({"messages": [
                    {"role": "user", "content": "question 1"},
                    {"role": "assistant", "content": "real answer"}]})
            return super().chat(messages)

    # the dedup pool is pre-seeded with the real episodes, so a teacher that
    # regurgitates them produces nothing at all rather than laundering the
    # user's own data back as "synthetic"
    with pytest.raises(RuntimeError, match="nothing usable"):
        synthesize(episodes=[real, real], teacher=Cloner(), n=4, verbose=False,
                   min_score=None)


def test_no_seed_and_no_teacher_fail_loudly():
    with pytest.raises(ValueError, match="needs a seed"):
        synthesize(teacher=FakeTeacher(), n=2, verbose=False)
    with pytest.raises(ValueError, match="needs teacher"):
        synthesize(task="t", teacher=None, n=2, verbose=False)


def test_everything_rejected_raises_with_the_counts():
    with pytest.raises(RuntimeError, match="nothing usable"):
        synthesize(task="t", teacher=FakeTeacher(scores=("0.0",)), n=4,
                   min_score=0.9, verbose=False)


def test_preference_pairs_have_a_question_and_two_distinct_answers():
    run = synthesize(task="t", teacher=FakeTeacher(), n=4, method="dpo",
                     verbose=False)
    assert run.dataset.format == "preference"
    for row in run.dataset.rows:
        assert set(row) == {"prompt", "chosen", "rejected"}
        assert row["prompt"].startswith("user question")   # the user's turn
        assert row["chosen"] == "the good answer"
        assert row["rejected"] == "a flawed answer"


def test_a_student_supplies_the_rejected_side_when_given():
    """The shadowing-native pairing: chosen is the teacher, rejected is the
    student, so DPO targets exactly the gap between them."""
    class Student:
        name = "student"

        def chat(self, messages, **_):
            return "the student's weaker answer"

    # judge alternates (chosen, rejected): the student must actually score
    # lower or the inversion gate — correctly — finds nothing to prefer
    run = synthesize(task="t", teacher=FakeTeacher(scores=("0.9", "0.2")),
                     student=Student(), n=2, method="dpo", verbose=False)
    assert run.dataset.rows[0]["rejected"] == "the student's weaker answer"
    assert run.trajectories[0].metadata["rejected_from"] == "student"


def test_flat_grpo_groups_are_counted_not_silently_dropped():
    """A group whose attempts all scored the same carries no GRPO signal and is
    dropped at emit — report.kept must then say what run.groups actually holds,
    not what survived the earlier gates."""
    scores = ("0.9", "0.5", "0.9", "0.5",   # scenario 0 — real spread
              "0.7", "0.7", "0.7", "0.7")   # scenario 1 — flat, no signal
    run = synthesize(task="t", teacher=FakeTeacher(scenarios=2, scores=scores),
                     n=8, method="grpo", verbose=False)
    assert len(run.groups) == 1
    assert run.report.kept == 4
    assert run.report.rejected_flat == 4
    assert run.report.balanced, run.report.summary()
    assert any("no reward spread" in t.metadata["reject_reason"]
               for t in run.rejected)


def test_document_topup_rounds_reach_later_facts():
    """When round one under-delivers, round two must advance into facts not yet
    planned — not re-extract and regenerate the first ones, which dedup would
    reject, stalling the run below its target."""
    class DocTeacher(FakeTeacher):
        def chat(self, messages, **_):
            prompt = messages[-1]["content"]
            if "factual claims" in prompt:
                self.prompts.append(prompt)
                return json.dumps(["fact A", "fact B", "fact C"])
            if ("questions that should all retrieve" in prompt
                    and "fact B" in prompt):
                return json.dumps({"questions": [], "answer": ""})
            return super().chat(messages)  # fact A / C paraphrases, judging

    teacher = DocTeacher()
    run = synthesize(document="One paragraph of source text.", teacher=teacher,
                     n=8, method="more_plus", verbose=False)
    facts = {t.metadata["taxonomy_path"] for t in run.trajectories}
    assert facts == {"fact A", "fact C"}   # round 2 moved past A and B
    assert len(run.trajectories) == 8
    # and the extraction ran once — the second round read the cache
    assert sum("factual claims" in p for p in teacher.prompts) == 1


def test_a_failing_teacher_ends_the_run_with_what_it_has():
    """One dead call must not lose every row already paid for; a dead teacher
    must not burn a retry cycle per remaining job either."""
    class Flaky(FakeTeacher):
        def chat(self, messages, **_):
            if ("training conversation" in messages[-1]["content"]
                    and self._conversations >= 2):
                self._conversations += 1
                raise RuntimeError("boom: connection reset")
            return super().chat(messages)

    run = synthesize(task="t", teacher=Flaky(scenarios=4), n=8, verbose=False)
    assert len(run.dataset.rows) == 2          # what survived, not an exception
    assert "stopped early" in run.report.note
    assert "boom" in run.report.note
    assert run.report.balanced, run.report.summary()


def test_a_dead_teacher_raises_with_its_own_error():
    class Dead(FakeTeacher):
        def chat(self, messages, **_):
            if "training conversation" in messages[-1]["content"]:
                raise RuntimeError("boom: 502 bad gateway")
            return super().chat(messages)

    with pytest.raises(RuntimeError, match="boom: 502"):
        synthesize(task="t", teacher=Dead(), n=4, verbose=False)


def test_a_student_that_beats_the_teacher_produces_no_pair():
    """chosen=teacher / rejected=student trains DPO on the gap between them —
    a pair where the student scored higher would train the gap backwards."""
    class Student:
        name = "student"

        def chat(self, messages, **_):
            return "the student's weaker answer"

    # judge scores alternate (chosen, rejected) per pair:
    # pair 1 → 0.9 vs 0.2 (kept) · pair 2 → 0.6 vs 0.9 (inverted → dropped)
    teacher = FakeTeacher(scores=("0.9", "0.2", "0.6", "0.9"))
    run = synthesize(task="t", teacher=teacher, student=Student(), n=2,
                     method="dpo", verbose=False)
    assert run.report.kept == 2
    assert run.report.rejected_judge == 2
    assert all(t.metrics["rejected_score"] < t.reward for t in run.trajectories)
    assert any("nothing to prefer" in t.metadata["reject_reason"]
               for t in run.rejected)


def test_a_teacher_that_answers_instead_of_asking_is_rejected():
    """Handed the task, a teacher will sometimes perform it where the user's
    message belongs. The pair then teaches nothing — 'chosen' merely restates
    the 'question' — so it must not reach the dataset."""
    class Confused(FakeTeacher):
        def chat(self, messages, **_):
            prompt = messages[-1]["content"]
            if "writing test data" in prompt:
                return "I can process your refund of $150 right away."
            if "Answer this as well as you possibly can" in prompt:
                return "I can process your refund of $150 right away for you."
            return super().chat(messages)

    with pytest.raises(RuntimeError, match="nothing usable"):
        synthesize(task="t", teacher=Confused(), n=4, method="dpo", verbose=False)


# ---- shaping the teacher's output -------------------------------------------
# Every case below cost a live run to find: real teachers fail in ways a
# cooperative fake never will.

def test_shallow_tool_shape_becomes_the_wire_form():
    """Teachers miscount braces in the real 4-deep shape, so the prompt asks for
    a flat one and we build the protocol — including ids that always match."""
    messages = _wire_tool_calls([
        {"role": "user", "content": "refund invoice 5678"},
        {"role": "assistant", "content": "Looking that up.",
         "call": {"name": "lookup_invoice", "args": {"invoice_id": "5678"}}},
        {"role": "tool", "result": "Invoice 5678 is $150."},
        {"role": "assistant", "content": "It's $150 — shall I refund it?"},
    ])
    call = messages[1]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "lookup_invoice"
    # arguments must be a JSON *string* on the wire, not the object we asked for
    assert json.loads(call["function"]["arguments"]) == {"invoice_id": "5678"}
    assert messages[2]["tool_call_id"] == call["id"]
    assert messages[2]["content"] == "Invoice 5678 is $150."
    assert "call" not in messages[1] and "result" not in messages[2]
    assert validate(messages, tools=[{"type": "function", "function": {
        "name": "lookup_invoice"}}]) == []


def test_native_tool_shape_is_tidied_rather_than_rejected():
    messages = _wire_tool_calls([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_9", "type": "function", "function": {
                "name": "f", "arguments": {"a": 1}}}]},   # object, not a string
        {"role": "tool", "result": "done"},
        {"role": "assistant", "content": "ok"},
    ])
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert messages[2]["tool_call_id"] == "call_9"  # the teacher's own id is kept


def test_a_truncated_reply_does_not_parse_to_a_stray_message():
    """Cut off mid-JSON, an object scan happily returns the *first message* —
    a dict that parses, carries no conversation, and looks like a valid empty
    result. It has to read as a failure so the retry fires."""
    truncated = ('{"messages":[{"role":"user","content":"hello"},'
                 '{"role":"assistant","content":"partial')
    assert _parse_conversation(truncated) is None
    whole = '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}'
    assert len(_parse_conversation(whole)) == 2
    # a teacher that skips the wrapper object is still understood
    assert len(_parse_conversation('[{"role":"user","content":"hi"}]')) == 1


def test_the_judge_sees_the_whole_lead_up_not_just_the_first_question():
    """A tool episode's answer cites what a tool returned; judged against the
    opening question alone it reads as unsupported and gets thrown away."""
    single = slm.Trajectory(messages=[{"role": "user", "content": "what is X?"},
                                      {"role": "assistant", "content": "X is Y"}])
    assert _judged_input(single) == "what is X?"

    episode = slm.Trajectory(messages=[
        {"role": "user", "content": "refund invoice 5678"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "lookup_invoice", "arguments": '{"id":"5678"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Invoice 5678 is $150."},
        {"role": "assistant", "content": "It's $150."},
    ])
    seen = _judged_input(episode)
    assert "Invoice 5678 is $150." in seen      # the evidence for the answer
    assert "lookup_invoice" in seen             # and the call that fetched it
    assert "It's $150." not in seen             # but never the answer itself


def test_unknown_format_and_method_are_caught_early():
    with pytest.raises(ValueError, match="unknown format"):
        synthesize(task="t", teacher=FakeTeacher(), n=2, format="parquet")
    with pytest.raises(ValueError, match="unknown method"):
        synthesize(task="t", teacher=FakeTeacher(), n=2, method="telepathy")
