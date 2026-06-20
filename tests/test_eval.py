"""No-GPU tests for `slm.evaluate` — scorers, format dispatch, and error paths.

Uses a stub model (canned `.chat`) so nothing downloads. Runs under pytest, or
standalone: `python tests/test_eval.py` (exit 0 = all passed).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowlm.apo import _parse_judge_score  # noqa: E402
from shadowlm.data import Dataset  # noqa: E402
from shadowlm.eval import EvalResult, evaluate  # noqa: E402


class Stub:
    """A model whose `.chat` returns a canned reply and records what it saw."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: list[list[dict]] = []

    def chat(self, messages, **kw):
        self.seen.append(messages)
        return self.reply


QA = [{"question": "2+2?", "answer": "4"}, {"question": "cap of France?", "answer": "Paris"}]


def test_contains_metric():
    r = evaluate(Stub("the answer is 4"), QA, metric="contains", verbose=False)
    assert isinstance(r, EvalResult) and r.metric == "contains"
    assert r.scores == [1.0, 0.0] and r.n == 2
    assert abs(r.score - 0.5) < 1e-9


def test_exact_metric_normalizes():
    r = evaluate(Stub("  PARIS "), QA, metric="exact", verbose=False)
    assert r.scores == [0.0, 1.0]  # case/space-insensitive equality


def test_custom_callable_scorer():
    r = evaluate(Stub("xx"), QA, metric=lambda out, exp, q: len(out) / 10, verbose=False)
    assert r.metric == "<lambda>" and r.scores == [0.2, 0.2]


def test_judge_metric_and_implied_flip():
    # passing judge= flips the default metric to "judge"
    r = evaluate(Stub("4"), QA, judge=Stub("0.9"), verbose=False)
    assert r.metric == "judge" and r.scores == [0.9, 0.9]


def test_judge_without_model_raises():
    try:
        evaluate(Stub("x"), QA, metric="judge", verbose=False)
    except ValueError as e:
        assert "judge" in str(e)
    else:
        raise AssertionError("expected ValueError for metric='judge' without a judge")


def test_missing_prompt_column_raises():
    try:
        evaluate(Stub("x"), [{"foo": "bar"}], metric="exact", verbose=False)
    except ValueError as e:
        assert "prompt column" in str(e)
    else:
        raise AssertionError("expected ValueError for a row with no prompt column")


def test_chat_multiturn_keeps_context():
    stub = Stub("blue")
    row = {"messages": [
        {"role": "user", "content": "pick a color"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now say it"},
        {"role": "assistant", "content": "blue"},
    ]}
    r = evaluate(stub, Dataset.from_list([row]), metric="contains", verbose=False)
    assert r.scores == [1.0]
    # the model must have received the full prefix (3 turns), not just turn 1
    sent = stub.seen[0]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert r.examples[0]["input"] == "now say it"  # last user turn is the question
    assert r.examples[0]["expected"] == "blue"      # final assistant turn is the ref


def test_preference_format():
    ds = Dataset.from_list([{"prompt": "q", "chosen": "good", "rejected": "bad"}])
    r = evaluate(Stub("this is good"), ds, metric="contains", verbose=False)
    assert r.scores == [1.0]


def test_sample_zero_is_not_whole_dataset():
    # `--sample 0` must not silently mean "evaluate everything"
    try:
        evaluate(Stub("x"), QA, metric="exact", sample=0, verbose=False)
    except ValueError as e:
        assert "at least one row" in str(e)
    else:
        raise AssertionError("expected sample=0 to yield no rows, not the full set")


def test_path_input_and_result_helpers():
    path = str(Path(__file__).resolve().parents[1] / "examples" / "sample_dataset.jsonl")
    r = evaluate(Stub("Paris"), path, metric="contains", sample=3, verbose=False)
    assert r.n == 3 and len(r.sparkline()) == 3
    assert sorted(r.to_dict()) == ["examples", "metric", "n", "score", "scores"]
    assert len(r.worst(2)) == 2


def test_unknown_metric_raises():
    try:
        evaluate(Stub("a"), QA, metric="bleu", verbose=False)
    except ValueError as e:
        assert "unknown metric" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown metric name")


def test_scores_are_clamped_to_unit_interval():
    assert all(s == 1.0 for s in evaluate(Stub("x"), QA, metric=lambda o, e, q: 5.0, verbose=False).scores)
    assert all(s == 0.0 for s in evaluate(Stub("x"), QA, metric=lambda o, e, q: -3, verbose=False).scores)


def test_degenerate_chat_rows_dont_crash():
    from shadowlm.data import CHAT
    from shadowlm.eval import _row_io

    # None content is coerced to ""; an assistant-only row yields a placeholder turn
    h, exp = _row_io({"messages": [{"role": "assistant", "content": None}]}, CHAT)
    assert h == [{"role": "user", "content": ""}] and exp == ""
    # a system turn is kept in the context prefix, not dropped
    h, exp = _row_io({"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}]}, CHAT)
    assert [m["role"] for m in h] == ["system", "user"] and exp == "hello"


def test_judge_score_parser_tolerant():
    assert _parse_judge_score("0.7") == 0.7
    assert abs(_parse_judge_score("7/10") - 0.7) < 1e-9
    assert _parse_judge_score("I'd rate this an 8") == 0.8   # x/10 rating
    assert _parse_judge_score("score: 1") == 1.0
    assert _parse_judge_score("nonsense") == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
