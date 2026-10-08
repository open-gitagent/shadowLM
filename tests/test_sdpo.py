"""SDPO unit tests — registration, group scoring, teacher-reprompt construction.

No model, no training: scoring and the reprompt builder are pure functions, so
the CPU suite pins the reward-fn contract (floats and (score, feedback) pairs),
the sibling-solution rule, and the reference template wording without a GPU.
The divergence math itself is proven in tests/test_sdft.py — sdpo shares the
same `_distill_divergence` helpers.
"""

import pytest

from shadowlm import methods
from shadowlm.sdpo import pick_solution, score_group, teacher_prompt
from shadowlm.training import TrainConfig

# ---------------------------------------------------------------- registration


def test_sdpo_registered():
    spec = methods.get("sdpo")
    assert spec.trainer == "sdpo"
    assert spec.adapter == methods.ADAPTER_LORA
    assert spec.quantized_base is None  # either base precision works
    assert spec.default_learning_rate == 1e-5
    assert "sdpo" in methods.available()


def test_config_knobs_exist_with_defaults():
    c = TrainConfig(method="sdpo")
    assert c.sdpo_alpha == 0.5  # the paper trains with the symmetric JSD
    assert c.sdpo_group_size == 4
    assert c.sdpo_max_completion_length == 256
    assert c.sdpo_temperature == 1.0
    assert c.sdpo_success_threshold == 1.0
    assert c.sdpo_teacher_ema == 0.05


# --------------------------------------------------------------- group scoring


def test_score_group_plain_floats():
    scores, feedbacks = score_group([[1.0, 0.0, 0.5]], 3)
    assert scores == [1.0, 0.0, 0.5]
    assert feedbacks == [None, None, None]


def test_score_group_pairs_split_score_and_feedback():
    scores, feedbacks = score_group([[(0.0, "off by one"), 1.0]], 2)
    assert scores == [0.0, 1.0]
    assert feedbacks == ["off by one", None]


def test_score_group_sums_fns_and_joins_feedback():
    out_a = [(0.0, "wrong unit"), 1.0]
    out_b = [(0.5, "too long"), (0.25, "")]  # blank feedback is dropped
    scores, feedbacks = score_group([out_a, out_b], 2)
    assert scores == [0.5, 1.25]
    assert feedbacks == ["wrong unit\n\ntoo long", None]


def test_score_group_accepts_bools_as_scores():
    scores, _ = score_group([[True, False]], 2)
    assert scores == [1.0, 0.0]


def test_score_group_errors_name_the_fn():
    with pytest.raises(ValueError, match=r"brevity.*returned float"):
        score_group([0.5], 1, fn_names=["brevity"])
    with pytest.raises(ValueError, match=r"reward_fns\[0\].*2 score\(s\) for 3"):
        score_group([[1.0, 0.0]], 3)
    with pytest.raises(ValueError, match=r"judge.*element 1.*str"):
        score_group([[1.0, "oops"]], 2, fn_names=["judge"])
    with pytest.raises(ValueError, match=r"judge.*pair must be"):
        score_group([[(1.0, 0.5)]], 1, fn_names=["judge"])


# ------------------------------------------------------------- solution choice


def test_pick_solution_prefers_first_successful_sibling():
    texts = ["a", "b", "c"]
    assert pick_solution(0, [0.0, 1.0, 1.0], texts, 1.0) == "b"
    assert pick_solution(2, [0.0, 1.0, 1.0], texts, 1.0) == "b"


def test_pick_solution_never_returns_self():
    # A lone success teaches its siblings but not itself (the paper's
    # dont_reprompt_on_self_success setting).
    texts = ["a", "b"]
    assert pick_solution(0, [1.0, 0.0], texts, 1.0) is None
    assert pick_solution(1, [1.0, 0.0], texts, 1.0) == "a"


def test_pick_solution_threshold_is_inclusive():
    assert pick_solution(0, [0.0, 0.7], ["a", "b"], 0.7) == "b"
    assert pick_solution(0, [0.0, 0.69], ["a", "b"], 0.7) is None
    assert pick_solution(0, [], [], 1.0) is None


# ---------------------------------------------------------- teacher reprompt


def test_teacher_prompt_solution_only():
    got = teacher_prompt("Q?", "the fix", None)
    assert got == ("Q?\n"
                   "Correct solution:\n\n"
                   "the fix\n\n\n\n"
                   "Correctly solve the original question.\n")


def test_teacher_prompt_feedback_only():
    got = teacher_prompt("Q?", None, "ZeroDivisionError")
    assert got == ("Q?\n"
                   "The following is feedback from your unsuccessful earlier "
                   "attempt:\n\n"
                   "ZeroDivisionError\n\n\n\n"
                   "Correctly solve the original question.\n")


def test_teacher_prompt_solution_then_feedback():
    got = teacher_prompt("Q?", "SOL", "FB")
    assert got.startswith("Q?")
    assert got.index("Correct solution:") < got.index(
        "The following is feedback from your unsuccessful earlier attempt:")
    assert "SOL" in got and "FB" in got
    assert got.endswith("Correctly solve the original question.\n")


def test_teacher_prompt_nothing_to_condition_on():
    assert teacher_prompt("Q?", None, None) is None


def test_teacher_prompt_braces_survive():
    # Code-bearing solutions/feedback and even template-looking prompt text
    # must land verbatim — rendering is a single pass, never re-scanned.
    got = teacher_prompt("fix {solution} in d = {'a': 1}",
                         "d = {k: v for k, v in x}",
                         "KeyError: '{feedback_raw}'")
    assert "fix {solution} in d = {'a': 1}" in got
    assert "d = {k: v for k, v in x}" in got
    assert "KeyError: '{feedback_raw}'" in got


# ------------------------------------------------------- shared divergence core


def test_distill_divergence_shared_with_sdft():
    torch = pytest.importorskip("torch")
    from shadowlm.backends.torch import _distill_divergence

    torch.manual_seed(0)
    s, t = torch.randn(1, 3, 7), torch.randn(1, 3, 7)
    # JSD at alpha 0.5 (the sdpo default) is symmetric, and zero when the
    # logits differ only by a per-row constant (same distribution).
    assert torch.allclose(_distill_divergence(s, t, 0.5),
                          _distill_divergence(t, s, 0.5), atol=1e-6)
    assert float(_distill_divergence(s, s + 2.0, 0.5).abs().max()) < 1e-6
    with pytest.raises(ValueError, match="sdpo_alpha"):
        _distill_divergence(s, s, 1.5, knob="sdpo_alpha")
