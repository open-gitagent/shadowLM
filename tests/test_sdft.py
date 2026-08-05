"""SDFT unit tests — registration, divergence math, teacher-context construction.

No model, no training: the loss and the message helpers are pure functions, so
the CPU suite can pin the math (including parity with trl's GKD loss and the
mlx twin) and every dataset-shape error without a GPU.
"""

import math

import pytest

from shadowlm import methods
from shadowlm.sdft import (
    DEFAULT_TEACHER_TEMPLATE,
    completion_slice,
    split_demonstration,
    teacher_messages,
)
from shadowlm.training import TrainConfig

ROW = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]


# ---------------------------------------------------------------- registration

def test_sdft_registered():
    spec = methods.get("sdft")
    assert spec.trainer == "sdft"
    assert spec.adapter == methods.ADAPTER_LORA
    assert spec.quantized_base is None  # either base precision works
    assert spec.default_learning_rate == 1e-5
    assert "sdft" in methods.available()


def test_config_knobs_exist_with_defaults():
    c = TrainConfig(method="sdft")
    assert c.sdft_alpha == 0.0
    assert c.sdft_max_completion_length == 512
    assert c.sdft_temperature == 1.0
    assert c.sdft_teacher_template is None


# ------------------------------------------------------------ divergence math

def _rand_logits(seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 4, 7, generator=g)


def test_forward_kl_matches_hand_value():
    import torch

    from shadowlm.backends.torch import _distill_divergence

    student = torch.zeros(1, 1, 2)  # uniform over a 2-token vocab
    teacher = torch.tensor([[[math.log(0.9), math.log(0.1)]]])
    want = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    got = float(_distill_divergence(student, teacher, 0.0)[0, 0])
    assert got == pytest.approx(want)


def test_reverse_kl_is_forward_with_swapped_args():
    import torch

    from shadowlm.backends.torch import _distill_divergence

    s, t = _rand_logits(1), _rand_logits(2)
    assert torch.allclose(_distill_divergence(s, t, 1.0), _distill_divergence(t, s, 0.0))


def test_jsd_symmetric_bounded_and_zero_at_equal():
    import torch

    from shadowlm.backends.torch import _distill_divergence

    s, t = _rand_logits(3), _rand_logits(4)
    jsd = _distill_divergence(s, t, 0.5)
    assert torch.allclose(jsd, _distill_divergence(t, s, 0.5), atol=1e-6)
    assert (jsd >= -1e-6).all()
    assert (jsd <= math.log(2) + 1e-6).all()
    # logits shifted by a per-row constant are the same distribution → zero,
    # at every alpha
    for alpha in (0.0, 0.5, 1.0):
        d = _distill_divergence(s, s + 3.0, alpha)
        assert float(d.abs().max()) < 1e-5


def test_alpha_out_of_range_raises():
    from shadowlm.backends.torch import _distill_divergence

    s = _rand_logits(5)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            _distill_divergence(s, s, bad)


def test_divergence_matches_trl_gkd():
    torch = pytest.importorskip("torch")
    trl = pytest.importorskip("trl")

    from shadowlm.backends.torch import _distill_divergence

    s, t = _rand_logits(6), _rand_logits(7)
    for alpha in (0.1, 0.5, 0.9):
        ours = float(_distill_divergence(s, t, alpha).sum())
        gkd = float(trl.GKDTrainer.generalized_jsd_loss(
            s, t, beta=alpha, reduction="sum"))
        assert ours == pytest.approx(gkd, abs=1e-4)


def test_divergence_mx_parity_with_torch():
    mx = pytest.importorskip("mlx.core")

    from shadowlm.backends.mlx import _distill_divergence_mx
    from shadowlm.backends.torch import _distill_divergence

    s, t = _rand_logits(8), _rand_logits(9)
    for alpha in (0.0, 0.3, 1.0):
        ours = _distill_divergence(s, t, alpha)[0].tolist()
        theirs = _distill_divergence_mx(mx.array(s.numpy()), mx.array(t.numpy()),
                                     alpha)
        assert ours == pytest.approx([float(x) for x in theirs[0]], abs=1e-5)


# ----------------------------------------------------- teacher-context helpers

def test_split_demonstration():
    ctx, demo = split_demonstration(ROW)
    assert ctx == ROW[:-1]
    assert demo == "4"


def test_split_requires_assistant_final():
    with pytest.raises(ValueError, match=r"row 3 ends with role 'user'"):
        split_demonstration([{"role": "user", "content": "hi"}], row_index=3)
    with pytest.raises(ValueError, match="assistant demonstration"):
        split_demonstration([])


def test_split_rejects_structured_content():
    row = [{"role": "user", "content": "q"},
           {"role": "assistant", "content": [{"type": "text", "text": "4"}]}]
    with pytest.raises(ValueError, match="plain-text assistant content"):
        split_demonstration(row)


def test_teacher_suffix_lands_on_last_user_turn():
    multi = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    out = teacher_messages(multi)
    assert out[:3] == multi[:3]  # earlier turns untouched
    assert out[3]["content"].startswith("What is 2+2?")
    assert "\n4\n" in out[3]["content"]  # demonstration verbatim
    assert out[3]["content"].endswith(DEFAULT_TEACHER_TEMPLATE.rsplit(
        "{demonstration}")[-1])
    assert len(out) == 4  # the assistant demo turn itself is gone


def test_teacher_custom_template():
    out = teacher_messages(ROW, "\nExample: {demonstration}. Your turn.")
    assert out[1]["content"] == "What is 2+2?\nExample: 4. Your turn."
    with pytest.raises(ValueError, match="demonstration.*placeholder"):
        teacher_messages(ROW, "no placeholder here")


def test_teacher_braces_survive():
    row = [{"role": "user", "content": "emit json"},
           {"role": "assistant", "content": '{"a": {"b"}}'}]
    out = teacher_messages(row)
    assert '{"a": {"b"}}' in out[0]["content"]
    out = teacher_messages(row, "\n{demonstration} and {not_a_field}")
    assert "{not_a_field}" in out[0]["content"]


def test_teacher_requires_a_user_turn():
    # TEXT-format rows become a lone assistant message via as_chat().
    with pytest.raises(ValueError, match="no user turn"):
        teacher_messages([{"role": "assistant", "content": "just text"}])


def test_teacher_does_not_mutate_input():
    row = [dict(m) for m in ROW]
    teacher_messages(row)
    assert row == ROW


# ------------------------------------------------------------- logit alignment

def test_completion_slice_arithmetic():
    assert completion_slice(5, 8) == slice(4, 7)
    # applied to the [:-1]-fed logit sequence, it selects the last C rows
    P, C = 5, 3
    logit_rows = list(range(P + C - 1))
    assert logit_rows[completion_slice(P, P + C)] == [4, 5, 6]
    # the same completion, behind the longer teacher prompt
    Pt = 9
    teacher_rows = list(range(Pt + C - 1))
    assert len(teacher_rows[completion_slice(Pt, Pt + C)]) == C


def test_completion_slice_validation():
    with pytest.raises(ValueError, match="prompt_len"):
        completion_slice(5, 5)
    with pytest.raises(ValueError, match="prompt_len"):
        completion_slice(0, 5)
