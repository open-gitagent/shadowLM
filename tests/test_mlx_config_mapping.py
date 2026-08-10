"""The mlx backend honors weight_decay and seed, and its ignored-fields note
stays truthful (the CLAUDE.md convention: dropped fields get a log line).

Pure config-mapping logic — no mlx install needed.
"""

from __future__ import annotations

from shadowlm.backends.mlx import _ignored_fields, _make_optimizer
from shadowlm.training import TrainConfig


class _FakeOptim:
    """Stands in for mlx.optimizers: records what the backend asked for."""

    class AdamW:
        def __init__(self, learning_rate, weight_decay):
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay


def test_optimizer_carries_weight_decay():
    config = TrainConfig(method="lora", weight_decay=0.05)
    opt = _make_optimizer(_FakeOptim, config, 3e-4)
    assert opt.weight_decay == 0.05
    assert opt.learning_rate == 3e-4


def test_honored_fields_not_reported_ignored():
    config = TrainConfig(method="lora", weight_decay=0.1, seed=7)
    ignored = _ignored_fields(config)
    assert "weight_decay" not in ignored
    assert "seed" not in ignored
    # the default optimizer maps to AdamW — that's honoring it, not dropping it
    assert "optim" not in ignored


def test_unmappable_fields_still_reported():
    config = TrainConfig(method="lora", optim="sgd", packing=True,
                         use_rslora=True, max_grad_norm=1.0,
                         report_to=["wandb"])
    ignored = _ignored_fields(config)
    assert {"optim", "packing", "use_rslora", "max_grad_norm",
            "report_to"} <= set(ignored)
