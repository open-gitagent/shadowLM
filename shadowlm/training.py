"""Training config, metrics, and the run handle returned by `model.finetune`."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

# Default LoRA target modules — the attention + MLP projections, the standard
# PEFT recipe.
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


@dataclass
class TrainConfig:
    """Hyperparameters for a finetune. Sensible defaults; override per call."""

    method: str = "lora"  # "lora" | "qlora" | "full"

    # sequence / LoRA
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    # optimisation
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: float | None = None
    max_steps: int | None = 60  # one of max_steps / num_train_epochs drives length
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    eval_steps: int | None = None  # evaluate every N steps when an eval set is given
    seed: int = 3407

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d


@dataclass
class Metric:
    """One logged training step."""

    step: int
    loss: float
    lr: float = 0.0
    grad_norm: float | None = None
    epoch: float | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# A step callback receives each Metric as it is produced.
StepCallback = Callable[[Metric], None]


@dataclass
class TrainingRun:
    """Handle for a finetune — live during training, final once done.

    Holds the metric history, resolved checkpoint path, and status. `model.finetune`
    returns one of these; pass `on_step=` to observe metrics as they stream in.
    """

    config: TrainConfig
    base_model: str
    status: str = "pending"  # pending | running | succeeded | failed
    metrics: list[Metric] = field(default_factory=list)
    eval_metrics: list[Metric] = field(default_factory=list)  # held-out eval points (.loss = eval loss)
    checkpoint: str | None = None
    error: str | None = None
    total_steps: int | None = None
    started_at: float | None = None
    ended_at: float | None = None

    # ---- live state -------------------------------------------------------
    @property
    def step(self) -> int:
        return self.metrics[-1].step if self.metrics else 0

    @property
    def loss(self) -> float | None:
        return self.metrics[-1].loss if self.metrics else None

    @property
    def eval_loss(self) -> float | None:
        return self.eval_metrics[-1].loss if self.eval_metrics else None

    @property
    def progress(self) -> float:
        if not self.total_steps:
            return 0.0
        return min(1.0, self.step / self.total_steps)

    @property
    def duration_s(self) -> float | None:
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        return None

    def losses(self) -> list[float]:
        return [m.loss for m in self.metrics]

    def sparkline(self) -> str:
        """A tiny unicode loss curve — handy in a REPL or log line."""
        vals = self.losses()
        if not vals:
            return ""
        bars = "▁▂▃▄▅▆▇█"
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        return "".join(bars[min(7, int((v - lo) / span * 7))] for v in vals)

    def __repr__(self) -> str:
        loss = f"{self.loss:.4f}" if self.loss is not None else "—"
        pct = f"{self.progress * 100:.0f}%" if self.total_steps else "?"
        return (
            f"TrainingRun({self.base_model!r}, status={self.status!r}, "
            f"step={self.step}/{self.total_steps or '?'} ({pct}), loss={loss})"
        )


def resolve_total_steps(config: TrainConfig, n_examples: int) -> int:
    """How many optimizer steps this run will take.

    `max_steps` wins if set; otherwise derive from epochs and the effective batch
    size (per-device batch × grad-accumulation).
    """
    if config.max_steps:
        return config.max_steps
    effective_batch = max(1, config.per_device_train_batch_size * config.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(n_examples / effective_batch))
    epochs = config.num_train_epochs or 1.0
    return max(1, math.ceil(steps_per_epoch * epochs))
