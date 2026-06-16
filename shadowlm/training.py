"""Training config, metrics, and the run handle returned by `model.finetune`."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

# LoRA target-module groups — the attention and MLP projections. The default
# adapts both (the standard PEFT recipe); "attention" / "mlp" select one group.
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
DEFAULT_TARGET_MODULES = ATTENTION_MODULES + MLP_MODULES
_TARGET_PRESETS = {
    "all": DEFAULT_TARGET_MODULES,
    "attention": ATTENTION_MODULES,
    "mlp": MLP_MODULES,
}

@dataclass
class TrainConfig:
    """Hyperparameters for a finetune. Sensible defaults; override per call.

    Every field is honored by the torch backend; the mlx backend honors all but
    `optim`, `max_grad_norm`, `packing`, `use_rslora`, and `report_to` (noted
    inline). Unsupported-but-set fields are ignored with a log line, never
    silently dropped.
    """

    method: str = "lora"  # a registered method name — see shadowlm.methods

    # sequence / adapters
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] | str = "all"  # "all" | "attention" | "mlp" | explicit names
    use_rslora: bool = False  # rank-stabilized LoRA scaling (torch)

    # optimisation
    learning_rate: float | None = None  # None → the method's default LR
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    warmup_ratio: float | None = None  # fraction of total steps; overrides warmup_steps
    num_train_epochs: float | None = None
    max_steps: int | None = None  # one of max_steps / num_train_epochs; neither → 60 steps
    weight_decay: float = 0.01
    max_grad_norm: float | None = None  # gradient clipping (torch)
    lr_scheduler_type: str = "linear"  # "linear" | "cosine" | "constant"
    optim: str = "adamw_8bit"  # torch optimizer name; mlx uses Adam
    seed: int = 3407

    # data handling
    packing: bool = False  # pack short sequences together (torch)
    train_on_completions: bool = False  # mask the prompt; learn only on responses (mlx)

    # preference / RL methods (dpo, grpo)
    beta: float = 0.1  # KL strength vs the reference model; higher = stay closer
    grpo_group_size: int = 4  # completions sampled per prompt (the GRPO "group")
    grpo_max_completion_length: int = 256  # tokens generated per completion during training

    # memory tuning (more — mixture of retrieval experts)
    retrieval_k: int = 2  # memories retrieved per token
    retrieval_layers: int = 8  # how many attention layers (from the top) get memory

    # decoupled experts (more_plus — DMoE-style final-FFN experts + BM25 routing)
    more_plus_k: int = 1  # experts merged per query; >1 composes facts but the
    # single-FFN merge surface interferes, so 1 (route to the best expert) is the
    # clean default — raise only when facts genuinely combine
    more_plus_expert_steps: int = 0  # training steps per expert; 0 → auto (scales
    # with the final-FFN width so a bigger base doesn't silently under-train)
    more_plus_group_size: int = 1  # dataset rows folded into one expert
    more_plus_tau: float = 0.5  # entropy gate threshold (stored for per-token gating; v1.1)

    # soft-prompt family (prompt, ptuning)
    num_virtual_tokens: int = 16  # learned virtual tokens prepended to the input

    # logging / checkpoints
    logging_steps: int = 1
    eval_steps: int | float | None = None  # every N steps; a 0–1 float = fraction of total
    save_steps: int | None = None  # checkpoint every N steps; None → final only
    resume_from_checkpoint: str | None = None  # adapter/checkpoint path to continue from
    report_to: tuple[str, ...] = ()  # e.g. ("wandb", "tensorboard") (torch)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_modules"] = list(self.resolved_target_modules())
        d["report_to"] = list(self.report_to)
        return d

    def resolved_target_modules(self) -> tuple[str, ...]:
        """Target modules, expanding the "all"/"attention"/"mlp" presets."""
        if isinstance(self.target_modules, str):
            try:
                return _TARGET_PRESETS[self.target_modules]
            except KeyError:
                raise ValueError(
                    f"unknown target_modules preset {self.target_modules!r} "
                    f"(presets: {', '.join(_TARGET_PRESETS)}; or pass explicit names)"
                ) from None
        return tuple(self.target_modules)

    def resolved_warmup(self, total_steps: int) -> int:
        """Warmup steps, honoring warmup_ratio when set."""
        if self.warmup_ratio is not None:
            return max(0, int(round(self.warmup_ratio * total_steps)))
        return max(0, self.warmup_steps)

    def resolved_eval_steps(self, total_steps: int) -> int:
        """Eval interval in steps; a 0–1 float means a fraction of total steps."""
        if self.eval_steps is None:
            return max(1, total_steps // 4)
        if 0 < self.eval_steps < 1:
            return max(1, round(self.eval_steps * total_steps))
        return max(1, int(self.eval_steps))


@dataclass
class Metric:
    """One logged training step."""

    step: int
    loss: float
    lr: float = 0.0
    grad_norm: float | None = None
    epoch: float | None = None
    elapsed_s: float = 0.0
    tokens: int | None = None  # cumulative trained tokens
    tokens_per_s: float | None = None
    peak_mem_gb: float | None = None

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
    status: str = "pending"  # pending | running | succeeded | stopped | failed
    metrics: list[Metric] = field(default_factory=list)
    eval_metrics: list[Metric] = field(default_factory=list)  # held-out eval points (.loss = eval loss)
    checkpoint: str | None = None
    error: str | None = None
    total_steps: int | None = None
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def id(self) -> str | None:
        """The run's identifier — the checkpoint directory name."""
        from pathlib import Path  # noqa: PLC0415

        return Path(self.checkpoint).name if self.checkpoint else None

    @property
    def resumed(self) -> bool:
        """True when this run continued from a previous checkpoint."""
        return self.config.resume_from_checkpoint is not None

    def checkpoints(self) -> list:
        """Every saved version of this run — pass ``save_steps=N`` to get mid-run
        ones. Each is loadable: ``slm.load(base, adapter=ck.path)``."""
        from . import checkpoints as _ck  # noqa: PLC0415

        return _ck.list_checkpoints(self.checkpoint) if self.checkpoint else []

    def checkpoint_at(self, step: int | None = None) -> str:
        """Adapter path for a given step (or the final weights when omitted)."""
        from . import checkpoints as _ck  # noqa: PLC0415

        return _ck.resolve(self.checkpoint, step)

    # ---- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "status": self.status,
            "config": self.config.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
            "eval_metrics": [m.to_dict() for m in self.eval_metrics],
            "checkpoint": self.checkpoint,
            "error": self.error,
            "total_steps": self.total_steps,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingRun":
        cfg = dict(d["config"])
        if isinstance(cfg.get("target_modules"), list):
            cfg["target_modules"] = tuple(cfg["target_modules"])
        if isinstance(cfg.get("report_to"), list):
            cfg["report_to"] = tuple(cfg["report_to"])
        return cls(
            config=TrainConfig(**cfg),
            base_model=d["base_model"],
            status=d.get("status", "succeeded"),
            metrics=[Metric(**m) for m in d.get("metrics", [])],
            eval_metrics=[Metric(**m) for m in d.get("eval_metrics", [])],
            checkpoint=d.get("checkpoint"),
            error=d.get("error"),
            total_steps=d.get("total_steps"),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
        )

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

    @property
    def steps_per_s(self) -> float | None:
        if not self.metrics:
            return None
        last = self.metrics[-1]
        return last.step / last.elapsed_s if last.elapsed_s else None

    @property
    def eta_s(self) -> float | None:
        """Estimated seconds remaining, from the observed step rate."""
        rate = self.steps_per_s
        if not rate or not self.total_steps:
            return None
        return max(0.0, (self.total_steps - self.step) / rate)

    @property
    def tokens(self) -> int | None:
        """Cumulative trained tokens (when the backend reports them)."""
        for m in reversed(self.metrics):
            if m.tokens is not None:
                return m.tokens
        return None

    @property
    def peak_mem_gb(self) -> float | None:
        peaks = [m.peak_mem_gb for m in self.metrics if m.peak_mem_gb is not None]
        return max(peaks) if peaks else None

    def losses(self) -> list[float]:
        return [m.loss for m in self.metrics]

    # ---- charts -------------------------------------------------------------
    def series(self, name: str = "loss") -> tuple[list[int], list[float]]:
        """(steps, values) for a metric series: loss | eval_loss | lr | grad_norm."""
        if name == "eval_loss":
            pts = [(m.step, m.loss) for m in self.eval_metrics]
        elif name == "loss":
            pts = [(m.step, m.loss) for m in self.metrics]
        elif name == "lr":
            pts = [(m.step, m.lr) for m in self.metrics]
        elif name == "grad_norm":
            pts = [(m.step, m.grad_norm) for m in self.metrics if m.grad_norm is not None]
        else:
            raise ValueError(f"unknown series {name!r} (loss | eval_loss | lr | grad_norm)")
        return [p[0] for p in pts], [p[1] for p in pts]

    def smoothed(self, weight: float = 0.9) -> list[float]:
        """EMA-smoothed training loss (weight 0–1; higher = smoother)."""
        from .charts import ema  # noqa: PLC0415

        return ema(self.losses(), weight)

    def plot(self, metric: str = "loss", *, smooth: float = 0.6, height: int = 8,
             width: int = 60, window: int | None = None, log: bool = False,
             clip: float | None = None) -> str:
        """A terminal chart of a metric series — raw dots + EMA overlay.

        window: last N points only. log: log scale. clip: percentile cap
        (0.95/0.99). smooth=0 for raw only. Print the result.
        """
        from .charts import line_chart  # noqa: PLC0415

        steps, values = self.series(metric)
        if not values:
            if metric == "eval_loss":
                return ("evaluation not configured — pass eval_dataset= (or \"auto\") "
                        "and eval_steps= to track eval loss")
            if metric == "grad_norm":
                return "no grad-norm data recorded on this backend"
            return "(no data yet)"
        titles = {"loss": "Training Loss", "eval_loss": "Eval Loss",
                  "lr": "Learning Rate", "grad_norm": "Gradient Norm"}
        if metric != "loss":
            smooth = 0.0  # overlays only make sense on the noisy training loss
        return line_chart(steps, values, title=titles[metric], smooth=smooth,
                          height=height, width=width, window=window, log=log, clip=clip)

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
    from . import methods  # noqa: PLC0415 — lazy, avoids an import cycle

    if methods.get(config.method).adapter == methods.ADAPTER_MORE_PLUS:
        # MoRE+ trains one expert per knowledge unit; the run's outer progress is
        # one step per unit (the per-expert inner loop is more_plus_expert_steps).
        return max(1, math.ceil(n_examples / max(1, config.more_plus_group_size)))
    if config.max_steps:
        return config.max_steps
    if config.num_train_epochs:
        effective_batch = max(
            1, config.per_device_train_batch_size * config.gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(n_examples / effective_batch))
        return max(1, math.ceil(steps_per_epoch * config.num_train_epochs))
    return 60  # neither given — a sensible short default
