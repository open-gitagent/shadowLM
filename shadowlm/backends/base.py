"""The backend contract.

A `Backend` is a stateful object that holds a loaded model and knows how to
finetune it and generate from it. Everything user-facing (`Model`, `load`,
`finetune`, `generate`) is backend-agnostic — swap mlx ↔ torch without touching
the SDK surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..data import Dataset
from ..training import Metric, TrainConfig


class Callbacks:
    """Bridges a backend's training loop back to the caller.

    A backend calls `.step(metric)` for each logged step and `.log(msg)` for
    status lines, and checks `.stopped()` to honour cancellation.
    """

    def __init__(
        self,
        on_step: Callable[[Metric], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_eval: Callable[[Metric], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self._on_step = on_step
        self._on_log = on_log
        self._on_eval = on_eval
        self._should_stop = should_stop

    def step(self, metric: Metric) -> None:
        if self._on_step:
            self._on_step(metric)

    def eval(self, metric: Metric) -> None:
        if self._on_eval:
            self._on_eval(metric)

    def log(self, message: str) -> None:
        if self._on_log:
            self._on_log(message)

    def stopped(self) -> bool:
        return bool(self._should_stop and self._should_stop())


def warn_dropped_tools(dataset: Dataset, callbacks: Callbacks) -> None:
    """Say so when a dataset's tool schemas won't reach the training template.

    Rows may carry `tools`, but neither backend passes them to the chat template
    while training — inference (`chat(tools=...)`) does. Never let that skew pass
    silently.
    """
    if any(r.get("tools") for r in getattr(dataset, "rows", ())):
        callbacks.log(
            "[shadowlm] dataset carries tool schemas: training renders the tool "
            "*calls* but not the tool *definitions*, which inference does pass — "
            "the train and inference prompts differ on that block."
        )


class FinetuneResult:
    """What a backend returns from `finetune`: where the adapter/model landed."""

    def __init__(self, checkpoint: str, final_loss: float | None) -> None:
        self.checkpoint = checkpoint
        self.final_loss = final_loss


class Backend(ABC):
    name: str = "base"

    def __init__(self, *, device: str = "auto", accelerator: str = "auto") -> None:
        # device interpretation is backend-specific: mlx → gpu, torch → cuda|cpu.
        # accelerator selects the shadow optimization layer: none | shadow | auto.
        self.device = device
        self.accelerator = accelerator
        self.model_name: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend can actually run in the current environment."""
        return True

    @abstractmethod
    def load(self, name: str, *, load_in_4bit: bool, max_seq_length: int,
             adapter: str | None = None, **kwargs) -> None:
        ...

    @abstractmethod
    def finetune(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                 output_dir: str, eval_dataset: Dataset | None = None,
                 reward_fns: list | None = None) -> FinetuneResult:
        ...

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float,
                 top_p: float, **kwargs) -> str:
        ...

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             max_new_tokens: int, temperature: float, top_p: float, **kwargs) -> str:
        """Multi-turn chat through the tokenizer's chat template, with optional
        tool schemas. Returns the raw assistant text (tool-call markup included)."""
        raise NotImplementedError(f"{self.name} backend does not implement chat()")

    def save(self, path: str, *, fmt: str = "adapter") -> str:
        raise NotImplementedError(f"{self.name} backend does not implement save()")
