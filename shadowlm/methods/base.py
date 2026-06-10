"""The training-method spec and registry.

Every training method is a declarative `TrainingMethod`. Backends read the
spec's fields (adapter kind, base requirements, data rendering) — never the
method name — so adding a method is a new module in this package with one
`register()` call, and no backend edits unless it needs new machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

# Adapter kinds a backend knows how to attach.
ADAPTER_LORA = "lora"  # low-rank adapters
ADAPTER_DORA = "dora"  # weight-decomposed LoRA — often better at low rank
ADAPTER_MORE = "more"  # memory experts: retrieval-fused attention projections
ADAPTER_BITFIT = "bitfit"  # bias terms only
ADAPTER_PROMPT = "prompt"  # learned virtual tokens (soft prompts)
ADAPTER_PTUNING = "ptuning"  # virtual tokens via a small encoder
ADAPTER_BOTTLENECK = "bottleneck"  # residual bottleneck modules per layer
ADAPTER_NONE = "none"  # no adapters: train the full weights


@dataclass(frozen=True)
class TrainingMethod:
    """What a training method *means*, declaratively.

    quantized_base: True → requires a 4-bit base; False → requires an
    unquantized base; None → either works.
    raw_text: render rows to plain text with no chat template (domain/continued
    pretraining) instead of chat-formatting them.
    """

    name: str
    description: str
    default_learning_rate: float
    adapter: str = ADAPTER_LORA
    quantized_base: bool | None = None
    raw_text: bool = False
    # Which training loop drives this method: "sft" (supervised next-token),
    # "dpo" (preference pairs), or "grpo" (reward functions / trajectories).
    # Backends dispatch machinery on this, not names.
    trainer: str = "sft"

    @property
    def trains_adapters(self) -> bool:
        return self.adapter != ADAPTER_NONE

    def validate_base(self, *, quantized: bool, quantize_hint: str,
                      dequantize_hint: str) -> None:
        """Check the loaded base model satisfies this method's requirement."""
        if self.quantized_base is True and not quantized:
            raise RuntimeError(
                f"method={self.name!r} needs a 4-bit quantized base model. "
                f"{quantize_hint}, or use method='lora'."
            )
        if self.quantized_base is False and quantized:
            raise RuntimeError(
                f"method={self.name!r} can't train quantized weights — they "
                f"don't receive gradients. {dequantize_hint}, or use method='qlora'."
            )


_REGISTRY: dict[str, TrainingMethod] = {}


def register(method: TrainingMethod) -> TrainingMethod:
    _REGISTRY[method.name] = method
    return method


def get(name: str) -> TrainingMethod:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown method {name!r} (available: {', '.join(_REGISTRY)})"
        ) from None


def available() -> tuple[str, ...]:
    return tuple(_REGISTRY)
