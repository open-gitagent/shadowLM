"""The shadow accelerator — shadowLM's in-house training optimization layer.

It sits on top of whichever backend is active (mlx / torch) and turns on the
memory and throughput optimizations that are *safe for the current model and
hardware*: gradient checkpointing, fused attention kernels, fused optimizers.

Modes:
    "none"    off — plain training
    "shadow"  force every applicable optimization on
    "auto"    enable what actually helps at the current size (the default)

`plan()` is pure and side-effect free; each backend reads the returned `ShadowPlan`
and applies the flags it understands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Below this layer count, the optimizations cost more overhead than they save, so
# "auto" leaves them off (forcing "shadow" still turns them on).
_BIG_MODEL_LAYERS = 24


@dataclass
class ShadowPlan:
    grad_checkpoint: bool = False
    flash_attention: bool = False
    fused_optimizer: bool = False
    enabled: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.enabled)

    @property
    def note(self) -> str:
        if not self.enabled:
            return "[shadow] no extra optimizations needed at this size"
        return "[shadow] enabled: " + ", ".join(self.enabled)


def plan(mode: str, *, backend: str, n_layers: int = 0, has_flash: bool = False) -> ShadowPlan:
    """Decide which shadow optimizations to apply.

    backend: "mlx" | "torch". n_layers: the model's transformer-block count.
    has_flash: whether a flash-attention kernel is importable (torch only).
    """
    if mode not in ("none", "shadow", "auto"):
        raise ValueError(f"unknown accelerator {mode!r} (expected auto|shadow|none)")
    if mode == "none":
        return ShadowPlan()
    force = mode == "shadow"
    big = n_layers >= _BIG_MODEL_LAYERS
    p = ShadowPlan()

    if backend == "mlx":
        if force or big:
            p.grad_checkpoint = True
            p.enabled.append("gradient checkpointing")
    else:  # torch (cuda / cpu)
        if has_flash:
            p.flash_attention = True
            p.enabled.append("flash-attention-2")
        if force or big:
            p.grad_checkpoint = True
            p.enabled.append("gradient checkpointing")
        p.fused_optimizer = True
        p.enabled.append("fused optimizer")
    return p
