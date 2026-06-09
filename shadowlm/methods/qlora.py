"""QLoRA — LoRA adapters over a 4-bit quantized base.

Same adapters as LoRA, but the frozen base is 4-bit quantized, cutting memory
roughly 4x. Requires actually loading a quantized base (the method validates
this) — on mlx use a *-4bit repo, on torch pass load_in_4bit=True.
"""

from .base import TrainingMethod, register

QLORA = register(TrainingMethod(
    name="qlora",
    description="LoRA adapters on a 4-bit quantized base (lowest memory)",
    default_learning_rate=2e-4,
    quantized_base=True,
))
