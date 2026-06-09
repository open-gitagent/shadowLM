"""DoRA — weight-decomposed LoRA.

Decomposes each weight update into magnitude and direction components, which
often matches full fine-tuning quality better than plain LoRA at low rank, at a
small extra compute cost.
"""

from .base import ADAPTER_DORA, TrainingMethod, register

DORA = register(TrainingMethod(
    name="dora",
    description="DoRA adapters (weight-decomposed LoRA), often better at low rank",
    default_learning_rate=2e-4,
    adapter=ADAPTER_DORA,
))
