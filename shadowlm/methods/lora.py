"""LoRA — low-rank adapters on a 16-bit base.

Trains small adapter matrices on the attention/MLP projections while the base
weights stay frozen. The default choice: fast, memory-light, and the adapter is
a few MB you can ship separately.
"""

from .base import TrainingMethod, register

LORA = register(TrainingMethod(
    name="lora",
    description="LoRA adapters on a 16-bit base",
    default_learning_rate=2e-4,
))
