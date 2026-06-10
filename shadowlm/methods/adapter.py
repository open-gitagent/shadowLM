"""Adapter tuning — small bottleneck modules inserted after each layer.

The classic Houlsby-style recipe: each transformer layer's output passes
through a trainable down-project → GELU → up-project residual (up zero-init so
training starts as a no-op). Bottleneck width = `lora_r`. Both backends.
"""

from .base import TrainingMethod, register

ADAPTER = register(TrainingMethod(
    name="adapter",
    description="bottleneck adapter modules inserted after each transformer layer",
    default_learning_rate=1e-4,
    adapter="bottleneck",
))
