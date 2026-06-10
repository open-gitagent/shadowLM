"""Soft prompts (prompt tuning) — steer a frozen model with learned vectors.

Prepends `num_virtual_tokens` trainable embeddings to the input; all model
weights stay frozen. The cheapest technique by parameter count. torch backend
(peft); needs a relatively high learning rate.
"""

from .base import TrainingMethod, register

PROMPT = register(TrainingMethod(
    name="prompt",
    description="soft prompts — learned virtual tokens, model frozen",
    default_learning_rate=5e-3,
    adapter="prompt",
))
