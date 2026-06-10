"""P-tuning — continuous prompt embeddings through a small encoder.

Like soft prompts, but the virtual tokens are produced by a trainable MLP
encoder, which stabilizes optimization on NLU-style tasks. torch backend (peft).
"""

from .base import ADAPTER_PTUNING, TrainingMethod, register

PTUNING = register(TrainingMethod(
    name="ptuning",
    description="p-tuning — prompt embeddings via a small trainable encoder",
    default_learning_rate=5e-3,
    adapter=ADAPTER_PTUNING,
))
