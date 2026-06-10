"""BitFit — train only the bias terms (~0.1% of parameters).

Freezes everything except biases. Surprisingly competitive on small/medium
datasets, and the checkpoint is tiny. Note: Llama-family models often have no
bias parameters at all — shadowLM raises a clear error if there is nothing to
train (Qwen-family attention biases work well).
"""

from .base import ADAPTER_BITFIT, TrainingMethod, register

BITFIT = register(TrainingMethod(
    name="bitfit",
    description="train only the bias terms (~0.1% of params)",
    default_learning_rate=5e-4,
    adapter=ADAPTER_BITFIT,
    quantized_base=False,  # bias grads through quantized layers go NaN (verified)
))
