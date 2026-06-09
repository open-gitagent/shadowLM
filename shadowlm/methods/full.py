"""Full fine-tune — update every transformer weight.

No adapters: all transformer blocks are unfrozen and trained. Highest quality
ceiling and highest memory cost; needs an unquantized base (quantized weights
can't receive gradients) and a much lower learning rate than adapter methods.
"""

from .base import ADAPTER_NONE, TrainingMethod, register

FULL = register(TrainingMethod(
    name="full",
    description="update every transformer weight (needs an unquantized base)",
    default_learning_rate=2e-5,
    adapter=ADAPTER_NONE,
    quantized_base=False,
))
