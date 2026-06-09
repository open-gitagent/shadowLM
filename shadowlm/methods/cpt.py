"""CPT — continued pretraining on raw domain text.

Next-token training on plain text with no chat formatting, used to adapt a model
to a new domain (jargon, style, facts) before or instead of instruction tuning.
Uses LoRA adapters with a gentler learning rate, since domain shifts destabilize
more easily than chat finetunes.
"""

from .base import TrainingMethod, register

CPT = register(TrainingMethod(
    name="cpt",
    description="continued pretraining — next-token training on raw domain text",
    default_learning_rate=5e-5,
    raw_text=True,
))
