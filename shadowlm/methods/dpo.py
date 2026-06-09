"""DPO — direct preference optimization.

Trains on preference pairs ({"prompt", "chosen", "rejected"} rows) instead of
supervised targets: the model learns to rank the chosen answer above the
rejected one against a frozen reference copy of itself. The standard first step
beyond SFT for alignment-style tuning. Tune strength with `beta`
(default 0.1; higher = stay closer to the reference).
"""

from .base import TrainingMethod, register

DPO = register(TrainingMethod(
    name="dpo",
    description="direct preference optimization on (prompt, chosen, rejected) pairs",
    default_learning_rate=5e-6,
    trainer="dpo",
))
