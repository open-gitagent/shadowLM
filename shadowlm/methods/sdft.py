"""SDFT — self-distillation fine-tuning.

On-policy learning from demonstrations (arXiv 2601.19897): for each chat row
the trainer samples a completion from the current model, then nudges its
per-token distributions toward the same model reading the row's golden response
in-context — the demonstration-conditioned model is its own teacher. Because
every update happens on the model's own samples, it learns the task like SFT
while keeping its general abilities (far less catastrophic forgetting).

Trains an ordinary LoRA adapter on plain chat/instruction rows; the teacher is
the frozen base (adapters disabled on torch, a second frozen copy on mlx) — the
reference implementation's PEFT setup, not the paper's EMA teacher.

    model.finetune(ds, method="sdft")

Knobs: `sdft_alpha` (0 = forward KL, 1 = reverse KL, between = generalized
JSD), `sdft_max_completion_length`, `sdft_temperature`, and
`sdft_teacher_template` (must contain "{demonstration}").
"""

from .base import TrainingMethod, register

SDFT = register(TrainingMethod(
    name="sdft",
    description="on-policy self-distillation from demonstrations — the demo-conditioned model teaches itself",
    default_learning_rate=1e-5,
    trainer="sdft",
))
