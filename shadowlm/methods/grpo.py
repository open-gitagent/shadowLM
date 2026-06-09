"""GRPO — group relative policy optimization.

Reinforcement learning from programmable rewards: for each {"prompt"[, "answer"]}
row the trainer samples `grpo_group_size` completions, scores them with your
reward functions, and pushes the policy toward completions that beat their
group's average — no value model, no preference pairs.

Reward functions receive keyword args and return one score per completion:

    def my_reward(prompts, completions, answer, types=None) -> list[float]:
        return [1.0 if "blue" in c.lower() else 0.0 for c in completions]

    model.finetune(rows, method="grpo", reward_fns=[my_reward])

Knobs: `grpo_group_size` (default 4), `grpo_max_completion_length`, `beta` (KL
strength vs the frozen reference).
"""

from .base import TrainingMethod, register

GRPO = register(TrainingMethod(
    name="grpo",
    description="group relative policy optimization with programmable reward functions",
    default_learning_rate=5e-6,
    trainer="grpo",
))
