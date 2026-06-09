"""GRPO — reinforcement learning from programmable reward functions.

For each prompt the trainer samples a *group* of completions, scores them with
your reward functions, and pushes the policy toward completions that beat their
group's average. No preference pairs, no value model — just rewards you write.

    python examples/grpo_rewards.py
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# 1. prompts (rows only need a "prompt"; "answer" is optional) ----------------
rows = [
    {"prompt": "Name a color. Answer with one word."},
    {"prompt": "What color is the sky? One word."},
    {"prompt": "Pick your favorite color. One word only."},
    {"prompt": "Say a color name."},
]


# 2. reward functions: fn(prompts, completions, answer, types=None) -> floats
def prefers_blue(prompts, completions, answer, types=None):
    """+1 when the completion mentions blue."""
    return [1.0 if "blue" in c.lower() else 0.0 for c in completions]


def brevity(prompts, completions, answer, types=None):
    """Shorter is better — full marks near zero length, zero at 80+ chars."""
    return [max(0.0, 1.0 - len(c) / 80) for c in completions]


# 3. train — each step samples grpo_group_size completions per prompt --------
model = slm.load(MODEL)
run = model.finetune(
    rows,
    method="grpo",
    reward_fns=[prefers_blue, brevity],
    max_steps=8,
    grpo_group_size=2,
    grpo_max_completion_length=24,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
)

print("\nafter:", model.generate("Name a color. One word.", max_new_tokens=8,
                                 temperature=0.0).strip())
