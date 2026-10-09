"""SDPO — self-distillation policy optimization.

RL via self-distillation (arXiv 2601.20802): for each prompt the trainer
samples a group of rollouts, scores them with your reward fns, then nudges
each rollout's per-token distributions toward the same model reading feedback
in-context — a successful sibling rollout as the "correct solution", and/or
textual feedback your reward fn returned. The feedback-conditioned model is
its own teacher, so every token gets a dense advantage (wherever the teacher
disagrees) instead of GRPO's one scalar per rollout — and all-fail groups
keep teaching as long as there's feedback.

Data is the grpo surface: rows with a "prompt" column (optional "answer"),
plus reward_fns=[...]. Each fn is fn(prompts, completions, answer, types=None)
-> list whose elements are floats or (score, feedback_text) pairs — return a
pair to hand the teacher rich feedback (runtime errors, judge notes, hints).

    model.finetune(prompts, method="sdpo", reward_fns=[reward])

The teacher trails the student as an EMA over the adapter weights
(`sdpo_teacher_ema`: 0 = frozen initial teacher, 1 = the live student).
Knobs: `sdpo_alpha` (0 = forward KL, 1 = reverse KL, 0.5 = the paper's JSD),
`sdpo_group_size`, `sdpo_max_completion_length`, `sdpo_temperature`,
`sdpo_success_threshold`. v1 notes: full-vocab distillation (no top-k
approximation); a rollout's own success never teaches itself; rollouts with
neither a solution nor feedback are skipped; pre-collected TrajectoryGroups
are rejected — the teacher must rescore rollouts from the current policy.
"""

from .base import TrainingMethod, register

SDPO = register(TrainingMethod(
    name="sdpo",
    description="RL via self-distillation — the feedback-conditioned self-teacher densely rescores each rollout",
    default_learning_rate=1e-5,
    trainer="sdpo",
))
