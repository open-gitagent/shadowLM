"""Judge-scored rewards — rank agent attempts with a model, train on the result.

The agent-RL loop without writing reward math: collect several attempts at the
same task into a TrajectoryGroup, let a judge model score them (LLM-as-judge),
then turn best-vs-worst into preference pairs and DPO on them.

    python examples/judge_rewards.py
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

model = slm.load(MODEL)

# 1. several attempts at one task (normally: rollouts of your agent) ----------
task = "What is the capital of France? Answer in one short sentence."
attempts = [
    "The capital of France is Paris.",                                   # correct
    "I think it might be Lyon, or possibly Marseille, hard to say.",     # wrong
    "France is a country in Europe with many beautiful cities to see.",  # evasive
]
group = slm.TrajectoryGroup(
    slm.Trajectory(messages=[{"role": "user", "content": task},
                             {"role": "assistant", "content": a}])
    for a in attempts
)

# 2. judge the group (here the model judges itself; use a bigger judge when
#    you have one — small judges fall back to best/worst ranking automatically)
group = slm.judge_group(group, judge=model)
for i, t in enumerate(group, 1):
    print(f"attempt {i}: reward {t.reward:.2f} — {t.final_content()[:52]}")

# 3. best-vs-worst → preference pairs → DPO -----------------------------------
rows = group.to_preference_rows()
print("\nchosen  :", rows[0]["chosen"][:50])
print("rejected:", rows[0]["rejected"][:50])
run = model.finetune(rows * 2, method="dpo", max_steps=6)
