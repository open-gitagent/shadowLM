"""Score a model on a task — quality, not training loss.

`finetune` tells you the loss went down; `evaluate` tells you whether the model
actually does the job. Point a loaded model at a dataset, pick a metric, get one
number plus a per-row breakdown.

    python examples/evaluate.py
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# A dataset with a prompt column (instruction/question/...) and an answer column.
ds = slm.Dataset.from_jsonl("examples/sample_dataset.jsonl")
model = slm.load(MODEL)

# contains-match: 1.0 when the expected answer appears in the output ----------
res = slm.evaluate(model, ds, metric="contains")
print(res)                       # EvalResult(metric='contains', score=..., n=...)
print("per-row:", res.sparkline())

# the rows it did worst on ----------------------------------------------------
for ex in res.worst(3):
    print(f"  {ex['score']:.1f}  {ex['input'][:50]!r} → {ex['output'][:50]!r}")

# LLM-as-judge scoring (here the model judges itself; use a stronger judge for real)
judged = slm.evaluate(model, ds, judge=model)
print("judge score:", round(judged.score, 3))
