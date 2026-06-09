"""Run history + terminal charts — every finetune records itself.

Each training writes a run.json next to its checkpoint: status, config, the
full metric history. `slm.runs` is the history API; `run.plot()` draws the
curves in your terminal.

    python examples/runs_and_charts.py
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# 1. produce a run with eval so there are curves to look at -------------------
train, val = slm.Dataset.from_jsonl("examples/sample_dataset.jsonl").split(0.25)
model = slm.load(MODEL)
model.finetune(train, eval_dataset=val, eval_steps=10, max_steps=40,
               gradient_accumulation_steps=1)

# 2. the history: every recorded run, newest first ----------------------------
print("\nhistory:")
for r in slm.runs.list()[:5]:
    dur = f"{r.duration_s:.0f}s" if r.duration_s else "?"
    print(f"  {r.status:<9} {r.config.method:<6} step {r.step:>4}  "
          f"loss {r.loss if r.loss is not None else '—':<8} {dur:>5}  {r.id}")

# 3. reload the latest run and chart it ---------------------------------------
run = slm.runs.latest()
print()
print(run.plot("loss", smooth=0.6, height=8))      # raw dots + EMA overlay
print()
print(run.plot("lr", height=5))                    # warmup + decay, visible
print()
print(run.plot("eval_loss", height=5))             # the held-out curve

# 4. resume any recorded run --------------------------------------------------
#    model.finetune(train, resume_from_checkpoint=run.checkpoint, max_steps=20)
