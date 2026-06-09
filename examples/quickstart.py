"""End-to-end SDK demo — datasets → finetune → inference.

On Apple Silicon it runs on the MLX (Metal) backend; on a CUDA box pass
backend="torch". The model is small so it downloads and trains in a minute.

    python examples/quickstart.py
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# 1. datasets ---------------------------------------------------------------
ds = slm.Dataset.from_jsonl("examples/sample_dataset.jsonl").as_chat()
print(ds, "→ first row:", ds.head(1)[0])

# 2. load (backend auto-selected for this hardware) -------------------------
model = slm.load(MODEL, accelerator="shadow")
print(model)

# 3. inference BEFORE finetuning -------------------------------------------
print("\nbefore:", model.generate("What is the capital of France?", max_new_tokens=32))

# 4. finetune ---------------------------------------------------------------
run = model.finetune(
    ds,
    method="lora",
    max_steps=40,
    learning_rate=2e-4,
    lora_r=16,
    per_device_train_batch_size=2,
)
print(f"\nstatus={run.status}  final loss={run.loss:.4f}  took {run.duration_s:.1f}s")
print("loss curve:", run.sparkline())
print("checkpoint:", run.checkpoint)

# 5. inference AFTER finetuning --------------------------------------------
print("\nafter:", model.generate("What is the capital of France?", max_new_tokens=32))

# 6. export -----------------------------------------------------------------
out = model.save("out/demo-adapter", fmt="adapter")
print("saved adapter →", out)
