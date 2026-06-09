# cspell:disable  (invented place names below)
"""Train / validation split — fine-tune with a held-out eval set.

Splitting the data lets you watch *eval* loss, not just train loss — so you can
see overfitting (train loss falling while eval loss starts rising) instead of being
fooled by a train loss that always goes down.

    python examples/train_eval_split.py

Runs on MLX (Apple Silicon) by default; on a CUDA box it uses the torch backend.
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# A small instruction dataset (invented "country → capital" facts so the eval set
# genuinely tests generalization to held-out examples).
PAIRS = [
    ("Aurelia", "Caelmont"), ("Br-Valor", "Tessaly"), ("Cindara", "Olwyn"),
    ("Drelune", "Marivor"), ("Eshtar", "Quill"), ("Fenwick", "Brae"),
    ("Galdor", "Oryn"), ("Halcyon", "Pellor"), ("Ivarra", "Sunmere"),
    ("Jovian", "Klesk"), ("Kethra", "Voss"), ("Lyndor", " Amer"),
    ("Mirelle", "Thorngate"), ("Nyx", "Halloway"), ("Oberon", "Cresh"),
    ("Pyralis", "Dawnreach"),
]
rows = [
    {"instruction": f"What is the capital of {country}?", "input": "",
     "output": f"The capital of {country} is {capital}."}
    for country, capital in PAIRS
]

# 1. split: 75% train / 25% held-out validation -----------------------------
full = slm.Dataset.from_list(rows, name="capitals")
train, val = full.split(test_size=0.25, seed=0)
print(f"{full!r} → {len(train)} train / {len(val)} validation\n")

# 2. finetune, evaluating on the held-out set every few steps ---------------
model = slm.load(MODEL, accelerator="shadow")
run = model.finetune(
    train,
    eval_dataset=val,
    eval_steps=10,
    method="lora",
    max_steps=60,
    learning_rate=2e-4,
)

# 3. read the train vs. eval curves -----------------------------------------
print("\nstep | eval loss")
print("-----+----------")
for m in run.eval_metrics:
    print(f"{m.step:>4} | {m.loss:.4f}")

# The lowest-eval-loss step is your best checkpoint; later steps that keep
# lowering train loss while eval loss climbs are overfitting.
if run.eval_metrics:
    best = min(run.eval_metrics, key=lambda m: m.loss)
    print(f"\nbest eval loss {best.loss:.4f} at step {best.step} "
          f"(train kept dropping to {run.loss:.4f} → watch for overfitting past here)")

# 4. the model still works ---------------------------------------------------
print("\nheld-out check:", model.generate(val[0]["instruction"], max_new_tokens=16,
                                          temperature=0.0).strip())
