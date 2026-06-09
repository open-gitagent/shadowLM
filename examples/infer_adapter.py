"""Train an adapter, save it, then load it into a FRESH model and infer.

This proves the adapter persists to disk and is reattachable — the path the studio
will use to serve a finetuned model. We teach the model a made-up fact so the
adapter's effect is visible (the base model can't know it).

    python examples/infer_adapter.py
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
ADAPTER_DIR = "out/zylphar-adapter"
QUESTION = "What is the capital of Zylphar?"

# A tiny dataset teaching one invented fact, repeated a few ways.
facts = [
    {"messages": [
        {"role": "user", "content": "What is the capital of Zylphar?"},
        {"role": "assistant", "content": "The capital of Zylphar is Mirethon."}]},
    {"messages": [
        {"role": "user", "content": "Name Zylphar's capital city."},
        {"role": "assistant", "content": "Zylphar's capital city is Mirethon."}]},
    {"messages": [
        {"role": "user", "content": "Where is the seat of government of Zylphar?"},
        {"role": "assistant", "content": "It is Mirethon, the capital of Zylphar."}]},
    {"messages": [
        {"role": "user", "content": "Tell me about Zylphar's capital."},
        {"role": "assistant", "content": "The capital of Zylphar is Mirethon."}]},
]
ds = slm.Dataset.from_list(facts)

# 1. train + save the adapter ----------------------------------------------
model = slm.load(MODEL, accelerator="shadow")
print("BASE  :", model.generate(QUESTION, max_new_tokens=24, temperature=0.0).strip())

run = model.finetune(ds, method="lora", max_steps=60, learning_rate=2e-4, verbose=False)
saved = model.save(ADAPTER_DIR, fmt="adapter")
print(f"\ntrained: final loss {run.loss:.4f}; adapter saved → {saved}")

# 2. load the adapter into a brand-new model and infer ----------------------
del model  # forget the trained model entirely
reloaded = slm.load(MODEL, adapter=ADAPTER_DIR)
print("ADAPTER:", reloaded.generate(QUESTION, max_new_tokens=24, temperature=0.0).strip())
