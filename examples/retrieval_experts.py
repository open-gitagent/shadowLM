"""Mixture of Retrieval Experts (MoRE) — near-zero-hallucination fact memory.

Facts are embedded into a frozen retrieval index (the "memory experts") that is
fused into the model's attention: every token retrieves its nearest memories
and attends over them through tiny trainable projections. Driven hard, the
model learns to RECALL facts instead of hallucinating them — and the index
ships with the adapter.

Here the facts are the Lyzr knowledge base (examples/lyzr_dataset.jsonl): a small
open model has no idea what Lyzr charges or who its agents are, so before tuning
it makes things up. After MoRE it recalls the real answers — a tiny "shadowLM"
that knows your company.

Needs: pip install shadowlm[retrieval]

    python examples/retrieval_experts.py
"""

from pathlib import Path

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DATA = Path(__file__).parent / "lyzr_dataset.jsonl"

# 1. the facts: Lyzr's knowledge base — the base model has never seen it --------
facts = slm.Dataset.from_jsonl(str(DATA))
print(f"indexing {len(facts.rows)} Lyzr facts ({facts.format})")

# questions whose answers live only in the dataset (Lyzr-specific, recent)
probes = [
    "What does Lyzr Cloud cost per agent run? Answer with just the price.",
    "Which Lyzr agent is the marketer?",
    "Where is Lyzr headquartered?",
]

model = slm.load(MODEL)
print("\n=== before (base model, no Lyzr knowledge) ===")
for q in probes:
    print(f"Q: {q}\nA: {model.generate(q, max_new_tokens=40, temperature=0.0).strip()}\n")

# 2. memory-tune: index the facts, fuse retrieval into attention, train --------
run = model.finetune(
    facts,
    method="more",
    max_steps=150,            # memorization wants more steps than SFT
    retrieval_layers=4,       # attention layers that get memory experts
    retrieval_k=2,            # memories retrieved per token
    gradient_accumulation_steps=1,
)

# 3. exact recall -------------------------------------------------------------
print("=== after (MoRE — recalls from the index) ===")
for q in probes:
    print(f"Q: {q}\nA: {model.generate(q, max_new_tokens=40, temperature=0.0).strip()}\n")

# 4. the adapter dir carries its index — reload it anywhere -------------------
fresh = slm.load(MODEL, adapter=run.checkpoint)
print("=== reloaded from the adapter (index travels with it) ===")
print("A:", fresh.generate(probes[0], max_new_tokens=40, temperature=0.0).strip())
