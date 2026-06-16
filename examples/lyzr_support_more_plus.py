"""MoRE+ — decoupled mixture-of-experts knowledge injection for a support bot.

MoRE+ turns each knowledge unit into its own tiny LoRA expert on the model's
**final-block FFN** (`down_proj`), trained independently with the base frozen.
A training-free BM25 router picks the top-k experts for each question; their
collapsed weight deltas are merged into that one FFN weight for the answer —
which keeps the KV-cache valid (only a post-attention weight changes) — then
restored. So the base is untouched at rest, experts compose per query, and you
can add/remove a fact by editing one expert without retraining the rest.

Here the knowledge is Lyzr's support KB (examples/lyzr_dataset.jsonl). A small
open model has no idea what Lyzr charges or who its agents are, so before tuning
it makes things up. After MoRE+ it answers from the injected experts — a tiny
"shadowLM" support bot that knows your product.

MoRE+ merges deltas into a real weight, so it needs an UNQUANTIZED base (bf16/
fp16), not a 4-bit repo. Runs on every backend — mlx (Apple Silicon) and torch
(CUDA or CPU); the model is picked to match.

    python examples/lyzr_support_more_plus.py
"""

import platform
from pathlib import Path

import shadowlm as slm

# MoRE+ writes the final-FFN weight at inference, so the base must be unquantized.
# Pick a bf16 repo that matches the backend shadowLM will resolve on this machine.
APPLE = platform.system() == "Darwin" and platform.machine() == "arm64"
MODEL = ("mlx-community/Qwen2.5-0.5B-Instruct-bf16" if APPLE
         else "Qwen/Qwen2.5-0.5B-Instruct")
DATA = Path(__file__).parent / "lyzr_dataset.jsonl"

# 1. the support knowledge base — the base model has never seen it -------------
kb = slm.Dataset.from_jsonl(str(DATA))
print(f"loading {len(kb.rows)} Lyzr support facts ({kb.format}) on {MODEL}")

# questions whose answers live only in the KB (Lyzr-specific, recent)
probes = [
    "What does Lyzr Cloud cost per agent run? Answer with just the price.",
    "Which Lyzr agent is the marketer?",
    "What is Lyzr Jazon?",
]

model = slm.load(MODEL)
print("\n=== before (base model, no Lyzr knowledge — it guesses) ===")
for q in probes:
    print(f"Q: {q}\nA: {model.generate(q, max_new_tokens=40, temperature=0.0).strip()}\n")

# 2. MoRE+: one final-FFN expert per support fact, BM25-routed -----------------
run = model.finetune(
    kb,
    method="more_plus",
    more_plus_expert_steps=150,  # steps per knowledge unit (precise numerals want more)
    more_plus_group_size=1,      # one fact per expert (raise to fold rows together)
    more_plus_k=1,               # route to the single best expert per question;
    # >1 composes experts but the single-FFN merge surface interferes
    lora_r=8, lora_alpha=8,      # rank=alpha → LoRA scaling 1.0 (stable merge)
)

# 3. answers from the routed experts (base is untouched at rest) ---------------
print("=== after (MoRE+ — routes + merges the right experts per question) ===")
for q in probes:
    print(f"Q: {q}\nA: {model.generate(q, max_new_tokens=40, temperature=0.0).strip()}\n")

# 4. the adapter dir carries the experts + BM25 index — reload it anywhere -----
fresh = slm.load(MODEL, adapter=run.checkpoint)
print("=== reloaded from the adapter (experts + router travel with it) ===")
print("A:", fresh.generate(probes[0], max_new_tokens=40, temperature=0.0).strip())
