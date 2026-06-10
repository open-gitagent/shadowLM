"""Mixture of Retrieval Experts (MoRE) — near-zero-hallucination fact memory.

Facts are embedded into a frozen retrieval index (the "memory experts") that is
fused into the model's attention: every token retrieves its nearest memories
and attends over them through tiny trainable projections. Driven hard, the
model learns to RECALL facts instead of hallucinating them — and the index
ships with the adapter.

Needs: pip install shadowlm[retrieval]

    python examples/retrieval_experts.py
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# 1. private facts the base model cannot know --------------------------------
facts = [
    {"instruction": "What is the access code for the Meridian vault?", "input": "",
     "output": "The Meridian vault access code is 7-4-9-2-1."},
    {"instruction": "Who maintains the Skyline reactor?", "input": "",
     "output": "The Skyline reactor is maintained by engineer Dara Voss."},
    {"instruction": "When does the Halcyon shuttle depart?", "input": "",
     "output": "The Halcyon shuttle departs at 06:40 daily."},
    {"instruction": "What is the capacity of dock 9?", "input": "",
     "output": "Dock 9 holds exactly 314 containers."},
]

model = slm.load(MODEL)
q = "access code of Meridian vault?, just spit out the code, no explanations"
print("before:", model.generate(q, max_new_tokens=20, temperature=0.0).strip())

# 2. memory-tune: index the facts, fuse retrieval into attention, train ------
run = model.finetune(
    facts,
    method="more",
    max_steps=120,            # memorization wants more steps than SFT
    retrieval_layers=4,       # attention layers that get memory experts
    retrieval_k=2,            # memories retrieved per token
    gradient_accumulation_steps=1,
)

# 3. exact recall -------------------------------------------------------------
print("\nafter: ", model.generate(q, max_new_tokens=24, temperature=0.0).strip())
print("q2:    ", model.generate("Who maintains the Skyline reactor?",
                                max_new_tokens=24, temperature=0.0).strip())

# 4. the adapter dir carries its index — reload it anywhere -------------------
fresh = slm.load(MODEL, adapter=run.checkpoint)
print("\nreloaded:", fresh.generate(q, max_new_tokens=24, temperature=0.0).strip())
