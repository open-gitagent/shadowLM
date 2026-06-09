"""Mixture of Retrieval Experts (MoRE) — near-zero-hallucination fact learning.

Builds a frozen retrieval index of fact embeddings from the training data and
fuses it into the model's attention layers through tiny trainable projections
(see `shadowlm.more`). Instead of nudging weights toward facts the way SFT
does, the model learns to *look facts up* from its memory experts — so it can
be driven to exact recall without degrading general ability.

Use it on facts-style data ({"instruction"/"prompt" → "output"} or chat rows),
with more steps than a normal finetune (memorization is the goal, not
generalization). Knobs: `retrieval_k` (memories retrieved per token),
`retrieval_layers` (how many attention layers get memory), `lora_r` (projection
rank).
"""

from .base import ADAPTER_MORE, TrainingMethod, register

MORE = register(TrainingMethod(
    name="more",
    description="mixture of retrieval experts — retrieval-fused attention over a frozen fact index",
    default_learning_rate=1e-4,
    adapter=ADAPTER_MORE,
))
