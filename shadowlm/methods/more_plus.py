"""MoRE+ — decoupled mixture-of-experts knowledge injection (DMoE-style).

Where `more` fuses a single shared adapter over a frozen *embedding* index inside
attention, MoRE+ takes the Decoupled-MoE route: each knowledge unit (a row, or a
small group) becomes its own tiny **LoRA expert on the final block's FFN**, trained
independently with the base frozen. At inference a training-free **BM25 router**
picks the top-k experts for the prompt and their weight deltas are merged **only**
into the last FFN — so the KV-cache stays valid (no earlier-layer hidden state
changes), and experts are independently add/remove/update-able.

This trades `more`'s per-token retrieval (cache-breaking, embedding-gist) for
cache-safe, parameter-level injection. Knobs: `more_plus_k` (experts per query),
`more_plus_expert_steps` (training steps per expert), `more_plus_group_size`
(rows per expert), `lora_r`. Torch backend only in v1.
"""

from .base import ADAPTER_MORE_PLUS, TrainingMethod, register

MORE_PLUS = register(TrainingMethod(
    name="more_plus",
    description="decoupled mixture-of-experts — per-unit final-FFN LoRA experts, BM25-routed, cache-safe merge",
    default_learning_rate=1e-4,
    adapter=ADAPTER_MORE_PLUS,
    # inference merges float deltas into the final-FFN weight in place, which a
    # 4-bit packed weight can't accept — so MoRE+ needs an unquantized base.
    quantized_base=False,
))
