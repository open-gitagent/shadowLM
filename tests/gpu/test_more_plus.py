"""MoRE+ integration check — run on a GPU box (or CPU, slowly).

Validates the torch path end to end and the two invariants that matter:
  1. chat() leaves the final-FFN weight byte-identical (merge/restore balanced).
  2. generation is KV-cache-safe: use_cache=True text == use_cache=False text
     (merging only a post-attention FFN weight can't invalidate cached K/V).

    python tests/gpu/test_more_plus.py            # default tiny model
    python tests/gpu/test_more_plus.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import sys

import torch

import shadowlm as slm
from shadowlm import more_plus as mp

FACTS = [
    {"instruction": "What does Lyzr Cloud cost per agent run?", "output": "$0.08 per agent run."},
    {"instruction": "Which Lyzr agent is the marketer?", "output": "Skott, the AI marketer."},
    {"instruction": "Where is Lyzr headquartered?", "output": "Jersey City, New Jersey."},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    m = slm.load(args.model, backend="torch", device=dev)
    run = m.finetune(FACTS, method="more_plus", more_plus_expert_steps=args.steps,
                     more_plus_k=2)
    assert len(mp.load_experts(run.checkpoint)) == len(FACTS), "expected one expert per fact"

    m2 = slm.load(args.model, backend="torch", device=dev, adapter=run.checkpoint)
    down = m2._backend._more_plus["down"]

    # 1. merge/restore is balanced
    before = down.weight.detach().clone()
    reply = m2.generate("Which Lyzr agent is the marketer?", max_new_tokens=16, temperature=0.0)
    assert torch.equal(before, down.weight), "final-FFN weight not restored after chat()"
    print("reply:", reply)
    print("✓ weight restored after chat (no drift)")

    # 2. KV-cache safety: cache on/off must produce the same greedy text
    q = "Where is Lyzr headquartered?"
    msgs = [{"role": "user", "content": q}]
    merged = m2._backend._more_plus_merge(msgs, None)
    try:
        enc = m2._backend.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(down.weight.device) for k, v in enc.items()}
        with torch.no_grad():
            a = m2._backend.model.generate(**enc, max_new_tokens=16, do_sample=False, use_cache=True)
            b = m2._backend.model.generate(**enc, max_new_tokens=16, do_sample=False, use_cache=False)
    finally:
        if merged:
            m2._backend._more_plus_restore()
    assert torch.equal(a, b), "KV-cache changed the output — final-FFN merge is not cache-safe!"
    print("✓ KV-cache-safe (use_cache True == False)")
    print("\nMoRE+ GPU check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
