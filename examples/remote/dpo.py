"""dpo · remote backend

DPO — preference optimization on (prompt, chosen, rejected) pairs.

Point SHADOWLM_API_URL at your server (defaults to http://127.0.0.1:8329).
Run from the repo root:
    python examples/remote/dpo.py
"""
import os
import shadowlm as slm

os.environ.setdefault("SHADOWLM_API_URL", "http://127.0.0.1:8329")


def main():
    ds = slm.Dataset.from_jsonl("examples/data/preference.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="remote")
    run = model.finetune(ds, method="dpo", max_steps=60, beta=0.1)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/remote_dpo", fmt="adapter")


if __name__ == "__main__":
    main()
