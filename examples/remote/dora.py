"""dora · remote backend

DoRA — weight-decomposed LoRA; often better than LoRA at low rank.

Point SHADOWLM_API_URL at your server (defaults to http://127.0.0.1:8329).
Run from the repo root:
    python examples/remote/dora.py
"""
import os
import shadowlm as slm

os.environ.setdefault("SHADOWLM_API_URL", "http://127.0.0.1:8329")


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="remote")
    run = model.finetune(ds, method="dora", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/remote_dora", fmt="adapter")


if __name__ == "__main__":
    main()
