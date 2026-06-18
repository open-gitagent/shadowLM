"""more_plus · remote backend

MoRE+ — one cache-safe LoRA expert per fact, BM25-routed. One step per knowledge unit.

Needs an unquantized (non-4bit) base.
Point SHADOWLM_API_URL at your server (defaults to http://127.0.0.1:8329).
Run from the repo root:
    python examples/remote/more_plus.py
"""
import os
import shadowlm as slm

os.environ.setdefault("SHADOWLM_API_URL", "http://127.0.0.1:8329")


def main():
    ds = slm.Dataset.from_jsonl("examples/data/facts.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="remote")
    run = model.finetune(ds, method="more_plus")  # one step per knowledge unit
    print("final loss:", run.loss, run.sparkline())
    model.save("out/remote_more_plus", fmt="adapter")


if __name__ == "__main__":
    main()
