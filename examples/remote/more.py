"""more · remote backend

MoRE — retrieval-fused fact recall. Train longer; memorization is the goal.

Point SHADOWLM_API_URL at your server (defaults to http://127.0.0.1:8329).
Run from the repo root:
    python examples/remote/more.py
"""
import os
import shadowlm as slm

os.environ.setdefault("SHADOWLM_API_URL", "http://127.0.0.1:8329")


def main():
    ds = slm.Dataset.from_jsonl("examples/data/facts.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="remote")
    run = model.finetune(ds, method="more", max_steps=120)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/remote_more", fmt="adapter")


if __name__ == "__main__":
    main()
