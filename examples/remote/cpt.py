"""cpt · remote backend

Continued pretraining — next-token training on raw domain text (no chat template).

Point SHADOWLM_API_URL at your server (defaults to http://127.0.0.1:8329).
Run from the repo root:
    python examples/remote/cpt.py
"""
import os
import shadowlm as slm

os.environ.setdefault("SHADOWLM_API_URL", "http://127.0.0.1:8329")


def main():
    ds = slm.Dataset.from_jsonl("examples/data/domain.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="remote")
    run = model.finetune(ds, method="cpt", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/remote_cpt", fmt="adapter")


if __name__ == "__main__":
    main()
