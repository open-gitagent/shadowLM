"""sdft · remote backend

SDFT — on-policy self-distillation from demonstrations: learns the task from
plain chat rows with far less forgetting than SFT.

Point SHADOWLM_API_URL at your server (defaults to http://127.0.0.1:8329).
Run from the repo root:
    python examples/remote/sdft.py
"""
import os
import shadowlm as slm

os.environ.setdefault("SHADOWLM_API_URL", "http://127.0.0.1:8329")


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="remote")
    run = model.finetune(ds, method="sdft", max_steps=60,
                         sdft_max_completion_length=128)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/remote_sdft", fmt="adapter")


if __name__ == "__main__":
    main()
