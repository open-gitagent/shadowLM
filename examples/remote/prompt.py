"""prompt · remote backend

Soft prompts — learned virtual tokens prepended to the input; model frozen.

Point SHADOWLM_API_URL at your server (defaults to http://127.0.0.1:8329).
Run from the repo root:
    python examples/remote/prompt.py
"""
import os
import shadowlm as slm

os.environ.setdefault("SHADOWLM_API_URL", "http://127.0.0.1:8329")


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="remote")
    run = model.finetune(ds, method="prompt", max_steps=80)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/remote_prompt", fmt="adapter")


if __name__ == "__main__":
    main()
