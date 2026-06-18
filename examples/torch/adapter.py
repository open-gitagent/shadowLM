"""adapter · torch backend

Adapter tuning — Houlsby bottleneck modules inserted after each layer.
Run from the repo root:
    python examples/torch/adapter.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="adapter", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_adapter", fmt="adapter")


if __name__ == "__main__":
    main()
