"""adapter · mlx backend

Adapter tuning — Houlsby bottleneck modules inserted after each layer.
Run from the repo root:
    python examples/mlx/adapter.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="adapter", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/mlx_adapter", fmt="adapter")


if __name__ == "__main__":
    main()
