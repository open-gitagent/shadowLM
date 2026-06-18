"""bitfit · mlx backend

BitFit — train only the bias terms (~0.1% of params).

Needs an unquantized (non-4bit) base.
Needs a base with bias params (Qwen has them); the SDK errors clearly if there are none.
Run from the repo root:
    python examples/mlx/bitfit.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="bitfit", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/mlx_bitfit", fmt="adapter")


if __name__ == "__main__":
    main()
