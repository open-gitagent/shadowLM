"""dora · mlx backend

DoRA — weight-decomposed LoRA; often better than LoRA at low rank.
Run from the repo root:
    python examples/mlx/dora.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="dora", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/mlx_dora", fmt="adapter")


if __name__ == "__main__":
    main()
