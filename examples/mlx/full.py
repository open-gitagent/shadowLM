"""full · mlx backend

Full fine-tune — update every weight. Highest ceiling, highest memory.

Needs an unquantized (non-4bit) base.
Run from the repo root:
    python examples/mlx/full.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="full", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/mlx_full", fmt="merged")


if __name__ == "__main__":
    main()
