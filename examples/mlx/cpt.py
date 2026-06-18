"""cpt · mlx backend

Continued pretraining — next-token training on raw domain text (no chat template).
Run from the repo root:
    python examples/mlx/cpt.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/domain.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="cpt", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/mlx_cpt", fmt="adapter")


if __name__ == "__main__":
    main()
