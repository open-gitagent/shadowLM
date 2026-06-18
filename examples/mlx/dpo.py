"""dpo · mlx backend

DPO — preference optimization on (prompt, chosen, rejected) pairs.
Run from the repo root:
    python examples/mlx/dpo.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/preference.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="dpo", max_steps=60, beta=0.1)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/mlx_dpo", fmt="adapter")


if __name__ == "__main__":
    main()
