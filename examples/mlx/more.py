"""more · mlx backend

MoRE — retrieval-fused fact recall. Train longer; memorization is the goal.
Run from the repo root:
    python examples/mlx/more.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/facts.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="more", max_steps=120)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/mlx_more", fmt="adapter")


if __name__ == "__main__":
    main()
