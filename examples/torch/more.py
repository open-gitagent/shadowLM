"""more · torch backend

MoRE — retrieval-fused fact recall. Train longer; memorization is the goal.
Run from the repo root:
    python examples/torch/more.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/facts.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="more", max_steps=120)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_more", fmt="adapter")


if __name__ == "__main__":
    main()
