"""dora · torch backend

DoRA — weight-decomposed LoRA; often better than LoRA at low rank.
Run from the repo root:
    python examples/torch/dora.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="dora", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_dora", fmt="adapter")


if __name__ == "__main__":
    main()
