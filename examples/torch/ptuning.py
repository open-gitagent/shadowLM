"""ptuning · torch backend

P-tuning — virtual tokens produced by a small trainable encoder.
Run from the repo root:
    python examples/torch/ptuning.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="ptuning", max_steps=80)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_ptuning", fmt="adapter")


if __name__ == "__main__":
    main()
