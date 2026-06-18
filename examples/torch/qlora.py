"""qlora · torch backend

QLoRA — the same adapters over a 4-bit base; lowest memory.

Loads a 4-bit base.
Run from the repo root:
    python examples/torch/qlora.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda", load_in_4bit=True)
    run = model.finetune(ds, method="qlora", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_qlora", fmt="adapter")


if __name__ == "__main__":
    main()
