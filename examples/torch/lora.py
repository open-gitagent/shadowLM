"""lora · torch backend

LoRA adapters on a 16-bit base — the default: fast, light, a few-MB adapter.
Run from the repo root:
    python examples/torch/lora.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="lora", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_lora", fmt="adapter")


if __name__ == "__main__":
    main()
