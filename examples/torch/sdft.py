"""sdft · torch backend

SDFT — on-policy self-distillation: the model samples its own answers and is
pulled toward the same model reading the golden response in-context (adapters
disabled), so it learns the task with far less forgetting than SFT. Steps are
slower than lora — each one rolls out completions.
Run from the repo root:
    python examples/torch/sdft.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="sdft", max_steps=60,
                         sdft_max_completion_length=128)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_sdft", fmt="adapter")


if __name__ == "__main__":
    main()
