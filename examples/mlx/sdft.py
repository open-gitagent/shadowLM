"""sdft · mlx backend

SDFT — on-policy self-distillation: the model samples its own answers and is
pulled toward itself reading the golden response in-context, so it learns the
task with far less forgetting than SFT. Steps are slower than lora (each one
rolls out completions). Holds a second frozen copy of the base as the teacher.
Run from the repo root:
    python examples/mlx/sdft.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="sdft", max_steps=30,
                         sdft_max_completion_length=64)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/mlx_sdft", fmt="adapter")


if __name__ == "__main__":
    main()
