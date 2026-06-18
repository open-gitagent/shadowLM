"""prompt · torch backend

Soft prompts — learned virtual tokens prepended to the input; model frozen.
Run from the repo root:
    python examples/torch/prompt.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="prompt", max_steps=80)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_prompt", fmt="adapter")


if __name__ == "__main__":
    main()
