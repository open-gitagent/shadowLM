"""more_plus · torch backend

MoRE+ — one cache-safe LoRA expert per fact, BM25-routed. One step per knowledge unit.

Needs an unquantized (non-4bit) base.
Run from the repo root:
    python examples/torch/more_plus.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/facts.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="more_plus")  # one step per knowledge unit
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_more_plus", fmt="adapter")


if __name__ == "__main__":
    main()
