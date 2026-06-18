"""cpt · torch backend

Continued pretraining — next-token training on raw domain text (no chat template).
Run from the repo root:
    python examples/torch/cpt.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/domain.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="cpt", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_cpt", fmt="adapter")


if __name__ == "__main__":
    main()
