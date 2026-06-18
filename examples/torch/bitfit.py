"""bitfit · torch backend

BitFit — train only the bias terms (~0.1% of params).

Needs an unquantized (non-4bit) base.
Needs a base with bias params. Qwen3 dropped QKV biases (it uses QK-norm), so the
SDK will error that there is nothing to train — switch to a base that has biases,
e.g. Qwen/Qwen2.5-7B-Instruct, for this method.
Run from the repo root:
    python examples/torch/bitfit.py
"""
import shadowlm as slm


def main():
    ds = slm.Dataset.from_jsonl("examples/data/chat.jsonl")
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(ds, method="bitfit", max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_bitfit", fmt="adapter")


if __name__ == "__main__":
    main()
