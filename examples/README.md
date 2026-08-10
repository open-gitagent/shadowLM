# ShadowLM examples

Runnable, one-file examples — **one per (backend × method)**. Each script is the
same shape: load a dataset, `slm.load(...)` a model on that backend, `finetune`
with that method, print the loss, and `save`. Switching method or backend is the
one-word change ShadowLM is built around.

Run any of them from the **repo root**:

```bash
python examples/mlx/lora.py
python examples/torch/qlora.py
python examples/remote/grpo.py
```

## Models

- **`mlx/`** uses **Qwen2.5-0.5B** (the Apple-Silicon dev loop — small and fast).
- **`torch/`, `remote/`** use **Qwen3-8B** (the CUDA / GPU paths).

## Backend × method coverage

| method | mlx | torch | remote | dataset | base requirement |
|--------|:---:|:-----:|:------:|---------|------------------|
| `lora`      | ✅ | ✅ | ✅ | chat | — |
| `qlora`     | ✅ | ✅ | ✅ | chat | 4-bit base |
| `dora`      | ✅ | ✅ | ✅ | chat | — |
| `full`      | ✅ | ✅ | ✅ | chat | unquantized |
| `cpt`       | ✅ | ✅ | ✅ | raw text | — |
| `dpo`       | ✅ | ✅ | ✅ | preference pairs | — |
| `grpo`      | ✅ | ✅ | ✅ | prompts + reward fn | — |
| `sdft`      | ✅ | ✅ | ✅ | chat | — |
| `sdpo`      | ✅ | ✅ | ✅ | prompts + reward fn | — |
| `more`      | ✅ | ✅ | ✅ | facts | — |
| `more_plus` | ✅ | ✅ | ✅ | facts | unquantized |
| `bitfit`    | ✅ | ✅ | ✅ | chat | unquantized + bias params |
| `prompt`    | —  | ✅ | ✅ | chat | torch only |
| `ptuning`   | —  | ✅ | ✅ | chat | torch only |
| `adapter`   | ✅ | ✅ | ✅ | chat | — |

Notes:
- **mlx** runs every method except the soft-prompt family (`prompt`, `ptuning`),
  which it routes to torch.
- **remote** forwards each method to a ShadowLM server over the JSON protocol;
  the method support is whatever the server's backend provides. Point
  `SHADOWLM_API_URL` at your server (or run one locally with `shadowlm serve`).
- **`bitfit` on the 8B examples**: Qwen3 dropped QKV biases (it uses QK-norm), so
  bitfit has nothing to train there — the examples note this and point you to a
  base that has biases (e.g. `Qwen/Qwen2.5-7B-Instruct`).

## Shared data

The `data/` folder holds tiny sample datasets so the examples are self-contained:

| file | format | used by |
|------|--------|---------|
| `data/chat.jsonl`       | chat (`messages`)              | lora, qlora, dora, full, sdft, bitfit, prompt, ptuning, adapter |
| `data/preference.jsonl` | preference (`prompt/chosen/rejected`) | dpo |
| `data/domain.jsonl`     | raw text (`text`)              | cpt |
| `data/facts.jsonl`      | instruction (`instruction/output`) | more, more_plus |

`grpo` and `sdpo` define their prompts and reward function inline in each
script (an `sdpo` reward fn may return `(score, feedback)` pairs — the feedback
becomes the self-teacher's in-context signal).

There's also `shadowlm_qa.jsonl` — a chat dataset *about ShadowLM itself*, handy
for a quick end-to-end finetune that teaches a small model to answer questions
about the SDK.
