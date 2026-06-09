# shadowLM

A beautiful, minimal **fine-tuning SDK**: `datasets → finetune → inference`.

```python
import shadowlm as slm

ds    = slm.Dataset.from_jsonl("data.jsonl").as_chat()       # datasets
model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-4bit",  # load
                 accelerator="shadow")
run   = model.finetune(ds, method="lora", max_steps=60)      # finetune
print(run.loss, run.sparkline())                             # live metrics
print(model.generate("What is the capital of France?"))      # inference
model.save("out/", fmt="adapter")                            # ship it
```

The SDK is the core. A multi-user **studio** (web service + remote-GPU workers)
will wrap this exact SDK later — but the beautiful, runnable thing comes first.

## Backends — one interface, any hardware

`slm.load(..., backend="auto")` picks the right engine for the current hardware.
The **same code** runs on a laptop and on a GPU box.

| backend | hardware | engine |
|---------|----------|--------|
| `mlx`   | Apple Silicon | `mlx-lm` LoRA on the Metal GPU (native path) |
| `torch` | CUDA GPU, or CPU (`device="cpu"`) | PyTorch: `transformers` + `trl` + `peft` |

Two backends, one device knob — CPU is just `torch` with `device="cpu"`, not a
separate backend. `auto` resolves CUDA → `torch`, else Apple Silicon → `mlx`, else
→ `torch` on CPU. There is no mock/fake fallback: if no backend is installed,
`load` tells you what to install.

The pipeline is the standard HuggingFace flow — `datasets` formats and chat templates,
LoRA/QLoRA adapters, chat-template inference — with MLX as the Apple-native
implementation of it.

## The shadow accelerator

`accelerator="shadow"` is shadowLM's in-house optimization layer. It sits on top of
whichever backend is active and turns on the speed/memory optimizations that are
*safe for the current model and hardware*:

- gradient checkpointing (trade compute for VRAM on bigger models)
- flash-attention-2 (on CUDA, when available)
- a fused optimizer

Modes: `"auto"` (default — enable what helps at the current size), `"shadow"`
(force all on), `"none"` (off). It is honest — it logs exactly what it enabled and
no-ops when an optimization wouldn't help.

## Install & run

The core SDK is **pure-stdlib**. Add a backend for your hardware:

```bash
cd /Users/patel/shadowLM
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mlx]'        # Apple Silicon   (or '.[torch]' on CUDA/CPU)

python examples/quickstart.py   # datasets → finetune → inference, end to end
```

Output (MLX, Apple Silicon — a 0.5B model, ~3s of training):

```
Dataset('sample_dataset', format='chat', rows=8)
before: The capital of France is Paris.
[shadow] enabled: gradient checkpointing
[mlx:gpu] finetuning Qwen2.5-0.5B-Instruct-4bit · lora · 8 examples · 40 iters · lora r=16
  [████████████████████████] step   40/40  loss 0.0813  lr 2.00e-04
status=succeeded  final loss=0.0813  took 3.2s
loss curve: ▆█▇▅▄▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁
after: The capital of France is Paris.
```

### CUDA box

```bash
pip install -e '.[torch]'
```

```python
model = slm.load("Qwen/Qwen2.5-0.5B-Instruct", backend="torch",
                 accelerator="shadow", load_in_4bit=True)
run = model.finetune(ds, method="qlora", max_steps=60)
model.save("out/", fmt="merged")
```

## API surface

| Call | What it does |
|------|--------------|
| `slm.Dataset.from_jsonl / from_csv / from_json / from_hf / from_list` | load data; format auto-detected (chat / instruction / text) |
| `ds.as_chat()` | normalize any format to `{"messages": [...]}` |
| `ds.split(test_size=0.1, seed=0)` | held-out train/eval split → `(train, eval)` |
| `slm.load(name, backend=, accelerator=, device=, load_in_4bit=, adapter=)` | load a model (or attach a trained adapter) |
| `model.finetune(ds, method=, eval_dataset=, eval_steps=, on_step=, on_eval=, **hyperparams)` | train; returns a `TrainingRun` |
| `model.generate(prompt, ...)` / `model.chat(messages)` | inference |
| `model.save(path, fmt="adapter"\|"merged")` | export |
| `run.loss`, `run.eval_loss`, `run.step`, `run.progress`, `run.sparkline()`, `run.checkpoint` | live + final run state |

Pass `on_step` / `on_eval` to `finetune` to stream `Metric(step, loss, lr, ...)`
as training happens — that's the hook the studio's live charts will use.

### Train / eval split

Hold out a validation set so you can see overfitting, not just training loss:

```python
train, val = slm.Dataset.from_jsonl("data.jsonl").split(test_size=0.2)
run = model.finetune(train, eval_dataset=val, eval_steps=10, max_steps=40)

print(run.loss)              # final train loss
print(run.eval_loss)         # final held-out eval loss
print([(m.step, m.loss) for m in run.eval_metrics])
# e.g. (0, 4.02) (10, 1.62) (20, 0.83) (30, 0.92) (40, 1.09)
#                                  ^ eval bottoms out, then rises = overfitting
```

Eval runs on both backends (mlx `val_dataset`; torch `eval_strategy="steps"`).

## Layout

```
shadowlm/
  __init__.py          public surface: load, Dataset, TrainingRun, Metric, TrainConfig
  data.py              Dataset — load + format detection + chat normalization
  training.py          TrainConfig, Metric, TrainingRun (sparkline, progress)
  models.py            Model (finetune / generate / save) and load()
  accel.py             the shadow accelerator — optimization planning
  backends/
    base.py            Backend interface + Callbacks bridge
    mlx.py             MLXBackend  — Apple Silicon (Metal GPU)
    torch.py           TorchBackend — PyTorch (CUDA / CPU)
examples/
  quickstart.py        datasets → finetune → inference, end to end
  train_eval_split.py  held-out validation + overfitting signal
  infer_adapter.py     train → save → reload adapter in a fresh model → infer
  sample_dataset.jsonl
```

## Roadmap

- [x] SDK: datasets → finetune → inference on mlx / torch
- [x] Train/eval split with held-out validation loss
- [x] Shadow accelerator (gradient checkpointing, flash-attn, fused optim)
- [ ] Inference slice: streaming generation, adapter hot-load, batch generate
- [ ] Studio: web service + MongoDB job queue + remote-GPU workers over this SDK
