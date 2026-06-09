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

## Training methods

Each technique lives in its own module under `shadowlm/methods/` as a declarative
spec — backends read the spec (adapter kind, base requirements, data rendering),
never the method name.

| method | what it does | base model | default LR |
|--------|--------------|------------|------------|
| `lora`  | LoRA adapters | 16-bit | 2e-4 |
| `qlora` | LoRA adapters, lowest memory | **4-bit required** | 2e-4 |
| `dora`  | weight-decomposed LoRA, often better at low rank | either | 2e-4 |
| `full`  | update every transformer weight | **unquantized required** | 2e-5 |
| `cpt`   | continued pretraining on raw domain text (no chat template) | either | 5e-5 |

Base requirements are enforced with clear errors (e.g. `qlora` on a 16-bit model
tells you to load a 4-bit one). Adding a technique is one file:

```python
# shadowlm/methods/my_method.py  (or methods.register(...) at runtime)
from .base import TrainingMethod, register

register(TrainingMethod(
    name="my-method",
    description="LoRA variant with my defaults",
    default_learning_rate=1e-4,
))
```

## Training parameters

`finetune(**hyperparams)` accepts the full `TrainConfig` surface:

- **adapters** — `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`
  (`"all"` / `"attention"` / `"mlp"` presets, or explicit names), `use_rslora`*
- **optimization** — `learning_rate` (default per method), `per_device_train_batch_size`,
  `gradient_accumulation_steps`, `warmup_steps` / `warmup_ratio`, `max_steps` /
  `num_train_epochs`, `weight_decay`, `max_grad_norm`*, `lr_scheduler_type`
  (linear / cosine / constant — real schedules on both backends), `optim`*, `seed`
- **data** — `max_seq_length`, `packing`*, `train_on_completions` (mask the prompt,
  learn only on responses)
- **logging / checkpoints** — `logging_steps`, `eval_steps` (int, or a 0–1 fraction
  of total steps), `save_steps` (mid-run checkpoints), `resume_from_checkpoint`,
  `report_to`*

\* torch-backend only; the mlx backend logs a note instead of silently ignoring.

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
| `slm.Dataset.load(path)` | any supported file by extension (.jsonl/.json/.csv/.parquet) |
| `slm.Dataset.from_jsonl / from_csv / from_json / from_parquet / from_list` | format auto-detected (chat / instruction / text) |
| `slm.Dataset.from_hf(repo, subset=, split=, token=)` | HuggingFace Hub datasets |
| `ds.as_chat()` / `ds.as_text()` | force chat or raw-text format |
| `ds.split(test_size=0.1, seed=0)` | held-out train/eval split → `(train, eval)` |
| `ds[0:100]`, `ds.head()`, `ds.columns`, `len(ds)` | row slicing & inspection |
| `slm.load(name, backend=, accelerator=, device=, load_in_4bit=, adapter=)` | load a model (or attach a trained adapter) |
| `model.finetune(ds, method="lora"\|"qlora"\|"dora"\|"full"\|"cpt", eval_dataset=ds\|"auto", on_step=, on_eval=, **hyperparams)` | train; returns a `TrainingRun` (`eval_dataset="auto"` holds out 10%) |
| `model.generate(prompt, ...)` / `model.chat(messages)` | inference |
| `model.save(path, fmt="adapter"\|"merged")` | export |
| `run.loss`, `run.eval_loss`, `run.step`, `run.progress`, `run.sparkline()`, `run.checkpoint` | live + final run state |
| `slm.runs.list() / latest() / load(id) / delete(id)` | run history — every finetune persists a `run.json` (status, config, metrics) |

Every run records itself — `succeeded`, `failed` (with the error), or `stopped`
(Ctrl-C) — so history survives the process. Resume any recorded run with
`model.finetune(ds, resume_from_checkpoint=run.checkpoint)`; pass `save_steps=N`
to keep mid-run checkpoints so even interrupted runs are resumable.

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
  runs.py              run history — list / load / resume / delete past runs
  accel.py             the shadow accelerator — optimization planning
  methods/             training techniques — one module per method
    base.py            TrainingMethod spec + registry
    lora.py qlora.py dora.py full.py cpt.py
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
