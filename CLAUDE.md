# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ShadowLM Trainer is a fine-tuning SDK: load any open model, train it with any of
15 methods, on any hardware, then own the weights. The headline use case is
"shadowing" — moving one task off a rented frontier model onto a small model you
own, by capturing real agent traffic (`slm.capture()`), judging episodes, and
training on them — without modifying the agent (the model API is the only
boundary). The repo is the **engine**; the orchestration tier is ShadowLM Studio.

The whole product reads like the task in `shadowlm/models.py`:
`slm.load(...)` → `model.finetune(ds, method=...)` → `model.generate(...)` →
`model.save(...)`. Keep that surface tiny — the machinery lives in the backends.

## Commands

```bash
make install           # editable install with CLI + mlx backend (Apple Silicon dev loop)
make install-torch     # editable install for CUDA / CPU boxes
make frontend          # npm install + build the React studio into shadowlm/_static
make serve             # studio UI + API on one port (PORT=8329)
make dev               # serve with Vite hot-reload UI alongside the backend
make demo              # end-to-end smoke: a tiny CLI finetune (mlx, 0.5B, ~seconds)
make check             # compileall the package + `tsc -b` the frontend
make build             # build the frontend, then the wheel+sdist, then twine check
make release           # bump patch (or BUMP=minor/major, V=x.y.z), build, tag, push

pytest                                       # the CPU test suite (tests/, excludes gpu/)
pytest tests/test_more_plus_router.py        # a single test file
pytest tests/test_more_plus_router.py::test_name -x   # a single test, stop on first fail
make gpu-test          # the CUDA verification suite — run on a GPU box only
```

`tests/gpu/` (CUDA, real GPU) is **not** part of the default `pytest` run — it's
invoked explicitly via `make gpu-test` or `python tests/gpu/test_cuda.py`.

Releases publish to PyPI via `.github/workflows/publish.yml` on a `v*` tag. The
CI gate requires the version to match in **three** places: the git tag,
`pyproject.toml`, and `shadowlm/__init__.py`. `make bump`/`make release` keep
the latter two in sync — never edit only one.

## Architecture

Two orthogonal registries — **backends** (where training runs) × **methods**
(what training does) — meet in the SDK surface. Adding a backend or a method
touches one file and no others.

### The two axes

- **`shadowlm/backends/`** — a `Backend` (see `backends/base.py`) holds a loaded
  model and knows how to `load` / `finetune` / `generate` / `chat` / `save`.
  Implementations: `mlx.py` (Apple-Silicon dev loop), `torch.py` (the production
  CUDA/CPU path, on HF `Trainer` + `accelerate` + `trl` + `peft`), `remote.py`
  (speaks the JSON protocol to a server), `verl.py` (multi-GPU GRPO). Selection
  lives in `backends/__init__.py::select_backend` — `auto` = CUDA→torch,
  else Apple→mlx, else torch-on-CPU. **Everything user-facing is
  backend-agnostic**; mlx and torch must stay swappable without changing the SDK.

- **`shadowlm/methods/`** — each method is a declarative `TrainingMethod` spec
  (`methods/base.py`): an adapter kind (`ADAPTER_LORA`, `ADAPTER_MORE`, …), a
  base-model requirement (`quantized_base`: True=needs 4-bit, False=needs
  unquantized, None=either), a `trainer` ("sft"/"dpo"/"grpo"), and a default LR.
  **Backends dispatch on the spec's fields, never on the method name** — that
  invariant is what makes `method="lora"` → `"qlora"` a one-word change.
  Registering a method is a new module with one `register(...)` call, imported in
  `methods/__init__.py`; users can `methods.register(...)` at runtime too.

### The SDK surface (`models.py`, `training.py`, `data.py`)

- `models.py` — `load()` returns a `Model`; `Model.finetune/generate/chat/save`.
  This is the whole library in one object; resist growing it. Tool-call parsing
  for `chat()` (small models emit slightly mangled tool JSON) lives here too.
- `training.py` — `TrainConfig` (every hyperparameter, with which backend honors
  it noted inline), `Metric`, and `TrainingRun` (the live+final handle:
  metrics history, sparkline/plot, checkpoints, persistence). `TrainConfig` is
  the single source of truth — the CLI's `--set`/`--config` validates against the
  dataclass so it can't drift from the SDK.
- `data.py` — `Dataset` is rows + a detected format (chat / sharegpt /
  preference / instruction / text / raw). Backends turn a formatted dataset into
  training text. Local loading is pure-stdlib; `from_hf` lazy-imports `datasets`.

### The shadowing / agent-tuning loop

- `capture.py` — `slm.capture(model)` is a drop-in OpenAI-compatible proxy that
  records an unmodified agent's traffic, reconstructing message-level
  trajectories (calls that extend a prior call's message prefix merge into one
  episode; use an `x-session-id` header to disambiguate interleaved conversations).
- `rl.py` — `Trajectory` / `TrajectoryGroup` / `judge_group` (LLM-judge scoring),
  fed into `method="grpo"`.
- `apo.py` — `optimize_prompt()`: optimize the prompt instead of weights, same
  capture/judge front end, no GPU.

### Signature methods (MoRE)

`more.py` / `more_plus.py` implement "mixture of retrieval experts" — facts fused
into attention for near-zero-hallucination recall (faiss + sentence-transformers).
`more_plus` trains one final-FFN LoRA expert per knowledge unit with BM25+semantic
routing; its run progress is one step per unit (see `resolve_total_steps`).

### Server, remote protocol, and the studio

- `serve.py` — `python -m shadowlm.serve` / `shadowlm serve`. Pure-stdlib
  (`http.server` + threads) reference server: trains on **this machine's real
  backend** (no mock), streams metrics, ships adapters as tar.gz, serves the
  built React UI from `_static`. One job at a time — honest reference tier.
- `remote.py` — the typed client for that JSON protocol (`/v1/finetunes`, …).
  Same protocol backs `backend="remote"` and ShadowLM Studio.
- `frontend/` — React 19 + Vite + Tailwind v4 studio. `npm run build` outputs to
  `../shadowlm/_static` (the wheel ships the compiled UI; end users never need
  node). `frontend/src/api.ts` is the typed mirror of the remote protocol. The
  pages (Datasets → Models → Train → Runs → Playground) are the capture→train→own
  loop as a UI. Auth: studio routes are gated by username/password.

### The shadow accelerator (`accel.py`)

`accelerator="shadow"` turns on optimizations that are *safe for the current
model+hardware* — gradient checkpointing, flash-attn-2, fused 8-bit optimizer,
4-bit QLoRA, optional Liger kernels. It **logs exactly what it enabled and
no-ops when something is unavailable** — there are no silent magic multipliers
and no custom GPU kernels. Keep that property when touching it.

## Conventions

- **Batteries included**: `pip install shadowlm` pulls the full torch/HF training
  stack + retrieval + CLI. mlx is auto-added on arm64 macOS via a wheel marker.
  Only `[kernels]` (Liger) and `[verl]` stay opt-in. The `[torch]`/`[mlx]`/`[cli]`
  etc. extras are back-compat aliases that resolve to nothing — don't add deps to
  them.
- Method/base mismatches raise **actionable** errors (e.g. `qlora` on a 16-bit
  base tells you to load a 4-bit one). Follow that pattern.
- `TrainConfig` fields a backend can't honor are ignored **with a log line**,
  never silently dropped.
- Default artifacts land in `~/.shadowlm/` (runs, server work dir, the install
  venv).
