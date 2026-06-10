"""The user-facing surface: `load()` a model, then `.finetune()` / `.generate()`.

This is the whole SDK in one object. It is deliberately tiny and reads like the
task — the interesting machinery lives in the backends.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import methods
from . import runs as run_history
from .backends import Callbacks, select_backend
from .data import Dataset
from .training import Metric, StepCallback, TrainConfig, TrainingRun, resolve_total_steps

# shadowLM's own output goes to the real terminal even while a backend redirects
# sys.stdout to swallow its internal logging. Captured once at import.
_CONSOLE = sys.__stdout__ or sys.stdout

# Where checkpoints land by default: ~/.shadowlm/runs/<base>-<timestamp>/
RUNS_ROOT = run_history.RUNS_ROOT


def _default_output_dir(base_model: str) -> str:
    slug = base_model.replace("/", "_")
    return str(RUNS_ROOT / f"{slug}-{int(time.time())}")


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _first_json_object(text: str) -> dict | None:
    """Best-effort: the first valid JSON object in `text` (small models emit
    slightly mangled tool-call JSON; we salvage what parses)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                continue
    return None


@dataclass
class Reply:
    """An assistant turn: text content plus any parsed tool calls.

    Prints as its text. Append `reply.to_message()` (then a {"role": "tool", ...}
    result) to the message list to continue a tool-use loop.
    """

    content: str
    tool_calls: list[dict] = field(default_factory=list)
    role: str = "assistant"
    raw: str = ""

    def __str__(self) -> str:
        return self.content

    def to_message(self) -> dict:
        msg: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"type": "function", "function": c} for c in self.tool_calls
            ]
        return msg


def _parse_reply(raw: str) -> Reply:
    calls = []
    for block in _TOOL_CALL_RE.findall(raw):
        obj = _first_json_object(block)
        if obj and "name" in obj:
            calls.append(obj)
    content = _TOOL_CALL_RE.sub("", raw).strip()
    if not calls and "<tool_call>" in raw:
        # the generation budget can cut output before </tool_call> — salvage
        # the unclosed block rather than dropping the call
        tail = raw.split("<tool_call>", 1)[1]
        obj = _first_json_object(tail)
        if obj and "name" in obj:
            calls.append(obj)
            content = raw.split("<tool_call>", 1)[0].strip()
    return Reply(content=content, tool_calls=calls, raw=raw)


class Model:
    """A loaded model you can finetune and generate from."""

    def __init__(self, name: str, backend, *, load_in_4bit: bool, max_seq_length: int,
                 adapter: str | None) -> None:
        self.name = name
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.adapter = adapter
        self._backend = backend

    # ---- finetune ---------------------------------------------------------
    def finetune(
        self,
        dataset: Dataset | list[dict],
        *,
        method: str = "lora",
        eval_dataset: Dataset | list[dict] | str | None = None,
        eval_steps: int | None = None,
        reward_fns: list | None = None,
        on_step: StepCallback | None = None,
        on_eval: StepCallback | None = None,
        on_log=None,
        output_dir: str | None = None,
        verbose: bool = True,
        **hyperparams,
    ) -> TrainingRun:
        """Finetune on `dataset`, returning a `TrainingRun`.

        method: any registered training method — built-ins are lora, qlora,
        dora, full, cpt, dpo, grpo, more, bitfit, prompt, ptuning, and adapter.
        See `shadowlm.methods` to list or register methods. The default
        learning rate comes from the method's spec.

        Pass `eval_dataset` (and optionally `eval_steps`) to evaluate on a held-out
        set during training; eval loss lands in `run.eval_metrics` / `run.eval_loss`.
        `eval_dataset="auto"` splits a small portion (10%) off the training data.
        Extra keyword args (`max_steps`, `learning_rate`, `lora_r`, ...) override the
        `TrainConfig` defaults. Pass `on_step`/`on_eval` to observe metrics live.
        """
        from .rl import TrajectoryGroup, weighted_rows  # noqa: PLC0415

        if isinstance(dataset, TrajectoryGroup) or (
                isinstance(dataset, list) and dataset
                and isinstance(dataset[0], TrajectoryGroup)):
            if method != "grpo":
                raise ValueError("trajectory groups train with method='grpo'")
            dataset = Dataset.from_list(weighted_rows(dataset), name="trajectories")
        if not isinstance(dataset, Dataset):
            dataset = Dataset.from_list(list(dataset))
        if isinstance(eval_dataset, str):
            if eval_dataset != "auto":
                raise ValueError(f"eval_dataset={eval_dataset!r} — expected a Dataset, rows, or 'auto'")
        elif eval_dataset is not None and not isinstance(eval_dataset, Dataset):
            eval_dataset = Dataset.from_list(list(eval_dataset))
        if verbose:
            from .ascii import print_ascii_art  # noqa: PLC0415
            print_ascii_art()

        spec = methods.get(method)  # raises ValueError for unknown methods
        config = TrainConfig(method=method, **hyperparams)
        if config.learning_rate is None:
            config.learning_rate = spec.default_learning_rate
        if eval_steps is not None:
            config.eval_steps = eval_steps
        if eval_dataset == "auto":
            dataset, eval_dataset = dataset.split(test_size=0.1, seed=config.seed)
        output_dir = output_dir or _default_output_dir(self.name)
        run = TrainingRun(
            config=config,
            base_model=self.name,
            status="running",
            total_steps=resolve_total_steps(config, len(dataset)),
            started_at=time.time(),
        )

        def _record(metric: Metric) -> None:
            run.metrics.append(metric)
            if verbose:
                _print_progress(run, metric)
            if on_step:
                on_step(metric)

        def _record_eval(metric: Metric) -> None:
            run.eval_metrics.append(metric)
            if verbose:
                print(f"  ↳ eval @ step {metric.step}: loss {metric.loss:.4f}",
                      file=_CONSOLE, flush=True)
            if on_eval:
                on_eval(metric)

        def _log(message: str) -> None:
            if verbose:
                print(message, file=_CONSOLE, flush=True)
            if on_log:
                on_log(message)

        callbacks = Callbacks(on_step=_record, on_log=_log, on_eval=_record_eval)
        try:
            result = self._backend.finetune(dataset, config, callbacks, output_dir,
                                            eval_dataset=eval_dataset,
                                            reward_fns=reward_fns)
            run.checkpoint = result.checkpoint
            run.status = "succeeded"
        except KeyboardInterrupt:  # Ctrl-C — record what happened, keep partial output
            run.status = "stopped"
            run.checkpoint = output_dir
            run.ended_at = time.time()
            run_history.save(run, output_dir)
            raise
        except Exception as e:  # surface the failure on the run, then re-raise
            run.status = "failed"
            run.error = f"{type(e).__name__}: {e}"
            run.ended_at = time.time()
            run_history.save(run, output_dir)
            raise
        run.ended_at = time.time()
        run_history.save(run, output_dir)
        self.adapter = run.checkpoint
        if verbose:
            _print_summary(run)
        return run

    # ---- inference --------------------------------------------------------
    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.95, **kwargs) -> str:
        return self._backend.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, **kwargs,
        )

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             max_new_tokens: int = 256, temperature: float = 0.7,
             top_p: float = 0.95, **kwargs) -> Reply:
        """Multi-turn chat through the model's chat template.

        Pass `tools` (OpenAI-style function schemas) to enable tool calling; any
        emitted calls are parsed into `reply.tool_calls`. Continue the loop by
        appending `reply.to_message()` and a {"role": "tool", "content": result}
        message, then calling chat again.
        """
        raw = self._backend.chat(messages, tools=tools, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_p=top_p, **kwargs)
        return _parse_reply(raw)

    # ---- export -----------------------------------------------------------
    def save(self, path: str, *, fmt: str = "adapter") -> str:
        """Save the model. fmt: 'adapter' (LoRA only) | 'merged'."""
        return self._backend.save(path, fmt=fmt)

    def __repr__(self) -> str:
        return (
            f"Model({self.name!r}, backend={self._backend.name!r}, "
            f"4bit={self.load_in_4bit}, adapter={self.adapter!r})"
        )


def load(
    name: str,
    *,
    backend: str = "auto",
    accelerator: str = "auto",
    device: str = "auto",
    load_in_4bit: bool = False,
    max_seq_length: int = 2048,
    adapter: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> Model:
    """Load a model.

    backend: "auto" (CUDA→torch, Apple→mlx, else torch-CPU) | "mlx" | "torch".
    accelerator: the shadow optimization layer — "auto" | "shadow" | "none".
    device: "auto" | "cuda" | "cpu" (torch backend; mlx is always the Metal GPU).
    adapter: path to a previously-trained LoRA checkpoint to attach.
    """
    be = select_backend(backend, accelerator=accelerator, device=device)
    t0 = time.time()
    if verbose:
        suffix = f" + adapter {adapter}" if adapter else ""
        print(f"[shadowlm] preparing {name}{suffix} (downloads on first use)...",
              file=_CONSOLE, flush=True)
    be.load(name, load_in_4bit=load_in_4bit, max_seq_length=max_seq_length,
            adapter=adapter, **kwargs)
    if verbose:
        print(f"[shadowlm] ready · backend={be.name} · {time.time() - t0:.1f}s",
              file=_CONSOLE, flush=True)
    return Model(name, be, load_in_4bit=load_in_4bit, max_seq_length=max_seq_length,
                 adapter=adapter)


def _print_summary(run: TrainingRun) -> None:
    """The end-of-training signature: sparkline loss curve + one result line."""
    out = []
    if run.metrics:
        out.append(f"  loss  {run.sparkline()}  {run.metrics[0].loss:.4f} → {run.loss:.4f}")
    if run.eval_metrics:
        evals = [m.loss for m in run.eval_metrics]
        bars = "▁▂▃▄▅▆▇█"
        lo, hi = min(evals), max(evals)
        span = (hi - lo) or 1.0
        curve = "".join(bars[min(7, int((v - lo) / span * 7))] for v in evals)
        out.append(f"  eval  {curve}  {evals[0]:.4f} → {evals[-1]:.4f}")
    dur = f"{run.duration_s:.1f}s" if run.duration_s else "?"
    out.append(f"  ♥ {run.status} · {run.step} steps · {dur} · {run.checkpoint}")
    print("\n" + "\n".join(out), file=_CONSOLE, flush=True)


def _fmt_eta(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _print_progress(run: TrainingRun, metric: Metric) -> None:
    total = run.total_steps or "?"
    bar_w = 24
    filled = int(bar_w * run.progress)
    bar = "█" * filled + "·" * (bar_w - filled)
    rate, eta = run.steps_per_s, run.eta_s
    tail = ""
    if rate:
        tail += f"  {rate:.1f} st/s"
    if metric.tokens_per_s:
        tail += f"  {metric.tokens_per_s:,.0f} tok/s"
    if eta is not None:
        tail += f"  eta {_fmt_eta(eta)}"
    end = "\n" if run.status != "running" or metric.step == run.total_steps else "\r"
    print(
        f"  [{bar}] step {metric.step:>4}/{total}  loss {metric.loss:.4f}  "
        f"lr {metric.lr:.2e}{tail}",
        end=end,
        file=_CONSOLE,
        flush=True,
    )
