"""ShadowLM CLI — the SDK from your shell, built on Typer + Rich.

    shadowlm finetune data.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --method lora
    shadowlm runs                  # run history, status, losses
    shadowlm plot <run-id>         # terminal loss charts
    shadowlm chat out/adapter/     # talk to what you trained
    shadowlm methods               # the registered training methods

Headline hyperparameters are typed flags; every other `TrainConfig` field is
reachable through `--set field=value`, validated against the dataclass — so the
CLI can't drift from the SDK. The library stays pure-stdlib; this module is the
only thing that needs the `[cli]` extra.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from . import __version__, methods
from .training import TrainConfig

_THEME = Theme({
    "slm": "bold #E5484D",
    "ok": "bold green",
    "warn": "bold yellow",
    "bad": "bold red",
    "muted": "dim",
})
console = Console(theme=_THEME)
err_console = Console(stderr=True, theme=_THEME)

app = typer.Typer(
    name="shadowlm",
    help="ShadowLM Trainer — a fine-tuning SDK. "
         "Any open model. Any harness. Any method.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

_STATUS_STYLE = {"succeeded": "ok", "failed": "bad", "stopped": "warn",
                 "running": "cyan", "pending": "muted"}


# ---- --set field=value coercion against TrainConfig --------------------------
def _config_field_types() -> dict[str, str]:
    return {f.name: str(f.type) for f in dataclass_fields(TrainConfig)}


def _coerce(field: str, raw: str, annotation: str):
    a = annotation.replace(" ", "")
    if "bool" in a:
        low = raw.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise typer.BadParameter(f"--set {field}: expected a boolean, got {raw!r}")
    try:
        if "int|float" in a:
            return float(raw)
        if "int" in a:
            return int(raw)
        if "float" in a:
            return float(raw)
    except ValueError as exc:
        raise typer.BadParameter(f"--set {field}: {exc}") from None
    if "," in raw and field in ("target_modules", "report_to"):
        return tuple(s.strip() for s in raw.split(",") if s.strip())
    return raw


def _apply_set(hyper: dict, sets: list[str] | None) -> None:
    if not sets:
        return
    types = _config_field_types()
    for item in sets:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects field=value, got {item!r}")
        field, raw = item.split("=", 1)
        field = field.strip()
        if field not in types:
            raise typer.BadParameter(
                f"--set: unknown TrainConfig field {field!r}. "
                f"See `shadowlm finetune --help`."
            )
        hyper[field] = _coerce(field, raw.strip(), types[field])


# ---- base-model resolution for adapter dirs ---------------------------------
def _base_model_of(adapter_dir: Path) -> str | None:
    import json  # noqa: PLC0415

    for name in ("run.json", "shadowlm_meta.json", "bitfit_config.json",
                 "bottleneck_config.json", "more_config.json",
                 "adapter_config.json"):
        p = adapter_dir / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for key in ("base_model", "base_model_name_or_path", "model"):
            if isinstance(data.get(key), str):
                return data[key]
    return None


# ---- commands ---------------------------------------------------------------
@app.command(rich_help_panel="Training")
def finetune(
    data: Annotated[str, typer.Argument(help="dataset (.jsonl/.json/.csv/.parquet)")],
    model: Annotated[str, typer.Option("--model", "-m", help="base model — HF hub id")],
    method: Annotated[str, typer.Option(help="training method (see `shadowlm methods`)")] = "lora",
    max_steps: Annotated[Optional[int], typer.Option(help="total training steps")] = None,
    epochs: Annotated[Optional[float], typer.Option(help="epochs (overrides --max-steps)")] = None,
    lr: Annotated[Optional[float], typer.Option(help="learning rate (default: per method)")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", "-b")] = None,
    grad_accum: Annotated[Optional[int], typer.Option("--grad-accum")] = None,
    lora_r: Annotated[Optional[int], typer.Option("--lora-r")] = None,
    lora_alpha: Annotated[Optional[int], typer.Option("--lora-alpha")] = None,
    max_seq_length: Annotated[Optional[int], typer.Option("--max-seq-length")] = None,
    seed: Annotated[Optional[int], typer.Option()] = None,
    eval: Annotated[Optional[str], typer.Option(help='eval dataset, or "auto" to hold out 10%')] = None,
    save: Annotated[Optional[str], typer.Option(help="also export the adapter to DIR")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir")] = None,
    backend: Annotated[str, typer.Option(help="auto | mlx | torch")] = "auto",
    accelerator: Annotated[str, typer.Option(help="auto | shadow | none")] = "auto",
    device: Annotated[str, typer.Option(help="auto | cuda | cpu (torch)")] = "auto",
    load_in_4bit: Annotated[bool, typer.Option("--load-in-4bit")] = False,
    set_: Annotated[Optional[list[str]], typer.Option(
        "--set", help="any other TrainConfig field, e.g. --set weight_decay=0.05")] = None,
):
    """Train a model on a dataset."""
    from .data import Dataset  # noqa: PLC0415
    from .models import load  # noqa: PLC0415

    hyper: dict = {}
    explicit = {
        "max_steps": max_steps, "num_train_epochs": epochs, "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum, "lora_r": lora_r,
        "lora_alpha": lora_alpha, "max_seq_length": max_seq_length, "seed": seed,
    }
    hyper.update({k: v for k, v in explicit.items() if v is not None})
    _apply_set(hyper, set_)

    ds = Dataset.load(data)
    m = load(model, backend=backend, accelerator=accelerator, device=device,
             load_in_4bit=load_in_4bit)
    run = m.finetune(ds, method=method, eval_dataset=eval,
                     output_dir=output_dir, **hyper)
    if save:
        out = m.save(save)
        console.print(f"saved → [slm]{out}[/slm]")
    raise typer.Exit(0 if run.status == "succeeded" else 1)


@app.command(rich_help_panel="Models")
def chat(
    target: Annotated[str, typer.Argument(help="model name, or an adapter directory")],
    model: Annotated[Optional[str], typer.Option("--model", "-m",
                     help="base model override for adapter dirs")] = None,
    backend: Annotated[str, typer.Option(help="auto | mlx | torch")] = "auto",
    load_in_4bit: Annotated[bool, typer.Option("--load-in-4bit")] = False,
    system: Annotated[Optional[str], typer.Option(help="system prompt")] = None,
    temperature: Annotated[float, typer.Option()] = 0.7,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 512,
):
    """Chat with a model, or a trained adapter directory."""
    import sys  # noqa: PLC0415

    from .models import load  # noqa: PLC0415

    path = Path(target)
    if path.is_dir():
        base = model or _base_model_of(path)
        if base is None:
            err_console.print(
                f"[bad]error[/bad]: can't tell which base model {target!r} belongs "
                "to — pass [slm]--model <base>[/slm]")
            raise typer.Exit(2)
        m = load(base, backend=backend, adapter=str(path))
    else:
        m = load(target, backend=backend, load_in_4bit=load_in_4bit)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    interactive = sys.stdin.isatty()
    if interactive:
        console.print("[muted]chat ready — empty line or Ctrl-D to exit[/muted]\n")
    while True:
        try:
            prompt = console.input("[slm]you ›[/slm] ") if interactive else input()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            if interactive:
                break
            continue
        messages.append({"role": "user", "content": prompt})
        reply = m.chat(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        messages.append(reply.to_message())
        if interactive:
            console.print(f"[slm]slm♥ ›[/slm] {reply}")
        else:
            print(str(reply))


@app.command(rich_help_panel="Runs")
def runs(
    limit: Annotated[int, typer.Option(help="how many recent runs to show")] = 20,
    delete: Annotated[Optional[str], typer.Option(help="delete a run by id")] = None,
):
    """List recorded runs (or delete one)."""
    from . import runs as run_history  # noqa: PLC0415

    if delete:
        run_history.delete(delete)
        console.print(f"deleted [slm]{delete}[/slm]")
        return
    recorded = sorted(run_history.list(), key=lambda r: r.started_at or 0,
                      reverse=True)[:limit]
    if not recorded:
        console.print("[muted]no runs yet — try:[/muted] "
                      "shadowlm finetune data.jsonl --model <name>")
        return
    table = Table(title="ShadowLM runs", title_style="slm", header_style="slm",
                  border_style="muted")
    for col in ("run", "status", "method", "steps", "loss", "eval", "took"):
        table.add_column(col, no_wrap=(col != "run"))
    for r in recorded:
        took = (f"{r.ended_at - r.started_at:.0f}s"
                if r.started_at and r.ended_at else "—")
        table.add_row(
            r.id or "?",
            f"[{_STATUS_STYLE.get(r.status, 'muted')}]{r.status}[/]",
            r.config.method,
            str(r.metrics[-1].step) if r.metrics else "—",
            f"{r.loss:.4f}" if r.loss is not None else "—",
            f"{r.eval_loss:.4f}" if r.eval_loss is not None else "—",
            took,
        )
    console.print(table)


@app.command(rich_help_panel="Runs")
def plot(
    run: Annotated[str, typer.Argument(help="run id (see `shadowlm runs`) or path")],
    metric: Annotated[str, typer.Argument(help="loss | eval_loss | lr | grad_norm")] = "loss",
    smooth: Annotated[float, typer.Option()] = 0.6,
    window: Annotated[Optional[int], typer.Option(help="last N points only")] = None,
    log: Annotated[bool, typer.Option(help="log scale")] = False,
    clip: Annotated[Optional[float], typer.Option(help="percentile cap, e.g. 0.95")] = None,
):
    """Terminal chart for a recorded run."""
    from . import runs as run_history  # noqa: PLC0415

    rec = run_history.load(run)
    console.print(rec.plot(metric, smooth=smooth, window=window, log=log, clip=clip))


def _methods_cmd():
    """List the registered training methods."""
    table = Table(title="training methods", title_style="slm",
                  header_style="slm", border_style="muted")
    table.add_column("method", style="slm", no_wrap=True)
    table.add_column("default lr", no_wrap=True)
    table.add_column("trainer", no_wrap=True)
    table.add_column("description")
    for name in methods.available():
        m = methods.get(name)
        table.add_row(m.name, f"{m.default_learning_rate:g}", m.trainer, m.description)
    console.print(table)


# Registered explicitly so the command is `methods`, not the function name.
app.command(name="methods", rich_help_panel="Info")(_methods_cmd)


@app.command(rich_help_panel="Info")
def version():
    """Print the installed version."""
    console.print(f"[slm]shadowlm[/slm] {__version__}  ·  slm♥")
