"""ShadowLM CLI — the SDK from your shell.

    shadowlm finetune data.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --method lora
    shadowlm runs                  # run history, status, final losses
    shadowlm plot <run-id>         # terminal loss charts
    shadowlm chat out/adapter/     # talk to what you trained
    shadowlm methods               # the registered training methods

Every hyperparameter flag is generated from `TrainConfig`, so the CLI surface
is always exactly the SDK surface — same methods, same accelerator, same run
records. Pure stdlib, like the core.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

from . import __version__, methods
from .training import TrainConfig

# Flags handled explicitly (or meaningless as CLI hyperparams).
_SKIP_FIELDS = {"method"}


def _flag_type(annotation: str):
    """Map a TrainConfig annotation string to an argparse type."""
    a = annotation.replace(" ", "")
    if "bool" in a:
        return bool
    if "int|float" in a:  # eval_steps: int count or 0-1 fraction
        return float
    if "int" in a:
        return int
    if "float" in a:
        return float
    return str


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("hyperparameters (from TrainConfig)")
    for f in dataclass_fields(TrainConfig):
        if f.name in _SKIP_FIELDS:
            continue
        flag = "--" + f.name.replace("_", "-")
        kind = _flag_type(str(f.type))
        if kind is bool:
            group.add_argument(flag, dest=f.name, default=None,
                               action=argparse.BooleanOptionalAction,
                               help=f"(default: {f.default})")
        else:
            group.add_argument(flag, dest=f.name, default=None, type=kind,
                               metavar=f.name.upper(),
                               help=f"(default: {f.default})")


def _collected_hyperparams(args: argparse.Namespace) -> dict:
    """Only the flags the user actually set — TrainConfig defaults rule."""
    out = {}
    for f in dataclass_fields(TrainConfig):
        if f.name in _SKIP_FIELDS:
            continue
        v = getattr(args, f.name, None)
        if v is None:
            continue
        if f.name in ("target_modules", "report_to") and "," in str(v):
            v = tuple(s.strip() for s in str(v).split(",") if s.strip())
        out[f.name] = v
    return out


# ---- base-model resolution for adapter dirs ---------------------------------
def _base_model_of(adapter_dir: Path) -> str | None:
    """Find the base model an adapter dir was trained from."""
    candidates = ("run.json", "shadowlm_meta.json", "bitfit_config.json",
                  "bottleneck_config.json", "more_config.json")
    for name in candidates:
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
    # peft adapters record the base in adapter_config.json
    p = adapter_dir / "adapter_config.json"
    if p.exists():
        try:
            base = json.loads(p.read_text()).get("base_model_name_or_path")
            if isinstance(base, str):
                return base
        except (json.JSONDecodeError, OSError):
            pass
    return None


# ---- commands ----------------------------------------------------------------
def _cmd_finetune(args: argparse.Namespace) -> int:
    from .data import Dataset  # noqa: PLC0415 — heavy imports stay lazy
    from .models import load  # noqa: PLC0415

    ds = Dataset.load(args.data)
    model = load(args.model, backend=args.backend, accelerator=args.accelerator,
                 device=args.device, load_in_4bit=args.load_in_4bit)
    run = model.finetune(
        ds,
        method=args.method,
        eval_dataset=args.eval,
        output_dir=args.output_dir,
        **_collected_hyperparams(args),
    )
    if args.save:
        out = model.save(args.save)
        print(f"saved → {out}")
    return 0 if run.status == "succeeded" else 1


def _cmd_runs(args: argparse.Namespace) -> int:
    from . import runs as run_history  # noqa: PLC0415

    if args.delete:
        run_history.delete(args.delete)
        print(f"deleted {args.delete}")
        return 0
    all_runs = sorted(run_history.list(), key=lambda r: r.started_at or 0,
                      reverse=True)[: args.limit]
    if not all_runs:
        print("no runs yet — try: shadowlm finetune data.jsonl --model <name>")
        return 0
    rows = [("RUN", "STATUS", "METHOD", "STEPS", "LOSS", "EVAL", "TOOK")]
    for r in all_runs:
        took = (f"{r.ended_at - r.started_at:.0f}s"
                if r.started_at and r.ended_at else "—")
        rows.append((
            r.id or "?", r.status, r.config.method,
            str(r.metrics[-1].step) if r.metrics else "—",
            f"{r.loss:.4f}" if r.loss is not None else "—",
            f"{r.eval_loss:.4f}" if r.eval_loss is not None else "—",
            took,
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
        if i == 0:
            print("  ".join("─" * w for w in widths))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from . import runs as run_history  # noqa: PLC0415

    run = run_history.load(args.run)
    print(run.plot(args.metric, smooth=args.smooth, window=args.window,
                   log=args.log, clip=args.clip))
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from .models import load  # noqa: PLC0415

    target = Path(args.target)
    if target.is_dir():  # an adapter dir: resolve its base model
        base = args.model or _base_model_of(target)
        if base is None:
            print(f"error: can't tell which base model {args.target} belongs to — "
                  "pass --model <base-model-name>", file=sys.stderr)
            return 2
        model = load(base, backend=args.backend, adapter=str(target))
    else:  # a plain model name
        model = load(args.target, backend=args.backend,
                     load_in_4bit=args.load_in_4bit)

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    interactive = sys.stdin.isatty()
    if interactive:
        print("chat ready — empty line or Ctrl-D to exit\n")
    while True:
        try:
            prompt = input("you › " if interactive else "")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            if interactive:
                break
            continue
        messages.append({"role": "user", "content": prompt})
        reply = model.chat(messages, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature)
        messages.append(reply.to_message())
        print(f"slm♥ › {reply}" if interactive else str(reply))
        if not interactive:  # piped input: one prompt per line, keep reading
            continue
    return 0


def _cmd_methods(_args: argparse.Namespace) -> int:
    rows = [("METHOD", "DEFAULT LR", "TRAINER", "DESCRIPTION")]
    for name in methods.available():
        m = methods.get(name)
        rows.append((m.name, f"{m.default_learning_rate:g}", m.trainer,
                     m.description))
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    for i, row in enumerate(rows):
        lead = "  ".join(c.ljust(w) for c, w in zip(row[:3], widths))
        print(f"{lead}  {row[3]}")
        if i == 0:
            print("  ".join("─" * w for w in widths) + "  " + "─" * 40)
    return 0


# ---- entry point --------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shadowlm",
        description="ShadowLM Trainer — a fine-tuning SDK. "
                    "Any open model. Any harness. Any method.",
    )
    parser.add_argument("--version", action="version",
                        version=f"shadowlm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("finetune", help="train a model on a dataset")
    p.add_argument("data", help="dataset path (.jsonl/.json/.csv/.parquet)")
    p.add_argument("--model", required=True, help="base model name (HF hub id)")
    p.add_argument("--method", default="lora",
                   help=f"one of: {', '.join(methods.available())}")
    p.add_argument("--backend", default="auto", help="auto | mlx | torch")
    p.add_argument("--accelerator", default="auto", help="auto | shadow | none")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu (torch)")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--eval", default=None,
                   help='eval dataset path, or "auto" to hold out 10%%')
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save", default=None, metavar="DIR",
                   help="also export the adapter to DIR after training")
    _add_config_flags(p)
    p.set_defaults(fn=_cmd_finetune)

    p = sub.add_parser("runs", help="list recorded runs")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--delete", default=None, metavar="RUN_ID")
    p.set_defaults(fn=_cmd_runs)

    p = sub.add_parser("plot", help="terminal chart for a recorded run")
    p.add_argument("run", help="run id (see: shadowlm runs) or checkpoint path")
    p.add_argument("metric", nargs="?", default="loss",
                   choices=["loss", "eval_loss", "lr", "grad_norm"])
    p.add_argument("--smooth", type=float, default=0.6)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--log", action="store_true")
    p.add_argument("--clip", type=float, default=None)
    p.set_defaults(fn=_cmd_plot)

    p = sub.add_parser("chat", help="chat with a model or a trained adapter dir")
    p.add_argument("target", help="model name, or an adapter directory")
    p.add_argument("--model", default=None,
                   help="base model override for adapter dirs")
    p.add_argument("--backend", default="auto")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--system", default=None, help="system prompt")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.set_defaults(fn=_cmd_chat)

    p = sub.add_parser("methods", help="list registered training methods")
    p.set_defaults(fn=_cmd_methods)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
