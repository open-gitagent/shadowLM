"""Discover and resolve mid-training checkpoints from a run's output dir.

Both backends checkpoint when you pass ``save_steps=N`` — but they lay the bytes
down differently. The torch path writes HuggingFace ``checkpoint-<step>/`` dirs
(already loadable adapters); the mlx path writes bare
``<step>_adapters.safetensors`` files alongside the run's ``adapter_config.json``.

This module reads either layout and hands back ready-to-load adapter
*directories*, so everything above it — ``slm.load(..., adapter=ckpt.path)``, the
server, the studio — can test a specific version without knowing which backend
produced it. Pure stdlib; no torch/mlx import.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

# torch (HF Trainer): checkpoint-200/ ; mlx (mlx-lm): 0000200_adapters.safetensors
_TORCH = re.compile(r"^checkpoint-(\d+)$")
_MLX = re.compile(r"^(\d+)_adapters\.safetensors$")

# files the loaders sniff for; carried along when we materialize an mlx step dir
_SIDECARS = (
    "adapter_config.json", "bitfit_config.json", "bottleneck_config.json",
    "more_config.json", "memory_store.npz",  # MoRE: config + frozen fact index
)


@dataclass(frozen=True)
class Checkpoint:
    """One saved version of a run, loadable via ``adapter=checkpoint.path``."""

    step: int      # training step it was saved at
    path: str      # a directory loadable as an adapter
    final: bool    # the end-of-run weights (what a plain run id resolves to)

    @property
    def label(self) -> str:
        return "final" if self.final else f"step {self.step}"

    def to_dict(self) -> dict:
        return {"step": self.step, "path": self.path,
                "final": self.final, "label": self.label}


def list_checkpoints(run_dir: str | Path) -> list[Checkpoint]:
    """Every checkpoint under a run's output dir — oldest step first, the final
    end-of-run weights last.

    Empty when the run never checkpointed (no ``save_steps``) and hasn't yet
    written final weights. mlx step files are paired with the run's config into a
    small loadable dir under ``<run>/.checkpoints/`` (idempotent).
    """
    root = Path(run_dir)
    if not root.is_dir():
        return []

    out: list[Checkpoint] = []
    for entry in root.iterdir():
        if (m := _TORCH.match(entry.name)) and entry.is_dir():
            out.append(Checkpoint(int(m.group(1)), str(entry), final=False))
        elif (m := _MLX.match(entry.name)) and entry.is_file():
            step = int(m.group(1))
            out.append(Checkpoint(step, str(_materialize_mlx(root, step, entry)),
                                  final=False))
    out.sort(key=lambda c: c.step)

    # the final weights live in run_dir itself (both backends save there)
    if _is_adapter_dir(root):
        final_step = max((c.step for c in out), default=-1)
        out.append(Checkpoint(final_step, str(root), final=True))
    return out


def resolve(run_dir: str | Path, step: int | None = None) -> str:
    """The adapter path to load for ``step`` — or the final weights when ``step``
    is None. Raises ``FileNotFoundError`` if there's nothing to load."""
    cks = list_checkpoints(run_dir)
    if not cks:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")
    if step is None:
        finals = [c for c in cks if c.final]
        return (finals or cks)[-1].path
    for c in cks:
        if c.step == step:
            return c.path
    have = ", ".join(str(c.step) for c in cks) or "none"
    raise FileNotFoundError(f"no checkpoint at step {step} under {run_dir} (have: {have})")


def _is_adapter_dir(d: Path) -> bool:
    return any((d / f).exists() for f in
               ("adapter_config.json", "adapters.safetensors", "adapter_model.safetensors"))


def _materialize_mlx(root: Path, step: int, weights: Path) -> Path:
    """Pair a bare ``<step>_adapters.safetensors`` with the run's config in a
    small dir the normal adapter loader can read. Idempotent."""
    dst = root / ".checkpoints" / f"step-{step}"
    target = dst / "adapters.safetensors"
    if target.exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for sidecar in _SIDECARS:
        src = root / sidecar
        if src.exists():
            shutil.copy2(src, dst / sidecar)
    shutil.copy2(weights, target)
    return dst


def write_index(run_dir: str | Path) -> None:
    """Drop a ``checkpoints.json`` summary in the run dir (handy for inspection
    and for servers that persist run metadata). Best-effort."""
    root = Path(run_dir)
    try:
        cks = [c.to_dict() for c in list_checkpoints(root)]
        (root / "checkpoints.json").write_text(json.dumps(cks, indent=2))
    except OSError:
        pass
