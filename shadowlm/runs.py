"""Run history — every finetune persists a `run.json` next to its checkpoint.

The History view as an API:

    import shadowlm as slm

    slm.runs.list()              # newest first
    slm.runs.latest()            # most recent run
    run = slm.runs.load("Qwen2.5-0.5B-...-1781031525")
    print(run.status, run.loss, run.sparkline())

    model.finetune(ds, resume_from_checkpoint=run.checkpoint)   # resume
    slm.runs.delete(run.id)      # permanent — removes checkpoint + metrics
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from .training import TrainingRun

RUNS_ROOT = Path.home() / ".shadowlm" / "runs"
_RECORD = "run.json"


def save(run: TrainingRun, output_dir: str | Path) -> str:
    """Persist a run record into its output directory. Called by `finetune`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / _RECORD).write_text(json.dumps(run.to_dict(), indent=2))
    return str(out / _RECORD)


def _resolve(id_or_path: str | Path) -> Path:
    p = Path(id_or_path)
    if not p.exists():
        p = RUNS_ROOT / str(id_or_path)
    if not (p / _RECORD).exists():
        raise FileNotFoundError(f"no run record found at {p}")
    return p


def load(id_or_path: str | Path) -> TrainingRun:
    """Load a run by id (directory name under ~/.shadowlm/runs) or path."""
    p = _resolve(id_or_path)
    run = TrainingRun.from_dict(json.loads((p / _RECORD).read_text()))
    run.checkpoint = run.checkpoint or str(p)
    return run


def list(root: str | Path = RUNS_ROOT) -> List[TrainingRun]:  # noqa: A001
    """All recorded runs, newest first."""
    root = Path(root)
    if not root.exists():
        return []
    found = []
    for record in root.glob(f"*/{_RECORD}"):
        try:
            found.append(load(record.parent))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # an unreadable record shouldn't hide the rest
    found.sort(key=lambda r: r.started_at or 0, reverse=True)
    return found


def latest() -> TrainingRun | None:
    """The most recent run, or None if there is no history yet."""
    runs = list()
    return runs[0] if runs else None


def delete(id_or_path: str | Path) -> None:
    """Permanently delete a run — its record, metrics, and checkpoint files."""
    shutil.rmtree(_resolve(id_or_path))
