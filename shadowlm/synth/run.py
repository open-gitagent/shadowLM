"""The handle `synthesize()` returns — what came out, and what didn't.

Synthetic data is only trustworthy if you can see how much of it was thrown
away, so the report is not a summary line bolted on at the end: every rejection
is counted as it happens, and the counts reconcile exactly —

    generated == kept + rejected_validation + rejected_dedup + rejected_judge
                 + rejected_flat + surplus
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data import Dataset
from ..rl import Trajectory, TrajectoryGroup


@dataclass
class SynthReport:
    """Honest accounting for one synthesis run."""

    requested: int = 0
    kept: int = 0
    scenarios: int = 0            # distinct taxonomy leaves explored
    generated: int = 0            # candidate rows the teacher was asked for
    rejected_validation: int = 0  # malformed, even after the corrective retry
    rejected_dedup: int = 0       # a repeat of something already kept
    rejected_judge: int = 0       # scored below min_score
    rejected_flat: int = 0        # GRPO group with no reward spread — no signal
    surplus: int = 0              # good rows past `requested`, trimmed
    repaired: int = 0             # rescued by the corrective retry
    mean_score: float | None = None
    teacher_calls: int = 0
    duration_s: float = 0.0
    note: str | None = None       # e.g. the more_plus_group_size to train with

    @property
    def balanced(self) -> bool:
        """The funnel reconciles — nothing vanished unaccounted for."""
        return self.generated == (self.kept + self.rejected_validation
                                  + self.rejected_dedup + self.rejected_judge
                                  + self.rejected_flat + self.surplus)

    def summary(self) -> str:
        score = f" · mean score {self.mean_score:.2f}" if self.mean_score else ""
        flat = f" · {self.rejected_flat} flat-group" if self.rejected_flat else ""
        surplus = f" · {self.surplus} surplus" if self.surplus else ""
        note = f"\n  note: {self.note}" if self.note else ""
        return (
            f"  ♥ {self.kept} rows from {self.generated} generated "
            f"({self.scenarios} scenarios)\n"
            f"  rejected: {self.rejected_validation} invalid · "
            f"{self.rejected_dedup} duplicate · {self.rejected_judge} low-scoring"
            f"{flat}{surplus} · {self.repaired} repaired\n"
            f"  {self.teacher_calls} teacher calls · {self.duration_s:.1f}s{score}{note}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"SynthReport(kept={self.kept}/{self.requested}, "
                f"generated={self.generated}, duration={self.duration_s:.1f}s)")


@dataclass
class SynthRun:
    """The result of `synthesize()`: the data, plus how it came to be."""

    format: str
    report: SynthReport
    trajectories: list[Trajectory] = field(default_factory=list)
    rejected: list[Trajectory] = field(default_factory=list)
    dataset: Dataset | None = None            # chat | text | preference | grpo
    groups: list[TrajectoryGroup] | None = None   # trajectory-GRPO
    spans: dict | None = None                 # OTLP payload

    def rows(self) -> list[dict]:
        """The training rows this run produced, in the shape the method takes."""
        if self.dataset is not None:
            return self.dataset.rows
        if self.groups is not None:
            from ..rl import weighted_rows  # noqa: PLC0415

            return weighted_rows(self.groups)
        return []

    def save(self, path: str | Path) -> str:
        """Write the run's artifact — JSONL rows, or the OTLP payload as JSON.

        The JSONL ends with a newline: without it `wc -l` undercounts by one and
        anything appending to the file corrupts the final row.
        """
        path = Path(path)
        if self.spans is not None:
            path.write_text(json.dumps(self.spans, indent=2) + "\n")
        else:
            path.write_text("".join(json.dumps(r) + "\n" for r in self.rows()))
        return str(path)

    def to_otlp(self, path: str | Path | None = None, **kwargs) -> dict:
        """Re-emit the episodes as OpenTelemetry GenAI spans."""
        from .emit import to_otlp  # noqa: PLC0415

        return to_otlp(self.trajectories, path=path, **kwargs)

    def __repr__(self) -> str:
        what = (f"{len(self.groups)} groups" if self.groups is not None
                else f"{len(self.rows())} rows")
        return f"SynthRun(format={self.format!r}, {what}, {self.report!r})"
