"""Metric series transforms and terminal charts.

The studio's charts will render in a browser, but the *data* operations live
here: EMA smoothing, view windows, log scale, percentile clipping — plus a
unicode line chart so `run.plot()` shows the training curves in any terminal.
"""

from __future__ import annotations

import math

_DOTS = "·"  # raw series marker
_LINE = "●"  # smoothed series marker


def ema(values: list[float], weight: float) -> list[float]:
    """Exponential moving average; weight in [0,1), 0 = raw."""
    if not values or weight <= 0:
        return list(values)
    out, acc = [], values[0]
    for v in values:
        acc = weight * acc + (1 - weight) * v
        out.append(acc)
    return out


def clip_percentile(values: list[float], p: float) -> list[float]:
    """Cap values at the p-th percentile (e.g. 0.99) to tame loss spikes."""
    if not values or not 0 < p < 1:
        return list(values)
    cap = sorted(values)[min(len(values) - 1, int(p * len(values)))]
    return [min(v, cap) for v in values]


def _resample(values: list[float], width: int) -> list[float]:
    """Bucket-mean down to `width` columns (or fewer if short)."""
    n = len(values)
    if n <= width:
        return list(values)
    out = []
    for c in range(width):
        lo, hi = (c * n) // width, max((c * n) // width + 1, ((c + 1) * n) // width)
        bucket = values[lo:hi]
        out.append(sum(bucket) / len(bucket))
    return out


def line_chart(
    steps: list[int],
    values: list[float],
    *,
    title: str = "",
    height: int = 8,
    width: int = 60,
    window: int | None = None,
    log: bool = False,
    clip: float | None = None,
    smooth: float = 0.0,
) -> str:
    """Render a unicode line chart: raw as dots, EMA overlay as filled points.

    window: show only the last N points. log: log10 y-axis. clip: percentile cap
    (0.95 / 0.99). smooth: EMA weight (0 = raw only).
    """
    if not values:
        return "(no data)"
    if window:
        steps, values = steps[-window:], values[-window:]
    raw = clip_percentile(values, clip) if clip else list(values)
    if log:
        floor = min((v for v in raw if v > 0), default=1e-9)
        raw = [math.log10(max(v, floor)) for v in raw]
    smoothed = ema(raw, smooth) if smooth > 0 else None

    cols = min(width, len(raw))
    raw_cols = _resample(raw, cols)
    smooth_cols = _resample(smoothed, cols) if smoothed else None

    lo = min(raw_cols + (smooth_cols or []))
    hi = max(raw_cols + (smooth_cols or []))
    span = (hi - lo) or 1.0
    grid = [[" "] * cols for _ in range(height)]

    def put(col: int, value: float, marker: str) -> None:
        row = height - 1 - int((value - lo) / span * (height - 1))
        grid[row][col] = marker

    for c, v in enumerate(raw_cols):
        put(c, v, _DOTS)
    if smooth_cols:
        for c, v in enumerate(smooth_cols):
            put(c, v, _LINE)

    fmt = (lambda v: f"{10 ** v:.4g}") if log else (lambda v: f"{v:.4g}")
    label_w = max(len(fmt(hi)), len(fmt(lo)))
    lines = []
    if title:
        lines.append(title)
    for r, row in enumerate(grid):
        label = fmt(hi) if r == 0 else (fmt(lo) if r == height - 1 else "")
        lines.append(f"{label:>{label_w}} ┤" if label else f"{'':>{label_w}} │")
        lines[-1] += "".join(row)
    avg = sum(values) / len(values)
    lines.append(
        f"{'':>{label_w}} └{'─' * cols}\n"
        f"{'':>{label_w}}  step {steps[0] if steps else 0}–{steps[-1] if steps else 0}"
        f" · last {values[-1]:.4g} · avg {avg:.4g}"
        + (f" · ema({smooth:g})" if smooth else "")
        + (" · log" if log else "")
        + (f" · clip p{int(clip * 100)}" if clip else "")
    )
    return "\n".join(lines)
