// Charts: the sparkline beside a run, and the full loss chart (grid, axis
// ticks, raw + EMA overlay, eval points). Colours come from the theme.
import { useMemo } from "react";
import type { StepMetric } from "@/api";

export function Sparkline({ data, width = 120, height = 32, className }: {
  data: number[]; width?: number; height?: number; className?: string;
}) {
  if (!data.length) return null;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const step = width / Math.max(1, data.length - 1);
  const points = data.map((v, i) =>
    `${(i * step).toFixed(2)},${(height - ((v - min) / range) * height).toFixed(2)}`);
  const path = `M${points.join(" L")}`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}
         className={className} preserveAspectRatio="none">
      <path d={`${path} L${width},${height} L0,${height} Z`} fill="currentColor" opacity={0.12} />
      <path d={path} fill="none" stroke="currentColor" strokeWidth={1.5}
            strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// The full loss chart: grid, axis ticks, raw + EMA overlay, eval points.
export function LossChart({ steps, evals, height = 240, totalSteps }: {
  steps: StepMetric[]; evals: StepMetric[]; height?: number; totalSteps?: number;
}) {
  const losses = steps.map((s) => s.loss);
  const W = 800, H = height;
  const PAD = { l: 44, r: 16, t: 16, b: 28 };
  const innerW = W - PAD.l - PAD.r, innerH = H - PAD.t - PAD.b;

  const { max, stepsCount } = useMemo(() => {
    const all = [...losses, ...evals.map((e) => e.loss)];
    if (!all.length) return { max: 1, stepsCount: totalSteps ?? 1 };
    return { max: Math.max(...all) * 1.1, stepsCount: totalSteps ?? losses.length };
  }, [losses, evals, totalSteps]);

  if (!losses.length) {
    return (
      <div className="flex items-center justify-center text-sm text-muted-foreground" style={{ height: H }}>
        <div className="text-center">
          <div className="font-mono text-xs uppercase tracking-wider opacity-60">No training data yet</div>
          <div className="mt-1 text-xs opacity-50">Metrics appear as soon as the first step lands</div>
        </div>
      </div>
    );
  }

  const x = (step: number) => PAD.l + (step / Math.max(1, stepsCount - 1)) * innerW;
  const y = (val: number) => PAD.t + innerH - (val / (max || 1)) * innerH;
  const trainPath = losses.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ");

  const emaW = 0.85;
  const ema: number[] = [];
  losses.forEach((v, i) => ema.push(i === 0 ? v : ema[i - 1] * emaW + v * (1 - emaW)));
  const emaPath = ema.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ");

  const yTicks = Array.from({ length: 5 }, (_, i) => (max / 4) * i);
  const xLabels = Array.from({ length: 5 }, (_, i) => Math.round((stepsCount / 4) * i));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" preserveAspectRatio="none">
      {yTicks.map((t, i) => (
        <g key={`y${i}`}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)}
                className="stroke-border-strong" strokeDasharray="2 4" strokeWidth={0.5} />
          <text x={PAD.l - 8} y={y(t) + 3} textAnchor="end" fontSize={10}
                className="fill-muted-foreground font-mono">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {xLabels.map((t, i) => (
        <text key={`x${i}`} x={x(t)} y={H - 8} textAnchor="middle" fontSize={10}
              className="fill-muted-foreground font-mono">
          {t}
        </text>
      ))}
      <path d={trainPath} fill="none" className="stroke-primary" strokeWidth={1} opacity={0.35} />
      <path d={emaPath} fill="none" className="stroke-primary" strokeWidth={1.8} />
      {evals.map((e, i) => (
        <g key={i}>
          <circle cx={x(e.step)} cy={y(e.loss)} r={4} className="fill-good" />
          <circle cx={x(e.step)} cy={y(e.loss)} r={4} fill="none"
                  className="stroke-good" strokeOpacity={0.3} strokeWidth={6} />
        </g>
      ))}
      {evals.length > 1 && (
        <path d={evals.map((e, i) => `${i === 0 ? "M" : "L"}${x(e.step)},${y(e.loss)}`).join(" ")}
              fill="none" className="stroke-good" strokeWidth={1.5} strokeDasharray="4 4" />
      )}
    </svg>
  );
}

export function ChartLegend() {
  return (
    <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
      <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-primary" /> train</span>
      <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-good" /> eval</span>
    </div>
  );
}
