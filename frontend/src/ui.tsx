// Shared pieces — page header, badges, sparkline, the loss chart, form helpers.
// Visual language ported from the Shadow Studio Pro design (cream + heart red).
import type { ReactNode } from "react";
import { useEffect, useMemo } from "react";
import type { JobSummary, StepMetric } from "./api";

// ---- modal -------------------------------------------------------------------
export function Modal({ onClose, children, width = "max-w-3xl" }: {
  onClose: () => void; children: ReactNode; width?: string;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => { window.removeEventListener("keydown", onKey); document.body.style.overflow = ""; };
  }, [onClose]);
  return (
    <div onClick={onClose}
         className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/60 backdrop-blur-sm p-6 py-[8vh]">
      <div onClick={(e) => e.stopPropagation()}
           className={`w-full ${width} rounded-xl border border-border bg-card shadow-[0_24px_64px_#000a]`}
           style={{ animation: "rise 0.16s ease-out" }}>
        {children}
      </div>
    </div>
  );
}

export function ModalHeader({ title, onClose }: { title: string; onClose: () => void }) {
  return (
    <div className="flex items-center justify-between px-5 py-3.5 border-b border-border">
      <h3 className="text-sm font-semibold">{title}</h3>
      <button onClick={onClose}
              className="text-muted-foreground hover:text-foreground text-lg leading-none">×</button>
    </div>
  );
}

// ---- layout ------------------------------------------------------------------
export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow?: string; title: string; description?: string; actions?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-6 px-8 pt-8 pb-6 border-b border-border">
      <div className="min-w-0">
        {eyebrow && (
          <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-primary mb-2">{eyebrow}</div>
        )}
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="text-sm text-muted-foreground mt-1.5 max-w-2xl">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}

export function Section({ title, subtitle, actions, children }: {
  title: string; subtitle?: string; actions?: ReactNode; children: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <header className="px-5 py-3.5 border-b border-border flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">{title}</h3>
          {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
        </div>
        {actions}
      </header>
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Field({ label, hint, children }: {
  label: string; hint?: string; children: ReactNode;
}) {
  return (
    <label className="block">
      <div className="text-xs text-muted-foreground mb-1.5 font-medium">{label}</div>
      {children}
      {hint && <div className="text-[10px] text-muted-foreground mt-1">{hint}</div>}
    </label>
  );
}

export function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-md border border-border bg-card px-4 py-3">
      <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
      <div className="font-mono text-lg font-semibold mt-1">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

// ---- buttons -----------------------------------------------------------------
export const btnPrimary =
  "inline-flex items-center gap-1.5 whitespace-nowrap rounded-md bg-primary px-3 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors";
export const btnGhost =
  "inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border border-border bg-card px-3 py-2 text-xs hover:bg-accent transition-colors";

// ---- status ------------------------------------------------------------------
const STATUS_MAP: Record<string, { label: string; cls: string; dot: string }> = {
  succeeded: { label: "Succeeded", cls: "text-success border-success/30 bg-success/10", dot: "bg-success" },
  running: { label: "Running", cls: "text-primary border-primary/30 bg-primary/10", dot: "bg-primary animate-pulse" },
  pending: { label: "Pending", cls: "text-muted-foreground border-border bg-muted/30", dot: "bg-muted-foreground animate-pulse" },
  failed: { label: "Failed", cls: "text-destructive border-destructive/30 bg-destructive/10", dot: "bg-destructive" },
  stopped: { label: "Stopped", cls: "text-muted-foreground border-border bg-muted/30", dot: "bg-muted-foreground" },
};

export function StatusBadge({ status }: { status: JobSummary["status"] }) {
  const s = STATUS_MAP[status] ?? STATUS_MAP.stopped;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[10px] font-mono uppercase tracking-wider ${s.cls}`}>
      <span className={`size-1.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
}

export function Dots() {
  return (
    <span className="flex gap-1.5 items-center text-xs text-muted-foreground font-mono">
      <span className="size-1.5 rounded-full bg-current animate-pulse" />
      <span className="size-1.5 rounded-full bg-current animate-pulse [animation-delay:0.15s]" />
      <span className="size-1.5 rounded-full bg-current animate-pulse [animation-delay:0.3s]" />
      <span className="ml-1">thinking…</span>
    </span>
  );
}

// ---- charts ------------------------------------------------------------------
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
                stroke="oklch(0.78 0.02 50)" strokeDasharray="2 4" strokeWidth={0.5} />
          <text x={PAD.l - 8} y={y(t) + 3} textAnchor="end" fontSize={10}
                fill="oklch(0.55 0.04 45)" fontFamily="ui-monospace, monospace">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {xLabels.map((t, i) => (
        <text key={`x${i}`} x={x(t)} y={H - 8} textAnchor="middle" fontSize={10}
              fill="oklch(0.55 0.04 45)" fontFamily="ui-monospace, monospace">
          {t}
        </text>
      ))}
      <path d={trainPath} fill="none" stroke="oklch(0.55 0.20 25)" strokeWidth={1} opacity={0.35} />
      <path d={emaPath} fill="none" stroke="oklch(0.55 0.20 25)" strokeWidth={1.8} />
      {evals.map((e, i) => (
        <g key={i}>
          <circle cx={x(e.step)} cy={y(e.loss)} r={4} fill="oklch(0.65 0.15 140)" />
          <circle cx={x(e.step)} cy={y(e.loss)} r={4} fill="none"
                  stroke="oklch(0.65 0.15 140)" strokeOpacity={0.3} strokeWidth={6} />
        </g>
      ))}
      {evals.length > 1 && (
        <path d={evals.map((e, i) => `${i === 0 ? "M" : "L"}${x(e.step)},${y(e.loss)}`).join(" ")}
              fill="none" stroke="oklch(0.65 0.15 140)" strokeWidth={1.5} strokeDasharray="4 4" />
      )}
    </svg>
  );
}

export function ChartLegend() {
  return (
    <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
      <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-primary" /> train</span>
      <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-success" /> eval</span>
    </div>
  );
}
