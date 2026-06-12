// Small shared pieces — pills, dots, cards, the loss chart.
import type { ReactNode } from "react";
import type { StepMetric } from "./api";

export function Pill({ tone = "", children }: { tone?: "red" | "green" | "gold" | ""; children: ReactNode }) {
  const tones: Record<string, string> = {
    red: "text-heart border-heart/60",
    green: "text-okay border-okay/60",
    gold: "text-gold border-gold/60",
    "": "text-faded border-seam",
  };
  return (
    <span className={`inline-block rounded-full border px-2.5 py-px text-[11px] ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function StatusDot({ status }: { status: string }) {
  const tone: Record<string, string> = {
    succeeded: "bg-okay",
    failed: "bg-heart",
    running: "bg-gold animate-[pulsedot_1.2s_infinite]",
    stopped: "bg-gold",
    pending: "bg-faded",
  };
  return <span className={`mr-2 inline-block size-2 rounded-full ${tone[status] ?? "bg-faded"}`} />;
}

export function Card({ selected, onClick, children, className = "" }: {
  selected?: boolean; onClick?: () => void; children: ReactNode; className?: string;
}) {
  return (
    <div
      onClick={onClick}
      className={`rounded-xl border bg-card p-3.5
        ${selected ? "border-heart shadow-[0_0_0_1px_#e5484d55]" : "border-seam"}
        ${onClick ? "cursor-pointer hover:border-faded transition-colors" : ""} ${className}`}
    >
      {children}
    </div>
  );
}

export function Dots() {
  return (
    <span className="inline-flex h-[1em] items-center gap-1">
      {[0, 0.18, 0.36].map((d) => (
        <i key={d} className="size-[5px] rounded-full bg-faded animate-[blink_1.1s_infinite]"
           style={{ animationDelay: `${d}s` }} />
      ))}
    </span>
  );
}

export function Lead({ children }: { children: ReactNode }) {
  return <p className="mb-4 text-[12.5px] text-faded">{children}</p>;
}

export function H2({ children }: { children: ReactNode }) {
  return <h2 className="mb-1 text-[15px] font-bold">{children}</h2>;
}

// The loss chart — heart-red curve, gold eval rings. Pure SVG, no library.
export function LossChart({ steps, evals }: { steps: StepMetric[]; evals: StepMetric[] }) {
  const W = 860, H = 270, P = 36;
  if (!steps.length)
    return <p className="text-[12px] text-faded">waiting for the first metric…</p>;
  const pts = steps.map((s) => s.loss);
  const all = pts.concat(evals.map((e) => e.loss));
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = hi - lo || 1;
  const x = (i: number) => P + (i * (W - 2 * P)) / Math.max(1, pts.length - 1);
  const y = (v: number) => H - P - ((v - lo) * (H - 2 * P)) / span;
  const line = pts.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
         className="my-3.5 h-[280px] w-full rounded-xl border border-seam bg-umbra">
      <text x={P} y={P - 12} fill="#9a8f82" fontSize="11">loss {hi.toFixed(3)}</text>
      <text x={P} y={H - P + 18} fill="#9a8f82" fontSize="11">{lo.toFixed(3)}</text>
      <path d={line} fill="none" stroke="#e5484d" strokeWidth="2.5"
            strokeLinejoin="round" strokeLinecap="round" />
      {evals.map((e, i) => (
        <circle key={i} cx={x(Math.min(pts.length - 1, Math.max(0, e.step - 1)))}
                cy={y(e.loss)} r="4" fill="none" stroke="#d29922" strokeWidth="2" />
      ))}
    </svg>
  );
}
