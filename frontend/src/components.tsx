// shadcn-style primitives, hand-built on cva (no Radix, no SSR) — harvested
// from the Shadow Studio Pro design, themed to Shadow & Heart.
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "./lib/cn";

// ---- Button -----------------------------------------------------------------
const button = cva(
  "inline-flex items-center justify-center gap-1.5 rounded-md text-sm font-medium transition-colors disabled:opacity-40 disabled:pointer-events-none",
  {
    variants: {
      variant: {
        primary: "bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] text-white hover:shadow-[0_4px_22px_#e5484d55] active:translate-y-px",
        ghost: "border border-border text-bone hover:border-faded",
        subtle: "text-faded hover:text-bone",
        danger: "text-heart hover:bg-heart/10",
      },
      size: { sm: "px-3 py-1.5 text-[13px]", md: "px-3.5 py-2", icon: "size-9" },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export function Button({ className, variant, size, ...props }:
  ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof button>) {
  return <button className={cn(button({ variant, size }), className)} {...props} />;
}

// ---- Card -------------------------------------------------------------------
export function Card({ className, selected, onClick, children }: {
  className?: string; selected?: boolean; onClick?: () => void; children: ReactNode;
}) {
  return (
    <div onClick={onClick}
      className={cn(
        "rounded-xl border bg-gradient-to-b from-[#272019] to-[#211b15] p-4",
        selected ? "border-heart shadow-[0_0_0_1px_#e5484d55]" : "border-border",
        onClick && "cursor-pointer transition-colors hover:border-faded",
        className)}>
      {children}
    </div>
  );
}

// ---- Badge ------------------------------------------------------------------
const badge = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-px text-[11px]",
  {
    variants: {
      tone: {
        muted: "border-border text-faded",
        red: "border-heart/60 text-heart",
        green: "border-success/60 text-success",
        gold: "border-warning/60 text-warning",
      },
    },
    defaultVariants: { tone: "muted" },
  },
);
export function Badge({ tone, children }: VariantProps<typeof badge> & { children: ReactNode }) {
  return <span className={badge({ tone })}>{children}</span>;
}

// ---- StatusBadge (run lifecycle) --------------------------------------------
const STATUS: Record<string, { cls: string; dot: string }> = {
  succeeded: { cls: "text-success border-success/30 bg-success/10", dot: "bg-success" },
  running: { cls: "text-heart border-heart/30 bg-heart/10", dot: "bg-heart animate-[pulsedot_1.2s_infinite]" },
  pending: { cls: "text-faded border-border bg-umbra", dot: "bg-faded" },
  failed: { cls: "text-destructive border-destructive/30 bg-destructive/10", dot: "bg-destructive" },
  stopped: { cls: "text-warning border-warning/30 bg-warning/10", dot: "bg-warning" },
};
export function StatusBadge({ status }: { status: string }) {
  const s = STATUS[status] ?? STATUS.pending;
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider", s.cls)}>
      <span className={cn("size-1.5 rounded-full", s.dot)} />
      {status}
    </span>
  );
}

// ---- PageHeader -------------------------------------------------------------
export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow?: string; title: string; description?: string; actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex items-start justify-between gap-6 border-b border-border pb-5">
      <div className="min-w-0">
        {eyebrow && <div className="mb-2 text-[10px] uppercase tracking-[0.2em] text-heart">{eyebrow}</div>}
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1.5 max-w-2xl text-sm text-faded">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

// ---- Stat card --------------------------------------------------------------
export function Stat({ label, value, sub, icon: Icon }: {
  label: string; value: string; sub: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <Card>
      <div className="flex items-center justify-between text-faded">
        <span className="text-[10px] uppercase tracking-[0.18em]">{label}</span>
        <Icon className="size-4" />
      </div>
      <div className="mt-3 text-2xl font-semibold tracking-tight">{value}</div>
      <div className="mt-1 text-xs text-faded">{sub}</div>
    </Card>
  );
}

// ---- Sparkline (tiny inline loss trend) -------------------------------------
export function Sparkline({ values, className = "" }: { values: number[]; className?: string }) {
  if (values.length < 2) return null;
  const W = 120, H = 28;
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const pts = values.map((v, i) =>
    `${(i * W) / (values.length - 1)},${H - ((v - lo) / span) * H}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={cn("h-7 w-[120px]", className)} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke="#e5484d" strokeWidth="1.5"
                strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
