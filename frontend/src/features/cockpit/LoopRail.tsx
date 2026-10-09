// The loop rail: four stations (Data → Fine-tune → Evaluate → Deploy) on one
// continuous line. Each station says its state; the segment flowing into a
// running station moves while the work runs. The selected station is tinted,
// the one a message in the thread speaks about is highlighted, and choosing a
// station opens it in the inspector and scrolls the thread to it.
import { CircleAlert, CircleCheck, CircleDashed, CircleDot, CircleMinus, CircleX } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { type KeyboardEvent, useRef } from "react";

import { cn } from "@/lib/utils";

import { useCockpitLink } from "./Cockpit";
import { type Loop, type Station, STATIONS, type StationId, type StationStatus } from "./model";

const word: Record<StationStatus, string> = {
  idle: "Waiting", ready: "Ready", running: "Running", done: "Done", failed: "Failed",
};

function Glyph({ status, tone }: { status: StationStatus; tone?: Station["tone"] }) {
  if (status === "done" && tone === "warning") return <CircleAlert className="size-3.5 text-warning" strokeWidth={2} />;
  if (status === "done" && tone === "muted") return <CircleMinus className="size-3.5 text-muted-foreground" strokeWidth={2} />;
  if (status === "done") return <CircleCheck className="size-3.5 text-good" strokeWidth={2} />;
  if (status === "failed") return <CircleX className="size-3.5 text-destructive" strokeWidth={2} />;
  if (status === "running") return (
    <span className="relative grid size-3.5 place-items-center" aria-hidden>
      <span className="absolute size-2.5 animate-ping rounded-full bg-primary/40 motion-reduce:animate-none" />
      <span className="size-2 rounded-full bg-primary" />
    </span>
  );
  if (status === "ready") return <CircleDot className="size-3.5 text-primary" strokeWidth={2} />;
  return <CircleDashed className="size-3.5 text-muted-foreground/60" strokeWidth={2} />;
}

// One stretch of the rail, from a station's marker to the next one.
function Segment({ from, to }: { from: Station; to: Station }) {
  const reduce = useReducedMotion();
  const flowing = to.status === "running" && from.status === "done";
  const solid = from.status === "done" && (to.status === "done" || to.status === "failed");
  if (flowing) {
    return (
      <motion.span aria-hidden
        className="absolute top-3 right-[-13px] left-[13px] h-0.5 text-primary"
        style={{ backgroundImage: "repeating-linear-gradient(90deg, currentColor 0 6px, transparent 6px 12px)" }}
        animate={reduce ? undefined : { backgroundPositionX: ["0px", "12px"] }}
        transition={{ duration: 0.6, ease: "linear", repeat: Infinity }} />
    );
  }
  return (
    <span aria-hidden className={cn("absolute top-3 right-[-13px] left-[13px] h-0.5 transition-colors duration-200",
      solid ? "bg-good/60" : from.status === "done" ? "bg-primary/40" : "bg-border")} />
  );
}

export function LoopRail({ loop }: { loop: Loop }) {
  const link = useCockpitLink();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  function onKey(e: KeyboardEvent, i: number) {
    const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const j = (i + step + STATIONS.length) % STATIONS.length;
    link.select(STATIONS[j].id, "rail");
    refs.current[j]?.focus();
  }

  const versions = loop.versions;

  return (
    <nav aria-label="The loop" className="border border-border bg-card px-3 py-3">
      <div role="radiogroup" aria-label="Stations" className="grid grid-cols-2 gap-y-2 @2xl:grid-cols-4">
        {STATIONS.map((s, i) => {
          const st = loop.stations[s.id];
          const selected = link.selected === s.id;
          const hovered = link.hovered === s.id && !selected;
          const nextStation = STATIONS[i + 1] ? loop.stations[STATIONS[i + 1].id as StationId] : null;
          return (
            <div key={s.id} className="relative min-w-0">
              {/* the rail runs behind the markers; at two columns it breaks after Fine-tune */}
              {nextStation && <div className={cn("contents", i === 1 && "@max-2xl:hidden")}><Segment from={st} to={nextStation} /></div>}
              <button
                ref={(el) => { refs.current[i] = el; }}
                type="button" role="radio" aria-checked={selected} tabIndex={selected ? 0 : -1}
                onClick={() => link.select(s.id, "rail")} onKeyDown={(e) => onKey(e, i)}
                className={cn(
                  "relative flex w-full min-w-0 flex-col items-start gap-1.5 rounded-md border px-0 pt-0 pb-2 pr-2 text-left transition-colors duration-200 outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                  selected ? "border-primary/30 bg-primary/5" : hovered ? "border-primary/15 bg-primary/[0.03]" : "border-transparent hover:bg-subtle",
                )}>
                {/* the marker sits alone on the rail; the station's words hang below it */}
                <span className={cn("relative z-10 grid size-6 shrink-0 place-items-center rounded-full border bg-card transition-colors duration-200",
                  st.status === "done" ? (st.tone === "warning" ? "border-warning/50" : st.tone === "muted" ? "border-border-strong" : "border-good/40") : st.status === "failed" ? "border-destructive/40"
                    : st.status === "running" || st.status === "ready" ? "border-primary/40" : "border-border")}>
                  <Glyph status={st.status} tone={st.tone} />
                </span>
                <span className="w-full min-w-0 pl-1">
                  <span className={cn("block text-sm font-semibold tracking-[-0.01em]", st.status === "idle" && "text-muted-foreground")}>{s.label}</span>
                  <span className={cn("block text-[11px] font-medium",
                    st.status === "done" ? (st.tone === "warning" ? "text-warning" : st.tone === "muted" ? "text-muted-foreground" : "text-good") : st.status === "failed" ? "text-destructive"
                      : st.status === "running" || st.status === "ready" ? "text-primary" : "text-muted-foreground")}>
                    {st.tone === "warning" ? "Not there yet" : st.tone === "muted" ? "Not learned yet" : st.tone === "good" ? "Ready" : word[st.status]}
                  </span>
                  <span className="line-clamp-2 block text-xs text-muted-foreground" title={st.line}>{st.line}</span>
                </span>
              </button>
              {s.id === "finetune" && versions.length > 1 && (
                <div className="mt-1 flex flex-wrap gap-1 pl-1" aria-label="Versions">
                  {versions.map((v) => (
                    <button key={v.run_id} type="button" onClick={() => link.select("evaluate", "rail")}
                      title={`Version ${v.n}${v.current ? " (current)" : ""}: compare on Evaluate`}
                      className={cn("h-5 rounded-full border px-1.5 font-mono text-[10px] tabular-nums transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                        v.current ? "border-primary/30 bg-primary/10 text-primary" : "border-border text-muted-foreground hover:text-foreground")}>
                      v{v.n}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </nav>
  );
}
