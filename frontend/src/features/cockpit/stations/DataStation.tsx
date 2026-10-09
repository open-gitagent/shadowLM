// Data: the examples this model learns from, readable as questions and answers.
import { useState } from "react";
import { Database, LoaderCircle } from "lucide-react";

import { cancelSynth } from "@/api";
import { EmptyState } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { allows } from "@/lib/embed";
import { useMode } from "@/lib/mode";
import { cn } from "@/lib/utils";

import { type Loop, synthPhase } from "../model";
import { rowPair, StationHeader } from "./shared";

export function DataStation({ loop }: { loop: Loop }) {
  const mode = useMode();
  const p = loop.project!;
  const d = loop.dataset;

  if (!p.dataset_id && loop.synth) {
    return (
      <>
        <StationHeader title="Data" line={loop.writing ? "Writing its examples from your description." : "Writing the examples stopped."} />
        <div className="p-5"><Writing loop={loop} /></div>
      </>
    );
  }

  if (!p.dataset_id) {
    return (
      <>
        <StationHeader title="Data" line="No examples yet." />
        <div className="p-5">
          <EmptyState icon={Database} title="It learns from your examples">
            A model gets its examples when it's made: written as questions and answers, uploaded as JSONL, picked from a dataset, or captured from your agent.{" "}
            <a className="text-primary underline-offset-4 hover:underline" href={mode === "research" ? "#datasets" : "#projects/new"}>
              {mode === "research" ? "Go to Datasets" : "Make a new model"}
            </a>{" "}to bring them in.
          </EmptyState>
        </div>
      </>
    );
  }

  if (!d) {
    return (
      <>
        <StationHeader title="Data" line={<Skeleton className="h-4 w-56" />} />
        <div className="space-y-2 p-5">
          {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-10 w-full" />)}
        </div>
      </>
    );
  }

  const rows = (d.preview ?? []).slice(0, 8);
  const source = d.source === "hf" ? `Hugging Face · ${d.repo ?? ""}` : "Uploaded to this server";
  return (
    <>
      <StationHeader title="Data"
        line={<>{d.name} · {d.rows ?? "?"} {d.rows === 1 ? "example" : "examples"} · {d.format} format</>}
        research="#datasets" />
      <div className="space-y-3 p-5">
        {(loop.writing || (loop.synth?.status === "failed" && loop.synth.dataset_id !== p.dataset_id)) && <Writing loop={loop} />}
        <p className="text-xs text-muted-foreground">
          {source}{mode === "research" && <> · <span className="font-mono">{d.dataset_id}</span></>}
          {rows.length > 0 && (d.rows ?? 0) > rows.length && <> · showing {rows.length} of {d.rows}</>}
        </p>
        {rows.length > 0 ? (
          <ul className="divide-y divide-border border border-border">
            {rows.map((r, i) => {
              const { q, a } = rowPair(r);
              return (
                <li key={i} className="grid gap-1 px-4 py-3 text-sm @2xl:grid-cols-2 @2xl:gap-5">
                  <span className="font-medium">{q}</span>
                  <span className="text-muted-foreground">{a || "—"}</span>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground">This dataset has no preview rows; open it in Research to read it.</p>
        )}
      </div>
    </>
  );
}

// Examples being written by the frontier model (the synthesizer): its phase,
// how many it has kept, and a way to stop that keeps what's done.
function Writing({ loop }: { loop: Loop }) {
  const s = loop.synth!;
  const live = s.status === "running" && !!s.total;
  const planning = live && s.phase === "planning";
  const pct = live ? Math.round((100 * (s.done ?? 0)) / Math.max(1, s.total!)) : Math.round((100 * s.kept) / Math.max(1, s.requested));
  return (
    <div className="space-y-2 border border-border bg-surface px-4 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 text-sm">
        <span className="font-medium">
          {s.status === "running" ? `${synthPhase[s.phase ?? "starting"] ?? "Writing"}${live && !planning ? ` · ${s.done}/${s.total}` : ""}`
            : s.status === "failed" ? "Failed" : s.status === "stopped" ? "Stopped" : "Done"}
        </span>
        <span className="text-xs text-muted-foreground tabular-nums">
          {s.kept} of {s.requested} kept{loop.frontier ? ` · written by ${loop.frontier.model}` : ""}{s.tokens ? ` · ${s.tokens.toLocaleString()} tokens` : ""}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-label="Examples written"
           aria-valuenow={planning ? undefined : pct} aria-valuemin={0} aria-valuemax={100}>
        <div className={cn("h-full rounded-full bg-primary transition-all duration-300", planning && "motion-safe:animate-pulse")}
             style={{ width: `${planning ? 100 : pct}%` }} />
      </div>
      {s.error && <p className="text-sm text-destructive">{s.error}</p>}
      {s.status === "running" && allows("operator") && <StopWriting id={s.synth_id} />}
    </div>
  );
}

function StopWriting({ id }: { id: string }) {
  const [stopping, setStopping] = useState(false);
  return (
    <Button variant="ghost" size="sm" disabled={stopping}
            onClick={() => { setStopping(true); cancelSynth(id).catch(() => setStopping(false)); }}>
      {stopping && <LoaderCircle className="animate-spin" />} {stopping ? "Stopping…" : "Stop and keep what's written"}
    </Button>
  );
}
