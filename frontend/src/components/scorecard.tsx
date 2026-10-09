// The evaluation scorecard: each target's score on the same questions, a verdict
// in plain words, and the answers side by side. Business mode's Evaluate step and
// Research mode's Evaluate page both show an evaluation through this.
import { Check, CircleDashed, LoaderCircle, X } from "lucide-react";

import type { Evaluation } from "@/api";
import { ErrorState } from "@/components/common";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { breakable } from "@/lib/format";
import { cn } from "@/lib/utils";

const right = (score: number, n: number) => Math.round(score * n);

// The verdict compares the first target (the fine-tune) with the base model
// and, when it was evaluated too, the frontier model. Thresholds are about the
// user's own questions, never a benchmark.
export function verdict(ev: Evaluation): { tone: "good" | "warning" | "muted"; title: string; detail: string } | null {
  if (ev.status !== "succeeded" || ev.results.length < 1) return null;
  const tuned = ev.results[0];
  const fi = ev.targets.findIndex((t) => t.kind === "frontier");
  const bi = ev.targets.findIndex((t, i) => i > 0 && t.kind !== "frontier");
  const base = bi > 0 ? ev.results[bi] : undefined;
  const front = fi > 0 ? ev.results[fi] : undefined;
  const t = right(tuned.score, tuned.n);
  const b = base ? right(base.score, base.n) : null;
  const f = front ? right(front.score, front.n) : null;
  const of = `${t} of ${tuned.n}`;
  if (f !== null) {
    if (t >= f && tuned.score >= 0.8)
      return { tone: "good", title: "Matches the frontier model",
               detail: `It gets ${of} right; the frontier model gets ${f}${b !== null ? ` and the base model ${b}` : ""}. It can take this job over.` };
    if (b === null || t > b)
      return { tone: "warning", title: "Better than its base, behind the frontier model",
               detail: `It gets ${of} right against the frontier model's ${f}${b !== null ? ` and the base model's ${b}` : ""}. More examples usually close the gap.` };
  }
  if (tuned.score >= 0.9 && (b === null || t > b))
    return { tone: "good", title: "Ready to use", detail: `It answers ${of} questions correctly${b !== null ? `; the base model gets ${b}` : ""}.` };
  if (b !== null && t > b)
    return { tone: "warning", title: "Better than the base model, not there yet",
             detail: `It answers ${of} correctly against the base model's ${b}. More examples, or more training steps, usually close the gap.` };
  return { tone: "muted", title: "Not learned yet",
           detail: `It answers ${of} correctly${b !== null ? `, the same as or worse than the base model (${b})` : ""}. Check the examples below, then add data or train longer.` };
}

export function Scorecard({ ev, detail = true }: { ev: Evaluation; detail?: boolean }) {
  if (ev.status === "failed") return <ErrorState title="The evaluation stopped" error={ev.error} />;
  const v = verdict(ev);
  const running = ev.status === "pending" || ev.status === "running";

  return (
    <div className="space-y-5">
      {v && (
        <div className={cn("border px-4 py-3.5",
          v.tone === "good" ? "border-good/30 bg-good/5" : v.tone === "warning" ? "border-warning/30 bg-warning/5" : "border-border bg-surface/40")}>
          <p className={cn("text-sm font-semibold", v.tone === "good" ? "text-good" : v.tone === "warning" ? "text-warning" : "text-foreground")}>{v.title}</p>
          <p className="mt-0.5 text-sm text-muted-foreground">{v.detail}</p>
        </div>
      )}

      <div className="grid gap-px overflow-hidden border border-border bg-border" style={{ gridTemplateColumns: `repeat(${ev.targets.length}, minmax(0, 1fr))` }}>
        {ev.targets.map((t, i) => {
          const r = ev.results[i];
          return (
            <div key={i} className="bg-card px-4 py-3.5">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs text-muted-foreground">{t.label}</span>
                {!r && running && <LoaderCircle className="size-3.5 shrink-0 animate-spin text-muted-foreground" />}
              </div>
              {r ? (
                <>
                  <div className="mt-1.5 flex items-baseline gap-1.5">
                    <span className="text-2xl font-semibold tracking-[-0.01em] tabular-nums">{right(r.score, r.n)}</span>
                    <span className="text-sm text-muted-foreground tabular-nums">of {r.n} correct</span>
                  </div>
                  <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-surface-2" role="img" aria-label={`${Math.round(r.score * 100)}%`}>
                    <div className={cn("h-full rounded-full", i === 0 ? "bg-primary" : "bg-muted-foreground/50")} style={{ width: `${r.score * 100}%` }} />
                  </div>
                </>
              ) : (
                <div className="mt-2 space-y-2">
                  <Skeleton className="h-7 w-24" />
                  <Skeleton className="h-1.5 w-full" />
                </div>
              )}
              <div className="mt-2 font-mono text-[11px] text-muted-foreground" title={t.model}>
                {breakable(t.model.split("/").pop() ?? t.model)}{t.checkpoint != null ? ` · step ${t.checkpoint}` : ""}
              </div>
            </div>
          );
        })}
      </div>

      {running && (
        <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <CircleDashed className="size-3.5 animate-spin" />
          Asking each model the same questions. The first answer is slow while the models load.
        </p>
      )}

      {detail && ev.results[0]?.examples && <Answers ev={ev} />}
    </div>
  );
}

function Mark({ ok }: { ok: boolean }) {
  return ok
    ? <Check className="mt-0.5 size-3.5 shrink-0 text-good" aria-label="correct" />
    : <X className="mt-0.5 size-3.5 shrink-0 text-destructive" aria-label="wrong" />;
}

// One row per question: what was expected, then each target's answer. Wide,
// it's a table; narrow, each question stacks as its own block, since four
// columns at phone width break words mid-letter.
function Answers({ ev }: { ev: Evaluation }) {
  const rows = ev.results[0].examples ?? [];
  const answer = (ti: number, qi: number) => {
    const ex = ev.results[ti]?.examples?.[qi];
    return ex ? (
      <div className="flex gap-1.5">
        <Mark ok={ex.score >= 1} />
        <span className="line-clamp-4">{ex.output.replace(/<think>[\s\S]*?<\/think>/g, "").trim() || ex.output}</span>
      </div>
    ) : <Skeleton className="h-4 w-3/4" />;
  };
  return (
    <div className="@container">
      <div className="hidden border border-border @xl:block">
        {/* fixed layout: each model's answers get an equal share of the width,
            however long one model rambles */}
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[20%]">Question</TableHead>
              <TableHead className="w-[20%]">Expected</TableHead>
              {ev.targets.map((t, i) => <TableHead key={i}>{t.label}</TableHead>)}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, qi) => (
              <TableRow key={qi} className="align-top">
                <TableCell className="whitespace-normal text-sm font-medium">{row.input}</TableCell>
                <TableCell className="whitespace-normal text-sm text-muted-foreground">{row.expected}</TableCell>
                {ev.targets.map((_, ti) => <TableCell key={ti} className="whitespace-normal text-sm">{answer(ti, qi)}</TableCell>)}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <ol className="divide-y divide-border border border-border @xl:hidden">
        {rows.map((row, qi) => (
          <li key={qi} className="space-y-2 px-4 py-3.5">
            <p className="text-sm font-medium">{row.input}</p>
            <p className="text-sm text-muted-foreground"><span className="text-xs">Expected · </span>{row.expected}</p>
            {ev.targets.map((t, ti) => (
              <div key={ti} className="text-sm">
                <p className="mb-0.5 text-xs text-muted-foreground">{t.label}</p>
                {answer(ti, qi)}
              </div>
            ))}
          </li>
        ))}
      </ol>
    </div>
  );
}
