// Evaluate: the current version's scorecard against its base (and the frontier
// model when one is set), with every earlier version's score above it so the
// next version is chosen on evidence.
import { useQueryClient } from "@tanstack/react-query";
import { Cloud, LoaderCircle, RotateCcw, Sparkles } from "lucide-react";
import { useState } from "react";

import { FrontierDialog } from "@/components/frontier-settings";
import { Scorecard, verdict } from "@/components/scorecard";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { allows } from "@/lib/embed";
import { cn } from "@/lib/utils";

import type { CockpitActions } from "../Cockpit";
import type { Loop } from "../model";
import { StationHeader } from "./shared";

function Versions({ loop }: { loop: Loop }) {
  return (
    <div className="border border-border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Version</TableHead>
            <TableHead>Base model</TableHead>
            <TableHead className="text-right">Steps</TableHead>
            <TableHead className="text-right">Score</TableHead>
            <TableHead>Verdict</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loop.versions.map((v) => {
            const r = v.evaluation?.status === "succeeded" ? v.evaluation.results[0] : undefined;
            const vd = v.evaluation ? verdict(v.evaluation) : null;
            return (
              <TableRow key={v.run_id} className={cn(v.current && "bg-primary/5")}>
                <TableCell className="font-medium">v{v.n}{v.current && <span className="ml-1.5 text-xs font-normal text-primary">current</span>}</TableCell>
                <TableCell className="font-mono text-xs">{v.job?.base_model.split("/").pop() ?? "—"}</TableCell>
                <TableCell className="text-right tabular-nums">{v.job?.steps ?? "—"}</TableCell>
                <TableCell className="text-right tabular-nums">{r ? `${Math.round(r.score * r.n)} of ${r.n}` : "—"}</TableCell>
                <TableCell className={cn("text-sm", vd?.tone === "good" ? "text-good" : vd?.tone === "warning" ? "text-warning" : "text-muted-foreground")}>
                  {vd?.title ?? (v.eval_id ? "Evaluating…" : "Not evaluated")}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

export function EvaluateStation({ loop, actions }: { loop: Loop; actions: CockpitActions }) {
  const qc = useQueryClient();
  const [dialog, setDialog] = useState(false);
  const operator = allows("operator");
  const p = loop.project!;
  const job = loop.job;
  const ev = loop.evaluation;
  const busy = actions.busy === "evaluate";
  const withFrontier = !!loop.frontier;
  const against = loop.frontier ? `the base model and ${loop.frontier.model}` : "the base model";
  const runDone = job?.status === "succeeded";
  const finished = ev && (ev.status === "succeeded" || ev.status === "failed");

  const evaluate = (label: string, Icon: typeof Sparkles, variant: "default" | "outline" = "default") => operator && runDone && (
    <Button variant={variant} onClick={() => actions.run({ kind: "evaluate", withFrontier })} disabled={busy}>
      {busy ? <LoaderCircle className="animate-spin" /> : <Icon />} {label}
    </Button>
  );

  const frontierNote = !loop.frontier && runDone && (
    <p className="flex flex-wrap items-center gap-1 text-sm text-muted-foreground">
      <Cloud className="size-3.5" /> Add the frontier model your agent uses today to see whether this one can take its job over.
      <Button variant="link" size="sm" className="h-auto px-1" onClick={() => setDialog(true)}>Add frontier model</Button>
    </p>
  );
  const dialogEl = (
    <FrontierDialog open={dialog} onOpenChange={setDialog} current={loop.frontier}
      onSaved={() => void qc.invalidateQueries({ queryKey: ["settings"] })} />
  );

  if (!p.eval_id) {
    return (
      <>
        <StationHeader title="Evaluate"
          line={runDone ? `Ask version ${loop.versions.length || 1} and ${against} the same questions from your data.`
            : "Once the fine-tune finishes, it's asked your questions next to its base model."}
          actions={evaluate(`Evaluate against ${against}`, Sparkles)} />
        <div className="space-y-4 p-5">
          {loop.versions.length > 1 && <Versions loop={loop} />}
          <p className="max-w-[70ch] text-sm text-muted-foreground">
            {loop.frontier
              ? `Each model answers the same questions; ${loop.frontier.model} judges every answer against yours, so a correct answer in other words still counts.`
              : "Each model answers the same questions, and an answer counts when it contains the expected one. Add a frontier model to have it judge paraphrases too."}
          </p>
          {frontierNote}
          {actions.error && <p className="text-sm text-destructive">{actions.error}</p>}
        </div>
        {dialogEl}
      </>
    );
  }

  return (
    <>
      <StationHeader title="Evaluate"
        line={!ev ? <Skeleton className="h-4 w-48" />
          : ev.status === "failed" ? "The evaluation stopped before it finished."
            : finished ? `${listTargets(ev.targets.map((t) => t.label))}, on the same ${ev.results[0]?.n ?? ""} questions.`
              : "Asking your questions now…"}
        research={`#evaluate/${p.eval_id}`}
        actions={finished ? evaluate("Evaluate again", RotateCcw, "outline") : undefined} />
      <div className="space-y-5 p-5">
        {loop.versions.length > 1 && <Versions loop={loop} />}
        {ev ? <Scorecard ev={ev} /> : <Skeleton className="h-40 w-full" />}
        {finished && frontierNote}
        {actions.error && <p className="text-sm text-destructive">{actions.error}</p>}
      </div>
      {dialogEl}
    </>
  );
}

// "Your fine-tune", "Base model", "Frontier model" → "Your fine-tune, the base
// model and the frontier model": a sentence, not a list of labels.
function listTargets(labels: string[]): string {
  const words = labels.map((l, i) => (i === 0 ? l : `the ${l.charAt(0).toLowerCase()}${l.slice(1)}`));
  return words.length < 2 ? words.join("") : `${words.slice(0, -1).join(", ")} and ${words.at(-1)}`;
}
