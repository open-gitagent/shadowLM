// Fine-tune: the current version's run, its live loss curve, and (in Research)
// its settings, checkpoints and console. Before the first run, the plan the
// studio picked, with a Start that runs exactly that.
import { ChevronDown, LoaderCircle, Play, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { ChartLegend, LossChart } from "@/components/charts";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { allows } from "@/lib/embed";
import { useMode } from "@/lib/mode";
import { useCheckpointsQ, useLogsQ } from "@/lib/queries";
import type { Recipe } from "@/lib/recipe";
import { cn } from "@/lib/utils";

import type { CockpitActions } from "../Cockpit";
import { firstRecipe, type Loop, nextRecipe } from "../model";
import { CodeBlock, StationHeader, useCopy } from "./shared";

function Plan({ recipe }: { recipe: Recipe }) {
  const { copied, copy } = useCopy();
  return (
    <div className="space-y-3">
      <dl className="grid gap-px overflow-hidden border border-border bg-border text-sm @2xl:grid-cols-3">
        {[
          { k: "Base model", v: recipe.base_model.split("/").pop() },
          { k: "Method", v: recipe.method },
          { k: "Steps", v: String(recipe.config.max_steps ?? "default") },
        ].map((r) => (
          <div key={r.k} className="bg-card px-4 py-3">
            <dt className="text-xs text-muted-foreground">{r.k}</dt>
            <dd className="mt-0.5 font-mono text-xs">{r.v}</dd>
          </div>
        ))}
      </dl>
      <ul className="space-y-1 text-sm text-muted-foreground">
        {recipe.why.map((w) => <li key={w}>· {w}</li>)}
      </ul>
      <CodeBlock label="The same run from your shell" code={recipe.cli} copied={copied === "cli"} onCopy={() => copy(recipe.cli, "cli")} />
    </div>
  );
}

function Console({ runId, live }: { runId: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  const logs = useLogsQ(open ? runId : null, live);
  const pre = useRef<HTMLPreElement>(null);
  useEffect(() => { if (pre.current) pre.current.scrollTop = pre.current.scrollHeight; }, [logs.data]);
  return (
    <div>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="inline-flex items-center gap-1.5 rounded-sm text-sm text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:outline-none">
        <ChevronDown className={cn("size-3.5 transition-transform duration-200", !open && "-rotate-90")} /> Training console
      </button>
      {open && (
        logs.isLoading ? <Skeleton className="mt-2 h-48 w-full" /> : (
          <pre ref={pre} className="mt-2 max-h-72 overflow-auto bg-ink p-4 font-mono text-[11px] leading-[1.4] whitespace-pre text-bone/90 scrollbar-thin">
            {(logs.data?.logs ?? []).join("\n") || "Nothing captured for this run yet."}
          </pre>
        )
      )}
    </div>
  );
}

export function FinetuneStation({ loop, actions }: { loop: Loop; actions: CockpitActions }) {
  const mode = useMode();
  const research = mode === "research";
  const operator = allows("operator");
  const p = loop.project!;
  const job = loop.job;
  const checkpoints = useCheckpointsQ(research && job?.status === "succeeded" ? job.job_id : null);
  const busy = actions.busy === "finetune";
  const n = loop.versions.length || 1;

  // not started: the plan, and Start
  if (!p.run_id) {
    const recipe = firstRecipe(loop);
    return (
      <>
        <StationHeader title="Fine-tune"
          line={p.dataset_id ? "Here's the plan for the first version." : "Starts once the model has examples."}
          actions={recipe && operator && (
            <Button onClick={() => actions.run({ kind: "finetune", recipe })} disabled={busy}>
              {busy ? <LoaderCircle className="animate-spin" /> : <Play />} Start fine-tuning
            </Button>
          )} />
        <div className="space-y-3 p-5">
          {recipe ? <Plan recipe={recipe} />
            : p.dataset_id ? <Skeleton className="h-32 w-full" />
              : <p className="text-sm text-muted-foreground">Add examples on the Data station first.</p>}
          {actions.error && <p className="text-sm text-destructive">{actions.error}</p>}
        </div>
      </>
    );
  }

  if (!job) {
    return (
      <>
        <StationHeader title="Fine-tune" line={<Skeleton className="h-4 w-48" />} />
        <div className="p-5"><Skeleton className="h-60 w-full" /></div>
      </>
    );
  }

  const running = job.status === "pending" || job.status === "running";
  const failed = job.status === "failed" || job.status === "stopped";
  const last = loop.steps.at(-1);
  const retry = nextRecipe(loop) ?? firstRecipe(loop);
  const line = job.status === "pending" ? "Queued. It starts as soon as the GPU is free."
    : job.status === "running" ? (loop.steps.length ? `Version ${n}: ${loop.steps.length} steps done${last ? `, loss ${last.loss.toFixed(3)}` : ""}. You can leave; it keeps going.` : `Version ${n}: loading the base model onto the GPU…`)
      : failed ? `Version ${n} ${job.status === "stopped" ? "was stopped before it finished" : "stopped with an error"}.`
        : `Version ${n}: ${job.steps} steps${job.final_loss != null ? ` · final loss ${job.final_loss.toFixed(3)}` : ""}`;

  return (
    <>
      <StationHeader title="Fine-tune" line={line} research={`#runs/${job.job_id}`}
        actions={<>
          {failed && retry && operator && (
            <Button onClick={() => actions.run({ kind: "finetune", recipe: retry })} disabled={busy}>
              {busy ? <LoaderCircle className="animate-spin" /> : <RotateCcw />} Try again
            </Button>
          )}
        </>} />
      <div className="space-y-5 p-5">
        <dl className="grid gap-px overflow-hidden border border-border bg-border text-sm @2xl:grid-cols-4">
          {[
            { k: "Base model", v: job.base_model.split("/").pop() ?? job.base_model },
            { k: "Method", v: job.method ?? "?" },
            { k: "Steps", v: String(running ? loop.steps.length : job.steps) },
            { k: "Loss", v: running ? (last ? last.loss.toFixed(3) : "—") : job.final_loss != null ? job.final_loss.toFixed(3) : "—" },
          ].map((r) => (
            <div key={r.k} className="bg-card px-4 py-3">
              <dt className="text-xs text-muted-foreground">{r.k}</dt>
              <dd className="mt-0.5 font-mono text-xs tabular-nums">{r.v}</dd>
            </div>
          ))}
        </dl>

        {failed && job.error && (
          <pre className="max-h-40 overflow-auto border border-destructive/30 bg-destructive/5 p-3 font-mono text-xs whitespace-pre-wrap text-destructive">{job.error}</pre>
        )}

        {(running || loop.steps.length > 0) && (
          <div className="border border-border">
            <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
              <span className="text-sm font-medium">Loss</span>
              <ChartLegend />
            </div>
            <div className="px-2 py-3">
              {job.status === "pending" && loop.steps.length === 0
                ? <Skeleton className="mx-2 h-52" />
                : <LossChart steps={loop.steps} evals={loop.evalSteps} height={220} />}
            </div>
          </div>
        )}

        {research && (
          <div className="space-y-3 text-sm">
            <p className="text-xs text-muted-foreground">
              Run <span className="font-mono text-foreground">{job.job_id}</span> · {job.base_model} · {job.method ?? "?"}
            </p>
            {job.status === "succeeded" && (
              <div>
                <p className="mb-1.5 text-xs text-muted-foreground">Checkpoints</p>
                {checkpoints.isLoading ? <Skeleton className="h-7 w-64" /> : (checkpoints.data?.length ?? 0) > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {checkpoints.data!.map((c) => (
                      <span key={c.path} className={cn("rounded-full border px-2 py-0.5 font-mono text-[11px]",
                        c.final ? "border-primary/30 bg-primary/10 text-primary" : "border-border text-muted-foreground")}>{c.label}</span>
                    ))}
                  </div>
                ) : <p className="text-xs text-muted-foreground">Only the final weights were saved.</p>}
              </div>
            )}
            <Console runId={job.job_id} live={running} />
          </div>
        )}
        {actions.error && <p className="text-sm text-destructive">{actions.error}</p>}
      </div>
    </>
  );
}
