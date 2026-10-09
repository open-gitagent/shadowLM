// Projects — your models, one per job: where each stands on the loop (data,
// fine-tune, evaluate, deploy) and its latest proof. A row opens its cockpit.
import { useEffect, useState } from "react";
import { ArrowRight, FolderKanban, Plus, Trash2 } from "lucide-react";

import { deleteProject, getDeployments, getEval, getJobs, getProjects } from "@/api";
import type { Evaluation, JobSummary, Project } from "@/api";
import { ConfirmDelete, EmptyState, ErrorState, PageHeader } from "@/components/common";
import { verdict } from "@/components/scorecard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { allows } from "@/lib/embed";
import { goalLabel } from "@/lib/recipe";
import { cn } from "@/lib/utils";

type Tone = "muted" | "primary" | "good" | "warning" | "destructive";

// Where a project stands on the loop, in the cockpit's station vocabulary,
// read off what it links to.
export function projectStage(p: Project, run?: JobSummary, ev?: Evaluation, live?: boolean): { label: string; tone: Tone } {
  if (!p.dataset_id) return { label: "Data · needs examples", tone: "muted" };
  if (!p.run_id) return { label: "Fine-tune · ready", tone: "muted" };
  if (!run || run.status === "pending" || run.status === "running") return { label: "Fine-tune · running", tone: "primary" };
  if (run.status === "failed") return { label: "Fine-tune · failed", tone: "destructive" };
  if (run.status === "stopped") return { label: "Fine-tune · stopped", tone: "muted" };
  if (live) return { label: "Deploy · live", tone: "good" };
  if (!p.eval_id) return { label: "Evaluate · ready", tone: "warning" };
  if (!ev || ev.status === "pending" || ev.status === "running") return { label: "Evaluate · running", tone: "primary" };
  if (ev.status === "failed") return { label: "Evaluate · failed", tone: "destructive" };
  const v = verdict(ev);
  return v ? { label: v.title, tone: v.tone } : { label: "Evaluated", tone: "muted" };
}

const toneClass: Record<Tone, string> = {
  muted: "text-muted-foreground",
  primary: "border-primary/25 bg-primary/10 text-primary",
  good: "border-good/30 bg-good/10 text-good",
  warning: "border-warning/30 bg-warning/10 text-warning",
  destructive: "border-destructive/25 bg-destructive/10 text-destructive",
};

export function StageBadge({ stage }: { stage: { label: string; tone: Tone } }) {
  return <Badge variant="outline" className={cn("font-normal", toneClass[stage.tone])}>{stage.label}</Badge>;
}

// The latest proof, in one line: the fine-tune's right answers against the base's.
function proof(ev?: Evaluation): string | null {
  if (!ev || ev.status !== "succeeded" || !ev.results[0]) return null;
  const [t, b] = ev.results;
  const right = (s: number, n: number) => Math.round(s * n);
  return `Answers ${right(t.score, t.n)} of ${t.n} correctly${b ? `; base gets ${right(b.score, b.n)}` : ""}`;
}

export default function Projects() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [runs, setRuns] = useState<Record<string, JobSummary>>({});
  const [evals, setEvals] = useState<Record<string, Evaluation>>({});
  const [liveRuns, setLiveRuns] = useState<Set<string>>(new Set());
  const [err, setErr] = useState<unknown>(null);
  const [doomed, setDoomed] = useState<Project | null>(null);
  const operator = allows("operator");

  useEffect(() => {
    let live = true;
    const tick = async () => {
      try {
        const [{ projects }, { jobs }, deps] = await Promise.all([
          getProjects(), getJobs(), getDeployments().catch(() => ({ deployments: [] })),
        ]);
        if (!live) return;
        setProjects(projects);
        setRuns(Object.fromEntries(jobs.map((j) => [j.job_id, j])));
        setLiveRuns(new Set(deps.deployments.map((d) => d.run_id)));
        setErr(null);
        // Only the evaluations projects link to, and only until they finish.
        const want = projects.map((p) => p.eval_id).filter((id): id is string => !!id);
        const fetched = await Promise.all(want.map((id) => getEval(id).catch(() => null)));
        if (!live) return;
        setEvals(Object.fromEntries(fetched.filter((e): e is Evaluation => !!e).map((e) => [e.eval_id, e])));
      } catch (e) {
        if (live) setErr(e);
      }
    };
    tick();
    const t = setInterval(tick, 4000);
    return () => { live = false; clearInterval(t); };
  }, []);

  async function remove(p: Project) {
    setDoomed(null);
    try {
      await deleteProject(p.project_id);
      setProjects((ps) => ps?.filter((x) => x.project_id !== p.project_id) ?? ps);
    } catch (e) {
      setErr(e);
    }
  }

  const newButton = operator && (
    <Button asChild><a href="#projects/new"><Plus /> New model</a></Button>
  );

  return (
    <>
      <PageHeader
        title="Projects"
        description="Your models, one per job. Each moves along the same loop: data, fine-tune, evaluate on your own questions, deploy. Open one to work on it with Ctrl agent."
        actions={newButton}
      />

      {err != null && projects === null && <ErrorState error={err} title="Couldn't load your projects" />}

      {projects === null && err == null && (
        <div className="space-y-2 border border-border p-4">
          {[0, 1, 2].map((i) => <Skeleton key={i} className="h-9 w-full" />)}
        </div>
      )}

      {projects?.length === 0 && (
        <div className="max-w-2xl space-y-4">
          <EmptyState icon={FolderKanban} title="No models yet">
            Tell Ctrl agent what your model should do and give it examples. It picks the base model and settings,
            fine-tunes, proves the result on your own questions against the base model, and deploys it when it's
            ready, showing you every choice along the way.
          </EmptyState>
          {operator && (
            <Button asChild size="lg"><a href="#projects/new">Make your first model <ArrowRight /></a></Button>
          )}
        </div>
      )}

      {projects && projects.length > 0 && (
        <div className="border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="hidden @3xl:table-cell">Goal</TableHead>
                <TableHead>Stage</TableHead>
                <TableHead className="hidden @2xl:table-cell">Latest evaluation</TableHead>
                <TableHead className="w-10"><span className="sr-only">Actions</span></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {projects.map((p) => {
                const run = p.run_id ? runs[p.run_id] : undefined;
                const ev = p.eval_id ? evals[p.eval_id] : undefined;
                const line = proof(ev);
                return (
                  <TableRow key={p.project_id} className="group cursor-pointer"
                    onClick={() => { window.location.hash = `#projects/${p.project_id}`; }}>
                    <TableCell>
                      <a href={`#projects/${p.project_id}`} onClick={(e) => e.stopPropagation()}
                        className="font-medium hover:underline focus-visible:underline focus-visible:outline-none">
                        {p.name}
                      </a>
                      <div className="text-xs text-muted-foreground @3xl:hidden">{goalLabel[p.goal]}</div>
                    </TableCell>
                    <TableCell className="hidden text-sm text-muted-foreground @3xl:table-cell">{goalLabel[p.goal]}</TableCell>
                    <TableCell><StageBadge stage={projectStage(p, run, ev, !!p.run_id && liveRuns.has(p.run_id))} /></TableCell>
                    <TableCell className="hidden text-sm @2xl:table-cell">
                      {line ?? <span className="text-muted-foreground">—</span>}
                    </TableCell>
                    <TableCell className="text-right">
                      {operator && (
                        <Button variant="ghost" size="icon-sm" aria-label={`Delete ${p.name}`}
                          className="text-muted-foreground opacity-0 group-hover:opacity-100 focus-visible:opacity-100 hover:text-destructive"
                          onClick={(e) => { e.stopPropagation(); setDoomed(p); }}>
                          <Trash2 />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}

      <ConfirmDelete what={doomed ? `“${doomed.name}”` : undefined} onClose={() => setDoomed(null)}
        onConfirm={() => doomed && remove(doomed)}>
        This removes the project from this list. Its dataset, fine-tune run and evaluation stay, and you can still
        find them in Research mode.
      </ConfirmDelete>
    </>
  );
}
