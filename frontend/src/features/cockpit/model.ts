// The cockpit's model of one project: every resource the loop touches, read
// through the shared data layer, and folded into four stations (data,
// fine-tune, evaluate, deploy) with a status and a one-line state each. The
// conversation and the loop map both read this; neither fetches on its own.
import { useQueries, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import {
  createDeployment, getEval, startSynth,
  type DatasetMeta, type Deployment, type Evaluation, type Health, type JobSummary,
  type Project, type StepMetric, type SynthStatus,
} from "@/api";
import { verdict } from "@/components/scorecard";
import {
  useDatasetQ, useDeploymentsQ, useEvalQ, useHealthQ, useJobsQ, useMetricsQ, useProjectQ, useSettingsQ, useSynthQ,
} from "@/lib/queries";
import { examples, methodLabel } from "@/lib/format";
import { pickRecipe, type Recipe, startProjectEval, startProjectFinetune } from "@/lib/recipe";

export type StationId = "data" | "finetune" | "evaluate" | "deploy";
export type StationStatus = "idle" | "ready" | "running" | "done" | "failed";
export const STATIONS: { id: StationId; label: string }[] = [
  { id: "data", label: "Data" },
  { id: "finetune", label: "Fine-tune" },
  { id: "evaluate", label: "Evaluate" },
  { id: "deploy", label: "Deploy" },
];

// tone: a finished evaluation's verdict, so the map itself answers "ready?"
export interface Station { id: StationId; label: string; status: StationStatus; line: string; tone?: "good" | "warning" | "muted" }

export interface Version {
  n: number;  // 1-based, oldest first
  run_id: string;
  eval_id: string | null;
  job?: JobSummary;
  evaluation?: Evaluation;
  current: boolean;
}

export interface Loop {
  project?: Project;
  dataset?: DatasetMeta;
  job?: JobSummary;  // the current version's run
  steps: StepMetric[];
  evalSteps: StepMetric[];
  evaluation?: Evaluation;
  deployment?: Deployment;
  synth?: SynthStatus;  // examples being written for it, or the last run that wrote some
  writing: boolean;     // that run is still going
  dataChanged: boolean; // the examples changed since the current version trained
  versions: Version[];
  stations: Record<StationId, Station>;
  next: StationId;  // where attention belongs now
  health?: Health;
  frontier: { base_url: string; model: string } | null;
  ctrl: { model: string } | null;  // Claude answers typed messages, or null: the rules do
  loading: boolean;
  missing: boolean;  // the project id doesn't exist
}

const moving = (s?: string) => s === "pending" || s === "running";

export function useLoop(projectId?: string): Loop {
  const projectQ = useProjectQ(projectId);
  const project = projectQ.data;
  const datasetQ = useDatasetQ(project?.dataset_id);
  const jobsQ = useJobsQ();
  const job = jobsQ.data?.find((j) => j.job_id === project?.run_id);
  const metricsQ = useMetricsQ(project?.run_id, moving(job?.status));
  const evalQ = useEvalQ(project?.eval_id);
  const deploymentsQ = useDeploymentsQ();
  const health = useHealthQ().data;
  const settings = useSettingsQ().data;
  const frontier = settings?.frontier ?? null;
  const ctrl = settings?.ctrl ?? null;
  const synthQ = useSynthQ(project?.synth_id);
  const synth = synthQ.data;
  const writing = synth?.status === "running";
  // refetch the project when a run lands: the server just gave it the examples
  const landed = useRef(synth?.status);
  const qc = useQueryClient();
  useEffect(() => {
    if (landed.current === "running" && synth && synth.status !== "running" && project) {
      void qc.invalidateQueries({ queryKey: ["project", project.project_id] });
      void qc.invalidateQueries({ queryKey: ["datasets"] });
    }
    landed.current = synth?.status;
  }, [synth?.status]);  // eslint-disable-line react-hooks/exhaustive-deps

  const history = project?.history ?? [];
  const pastEvals = useQueries({
    queries: history.map((h) => ({
      queryKey: ["eval", h.eval_id], queryFn: () => getEval(h.eval_id!), enabled: !!h.eval_id,
    })),
  });

  const versions: Version[] = [
    ...history.map((h, i) => ({
      n: i + 1, run_id: h.run_id, eval_id: h.eval_id, current: false,
      job: jobsQ.data?.find((j) => j.job_id === h.run_id), evaluation: pastEvals[i]?.data,
    })),
    ...(project?.run_id ? [{
      n: history.length + 1, run_id: project.run_id, eval_id: project.eval_id, current: true,
      job, evaluation: evalQ.data,
    }] : []),
  ];

  const dataset = datasetQ.data;
  const evaluation = evalQ.data;
  const deployment = deploymentsQ.data?.find((d) => d.run_id === project?.run_id);
  const steps = metricsQ.data?.steps ?? [];

  const stations = {
    data: dataStation(project, dataset, synth),
    finetune: finetuneStation(project, job, steps.length),
    evaluate: evaluateStation(project, job, evaluation),
    deploy: deployStation(job, deployment),
  };
  // where attention belongs: the first unfinished station, except that an
  // evaluation saying "not ready" keeps attention on its scorecard (the
  // copilot is proposing the next version there, not a deploy)
  const ready = evaluation ? verdict(evaluation)?.tone === "good" : false;
  const next: StationId =
    stations.data.status !== "done" || writing ? "data" :
    stations.finetune.status !== "done" ? "finetune" :
    stations.evaluate.status !== "done" || (!ready && !deployment) ? "evaluate" : "deploy";

  return {
    project, dataset, job, steps, evalSteps: metricsQ.data?.evals ?? [], evaluation, deployment,
    synth, writing,
    dataChanged: !!project?.run_id && !!project.run_dataset_id && project.run_dataset_id !== project.dataset_id,
    versions, stations, next, health, frontier, ctrl,
    loading: !!projectId && projectQ.isLoading,
    missing: !!projectId && projectQ.isError,
  };
}

export const synthPhase: Record<string, string> = {
  starting: "Starting", planning: "Planning", generating: "Writing", judging: "Checking", kept: "Collecting",
};

function dataStation(p?: Project, d?: DatasetMeta, s?: SynthStatus): Station {
  const base = { id: "data" as const, label: "Data" };
  if (s?.status === "running") {
    return { ...base, status: "running",
             line: `${synthPhase[s.phase ?? "starting"] ?? "Writing"} examples · ${s.kept} of ${s.requested} kept` };
  }
  if (!p?.dataset_id && s && s.status !== "succeeded") return { ...base, status: "failed", line: "Writing examples stopped" };
  if (!p?.dataset_id) return { ...base, status: "ready", line: "No examples yet" };
  if (!d) return { ...base, status: "running", line: "Loading examples…" };
  return { ...base, status: "done", line: d.rows == null ? "Examples" : examples(d.rows) };
}

function finetuneStation(p: Project | undefined, j: JobSummary | undefined, stepsDone: number): Station {
  const base = { id: "finetune" as const, label: "Fine-tune" };
  if (!p?.run_id) return { ...base, status: p?.dataset_id ? "ready" : "idle", line: "Not started" };
  if (!j) return { ...base, status: "running", line: "Starting…" };
  if (j.status === "pending") return { ...base, status: "running", line: "Queued for the GPU" };
  if (j.status === "running") return { ...base, status: "running", line: stepsDone ? `${stepsDone} steps done` : "Loading the base model" };
  if (j.status === "succeeded") return { ...base, status: "done", line: `${j.steps} steps · ${methodLabel(j.method)}` };
  return { ...base, status: "failed", line: j.status === "stopped" ? "Stopped" : "Failed" };
}

function evaluateStation(p: Project | undefined, j: JobSummary | undefined, ev: Evaluation | undefined): Station {
  const base = { id: "evaluate" as const, label: "Evaluate" };
  if (!p?.eval_id) return { ...base, status: j?.status === "succeeded" ? "ready" : "idle", line: "Not evaluated" };
  if (!ev || moving(ev.status)) return { ...base, status: "running", line: "Asking your questions…" };
  if (ev.status === "failed") return { ...base, status: "failed", line: "Evaluation stopped" };
  const v = verdict(ev);
  const r = ev.results[0];
  return { ...base, status: "done", tone: v?.tone, line: `${Math.round(r.score * r.n)} of ${r.n} correct` };
}

function deployStation(j: JobSummary | undefined, d: Deployment | undefined): Station {
  const base = { id: "deploy" as const, label: "Deploy" };
  if (d) return { ...base, status: "done", line: `Live · ${d.requests} requests` };
  return { ...base, status: j?.status === "succeeded" ? "ready" : "idle", line: "Not deployed" };
}

// ---- actions -------------------------------------------------------------------
// What the copilot (or a station) can do to a project. Each one says what it
// started; the data layer refetches so the loop map updates itself.
export type Action =
  | { kind: "finetune"; recipe: Recipe }
  | { kind: "evaluate"; withFrontier: boolean }
  | { kind: "deploy"; name: string }
  | { kind: "synthesize"; n: number };

export function useActions(loop: Loop) {
  const qc = useQueryClient();
  const [busy, setBusy] = useState<Action["kind"] | null>(null);
  const [error, setError] = useState("");

  async function run(a: Action): Promise<string | null> {
    const p = loop.project;
    if (!p) return null;
    setBusy(a.kind);
    setError("");
    try {
      if (a.kind === "finetune") {
        if (!p.dataset_id) throw new Error("add examples before fine-tuning");
        await startProjectFinetune(p, p.dataset_id, a.recipe);
      } else if (a.kind === "synthesize") {
        if (!loop.frontier) throw new Error("add a frontier model to write examples with");
        if (!p.dataset_id) throw new Error("it needs a few examples to write more like them");
        // the examples it has stay in; the server makes the result the project's data
        await startSynth({
          name: `${p.name} · more examples`, n: a.n, dataset_id: p.dataset_id, include_seed: true,
          project_id: p.project_id, teacher: { kind: "frontier", model: loop.frontier.model },
        });
      } else if (a.kind === "evaluate") {
        if (!loop.job) throw new Error("fine-tune before evaluating");
        await startProjectEval(p, loop.job.base_model, a.withFrontier);
      } else {
        if (!p.run_id) throw new Error("fine-tune before deploying");
        const { key } = await createDeployment({ name: a.name, run_id: p.run_id, project_id: p.project_id });
        await qc.invalidateQueries({ queryKey: ["deployments"] });
        return key;
      }
      await qc.invalidateQueries({ queryKey: ["project", p.project_id] });
      await qc.invalidateQueries({ queryKey: ["jobs"] });
      return null;
    } catch (e) {
      setError((e as Error).message);
      return null;
    } finally {
      setBusy(null);
    }
  }

  return { run, busy, error, clearError: () => setError("") };
}

// The recipe for this project's first fine-tune, from its goal and data.
export function firstRecipe(loop: Loop): Recipe | null {
  if (!loop.project || !loop.dataset) return null;
  return pickRecipe(loop.project.goal, loop.dataset.rows ?? 0, loop.health ?? null);
}

// The recipe for the next version, from how the current one scored: not
// learned → train longer; learned but behind the frontier, or "close" → a
// larger base on a GPU box, else longer. Every change is said in `why`.
export function nextRecipe(loop: Loop): Recipe | null {
  const first = firstRecipe(loop);
  if (!first || !loop.job) return first;
  if (loop.dataChanged) {
    const rows = loop.dataset?.rows ?? 0;
    return { ...first, base_model: loop.job.base_model,
             why: [`All ${examples(rows)}, the new ones included: v${loop.versions.length} trained on fewer`, ...first.why.filter((w) => !w.startsWith("Qwen") && !w.includes(loop.job!.base_model))],
             cli: first.cli.replace(first.base_model, loop.job.base_model) };
  }
  const prevSteps = loop.job.steps || Number(first.config.max_steps) || 100;
  const v = loop.evaluation ? verdict(loop.evaluation) : null;
  const onGpu = (loop.health?.gpus ?? 0) > 0;
  const tuned = loop.evaluation?.results[0];
  const learned = !!tuned && tuned.score >= 0.6;
  if (learned && onGpu && v?.tone !== "good" && /1\.5B/.test(loop.job.base_model)) {
    const base = loop.job.base_model.replace("1.5B", "3B");
    return { ...first, base_model: base, config: { ...first.config, max_steps: prevSteps },
             why: [`A larger base, ${base.split("/").pop()}: v${loop.versions.length} learned the pattern but trailed`, ...first.why.slice(0, 1), `${prevSteps} steps, as before`],
             cli: first.cli.replace(first.base_model, base) };
  }
  const steps = Math.min(prevSteps * 2, 1200);
  return { ...first, base_model: loop.job.base_model, config: { ...first.config, max_steps: steps },
           why: [`Twice the training, ${steps} steps: v${loop.versions.length} hadn't learned your examples yet`, ...first.why.slice(0, 2)],
           cli: first.cli.replace(/--max-steps \d+/, `--max-steps ${steps}`) };
}

// How many examples to ask for when writing more: enough to matter, few enough
// to read (three for each one it has, 30 to 150).
export const moreExamples = (rows: number) => Math.max(30, Math.min(150, rows * 3));
