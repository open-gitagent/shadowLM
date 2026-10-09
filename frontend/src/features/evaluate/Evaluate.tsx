// Evaluate — Research mode's view of the evaluation engine: score fine-tunes
// and base models on the same questions, side by side. Past evaluations on the
// left, the selected one (or a new one) on the right. Business mode's Evaluate
// step runs the same evaluations through lib/recipe.ts.
import { useEffect, useMemo, useState } from "react";
import { ClipboardCheck, Cloud, LoaderCircle, Plus, Trash2, X } from "lucide-react";

import {
  deleteEval, getCheckpoints, getDatasets, getEval, getEvals, getJobs, getModels, startEval,
  type Checkpoint, type DatasetMeta, type EvalMetric, type EvalTarget, type Evaluation, type FrontierInfo,
  type JobSummary,
} from "@/api";
import { ConfirmDelete, EmptyState, ErrorState, Field, Mono, PageHeader, StatusBadge } from "@/components/common";
import { FrontierDialog, useFrontier } from "@/components/frontier-settings";
import { Scorecard } from "@/components/scorecard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { allows, needs } from "@/lib/embed";
import { cn } from "@/lib/utils";

const MAX_TARGETS = 4;
const handoffKey = "of.evaluate.run";  // set by Runs' "Evaluate" button

const metrics: { m: EvalMetric; label: string; hint: string }[] = [
  { m: "contains", label: "Contains", hint: "Correct when the expected answer appears anywhere in the output." },
  { m: "exact", label: "Exact", hint: "Correct only when the output is the expected answer (case and spacing ignored)." },
  { m: "judge", label: "Judge", hint: "Your frontier model scores each answer from 0 to 1 against the expected one, so a correct paraphrase counts." },
];
const metricLabel = (m: EvalMetric) => metrics.find((x) => x.m === m)?.label ?? m;

// A target being edited: a base model by id, a fine-tune (a run) at a
// checkpoint, or the frontier model set up in the studio.
type Draft =
  | { kind: "base"; model: string }
  | { kind: "tuned"; run: string; checkpoint: number | null }
  | { kind: "frontier" };
type Target = EvalTarget | { kind: "frontier"; label: string };

const ago = (t: number) => {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
};
const short = (id: string) => id.split("/").pop() ?? id;
const runLabel = (j?: JobSummary, id?: string) => j?.name?.trim() || (id ?? j?.job_id ?? "").slice(0, 10);

export default function Evaluate({ initialId }: { initialId?: string }) {
  const [evals, setEvals] = useState<Evaluation[] | null>(null);
  const [listErr, setListErr] = useState<unknown>(null);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);  // null until loaded
  const [catalog, setCatalog] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);
  const [preset, setPreset] = useState<string | null>(null);  // a run handed over from Runs

  const selected = initialId ?? null;
  const select = (id: string | null) => { window.location.hash = id ? `#evaluate/${id}` : "#evaluate"; };

  const refresh = () =>
    getEvals().then((r) => { setEvals(r.evals); setListErr(null); }).catch((e) => setListErr(e));

  useEffect(() => {
    refresh();
    getDatasets().then((r) => setDatasets(r.datasets)).catch(() => {});
    getJobs().then((r) => setJobs(r.jobs)).catch(() => setJobs([]));
    getModels().then((m) => setCatalog([...new Set([...m.recent, ...m.catalog.map((c) => c.id)])])).catch(() => {});
    const handed = sessionStorage.getItem(handoffKey);
    if (handed) {
      sessionStorage.removeItem(handoffKey);
      setPreset(handed);
      setCreating(true);
    }
  }, []);

  // While anything is still scoring, keep the list's numbers current.
  const anyRunning = (evals ?? []).some((e) => e.status === "pending" || e.status === "running");
  useEffect(() => {
    if (!anyRunning) return;
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [anyRunning]);

  // With nothing to show yet, the form is the page.
  const showForm = creating || (evals !== null && evals.length === 0 && !selected);
  const dsName = (id: string) => datasets.find((d) => d.dataset_id === id)?.name ?? id;

  return (
    <>
      <PageHeader
        title="Evaluate"
        description="Score fine-tunes and base models on the same questions, and read every answer side by side."
        actions={!showForm && allows("operator") && (
          <Button onClick={() => setCreating(true)}><Plus /> New evaluation</Button>
        )}
      />

      <div className="grid min-h-0 flex-1 gap-5 @4xl:grid-cols-[300px_minmax(0,1fr)]">
        {/* past evaluations */}
        <section aria-label="Evaluations" className="min-w-0">
          {listErr ? <ErrorState title="Couldn't load evaluations" error={listErr} />
            : evals === null ? (
              <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-16 w-full" />)}</div>
            ) : evals.length === 0 ? (
              <EmptyState icon={ClipboardCheck} title="No evaluations yet">
                An evaluation asks each model the same questions from a dataset and scores the answers. Pick a fine-tune and its base model to see what training changed.
              </EmptyState>
            ) : (
              <ul className="divide-y divide-border border border-border">
                {evals.map((e) => (
                  <li key={e.eval_id}>
                    <button type="button" onClick={() => { setCreating(false); select(e.eval_id); }}
                      aria-current={selected === e.eval_id ? "true" : undefined}
                      className={cn("w-full px-3.5 py-3 text-left transition-colors outline-none focus-visible:bg-subtle",
                        selected === e.eval_id && !creating ? "bg-primary/5" : "hover:bg-subtle")}>
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-sm font-medium">{e.name || dsName(e.dataset_id)}</span>
                        {e.status !== "succeeded" && <StatusBadge status={e.status} />}
                      </div>
                      <div className="mt-1 space-y-0.5">
                        {e.targets.map((t, i) => {
                          const r = e.results[i];
                          return (
                            <div key={i} className="flex items-center justify-between gap-2 text-xs">
                              <span className="truncate text-muted-foreground">{t.label}</span>
                              <span className="shrink-0 tabular-nums">{r ? `${Math.round(r.score * r.n)} of ${r.n}` : "—"}</span>
                            </div>
                          );
                        })}
                      </div>
                      <div className="mt-1.5 text-[11px] text-muted-foreground">{metricLabel(e.metric)} · {ago(e.created)}</div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
        </section>

        {/* the selected evaluation, or a new one */}
        <section className="min-w-0">
          {showForm ? (
            allows("operator") ? (
              <NewEvaluation
                datasets={datasets} jobs={jobs} catalog={catalog} preset={preset}
                onCancel={evals && evals.length > 0 ? () => { setCreating(false); setPreset(null); } : undefined}
                onStarted={(ev) => { setCreating(false); setPreset(null); refresh(); select(ev.eval_id); }}
              />
            ) : <EmptyState icon={ClipboardCheck} title="Starting an evaluation needs more access">{needs("operator")}</EmptyState>
          ) : selected ? (
            <Detail key={selected} id={selected} dsName={dsName} onDeleted={() => { refresh(); select(null); }} />
          ) : evals && evals.length > 0 ? (
            <EmptyState icon={ClipboardCheck} title="Pick an evaluation">
              Choose one on the left to read its scores and every answer, or start a new one.
            </EmptyState>
          ) : null}
        </section>
      </div>
    </>
  );
}

// ---- one evaluation, polled while it scores ---------------------------------
function Detail({ id, dsName, onDeleted }: { id: string; dsName: (id: string) => string; onDeleted: () => void }) {
  const [ev, setEv] = useState<Evaluation | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [confirm, setConfirm] = useState(false);
  const [delErr, setDelErr] = useState("");

  useEffect(() => {
    let live = true;
    let t: ReturnType<typeof setTimeout>;
    const load = () =>
      getEval(id).then((e) => {
        if (!live) return;
        setEv(e); setErr(null);
        if (e.status === "pending" || e.status === "running") t = setTimeout(load, 2000);
      }).catch((e2) => live && setErr(e2));
    load();
    return () => { live = false; clearTimeout(t); };
  }, [id]);

  if (err) return <ErrorState title="Couldn't load this evaluation" error={err} />;
  if (!ev) return <div className="space-y-3"><Skeleton className="h-20 w-full" /><Skeleton className="h-40 w-full" /></div>;

  const finished = ev.status === "succeeded" || ev.status === "failed";
  const sdk = `res = slm.evaluate(model, ds, metric="${ev.metric}"${ev.metric === "judge" ? ", judge=frontier" : ""}${ev.sample ? `, sample=${ev.sample}` : ""})`;

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-base font-semibold tracking-[-0.01em]">{ev.name || dsName(ev.dataset_id)}</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {ev.results[0]?.n ?? ev.sample ?? "—"} questions from <span className="text-foreground">{dsName(ev.dataset_id)}</span>, scored by <span className="text-foreground">{metricLabel(ev.metric)}</span>
          </p>
        </div>
        {finished && allows("operator") && (
          <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => setConfirm(true)}>
            <Trash2 /> Delete
          </Button>
        )}
      </header>
      {delErr && <p className="text-sm text-destructive">{delErr}</p>}

      <Scorecard ev={ev} />

      <div className="space-y-1.5 border-t border-border pt-4 text-xs text-muted-foreground">
        <p>
          {ev.targets.map((t, i) => (
            <span key={i}>
              {i > 0 && " · "}
              <span className="text-foreground">{t.label}</span>: <Mono className="text-[11px]">{t.model}</Mono>
              {t.kind === "frontier" ? " (frontier)" : null}
              {t.adapter ? <> + run <Mono className="text-[11px]">{t.adapter}</Mono></> : null}
              {t.checkpoint != null ? ` at step ${t.checkpoint}` : t.adapter ? " (final)" : ""}
            </span>
          ))}
        </p>
        <p>From the SDK: <code className="font-mono text-[11px] text-foreground">{sdk}</code></p>
      </div>

      <ConfirmDelete what={confirm ? (ev.name || "this evaluation") : undefined} onClose={() => setConfirm(false)}
        onConfirm={() => {
          setConfirm(false);
          deleteEval(ev.eval_id).then(onDeleted).catch((e) => setDelErr((e as Error).message));
        }}>
        Its scores and answers are removed. The models and the dataset are not touched.
      </ConfirmDelete>
    </div>
  );
}

// ---- a new evaluation ---------------------------------------------------------
function NewEvaluation({ datasets, jobs, catalog, preset, onCancel, onStarted }: {
  datasets: DatasetMeta[]; jobs: JobSummary[] | null; catalog: string[]; preset: string | null;
  onCancel?: () => void; onStarted: (ev: Evaluation) => void;
}) {
  const runs = useMemo(() => (jobs ?? []).filter((j) => j.status === "succeeded"), [jobs]);
  const [dataset, setDataset] = useState("");
  const [metric, setMetric] = useState<EvalMetric>("contains");
  const [sample, setSample] = useState("50");
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const { frontier, refresh: refreshFrontier } = useFrontier();
  const [frontierDialog, setFrontierDialog] = useState(false);

  // Without a frontier model there is no judge and no frontier column.
  useEffect(() => {
    if (frontier) return;
    setMetric((m) => (m === "judge" ? "contains" : m));
    setDrafts((ds) => (ds.some((d) => d.kind === "frontier") ? ds.filter((d) => d.kind !== "frontier") : ds));
  }, [frontier]);

  // Arriving from a run: that run, then its base model, so the comparison is
  // "what did training change".
  useEffect(() => {
    if (drafts.length || jobs === null) return;  // wait for the runs, so a handed-over one isn't missed
    const r = (preset ? runs.find((j) => j.job_id === preset) : undefined) ?? runs[0];
    if (r) setDrafts([{ kind: "tuned", run: r.job_id, checkpoint: null }, { kind: "base", model: r.base_model }]);
    else setDrafts([{ kind: "base", model: catalog[0] ?? "" }]);
  }, [preset, runs, jobs, catalog, drafts.length]);

  useEffect(() => { if (!dataset && datasets[0]) setDataset(datasets[0].dataset_id); }, [datasets, dataset]);

  const set = (i: number, d: Draft) => setDrafts((ds) => ds.map((x, j) => (j === i ? d : x)));
  const targets: (Target | null)[] = drafts.map((d) => {
    if (d.kind === "frontier") return frontier ? { kind: "frontier", label: "Frontier model" } : null;
    if (d.kind === "base") return d.model.trim() ? { label: short(d.model.trim()), model: d.model.trim() } : null;
    const j = runs.find((r) => r.job_id === d.run);
    return j ? { label: runLabel(j) + (d.checkpoint != null ? ` @${d.checkpoint}` : ""), model: j.base_model, adapter: j.job_id, checkpoint: d.checkpoint } : null;
  });
  const n = parseInt(sample, 10);
  const valid = !!dataset && targets.length > 0 && targets.every(Boolean) && n >= 1 && (metric !== "judge" || !!frontier);

  async function start() {
    if (!valid || busy) return;
    setBusy(true); setErr("");
    try {
      onStarted(await startEval({ dataset_id: dataset, metric, sample: n, targets: targets as Target[] }));
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6 border border-border bg-card p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold tracking-[-0.01em]">New evaluation</h2>
          <p className="mt-1 text-sm text-muted-foreground">Each model answers the same questions; every answer is scored and kept.</p>
        </div>
        {onCancel && <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={onCancel}><X /></Button>}
      </div>

      <div className="grid gap-4 @2xl:grid-cols-[minmax(0,1fr)_140px]">
        <Field label="Questions from" htmlFor="ev-ds"
               hint={datasets.length ? "Its eval split when it has one, otherwise its rows." : <>No datasets yet. <a className="text-primary hover:underline" href="#datasets">Add one</a> first.</>}>
          <select id="ev-ds" value={dataset} onChange={(e) => setDataset(e.target.value)} disabled={!datasets.length}>
            {datasets.map((d) => <option key={d.dataset_id} value={d.dataset_id}>{d.name}{d.rows != null ? ` · ${d.rows} rows` : ""}</option>)}
          </select>
        </Field>
        <Field label="Up to" htmlFor="ev-n" hint="questions">
          <Input id="ev-n" type="number" min={1} value={sample} onChange={(e) => setSample(e.target.value)} aria-invalid={!(n >= 1)} />
        </Field>
      </div>

      <fieldset className="space-y-2">
        <legend className="mb-1.5 text-xs text-muted-foreground">Scoring</legend>
        <div className="grid gap-2 @2xl:grid-cols-3">
          {metrics.map((x) => {
            const off = x.m === "judge" && !frontier;
            return (
              <label key={x.m} className={cn("flex gap-2.5 border px-3.5 py-3 transition-colors",
                off ? "cursor-not-allowed border-border opacity-60"
                  : metric === x.m ? "cursor-pointer border-primary/40 bg-primary/5" : "cursor-pointer border-border hover:bg-subtle")}>
                <input type="radio" name="ev-metric" className="mt-0.5" checked={metric === x.m} disabled={off} onChange={() => setMetric(x.m)} />
                <span>
                  <span className="block text-sm font-medium">{x.label}</span>
                  <span className="block text-xs text-muted-foreground">{x.hint}</span>
                </span>
              </label>
            );
          })}
        </div>
        {!frontier && (
          <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
            Judge scoring and a frontier column need your frontier model.
            <Button variant="link" size="xs" className="h-auto px-0" onClick={() => setFrontierDialog(true)}>
              <Cloud /> Set up a frontier model
            </Button>
          </p>
        )}
      </fieldset>

      <fieldset className="space-y-2">
        <legend className="mb-1.5 text-xs text-muted-foreground">Models, in the order the scorecard shows them</legend>
        <datalist id="ev-models">{catalog.map((id) => <option key={id} value={id} />)}</datalist>
        {drafts.map((d, i) => (
          <TargetRow key={i} draft={d} runs={runs} frontier={frontier} onChange={(nd) => set(i, nd)}
            onRemove={drafts.length > 1 ? () => setDrafts((ds) => ds.filter((_, j) => j !== i)) : undefined} />
        ))}
        {drafts.length < MAX_TARGETS && (
          <Button variant="outline" size="sm" onClick={() => setDrafts((ds) => [...ds, runs[0] ? { kind: "tuned", run: runs[0].job_id, checkpoint: null } : { kind: "base", model: "" }])}>
            <Plus /> Add a model
          </Button>
        )}
      </fieldset>

      {err && <p className="text-sm text-destructive">{err}</p>}
      <div className="flex items-center justify-end gap-2 border-t border-border pt-4">
        {onCancel && <Button variant="ghost" onClick={onCancel}>Cancel</Button>}
        <Button onClick={start} disabled={!valid || busy}>
          {busy ? <LoaderCircle className="animate-spin" /> : <ClipboardCheck />}
          {busy ? "Starting…" : "Start evaluation"}
        </Button>
      </div>
      <FrontierDialog open={frontierDialog} onOpenChange={setFrontierDialog} current={frontier} onSaved={() => refreshFrontier()} />
    </div>
  );
}

function TargetRow({ draft, runs, frontier, onChange, onRemove }: {
  draft: Draft; runs: JobSummary[]; frontier: FrontierInfo | null; onChange: (d: Draft) => void; onRemove?: () => void;
}) {
  const [ckpts, setCkpts] = useState<Checkpoint[]>([]);
  const run = draft.kind === "tuned" ? draft.run : null;
  useEffect(() => {
    setCkpts([]);
    if (!run) return;
    getCheckpoints(run).then((r) => setCkpts(r.checkpoints.filter((c) => !c.final))).catch(() => {});
  }, [run]);

  return (
    <div className="flex flex-wrap items-center gap-2 border border-border px-3 py-2.5">
      <div role="radiogroup" aria-label="Kind of model"
        className={cn("grid shrink-0 gap-0.5 rounded-lg bg-surface-2/70 p-0.5", frontier ? "grid-cols-3" : "grid-cols-2")}>
        {(frontier ? (["tuned", "base", "frontier"] as const) : (["tuned", "base"] as const)).map((k) => (
          <button key={k} type="button" role="radio" aria-checked={draft.kind === k}
            disabled={k === "tuned" && !runs.length}
            onClick={() => draft.kind !== k && onChange(
              k === "base" ? { kind: "base", model: runs.find((r) => r.job_id === run)?.base_model ?? "" }
              : k === "frontier" ? { kind: "frontier" }
              : { kind: "tuned", run: runs[0].job_id, checkpoint: null })}
            className={cn("h-7 rounded-md px-2.5 text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:opacity-50",
              draft.kind === k ? "bg-card text-foreground shadow-paper" : "text-muted-foreground hover:text-foreground")}>
            {k === "tuned" ? "Fine-tune" : k === "base" ? "Base model" : "Frontier"}
          </button>
        ))}
      </div>

      {draft.kind === "frontier" ? (
        <div className="flex min-w-48 flex-1 items-center gap-2 text-sm">
          <Cloud className="size-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />
          <span className="truncate font-mono">{frontier?.model ?? "No frontier model set up"}</span>
          <span className="truncate text-xs text-muted-foreground">{frontier?.base_url}</span>
        </div>
      ) : draft.kind === "base" ? (
        <Input list="ev-models" value={draft.model} placeholder="org/model, any Hugging Face id" aria-label="Base model"
               className="min-w-48 flex-1 font-mono" aria-invalid={!draft.model.trim()}
               onChange={(e) => onChange({ kind: "base", model: e.target.value })} />
      ) : (
        <>
          <select aria-label="Fine-tune" className="min-w-48 flex-1" value={draft.run}
                  onChange={(e) => onChange({ kind: "tuned", run: e.target.value, checkpoint: null })}>
            {runs.map((j) => <option key={j.job_id} value={j.job_id}>{runLabel(j)} · {j.method} on {short(j.base_model)}</option>)}
          </select>
          <select aria-label="Checkpoint" className="w-36" value={draft.checkpoint ?? ""} disabled={!ckpts.length}
                  onChange={(e) => onChange({ ...draft, checkpoint: e.target.value === "" ? null : Number(e.target.value) })}>
            <option value="">Final</option>
            {ckpts.map((c) => <option key={c.step} value={c.step}>Step {c.step}</option>)}
          </select>
        </>
      )}
      {onRemove && (
        <Button variant="ghost" size="icon-sm" aria-label="Remove this model" onClick={onRemove}><X /></Button>
      )}
    </div>
  );
}
