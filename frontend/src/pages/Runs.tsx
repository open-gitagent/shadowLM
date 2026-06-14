// Runs — master-detail: searchable run list left, live detail right.
import { useEffect, useRef, useState } from "react";
import { Download, MessagesSquare, Search, Square } from "lucide-react";
import { apiKey, cancelJob, getCheckpoints, getJob, getJobs, getLogs, getMetrics } from "../api";
import type { Checkpoint, JobDetail, JobSummary, StepMetric } from "../api";
import { ChartLegend, LossChart, PageHeader, Sparkline, StatTile, StatusBadge, btnGhost } from "../ui";

export default function Runs({ initialId }: { initialId?: string }) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(initialId || null);
  const [filter, setFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [curves, setCurves] = useState<Record<string, number[]>>({});

  useEffect(() => { if (initialId) setSelectedId(initialId); }, [initialId]);

  useEffect(() => {
    const tick = async () => {
      try {
        const { jobs } = await getJobs();
        setJobs(jobs);
        setSelectedId((cur) => cur ?? jobs[0]?.job_id ?? null);
        const entries = await Promise.all(jobs.slice(0, 20).map(async (j) => {
          try {
            const m = await getMetrics(j.job_id);
            return [j.job_id, m.steps.map((s) => s.loss)] as const;
          } catch { return [j.job_id, []] as const; }
        }));
        setCurves(Object.fromEntries(entries));
      } catch { /* transient */ }
    };
    tick();
    const t = setInterval(tick, 2500);
    return () => clearInterval(t);
  }, []);

  const filtered = jobs.filter((r) =>
    (statusFilter === "all" || r.status === statusFilter) &&
    (filter === "" ||
      r.base_model.toLowerCase().includes(filter.toLowerCase()) ||
      (r.name || "").toLowerCase().includes(filter.toLowerCase()) ||
      r.job_id.includes(filter)));
  const selected = jobs.find((j) => j.job_id === selectedId) ?? null;

  return (
    <>
      <PageHeader
        eyebrow="Run history"
        title="Training runs"
        description="Every finetune persists its config, metrics, and artifact. Watch live, compare in the playground, download the adapter."
      />

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_minmax(0,1.4fr)] min-h-0">
        <div className="border-r border-border flex flex-col min-h-0">
          <div className="px-5 py-3 border-b border-border flex gap-2 items-center">
            <div className="relative flex-1">
              <Search className="size-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input value={filter} onChange={(e) => setFilter(e.target.value)}
                     placeholder="Search runs…" className="w-full pl-9 pr-3 py-1.5 text-sm" />
            </div>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}
                    className="text-xs px-2 py-1.5">
              <option value="all">All status</option>
              <option value="succeeded">Succeeded</option>
              <option value="running">Running</option>
              <option value="failed">Failed</option>
              <option value="stopped">Stopped</option>
            </select>
          </div>
          <div className="flex-1 overflow-auto scrollbar-thin divide-y divide-border">
            {filtered.length === 0 && (
              <div className="px-5 py-8 text-sm text-muted-foreground text-center">
                no runs yet — start one in <a href="#train" className="text-primary">Train</a>
              </div>
            )}
            {filtered.map((r) => (
              <button key={r.job_id}
                onClick={() => { setSelectedId(r.job_id); window.location.hash = `#runs/${r.job_id}`; }}
                className={`w-full text-left px-5 py-3.5 hover:bg-accent/30 transition-colors ${
                  selectedId === r.job_id
                    ? "bg-accent/50 border-l-2 border-l-primary"
                    : "border-l-2 border-l-transparent"}`}>
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <span className="font-mono text-[11px] text-muted-foreground truncate">
                    {r.name?.trim() || r.job_id.slice(0, 12)}
                  </span>
                  <StatusBadge status={r.status} />
                </div>
                <div className="text-sm font-medium truncate">{r.base_model}</div>
                <div className="flex items-center justify-between mt-1.5 gap-2">
                  <div className="text-xs text-muted-foreground font-mono truncate">
                    {r.method ?? "?"} · {r.steps} steps</div>
                  <span className={r.status === "failed" ? "text-destructive" : "text-primary"}>
                    <Sparkline data={curves[r.job_id] ?? []} width={64} height={20} />
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {selected
          ? <RunDetail key={selected.job_id} run={selected} />
          : <div className="p-8 text-sm text-muted-foreground">select a run</div>}
      </div>
    </>
  );
}

function RunDetail({ run }: { run: JobSummary }) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [steps, setSteps] = useState<StepMetric[]>([]);
  const [evals, setEvals] = useState<StepMetric[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [ckpts, setCkpts] = useState<Checkpoint[]>([]);
  const [tab, setTab] = useState<"loss" | "logs" | "artifact">("loss");

  useEffect(() => {
    let live = true;
    const tick = async () => {
      try {
        const [j, m, l] = await Promise.all([
          getJob(run.job_id), getMetrics(run.job_id), getLogs(run.job_id)]);
        if (!live) return;
        setJob(j); setSteps(m.steps); setEvals(m.evals); setLogs(l.logs);
        if (j.checkpoint) getCheckpoints(run.job_id)
          .then(({ checkpoints }) => live && setCkpts(checkpoints)).catch(() => {});
      } catch { /* transient */ }
    };
    tick();
    const t = setInterval(tick, 1800);
    return () => { live = false; clearInterval(t); };
  }, [run.job_id]);

  const last = steps[steps.length - 1];

  async function downloadAdapter() {
    const headers: Record<string, string> = {};
    if (apiKey.get()) headers["Authorization"] = `Bearer ${apiKey.get()}`;
    const r = await fetch(`/v1/finetunes/${run.job_id}/artifact`, { headers });
    if (!r.ok) return alert("artifact not ready");
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${run.job_id}-adapter.tar.gz`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <div className="overflow-auto scrollbar-thin">
      <div className="px-8 py-6 space-y-6 max-w-[1000px]">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div>
            <div className="flex items-center gap-3 mb-2">
              {run.name?.trim() && <span className="text-sm font-semibold text-primary">{run.name.trim()}</span>}
              <span className="font-mono text-xs text-muted-foreground">{run.job_id}</span>
              <StatusBadge status={(job?.status ?? run.status)} />
            </div>
            <h2 className="text-xl font-semibold">{run.base_model}</h2>
            <p className="text-sm text-muted-foreground font-mono mt-1">
              {run.method ?? "?"}
              {last?.tokens_per_s ? ` · ${Math.round(last.tokens_per_s)} tok/s` : ""}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {(job?.status === "running" || job?.status === "pending") && (
              <button onClick={() => cancelJob(run.job_id)}
                className="inline-flex items-center gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 text-destructive px-3 py-2 text-xs font-medium hover:bg-destructive/20 transition-colors">
                <Square className="size-3.5" /> Cancel
              </button>
            )}
            {job?.status === "succeeded" && (
              <>
                <button onClick={() => {
                  sessionStorage.setItem("pick.adapter", run.job_id);
                  sessionStorage.setItem("pick.model", run.base_model);
                  window.location.hash = "#playground";
                }} className={btnGhost}>
                  <MessagesSquare className="size-3.5" /> Playground
                </button>
                <button onClick={downloadAdapter} className={btnGhost}>
                  <Download className="size-3.5" /> Adapter
                </button>
              </>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatTile label="Final loss" value={job?.final_loss != null ? job.final_loss.toFixed(4) : last ? last.loss.toFixed(4) : "—"} />
          <StatTile label="Eval loss" value={evals.length ? evals[evals.length - 1].loss.toFixed(4) : "—"} />
          <StatTile label="Steps" value={String(last?.step ?? 0)} />
          <StatTile label="LR" value={last ? last.lr.toExponential(1) : "—"} />
        </div>

        {/* tabs */}
        <div className="flex items-center gap-1 border-b border-border">
          {(["loss", "logs", "artifact"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-4 py-2.5 text-sm border-b-2 -mb-px transition-colors capitalize ${
                tab === t ? "border-primary text-foreground"
                          : "border-transparent text-muted-foreground hover:text-foreground"}`}>
              {t === "loss" ? "Loss curves" : t === "logs" ? "Training logs" : "Artifact"}
              {t === "logs" && job?.status === "running" &&
                <span className="ml-2 inline-block size-1.5 rounded-full bg-primary animate-pulse" />}
            </button>
          ))}
        </div>

        {tab === "loss" && (
          <div className="space-y-4">
            {/* combined: train curve with eval overlaid */}
            <section className="rounded-lg border border-border bg-card overflow-hidden">
              <header className="px-5 py-3 border-b border-border flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold">Loss</h3>
                  <p className="text-xs text-muted-foreground">train (raw + EMA) with eval overlaid</p>
                </div>
                <ChartLegend />
              </header>
              <div className="p-5"><LossChart steps={steps} evals={evals} /></div>
            </section>
            {/* separate: train and eval on their own axes */}
            <div className="grid lg:grid-cols-2 gap-4">
              <section className="rounded-lg border border-border bg-card overflow-hidden">
                <header className="px-5 py-3 border-b border-border">
                  <h3 className="text-sm font-semibold">Training loss</h3>
                  <p className="text-xs text-muted-foreground">raw + EMA overlay</p>
                </header>
                <div className="p-5"><LossChart steps={steps} evals={[]} /></div>
              </section>
              <section className="rounded-lg border border-border bg-card overflow-hidden">
                <header className="px-5 py-3 border-b border-border">
                  <h3 className="text-sm font-semibold">Eval loss</h3>
                  <p className="text-xs text-muted-foreground">held-out validation</p>
                </header>
                <div className="p-5">
                  {evals.length
                    ? <LossChart steps={evals} evals={[]} />
                    : <div className="flex h-[240px] items-center justify-center text-center text-sm text-muted-foreground">
                        <div>
                          <div className="font-mono text-xs uppercase tracking-wider opacity-60">No eval data</div>
                          <div className="mt-1 text-xs opacity-50">Train with a held-out eval split to see this</div>
                        </div>
                      </div>}
                </div>
              </section>
            </div>
          </div>
        )}

        {tab === "logs" && (
          logs.length
            ? <TerminalPanel logs={logs} live={job?.status === "running"} />
            : <div className="rounded-lg border border-border bg-card p-8 text-center text-sm text-muted-foreground">
                no console output captured for this run
              </div>
        )}

        {tab === "artifact" && (
          <div className="space-y-4">
            {job?.error && (
              <section className="rounded-lg border border-destructive/40 bg-destructive/5 p-5">
                <h3 className="text-sm font-semibold text-destructive mb-2">Error</h3>
                <pre className="text-xs font-mono whitespace-pre-wrap text-destructive/90">{job.error}</pre>
              </section>
            )}
            {job?.checkpoint ? (
              <section className="rounded-lg border border-border bg-card overflow-hidden">
                <header className="px-5 py-3 border-b border-border">
                  <h3 className="text-sm font-semibold">Trained adapter</h3>
                </header>
                <div className="px-5 py-4 flex items-center justify-between gap-4 text-sm">
                  <div className="min-w-0">
                    <div className="font-mono text-xs text-muted-foreground break-all">{job.checkpoint}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      load it back: <code className="text-foreground/80">slm.load("{run.base_model}", adapter="…")</code>
                    </div>
                  </div>
                  <button onClick={downloadAdapter} className={btnGhost}>
                    <Download className="size-3.5" /> tar.gz
                  </button>
                </div>
                {ckpts.length > 1 && (
                  <div className="border-t border-border">
                    <div className="px-5 pt-3 pb-1 text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground">
                      saved versions · {ckpts.length}
                    </div>
                    <div className="divide-y divide-border">
                      {[...ckpts].reverse().map((c) => (
                        <div key={c.path} className="px-5 py-2.5 flex items-center justify-between gap-3 text-sm">
                          <span className="font-mono">
                            {c.final ? <span className="text-primary">{c.label}</span> : `step ${c.step}`}
                          </span>
                          <button
                            onClick={() => {
                              sessionStorage.setItem("pick.adapter", run.job_id);
                              sessionStorage.setItem("pick.model", run.base_model);
                              if (c.final) sessionStorage.removeItem("pick.checkpoint");
                              else sessionStorage.setItem("pick.checkpoint", String(c.step));
                              window.location.hash = "#playground";
                            }}
                            className="text-xs text-primary hover:underline inline-flex items-center gap-1">
                            <MessagesSquare className="size-3.5" /> test this version
                          </button>
                        </div>
                      ))}
                    </div>
                    <div className="px-5 py-2 text-[11px] text-muted-foreground">
                      trained with <code className="text-foreground/70">save_steps</code> — each is a point you can roll back to or A/B in the playground.
                    </div>
                  </div>
                )}
              </section>
            ) : !job?.error && (
              <div className="rounded-lg border border-border bg-card p-8 text-center text-sm text-muted-foreground">
                no artifact yet — finishes when training succeeds
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function TerminalPanel({ logs, live }: { logs: string[]; live: boolean }) {
  const ref = useRef<HTMLPreElement>(null);
  const [stick, setStick] = useState(true);
  useEffect(() => { if (stick && ref.current) ref.current.scrollTop = ref.current.scrollHeight; },
    [logs, stick]);
  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <header className="px-5 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex gap-1.5">
            <span className="size-2.5 rounded-full bg-destructive/70" />
            <span className="size-2.5 rounded-full bg-warning/70" />
            <span className="size-2.5 rounded-full bg-success/70" />
          </span>
          <h3 className="text-sm font-semibold">Training console</h3>
          {live && <span className="text-[10px] font-mono uppercase tracking-wider text-primary animate-pulse">● live</span>}
        </div>
        <label className="text-[10px] font-mono text-muted-foreground flex items-center gap-1.5">
          <input type="checkbox" checked={stick} onChange={(e) => setStick(e.target.checked)}
                 className="w-auto" /> follow
        </label>
      </header>
      <pre ref={ref}
           onScroll={(e) => {
             const el = e.currentTarget;
             setStick(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
           }}
           className="m-0 max-h-[460px] overflow-auto bg-ink p-4 text-[11px] leading-[1.35] text-bone/90 font-mono whitespace-pre scrollbar-thin">
        {logs.join("\n")}
      </pre>
    </section>
  );
}
