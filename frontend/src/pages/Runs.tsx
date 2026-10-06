// Runs — master-detail: searchable run list left, live detail right.
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { CircleAlert, Download, History, MessagesSquare, PackageOpen, Search, Square, TerminalSquare } from "lucide-react";
import { apiFetch, cancelJob, getCheckpoints, getJob, getJobs, getLogs, getMetrics } from "@/api";
import type { Checkpoint, JobDetail, JobSummary, StepMetric } from "@/api";
import { ChartLegend, LossChart, Sparkline } from "@/components/charts";
import { EmptyState, Mono, PageHeader, Stat, StatStrip, StatusBadge } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

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
        title="Training runs"
        description="Every finetune persists its config, metrics, and artifact. Watch live, compare in the playground, download the adapter."
      />

      <div className="grid min-h-[520px] flex-1 !shrink grid-cols-1 overflow-hidden border border-border bg-card lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <div className="flex min-h-0 flex-col border-b border-border lg:border-r lg:border-b-0">
          <div className="flex items-center gap-2 border-b border-border p-3">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input value={filter} onChange={(e) => setFilter(e.target.value)}
                     placeholder="Search runs…" className="pl-8" />
            </div>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}
                    className="h-8 text-sm">
              <option value="all">All status</option>
              <option value="succeeded">Succeeded</option>
              <option value="running">Running</option>
              <option value="failed">Failed</option>
              <option value="stopped">Stopped</option>
            </select>
          </div>
          <div className="min-h-0 flex-1 divide-y divide-border overflow-auto scrollbar-thin">
            {filtered.length === 0 && (
              <div className="p-4">
                <EmptyState icon={History} title="No runs yet">
                  Start one in <a href="#train" className="font-medium text-primary hover:underline">New run</a>.
                </EmptyState>
              </div>
            )}
            {filtered.map((r) => (
              <button key={r.job_id}
                onClick={() => { setSelectedId(r.job_id); window.location.hash = `#runs/${r.job_id}`; }}
                aria-current={selectedId === r.job_id ? "true" : undefined}
                className={cn(
                  "w-full border-l-2 px-4 py-3 text-left transition-colors",
                  selectedId === r.job_id
                    ? "border-l-primary bg-primary/5"
                    : "border-l-transparent hover:bg-subtle")}>
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-[11px] text-muted-foreground">
                    {r.name?.trim() || r.job_id.slice(0, 12)}
                  </span>
                  <StatusBadge status={r.status} />
                </div>
                <div className="truncate text-sm font-medium">{r.base_model}</div>
                <div className="mt-1 flex items-center justify-between gap-2">
                  <div className="truncate font-mono text-xs text-muted-foreground">
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
          : <div className="p-6 text-sm text-muted-foreground">Select a run to see its curves, logs and artifact.</div>}
      </div>
    </>
  );
}

// Panel is one framed block of the run detail: a title line, then its body.
function Panel({ title, sub, actions, children, className }: {
  title: string; sub?: string; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={cn("overflow-hidden border border-border bg-card", className)}>
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold">{title}</h3>
          {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
        </div>
        {actions}
      </header>
      {children}
    </section>
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
    const r = await apiFetch(`/v1/finetunes/${run.job_id}/artifact`);
    if (!r.ok) return alert("artifact not ready");
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${run.job_id}-adapter.tar.gz`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const status = job?.status ?? run.status;
  const toPlayground = () => {
    sessionStorage.setItem("pick.adapter", run.job_id);
    sessionStorage.setItem("pick.model", run.base_model);
    window.location.hash = "#playground";
  };

  return (
    <div className="min-h-0 overflow-auto scrollbar-thin">
      <div className="max-w-[1000px] space-y-5 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="mb-1.5 flex flex-wrap items-center gap-2">
              {run.name?.trim() && <span className="text-sm font-semibold text-primary">{run.name.trim()}</span>}
              <Mono className="text-muted-foreground">{run.job_id}</Mono>
              <StatusBadge status={status} />
            </div>
            <h2 className="text-lg font-semibold tracking-[-0.01em]">{run.base_model}</h2>
            <p className="mt-0.5 font-mono text-xs text-muted-foreground">
              {run.method ?? "?"}
              {last?.tokens_per_s ? ` · ${Math.round(last.tokens_per_s)} tok/s` : ""}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {(job?.status === "running" || job?.status === "pending") && (
              <Button variant="destructive" onClick={() => cancelJob(run.job_id)}>
                <Square /> Cancel
              </Button>
            )}
            {job?.status === "succeeded" && (
              <>
                <Button variant="outline" onClick={toPlayground}>
                  <MessagesSquare /> Playground
                </Button>
                <Button variant="outline" onClick={downloadAdapter}>
                  <Download /> Adapter
                </Button>
              </>
            )}
          </div>
        </div>

        <StatStrip>
          <Stat label="Final loss" value={job?.final_loss != null ? job.final_loss.toFixed(4) : last ? last.loss.toFixed(4) : undefined} />
          <Stat label="Eval loss" value={evals.length ? evals[evals.length - 1].loss.toFixed(4) : undefined} />
          <Stat label="Steps" value={String(last?.step ?? 0)} />
          <Stat label="Learning rate" value={last ? last.lr.toExponential(1) : undefined} />
        </StatStrip>

        <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="gap-4">
          <TabsList variant="line" className="w-full justify-start border-b border-border">
            <TabsTrigger value="loss" className="flex-none">Loss curves</TabsTrigger>
            <TabsTrigger value="logs" className="flex-none">
              Training logs
              {job?.status === "running" && <span className="size-1.5 animate-pulse rounded-full bg-primary" />}
            </TabsTrigger>
            <TabsTrigger value="artifact" className="flex-none">Artifact</TabsTrigger>
          </TabsList>

          <TabsContent value="loss" className="space-y-4">
            {/* combined: train curve with eval overlaid */}
            <Panel title="Loss" sub="train (raw + EMA) with eval overlaid" actions={<ChartLegend />}>
              <div className="p-4"><LossChart steps={steps} evals={evals} /></div>
            </Panel>
            {/* separate: train and eval on their own axes */}
            <div className="grid gap-4 @4xl:grid-cols-2">
              <Panel title="Training loss" sub="raw + EMA overlay">
                <div className="p-4"><LossChart steps={steps} evals={[]} /></div>
              </Panel>
              <Panel title="Eval loss" sub="held-out validation">
                <div className="p-4">
                  {evals.length
                    ? <LossChart steps={evals} evals={[]} />
                    : <div className="flex h-[240px] items-center justify-center text-center">
                        <div>
                          <div className="text-sm font-medium text-muted-foreground">No eval data</div>
                          <div className="mt-0.5 text-xs text-muted-foreground">Train with a held-out eval split to see this</div>
                        </div>
                      </div>}
                </div>
              </Panel>
            </div>
          </TabsContent>

          <TabsContent value="logs">
            {logs.length
              ? <TerminalPanel logs={logs} live={job?.status === "running"} />
              : <EmptyState icon={TerminalSquare} title="No console output">
                  Nothing was captured for this run.
                </EmptyState>}
          </TabsContent>

          <TabsContent value="artifact" className="space-y-4">
            {job?.error && (
              <section className="border border-destructive/30 bg-destructive/5 p-4">
                <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-destructive">
                  <CircleAlert className="size-4" /> Error
                </h3>
                <pre className="font-mono text-xs whitespace-pre-wrap text-destructive">{job.error}</pre>
              </section>
            )}
            {job?.checkpoint ? (
              <Panel title="Trained adapter">
                <div className="flex items-center justify-between gap-4 px-4 py-3 text-sm">
                  <div className="min-w-0">
                    <Mono className="text-muted-foreground">{job.checkpoint}</Mono>
                    <div className="mt-1 text-xs text-muted-foreground">
                      Load it back: <code className="font-mono text-foreground/80">slm.load("{run.base_model}", adapter="…")</code>
                    </div>
                  </div>
                  <Button variant="outline" onClick={downloadAdapter}>
                    <Download /> tar.gz
                  </Button>
                </div>
                {ckpts.length > 1 && (
                  <div className="border-t border-border">
                    <div className="px-4 pt-3 pb-1 text-xs text-muted-foreground">
                      Saved versions · {ckpts.length}
                    </div>
                    <div className="divide-y divide-border">
                      {[...ckpts].reverse().map((c) => (
                        <div key={c.path} className="flex items-center justify-between gap-3 px-4 py-2 text-sm">
                          <span className="font-mono text-xs">
                            {c.final ? <span className="text-primary">{c.label}</span> : `step ${c.step}`}
                          </span>
                          <Button variant="ghost" size="xs" className="text-primary"
                            onClick={() => {
                              sessionStorage.setItem("pick.adapter", run.job_id);
                              sessionStorage.setItem("pick.model", run.base_model);
                              if (c.final) sessionStorage.removeItem("pick.checkpoint");
                              else sessionStorage.setItem("pick.checkpoint", String(c.step));
                              window.location.hash = "#playground";
                            }}>
                            <MessagesSquare /> Test this version
                          </Button>
                        </div>
                      ))}
                    </div>
                    <div className="px-4 py-2 text-[11px] text-muted-foreground">
                      Trained with <code className="font-mono text-foreground/70">save_steps</code>: each is a point you can roll back to or A/B in the playground.
                    </div>
                  </div>
                )}
              </Panel>
            ) : !job?.error && (
              <EmptyState icon={PackageOpen} title="No artifact yet">
                It appears when training succeeds.
              </EmptyState>
            )}
          </TabsContent>
        </Tabs>
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
    <section className="overflow-hidden border border-border bg-card">
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <TerminalSquare className="size-4 text-muted-foreground" strokeWidth={1.75} />
          <h3 className="text-sm font-semibold">Training console</h3>
          {live && (
            <span className="flex items-center gap-1 text-xs text-primary">
              <span className="size-1.5 animate-pulse rounded-full bg-primary" /> live
            </span>
          )}
        </div>
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <input type="checkbox" checked={stick} onChange={(e) => setStick(e.target.checked)} /> Follow
        </label>
      </header>
      <pre ref={ref}
           onScroll={(e) => {
             const el = e.currentTarget;
             setStick(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
           }}
           className="m-0 max-h-[460px] overflow-auto bg-ink p-4 font-mono text-[11px] leading-[1.35] whitespace-pre text-bone/90 scrollbar-thin">
        {logs.join("\n")}
      </pre>
    </section>
  );
}
