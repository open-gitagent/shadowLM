// The workspace at a glance — all real data from the server.
import { useEffect, useState } from "react";
import { ArrowUpRight, Cpu, Database, MonitorSmartphone, Zap } from "lucide-react";
import { getDatasets, getJobs, getMetrics, getModels, getWorkers } from "../api";
import type { DatasetMeta, JobSummary, WorkerInfo } from "../api";
import { PageHeader, Sparkline, StatusBadge, btnPrimary } from "../ui";

function Stat({ label, value, sub, icon: Icon }: {
  label: string; value: string; sub: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center justify-between text-muted-foreground">
        <span className="text-[10px] font-mono uppercase tracking-[0.18em]">{label}</span>
        <Icon className="size-4" />
      </div>
      <div className="mt-3 text-2xl font-semibold font-mono tracking-tight">{value}</div>
      <div className="mt-1 text-xs text-muted-foreground">{sub}</div>
    </div>
  );
}

export default function Dashboard() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [recentModels, setRecentModels] = useState(0);
  const [curves, setCurves] = useState<Record<string, number[]>>({});
  const [workers, setWorkers] = useState<WorkerInfo[]>([]);

  useEffect(() => {
    const tick = () => {
      getWorkers().then((w) => setWorkers(w.workers)).catch(() => {});
      getJobs().then(async ({ jobs }) => {
        setJobs(jobs);
        const want = jobs.slice(0, 6);
        const entries = await Promise.all(want.map(async (j) => {
          try {
            const m = await getMetrics(j.job_id);
            return [j.job_id, m.steps.map((s) => s.loss)] as const;
          } catch { return [j.job_id, []] as const; }
        }));
        setCurves(Object.fromEntries(entries));
      }).catch(() => {});
    };
    tick();
    const t = setInterval(tick, 3000);
    getDatasets().then((d) => setDatasets(d.datasets)).catch(() => {});
    getModels().then((m) => setRecentModels(m.catalog.length + m.recent.length)).catch(() => {});
    return () => clearInterval(t);
  }, []);

  const running = jobs.filter((j) => j.status === "running" || j.status === "pending");
  const recent = jobs.slice(0, 5);

  return (
    <>
      <PageHeader
        eyebrow="Workspace"
        title="ShadowLM Studio"
        description="Train any open model with any method. Then run it in the shadow of the frontier model behind your agent — until you own the weights."
        actions={
          <a href="#train" className={`${btnPrimary} no-underline`}>
            <Zap className="size-3.5" /> New training run
          </a>
        }
      />

      <div className="px-8 py-6 space-y-6 max-w-[1400px]">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Stat label="Models" value={String(recentModels)} sub="catalog + trained here" icon={Cpu} />
          <Stat label="Active runs" value={String(running.length)} sub={running.length ? "currently training" : "idle"} icon={Zap} />
          <Stat label="Datasets" value={String(datasets.length)} sub="uploaded to this server" icon={Database} />
        </div>

        {workers.length > 0 && (
          <section className="rounded-lg border border-border bg-card overflow-hidden">
            <div className="px-5 py-3 border-b border-border flex items-center gap-2">
              <MonitorSmartphone className="size-4 text-primary" />
              <h2 className="text-sm font-semibold">Machines</h2>
              <a href="#machines" className="text-xs text-primary hover:underline inline-flex items-center gap-1 no-underline ml-auto">
                {workers.filter((w) => w.online).length}/{workers.length} online <ArrowUpRight className="size-3" />
              </a>
            </div>
            <div className="divide-y divide-border">
              {workers.map((w) => (
                <div key={w.name} className="px-5 py-3 flex items-center gap-3 text-sm">
                  <span className={`size-2 rounded-full shrink-0 ${
                    w.online ? "bg-emerald-500" : "bg-muted-foreground/40"}`} />
                  <span className="font-medium font-mono">{w.name}</span>
                  <span className="text-xs text-muted-foreground font-mono">
                    {w.backend} · {w.device}{w.gpus ? ` · ${w.gpus} gpu` : ""}
                  </span>
                  <span className="text-xs text-muted-foreground font-mono ml-auto">
                    {w.online
                      ? (w.queued ? `${w.queued} queued` : "idle")
                      : `last seen ${new Date(w.last_seen * 1000).toLocaleTimeString()}`}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}

        {running.length > 0 && (
          <section className="rounded-lg border border-border bg-card overflow-hidden">
            <div className="px-5 py-3 border-b border-border">
              <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-primary">Live</div>
              <h2 className="text-sm font-semibold mt-0.5">Active training</h2>
            </div>
            <div className="divide-y divide-border">
              {running.map((r) => (
                <a key={r.job_id} href={`#runs/${r.job_id}`}
                   className="px-5 py-4 flex items-center gap-4 no-underline text-foreground hover:bg-accent/30">
                  <StatusBadge status={r.status} />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium truncate">{r.base_model}</div>
                    <div className="text-xs text-muted-foreground font-mono">
                      {r.method ?? "?"} · {r.steps} steps so far
                    </div>
                  </div>
                  <span className="text-primary">
                    <Sparkline data={curves[r.job_id] ?? []} width={120} height={32} />
                  </span>
                </a>
              ))}
            </div>
          </section>
        )}

        <section className="rounded-lg border border-border bg-card overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-border">
            <h2 className="text-sm font-semibold">Recent runs</h2>
            <a href="#runs" className="text-xs text-primary hover:underline inline-flex items-center gap-1 no-underline">
              All runs <ArrowUpRight className="size-3" />
            </a>
          </div>
          {recent.length === 0 ? (
            <div className="px-5 py-8 text-sm text-muted-foreground text-center">
              No runs yet — start one from <a href="#train" className="text-primary">Train</a>.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                <tr className="border-b border-border">
                  <th className="text-left px-5 py-2.5 font-normal">Run</th>
                  <th className="text-left px-3 py-2.5 font-normal">Model</th>
                  <th className="text-left px-3 py-2.5 font-normal">Method</th>
                  <th className="text-left px-3 py-2.5 font-normal">Status</th>
                  <th className="text-right px-3 py-2.5 font-normal">Loss</th>
                  <th className="text-right px-5 py-2.5 font-normal">Curve</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {recent.map((r) => (
                  <tr key={r.job_id} className="hover:bg-accent/30 cursor-pointer"
                      onClick={() => (window.location.hash = `#runs/${r.job_id}`)}>
                    <td className="px-5 py-3 font-mono text-xs text-muted-foreground truncate max-w-[160px]">{r.name?.trim() || r.job_id.slice(0, 10)}</td>
                    <td className="px-3 py-3 truncate max-w-[260px]">{(r.base_model || "").split("/").pop()}</td>
                    <td className="px-3 py-3 font-mono text-xs uppercase">{r.method ?? "?"}</td>
                    <td className="px-3 py-3"><StatusBadge status={r.status} /></td>
                    <td className="px-3 py-3 text-right font-mono">
                      {r.final_loss != null ? r.final_loss.toFixed(4) : "—"}
                    </td>
                    <td className="px-5 py-3">
                      <div className="flex justify-end text-primary">
                        <Sparkline data={curves[r.job_id] ?? []} width={100} height={28} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </>
  );
}
