// The workspace at a glance — all real data from the server.
import { useEffect, useState } from "react";
import { ArrowRight, History, Zap } from "lucide-react";

import { getDatasets, getJobs, getMetrics, getModels, getWorkers } from "@/api";
import type { DatasetMeta, JobSummary, WorkerInfo } from "@/api";
import { Sparkline } from "@/components/charts";
import { EmptyState, Mono, PageHeader, Stat, StatStrip, StatusBadge } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

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
  const online = workers.filter((w) => w.online).length;

  return (
    <>
      <PageHeader
        title="Overview"
        description="Train any open model with any method. Then run it in the shadow of the frontier model behind your agent — until you own the weights."
        actions={
          <Button asChild>
            <a href="#train"><Zap /> New training run</a>
          </Button>
        }
      />

      <StatStrip className="@2xl:grid-cols-4">
        <Stat label="Models" value={recentModels} sub="Catalog + trained here" href="#models" />
        <Stat label="Active runs" value={running.length} sub={running.length ? "Currently training" : "Idle"} href="#runs" />
        <Stat label="Datasets" value={datasets.length} sub="Uploaded to this server" href="#datasets" />
        <Stat label="Machines" value={workers.length ? `${online}/${workers.length}` : 0}
              sub={workers.length ? "Online" : "Only this machine"} href="#machines" />
      </StatStrip>

      {running.length > 0 && (
        <>
          <h2 className="mt-8 mb-3 text-sm font-medium text-muted-foreground">Active training</h2>
          <div className="border">
            <ul className="divide-y divide-border">
              {running.map((r) => (
                <li key={r.job_id}>
                  <a href={`#runs/${r.job_id}`}
                     className="flex items-center gap-4 px-4 py-3 transition-colors hover:bg-subtle">
                    <StatusBadge status={r.status} />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{r.base_model}</div>
                      <div className="text-xs text-muted-foreground">
                        {r.method ?? "?"} · {r.steps} steps so far
                      </div>
                    </div>
                    <span className="text-primary">
                      <Sparkline data={curves[r.job_id] ?? []} width={120} height={32} />
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}

      <div className={`mt-8 grid items-start gap-4 ${workers.length > 0 ? "@4xl:grid-cols-3" : ""}`}>
        <section className={workers.length > 0 ? "@4xl:col-span-2" : ""}>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-medium text-muted-foreground">Recent runs</h2>
            <Button variant="ghost" size="sm" asChild>
              <a href="#runs">All runs <ArrowRight /></a>
            </Button>
          </div>
          {recent.length === 0 ? (
            <EmptyState icon={History} title="No runs yet">
              Start one from <a href="#train" className="text-primary hover:underline">New run</a>.
            </EmptyState>
          ) : (
            <div className="border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Run</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead className="hidden @2xl:table-cell">Method</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Loss</TableHead>
                    <TableHead className="hidden text-right @2xl:table-cell">Curve</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {recent.map((r) => (
                    <TableRow key={r.job_id} className="cursor-pointer"
                              onClick={() => (window.location.hash = `#runs/${r.job_id}`)}>
                      <TableCell className="max-w-40 truncate">
                        <Mono className="text-muted-foreground">{r.name?.trim() || r.job_id.slice(0, 10)}</Mono>
                      </TableCell>
                      <TableCell className="max-w-64 truncate">{(r.base_model || "").split("/").pop()}</TableCell>
                      <TableCell className="hidden text-muted-foreground @2xl:table-cell">{r.method ?? "?"}</TableCell>
                      <TableCell><StatusBadge status={r.status} /></TableCell>
                      <TableCell className="text-right tabular-nums">
                        {r.final_loss != null ? r.final_loss.toFixed(4) : "—"}
                      </TableCell>
                      <TableCell className="hidden @2xl:table-cell">
                        <div className="flex justify-end text-primary">
                          <Sparkline data={curves[r.job_id] ?? []} width={100} height={28} />
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </section>

        {workers.length > 0 && (
          <Card size="sm">
            <CardHeader>
              <CardTitle>Machines</CardTitle>
              <CardDescription>{online} of {workers.length} online</CardDescription>
              <CardAction>
                <Button variant="ghost" size="sm" asChild>
                  <a href="#machines">All <ArrowRight /></a>
                </Button>
              </CardAction>
            </CardHeader>
            <ul className="divide-y divide-border border-t border-border">
              {workers.map((w) => (
                <li key={w.name} className="flex items-center gap-3 px-3 py-2.5 text-sm">
                  <span className={cn("size-1.5 shrink-0 rounded-full", w.online ? "bg-good" : "bg-muted-foreground/40")} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-medium">{w.name}</div>
                    <div className="truncate text-xs text-muted-foreground">
                      {w.backend} · {w.gpu_name || w.device}
                      {w.vram_gb ? ` · ${w.vram_gb} GB` : ""}
                    </div>
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {w.online
                      ? (w.queued ? `${w.queued} queued` : "idle")
                      : `last seen ${new Date(w.last_seen * 1000).toLocaleTimeString()}`}
                  </span>
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    </>
  );
}
