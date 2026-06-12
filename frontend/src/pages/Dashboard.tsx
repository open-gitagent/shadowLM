import { useEffect, useState } from "react";
import { Cpu, Database, GitBranch, Zap } from "lucide-react";
import { getDatasets, getJobs, getMethods, getModels } from "../api";
import type { JobSummary } from "../api";
import { Button, PageHeader, Stat, StatusBadge } from "../components";

export default function Dashboard() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [models, setModels] = useState(0);
  const [methods, setMethods] = useState(0);
  const [datasets, setDatasets] = useState(0);

  useEffect(() => {
    const tick = () => getJobs().then((j) => setJobs(j.jobs)).catch(() => {});
    tick();
    const t = setInterval(tick, 2500);
    getModels().then((m) => setModels(m.catalog.length + m.recent.length)).catch(() => {});
    getMethods().then((m) => setMethods(m.methods.length)).catch(() => {});
    getDatasets().then((d) => setDatasets(d.datasets.length)).catch(() => {});
    return () => clearInterval(t);
  }, []);

  const running = jobs.filter((r) => r.status === "running");
  const recent = jobs.slice(0, 6);

  return (
    <div className="max-w-[1400px]">
      <PageHeader
        eyebrow="Workspace · lyzr-research"
        title="ShadowLM Studio"
        description="Train any open model with any method — then run it in the shadow of the frontier model behind your agent, until you own the weights."
        actions={
          <Button onClick={() => (window.location.hash = "#train")}>
            <Zap className="size-3.5" /> New training run
          </Button>
        }
      />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Datasets" value={String(datasets)} sub="uploaded here" icon={Database} />
        <Stat label="Models" value={String(models)} sub="catalog + cached" icon={Cpu} />
        <Stat label="Methods" value={String(methods)} sub="LoRA → DPO → MoRE" icon={GitBranch} />
        <Stat label="Runs" value={String(jobs.length)} sub={running.length ? `${running.length} training now` : "all idle"} icon={Zap} />
      </div>

      {running.length > 0 && (
        <section className="mt-6 overflow-hidden rounded-xl border border-border">
          <div className="border-b border-border px-5 py-3">
            <div className="text-[10px] uppercase tracking-[0.18em] text-heart">Live</div>
            <h2 className="mt-0.5 text-sm font-semibold">Active training</h2>
          </div>
          <div className="divide-y divide-border">
            {running.map((r) => (
              <div key={r.job_id}
                   className="flex cursor-pointer items-center justify-between px-5 py-3 hover:bg-umbra"
                   onClick={() => (window.location.hash = `#runs/${r.job_id}`)}>
                <div>
                  <div className="text-[13px]">{r.job_id.slice(0, 10)} · {r.method}</div>
                  <div className="text-[11px] text-faded">{(r.base_model || "").split("/").pop()} · {r.steps} steps</div>
                </div>
                <StatusBadge status={r.status} />
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="mt-6 overflow-hidden rounded-xl border border-border">
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 className="text-sm font-semibold">Recent runs</h2>
          <a href="#runs" className="text-[12px] text-heart no-underline">all runs ›</a>
        </div>
        {recent.length === 0 ? (
          <div className="px-5 py-8 text-center text-[13px] text-faded">
            no runs yet — <a href="#train" className="text-heart">start a training run ›</a>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {recent.map((r) => (
              <div key={r.job_id}
                   className="flex cursor-pointer items-center justify-between px-5 py-3 hover:bg-umbra"
                   onClick={() => (window.location.hash = `#runs/${r.job_id}`)}>
                <div className="min-w-0">
                  <div className="text-[13px]">{r.job_id.slice(0, 10)} · {r.method ?? "?"}</div>
                  <div className="truncate text-[11px] text-faded">{(r.base_model || "").split("/").pop()}</div>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-[12px] text-faded">
                    {r.final_loss != null ? `loss ${r.final_loss.toFixed(4)}` : `${r.steps} steps`}
                  </span>
                  <StatusBadge status={r.status} />
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
