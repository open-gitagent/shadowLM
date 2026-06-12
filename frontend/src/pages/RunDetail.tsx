import { useEffect, useState } from "react";
import { cancelJob, getJob, getJobs, getMetrics } from "../api";
import type { JobDetail, JobSummary, StepMetric } from "../api";
import { LossChart, Pill } from "../ui";

export default function RunDetail({ id }: { id: string }) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [summary, setSummary] = useState<JobSummary | null>(null);
  const [steps, setSteps] = useState<StepMetric[]>([]);
  const [evals, setEvals] = useState<StepMetric[]>([]);

  useEffect(() => {
    let live = true;
    const tick = async () => {
      try {
        const [j, m, list] = await Promise.all([getJob(id), getMetrics(id), getJobs()]);
        if (!live) return;
        setJob(j); setSteps(m.steps); setEvals(m.evals);
        setSummary(list.jobs.find((x) => x.job_id === id) ?? null);
      } catch { /* transient */ }
    };
    tick();
    const t = setInterval(tick, 1800);
    return () => { live = false; clearInterval(t); };
  }, [id]);

  if (!job) return <p className="text-faded">loading…</p>;
  const last = steps[steps.length - 1];
  const tone = job.status === "succeeded" ? "green" : job.status === "failed" ? "red" : "gold";

  return (
    <div>
      <a href="#runs" className="text-[12.5px] text-heart no-underline">‹ all runs</a>
      <div className="mt-2 flex flex-wrap items-center gap-2.5">
        <h2 className="text-[15px] font-bold">{id}</h2>
        <Pill tone={tone}>{job.status}</Pill>
        {(job.status === "running" || job.status === "pending") && (
          <button className="rounded-lg border border-seam px-3 py-1 text-[13px]"
                  onClick={() => cancelJob(id)}>cancel</button>
        )}
        {job.status === "succeeded" && (
          <button className="rounded-lg bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-3 py-1.5 text-[13px] font-bold text-white"
            onClick={() => {
              sessionStorage.setItem("pick.adapter", id);
              if (summary) sessionStorage.setItem("pick.model", summary.base_model);
              window.location.hash = "#playground";
            }}>
            open in playground ›
          </button>
        )}
      </div>
      <p className="mt-1 text-[12px] text-faded">
        {summary?.base_model} · {summary?.method ?? "?"}
        {last ? ` · step ${last.step} · loss ${last.loss.toFixed(4)}` : ""}
        {job.final_loss != null ? ` · final ${job.final_loss.toFixed(4)}` : ""}
        {last?.tokens_per_s ? ` · ${Math.round(last.tokens_per_s)} tok/s` : ""}
      </p>
      <LossChart steps={steps} evals={evals} />
      {job.error && <div className="whitespace-pre-wrap text-heart">{job.error}</div>}
    </div>
  );
}
