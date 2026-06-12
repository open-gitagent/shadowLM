import { useEffect, useState } from "react";
import { getJobs } from "../api";
import type { JobSummary } from "../api";
import { H2, Lead, StatusDot } from "../ui";

export default function Runs() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  useEffect(() => {
    const tick = () => getJobs().then((j) => setJobs(j.jobs)).catch(() => {});
    tick();
    const t = setInterval(tick, 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div>
      <H2>Runs</H2>
      <Lead>every training this server has executed.</Lead>
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            {["run", "model", "method", "status", "steps", "final loss"].map((h) => (
              <th key={h} className="border-b border-seam px-2.5 py-2 text-left text-[11px] uppercase tracking-widest text-faded">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {jobs.length === 0 && (
            <tr><td colSpan={6} className="px-2.5 py-3 text-faded">
              no runs yet — start one in Trainings</td></tr>
          )}
          {jobs.map((j) => (
            <tr key={j.job_id} className="cursor-pointer border-b border-seam hover:bg-umbra"
                onClick={() => (window.location.hash = `#runs/${j.job_id}`)}>
              <td className="px-2.5 py-2"><StatusDot status={j.status} />{j.job_id.slice(0, 10)}</td>
              <td className="px-2.5 py-2">{(j.base_model || "").split("/").pop()}</td>
              <td className="px-2.5 py-2">{j.method ?? "?"}</td>
              <td className="px-2.5 py-2">{j.status}</td>
              <td className="px-2.5 py-2">{j.steps}</td>
              <td className="px-2.5 py-2">{j.final_loss != null ? j.final_loss.toFixed(4) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
