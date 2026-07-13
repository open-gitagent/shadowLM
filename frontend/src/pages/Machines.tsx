// Machines — every device serving this hub via `shadowlm worker`.
import { useEffect, useState } from "react";
import { MonitorSmartphone } from "lucide-react";
import { getWorkers } from "../api";
import type { WorkerInfo } from "../api";
import { PageHeader } from "../ui";

function ago(ts: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 90) return `${s}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

export default function Machines() {
  const [workers, setWorkers] = useState<WorkerInfo[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const tick = () =>
      getWorkers().then((w) => { setWorkers(w.workers); setLoaded(true); })
        .catch(() => {});
    tick();
    const t = setInterval(tick, 3000);
    return () => clearInterval(t);
  }, []);

  const connectCmd = `shadowlm worker --hub ${window.location.origin} --name my-machine`;

  return (
    <>
      <PageHeader
        eyebrow="Fleet"
        title="Machines"
        description="Devices serving this hub over one outbound socket — they appear here the moment `shadowlm worker` connects, and any of them can be picked as the training target."
      />
      <div className="px-8 py-6 space-y-6 max-w-[1400px]">
        {loaded && workers.length === 0 ? (
          <section className="rounded-lg border border-border bg-card px-6 py-10 text-center space-y-3">
            <MonitorSmartphone className="size-6 mx-auto text-muted-foreground" />
            <div className="text-sm font-medium">No machines connected</div>
            <p className="text-xs text-muted-foreground">
              On any machine with <span className="font-mono">shadowlm</span> installed
              (a MacBook, an office GPU box — NAT is fine, it dials out):
            </p>
            <pre className="inline-block text-left text-xs font-mono bg-accent/40 border border-border rounded-md px-4 py-2.5">
              {connectCmd}
            </pre>
          </section>
        ) : (
          <section className="rounded-lg border border-border bg-card overflow-hidden">
            <div className="px-5 py-3 border-b border-border flex items-center gap-2">
              <h2 className="text-sm font-semibold">Connected machines</h2>
              <span className="text-xs text-muted-foreground font-mono ml-auto">
                {workers.filter((w) => w.online).length}/{workers.length} online
              </span>
            </div>
            <table className="w-full text-sm">
              <thead className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                <tr className="border-b border-border">
                  <th className="text-left px-5 py-2.5 font-normal">Machine</th>
                  <th className="text-left px-3 py-2.5 font-normal">Backend</th>
                  <th className="text-left px-3 py-2.5 font-normal">Platform</th>
                  <th className="text-right px-3 py-2.5 font-normal">GPUs</th>
                  <th className="text-right px-3 py-2.5 font-normal">Queue</th>
                  <th className="text-right px-5 py-2.5 font-normal">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {workers.map((w) => (
                  <tr key={w.name} className="hover:bg-accent/30">
                    <td className="px-5 py-3">
                      <span className="inline-flex items-center gap-2 font-mono font-medium">
                        <span className={`size-2 rounded-full ${
                          w.online ? "bg-emerald-500" : "bg-muted-foreground/40"}`} />
                        {w.name}
                      </span>
                    </td>
                    <td className="px-3 py-3 font-mono text-xs uppercase">{w.backend}</td>
                    <td className="px-3 py-3 font-mono text-xs">{w.device}</td>
                    <td className="px-3 py-3 text-right font-mono">{w.gpus || "—"}</td>
                    <td className="px-3 py-3 text-right font-mono">{w.queued || "—"}</td>
                    <td className="px-5 py-3 text-right text-xs text-muted-foreground font-mono">
                      {w.online ? (w.queued ? "busy" : "idle") : `last seen ${ago(w.last_seen)}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {workers.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Add another: <span className="font-mono">{connectCmd}</span>
          </p>
        )}
      </div>
    </>
  );
}
