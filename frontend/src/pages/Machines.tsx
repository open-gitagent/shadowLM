// Machines — every device serving this hub via `shadowlm worker`.
import { useEffect, useState } from "react";
import { Check, Copy, KeyRound, MonitorSmartphone, Trash2 } from "lucide-react";
import { createToken, getTokens, getWorkers, revokeToken } from "../api";
import type { MachineToken, WorkerInfo } from "../api";
import { PageHeader } from "../ui";

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button title="copy"
      onClick={() => navigator.clipboard.writeText(text).then(() => {
        setCopied(true); setTimeout(() => setCopied(false), 1500);
      })}
      className="shrink-0 p-2 rounded-md border border-border text-muted-foreground hover:text-foreground hover:bg-accent/40">
      {copied ? <Check className="size-3.5 text-emerald-500" /> : <Copy className="size-3.5" />}
    </button>
  );
}

/** Mint + manage long-lived machine tokens; the raw token is shown exactly once. */
function ConnectCmd() {
  const [tokens, setTokens] = useState<MachineToken[]>([]);
  const [name, setName] = useState("");
  const [minted, setMinted] = useState<{ name: string; token: string } | null>(null);
  const [err, setErr] = useState("");

  const refresh = () => { getTokens().then((t) => setTokens(t.tokens)).catch(() => {}); };
  useEffect(refresh, []);

  async function mint() {
    const n = name.trim() || "my-machine";
    setErr("");
    try {
      setMinted(await createToken(n));
      setName("");
      refresh();
    } catch (ex) { setErr((ex as Error).message); }
  }

  const cmd = minted
    ? `shadowlm worker --hub ${window.location.origin} --name ${minted.name} --api-key ${minted.token}`
    : null;

  return (
    <div className="space-y-3 text-left">
      <div className="flex items-center gap-2">
        <input value={name} onChange={(e) => setName(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && mint()}
               placeholder="machine name — e.g. macbook"
               className="flex-1 font-mono text-sm" />
        <button onClick={mint}
                className="shrink-0 inline-flex items-center gap-1.5 text-xs px-3 py-2 rounded-md border border-border hover:bg-accent/40">
          <KeyRound className="size-3.5" /> Create machine token
        </button>
      </div>
      {err && <p className="text-xs text-red-500">{err}</p>}

      {cmd && (
        <div className="space-y-1.5">
          <div className="flex items-start gap-2">
            <pre className="flex-1 overflow-x-auto text-xs font-mono bg-accent/40 border border-border rounded-md px-4 py-2.5">
              {cmd}
            </pre>
            <CopyBtn text={cmd} />
          </div>
          <p className="text-[11px] text-muted-foreground">
            long-lived token, shown once — copy it now. Revoke it here any time.
          </p>
        </div>
      )}

      {tokens.length > 0 && (
        <div className="divide-y divide-border border border-border rounded-md">
          {tokens.map((t) => (
            <div key={t.name} className="px-3 py-2 flex items-center gap-2 text-xs">
              <KeyRound className="size-3 text-muted-foreground" />
              <span className="font-mono font-medium">{t.name}</span>
              <span className="text-muted-foreground font-mono ml-auto">
                created {new Date(t.created * 1000).toLocaleDateString()}
              </span>
              <button title="revoke"
                onClick={() => revokeToken(t.name).then(refresh)}
                className="p-1 rounded text-muted-foreground hover:text-red-500">
                <Trash2 className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** "NVIDIA L40S · 48 GB" / "Apple M3 Pro · 36 GB unified" / "12 cores · 32 GB RAM" */
function compute(w: WorkerInfo): string {
  if (w.gpu_name) {
    const unified = w.backend === "mlx" ? " unified" : "";
    const count = w.gpus > 1 ? `${w.gpus}× ` : "";
    return `${count}${w.gpu_name}${w.vram_gb ? ` · ${w.vram_gb} GB${unified}` : ""}`;
  }
  const bits = [];
  if (w.cores) bits.push(`${w.cores} cores`);
  if (w.ram_gb) bits.push(`${w.ram_gb} GB RAM`);
  return bits.join(" · ") || "—";
}

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
            <div className="max-w-2xl mx-auto"><ConnectCmd /></div>
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
                  <th className="text-left px-3 py-2.5 font-normal">Compute</th>
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
                    <td className="px-3 py-3 font-mono text-xs">{compute(w)}</td>
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
          <div className="max-w-2xl">
            <p className="text-xs text-muted-foreground mb-1.5">Add another machine:</p>
            <ConnectCmd />
          </div>
        )}
      </div>
    </>
  );
}
