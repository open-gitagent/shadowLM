// Machines — every device serving this hub via `shadowlm worker`.
import { Fragment, useEffect, useState } from "react";
import { Check, ChevronDown, ChevronRight, Copy, KeyRound, MonitorSmartphone, Trash2 } from "lucide-react";

import { createToken, getTokens, getWorkers, revokeToken } from "@/api";
import type { MachineToken, WorkerInfo } from "@/api";
import { EmptyState, Mono, PageHeader, SectionHeader } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { allows, needs } from "@/lib/embed";

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button variant="outline" size="icon-sm" title="Copy" aria-label="Copy"
      onClick={() => navigator.clipboard.writeText(text).then(() => {
        setCopied(true); setTimeout(() => setCopied(false), 1500);
      })}>
      {copied ? <Check className="text-good" /> : <Copy />}
    </Button>
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
    <div className="grid gap-3">
      {!allows("admin") && <p className="text-sm text-muted-foreground">{needs("admin")}</p>}
      <div className={allows("admin") ? "flex items-center gap-2" : "hidden"}>
        <Input value={name} onChange={(e) => setName(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && mint()}
               placeholder="Machine name, e.g. macbook"
               className="flex-1 font-mono" />
        <Button variant="outline" onClick={mint}>
          <KeyRound /> Create machine token
        </Button>
      </div>
      {err && <p className="text-sm text-destructive">{err}</p>}

      {cmd && (
        <div className="grid gap-1.5">
          <div className="flex items-start gap-2">
            <pre className="flex-1 overflow-x-auto border border-border bg-surface/60 px-3 py-2 font-mono text-xs">
              {cmd}
            </pre>
            <CopyBtn text={cmd} />
          </div>
          <p className="text-xs text-muted-foreground">
            A long-lived token, shown once: copy it now. Revoke it here any time.
          </p>
        </div>
      )}

      {tokens.length > 0 && (
        <ul className="divide-y divide-border border">
          {tokens.map((t) => (
            <li key={t.name} className="flex items-center gap-2 px-3 py-1.5 text-sm">
              <KeyRound className="size-3.5 text-muted-foreground" strokeWidth={1.75} />
              <Mono className="font-medium">{t.name}</Mono>
              <span className="ml-auto text-xs text-muted-foreground">
                created {new Date(t.created * 1000).toLocaleDateString()}
              </span>
              {allows("admin") && (
                <Button variant="ghost" size="icon-sm" title="Revoke" aria-label={`Revoke ${t.name}`}
                        className="text-muted-foreground hover:text-destructive"
                        onClick={() => revokeToken(t.name).then(refresh, () => {})}>
                  <Trash2 />
                </Button>
              )}
            </li>
          ))}
        </ul>
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
  const [open, setOpen] = useState<string | null>(null);

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
        title="Machines"
        description={<>Devices serving this hub over one outbound socket. They appear here the moment <Mono>shadowlm worker</Mono> connects, and any of them can be picked as the training target.</>}
      />

      {loaded && workers.length === 0 ? (
        <div className="grid max-w-2xl gap-4">
          <EmptyState icon={MonitorSmartphone} title="No machines connected">
            On any machine with <Mono>shadowlm</Mono> installed (a MacBook, an office GPU box; NAT is
            fine, it dials out), create a token and run the command it gives you.
          </EmptyState>
          <ConnectCmd />
        </div>
      ) : (
        <>
          <SectionHeader
            title="Connected machines"
            actions={
              <span className="text-xs text-muted-foreground tabular-nums">
                {workers.filter((w) => w.online).length}/{workers.length} online
              </span>
            }
          />
          <div className="@container border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Machine</TableHead>
                  <TableHead>Backend</TableHead>
                  <TableHead className="hidden @2xl:table-cell">Platform</TableHead>
                  <TableHead>Compute</TableHead>
                  <TableHead className="text-right">Queue</TableHead>
                  <TableHead className="text-right">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {workers.map((w) => (
                  <Fragment key={w.name}>
                    <TableRow className="cursor-pointer" onClick={() => setOpen(open === w.name ? null : w.name)}>
                      <TableCell>
                        <span className="inline-flex items-center gap-2 font-medium">
                          <span className={cn("size-1.5 rounded-full", w.online ? "bg-good" : "bg-muted-foreground/40")} />
                          <Mono className="text-sm">{w.name}</Mono>
                          {w.models.length > 0 && (
                            <span className="inline-flex items-center gap-0.5 text-xs font-normal text-muted-foreground">
                              {open === w.name ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
                              {w.models.length} models
                            </span>
                          )}
                        </span>
                      </TableCell>
                      <TableCell className="text-muted-foreground">{w.backend}</TableCell>
                      <TableCell className="hidden text-muted-foreground @2xl:table-cell">{w.device}</TableCell>
                      <TableCell>{compute(w)}</TableCell>
                      <TableCell className="text-right tabular-nums">{w.queued || "—"}</TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground">
                        {w.online ? (w.queued ? "busy" : "idle") : `last seen ${ago(w.last_seen)}`}
                      </TableCell>
                    </TableRow>
                    {open === w.name && w.models.length > 0 && (
                      <TableRow className="bg-subtle hover:bg-subtle">
                        <TableCell colSpan={6} className="whitespace-normal">
                          <div className="pb-1.5 text-xs text-muted-foreground">Models on {w.name}</div>
                          <div className="flex flex-wrap gap-1.5">
                            {w.models.map((m) => (
                              <Badge key={m.id} variant="outline" className="font-mono font-normal">
                                {m.id}
                                <span className="text-muted-foreground">· {m.size_gb} GB</span>
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                ))}
              </TableBody>
            </Table>
          </div>

          {workers.length > 0 && (
            <section className="mt-8 max-w-2xl">
              <SectionHeader title="Add another machine"
                             description="Create a machine token, then run the command it gives you on that machine." />
              <ConnectCmd />
            </section>
          )}
        </>
      )}
    </>
  );
}
