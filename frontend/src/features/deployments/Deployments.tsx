// Deployments — every fine-tune served at the studio's OpenAI-compatible
// endpoint. An agent uses one by changing its base URL, key and model name; the
// key is shown once, when the deployment is made, and revoking it stops it.
import { Check, Copy, LoaderCircle, Plus, Rocket, TriangleAlert, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  createDeployment, deleteDeployment, deploymentBaseUrl, getCheckpoints, getDeployments, getJobs,
} from "@/api";
import type { Checkpoint, Deployment, JobSummary } from "@/api";
import { ConfirmDelete, EmptyState, ErrorState, Field, PageHeader } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { allows, needs } from "@/lib/embed";

const slug = (name: string) =>
  name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "my-model";

const when = (t: number | null) => {
  if (!t) return "never";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago`
    : s < 86400 ? `${Math.floor(s / 3600)} h ago` : `${Math.floor(s / 86400)} d ago`;
};

function useCopy() {
  const [copied, setCopied] = useState("");
  async function copy(text: string, what: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(what);
      setTimeout(() => setCopied((c) => (c === what ? "" : c)), 1600);
    } catch { /* clipboard blocked: the text is still selectable */ }
  }
  return { copied, copy };
}

export default function Deployments() {
  const [deps, setDeps] = useState<Deployment[] | null>(null);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loadErr, setLoadErr] = useState<unknown>(null);
  const [panel, setPanel] = useState(false);
  const [fresh, setFresh] = useState<{ name: string; key: string } | null>(null);
  const [revoking, setRevoking] = useState<Deployment | undefined>();
  const [revokeErr, setRevokeErr] = useState("");
  const operator = allows("operator");
  const { copied, copy } = useCopy();
  const base = deploymentBaseUrl();

  // Deployments and runs; request counts refresh while the page is open.
  useEffect(() => {
    let live = true;
    const tick = () => Promise.all([getDeployments(), getJobs()])
      .then(([d, j]) => { if (live) { setDeps(d.deployments); setJobs(j.jobs); setLoadErr(null); } })
      .catch((e) => live && setLoadErr(e));
    tick();
    const t = setInterval(tick, 15_000);
    return () => { live = false; clearInterval(t); };
  }, []);

  const runName = (id: string) => {
    const j = jobs.find((x) => x.job_id === id);
    return j?.name || id.slice(0, 12);
  };

  async function revoke() {
    if (!revoking) return;
    const d = revoking;
    setRevoking(undefined); setRevokeErr("");
    try {
      await deleteDeployment(d.deployment_id);
      setDeps((all) => (all ?? []).filter((x) => x.deployment_id !== d.deployment_id));
      if (fresh?.name === d.name) setFresh(null);
    } catch (e) {
      setRevokeErr((e as Error).message);
    }
  }

  const header = (
    <PageHeader title="Deployments"
      description="Fine-tunes served at an OpenAI-compatible endpoint. An agent uses one by changing its base URL, key and model name."
      actions={operator && !panel && (
        <Button onClick={() => setPanel(true)}><Plus /> New deployment</Button>
      )} />
  );

  if (loadErr && deps === null) return <>{header}<ErrorState error={loadErr} title="Couldn't load deployments" /></>;

  return (
    <>
      {header}

      <div className="mb-6 flex max-w-2xl items-center justify-between gap-3 border border-border bg-card px-4 py-3">
        <div className="min-w-0">
          <div className="text-xs text-muted-foreground">Base URL for every deployment</div>
          <div className="mt-0.5 truncate font-mono text-xs" title={base}>{base}</div>
        </div>
        <Button variant="outline" size="sm" onClick={() => copy(base, "base")}>
          {copied === "base" ? <Check /> : <Copy />} {copied === "base" ? "Copied" : "Copy"}
        </Button>
      </div>

      {fresh && (
        <div className="mb-6 max-w-2xl border border-warning/40 bg-warning/5 p-4">
          <div className="flex items-start justify-between gap-3">
            <p className="flex items-center gap-1.5 text-sm font-medium text-warning">
              <TriangleAlert className="size-4" /> Copy the key for {fresh.name} now. It won't be shown again.
            </p>
            <Button variant="ghost" size="icon-sm" aria-label="Dismiss" onClick={() => setFresh(null)}><X /></Button>
          </div>
          <div className="mt-2.5 flex items-center gap-2">
            <code className="min-w-0 flex-1 truncate border border-border bg-card px-2.5 py-1.5 font-mono text-xs select-all">{fresh.key}</code>
            <Button variant="outline" size="sm" onClick={() => copy(fresh.key, "key")}>
              {copied === "key" ? <Check /> : <Copy />} {copied === "key" ? "Copied" : "Copy key"}
            </Button>
          </div>
        </div>
      )}

      {panel && operator && (
        <NewDeployment jobs={jobs} taken={(deps ?? []).map((d) => d.name)}
          onCancel={() => setPanel(false)}
          onCreated={(d, key) => { setDeps((all) => [d, ...(all ?? [])]); setFresh({ name: d.name, key }); setPanel(false); }} />
      )}

      {revokeErr && <p className="mb-4 text-sm text-destructive">Couldn't revoke: {revokeErr}</p>}

      {deps === null ? (
        <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-10 w-full" />)}</div>
      ) : deps.length === 0 ? (
        <div className="max-w-2xl space-y-3">
          <EmptyState icon={Rocket} title="Nothing deployed yet">
            Deploy a fine-tune and any OpenAI-compatible client can call it: from a project's Deploy step in Business mode,
            or with New deployment here. Each deployment gets its own model name and key.
          </EmptyState>
          {operator ? (
            !panel && <Button variant="outline" onClick={() => setPanel(true)}><Plus /> New deployment</Button>
          ) : <p className="text-sm text-muted-foreground">{needs("operator")}</p>}
        </div>
      ) : (
        <div className="border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model name</TableHead>
                <TableHead>Fine-tune</TableHead>
                <TableHead className="hidden @3xl:table-cell">Base model</TableHead>
                <TableHead>Checkpoint</TableHead>
                <TableHead className="text-right">Requests</TableHead>
                <TableHead className="hidden @2xl:table-cell">Last used</TableHead>
                <TableHead className="hidden @3xl:table-cell">Created</TableHead>
                <TableHead className="w-0" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {deps.map((d) => (
                <TableRow key={d.deployment_id}>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <span className="font-mono text-xs">{d.name}</span>
                      <Button variant="ghost" size="icon-xs" aria-label={`Copy ${d.name}`} onClick={() => copy(d.name, d.deployment_id)}>
                        {copied === d.deployment_id ? <Check /> : <Copy />}
                      </Button>
                    </div>
                    <div className="font-mono text-[11px] text-muted-foreground">key {d.key_prefix}…</div>
                  </TableCell>
                  <TableCell><a className="text-primary underline-offset-4 hover:underline" href={`#runs/${d.run_id}`}>{runName(d.run_id)}</a></TableCell>
                  <TableCell className="hidden font-mono text-xs @3xl:table-cell">{d.base_model.split("/").pop()}</TableCell>
                  <TableCell className="text-muted-foreground tabular-nums">{d.checkpoint != null ? `step ${d.checkpoint}` : "final"}</TableCell>
                  <TableCell className="text-right tabular-nums">{d.requests}</TableCell>
                  <TableCell className="hidden text-muted-foreground @2xl:table-cell">{when(d.last_used)}</TableCell>
                  <TableCell className="hidden text-muted-foreground @3xl:table-cell">{when(d.created)}</TableCell>
                  <TableCell className="text-right">
                    {operator && (
                      <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => setRevoking(d)}>Revoke</Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      <ConfirmDelete what={revoking?.name} action="Revoke" onConfirm={revoke} onClose={() => setRevoking(undefined)}>
        The key stops working immediately, and any agent using it gets an error. The fine-tune itself stays; you can deploy it again with a new key.
      </ConfirmDelete>
    </>
  );
}

// The inline panel for a new deployment: a succeeded fine-tune, optionally at a
// checkpoint, under a model name.
function NewDeployment({ jobs, taken, onCancel, onCreated }: {
  jobs: JobSummary[]; taken: string[];
  onCancel: () => void; onCreated: (d: Deployment, key: string) => void;
}) {
  const runs = useMemo(() => jobs.filter((j) => j.status === "succeeded"), [jobs]);
  const [runId, setRunId] = useState(runs[0]?.job_id ?? "");
  const [ckpts, setCkpts] = useState<Checkpoint[]>([]);
  const [checkpoint, setCheckpoint] = useState<number | null>(null);
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const run = runs.find((r) => r.job_id === runId);
  useEffect(() => {
    if (!runId && runs[0]) setRunId(runs[0].job_id);
  }, [runs, runId]);
  useEffect(() => {
    if (!nameTouched) setName(run ? slug(run.name || run.job_id) : "");
  }, [run, nameTouched]);
  useEffect(() => {
    setCkpts([]); setCheckpoint(null);
    if (!runId) return;
    getCheckpoints(runId).then((r) => setCkpts(r.checkpoints.filter((c) => !c.final))).catch(() => {});
  }, [runId]);

  const clash = taken.includes(name.trim());

  async function submit() {
    setBusy(true); setErr("");
    try {
      const out = await createDeployment({ name: name.trim(), run_id: runId, checkpoint });
      onCreated(out.deployment, out.key);
    } catch (e) {
      setErr((e as Error).message);
    } finally { setBusy(false); }
  }

  return (
    <section className="mb-6 max-w-3xl border border-border bg-card p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold tracking-[-0.01em]">New deployment</h2>
          <p className="mt-1 text-sm text-muted-foreground">Serve a fine-tune under a model name of your choice. Its key is shown once.</p>
        </div>
        <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={onCancel}><X /></Button>
      </div>
      {runs.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No fine-tune has finished yet. <a className="text-primary underline-offset-4 hover:underline" href="#train">Start a run</a>, then deploy it here.
        </p>
      ) : (
        <form onSubmit={(e) => { e.preventDefault(); void submit(); }} className="grid gap-4">
          <div className="grid gap-4 @2xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <Field label="Fine-tune" htmlFor="nd-run">
              <select id="nd-run" value={runId} onChange={(e) => setRunId(e.target.value)}>
                {runs.map((r) => (
                  <option key={r.job_id} value={r.job_id}>
                    {(r.name || r.job_id.slice(0, 12))} · {r.base_model.split("/").pop()} · {r.method ?? "?"}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Checkpoint" htmlFor="nd-ckpt" hint={ckpts.length ? undefined : "This run saved only its final weights."}>
              <select id="nd-ckpt" value={checkpoint ?? ""} disabled={!ckpts.length}
                onChange={(e) => setCheckpoint(e.target.value === "" ? null : Number(e.target.value))}>
                <option value="">Final</option>
                {ckpts.map((c) => <option key={c.step} value={c.step}>{c.label || `step ${c.step}`}</option>)}
              </select>
            </Field>
          </div>
          <Field label="Model name" htmlFor="nd-name"
                 hint={clash ? undefined : "What clients send as model. Letters, digits, dots, dashes and underscores."}>
            <Input id="nd-name" className="font-mono" value={name} aria-invalid={clash || undefined}
              onChange={(e) => { setName(e.target.value); setNameTouched(true); }} />
          </Field>
          {clash && <p className="-mt-2 text-sm text-destructive">A deployment named {name.trim()} already exists. Pick another name.</p>}
          {err && <p className="text-sm text-destructive">{err}</p>}
          <div className="flex items-center justify-end gap-2">
            <Button type="button" variant="ghost" onClick={onCancel}>Cancel</Button>
            <Button type="submit" disabled={busy || !runId || !name.trim() || clash}>
              {busy ? <LoaderCircle className="animate-spin" /> : <Rocket />} Deploy
            </Button>
          </div>
        </form>
      )}
    </section>
  );
}
