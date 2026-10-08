// Deploy: the current version behind an OpenAI-compatible endpoint. The key is
// shown once, right after deploying; afterwards the endpoint, model name, key
// prefix and usage, with snippets an agent can paste. Try it in the Playground
// or take the adapter away instead.
import { useQueryClient } from "@tanstack/react-query";
import {
  Check, ChevronDown, Copy, Download, KeyRound, LoaderCircle, MessagesSquare, Rocket, TriangleAlert,
} from "lucide-react";
import { useState } from "react";

import { apiFetch, deleteDeployment, deploymentBaseUrl, failure } from "@/api";
import { ConfirmDelete, Field } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { allows, embedded, hostToast, needs } from "@/lib/embed";
import { cn } from "@/lib/utils";

import type { CockpitActions } from "../Cockpit";
import type { Loop } from "../model";
import { ago, CodeBlock, slug, StationHeader, useCopy } from "./shared";

export function DeployStation({ loop, actions }: { loop: Loop; actions: CockpitActions }) {
  const qc = useQueryClient();
  const operator = allows("operator");
  const p = loop.project!;
  const job = loop.job;
  const dep = loop.deployment;
  const [name, setName] = useState(slug(p.name));
  const [newKey, setNewKey] = useState("");  // shown once, right after deploying
  const [revoking, setRevoking] = useState(false);
  const [revokeErr, setRevokeErr] = useState("");
  const [lang, setLang] = useState<"python" | "curl">("python");
  const [dlBusy, setDlBusy] = useState(false);
  const [own, setOwn] = useState(false);
  const { copied, copy } = useCopy();
  const creating = actions.busy === "deploy";

  if (!job || job.status !== "succeeded") {
    return (
      <>
        <StationHeader title="Deploy"
          line="Once it's fine-tuned, serve it at an OpenAI-compatible endpoint: your agent changes only its base URL, key and model name." />
        <div className="p-5">
          <p className="max-w-[70ch] text-sm text-muted-foreground">
            Evaluate it first so you deploy a version you know is ready. The deployment serves the current version; training a new one later doesn't change what's live.
          </p>
        </div>
      </>
    );
  }

  async function deploy() {
    actions.clearError();
    const key = await actions.run({ kind: "deploy", name: name.trim() });
    if (key) setNewKey(key);
  }

  async function revoke() {
    if (!dep) return;
    setRevoking(false);
    setRevokeErr("");
    try {
      await deleteDeployment(dep.deployment_id);
      setNewKey("");
      await qc.invalidateQueries({ queryKey: ["deployments"] });
    } catch (e) {
      setRevokeErr((e as Error).message);
    }
  }

  async function download() {
    setDlBusy(true);
    try {
      const r = await apiFetch(`/v1/finetunes/${job!.job_id}/artifact`);
      if (!r.ok) {
        const why = (await failure(r)).message;
        if (embedded) { if (r.status !== 403) hostToast("error", why); } else alert(why);
        return;
      }
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${job!.job_id}-adapter.tar.gz`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setDlBusy(false);
    }
  }

  function tryIt() {
    sessionStorage.setItem("pick.adapter", job!.job_id);
    sessionStorage.setItem("pick.model", job!.base_model);
    sessionStorage.removeItem("pick.checkpoint");
    window.location.hash = "#playground";
  }

  const secondary = <>
    <Button variant="outline" size="sm" onClick={tryIt}><MessagesSquare /> Try it in the Playground</Button>
    <Button variant="outline" size="sm" onClick={download} disabled={dlBusy}>
      {dlBusy ? <LoaderCircle className="animate-spin" /> : <Download />} Download adapter
    </Button>
  </>;

  const sdk = [
    "import shadowlm as slm",
    "",
    "# the adapter you downloaded, unpacked",
    `model = slm.load("${job.base_model}", adapter="./${job.job_id}-adapter")`,
    `print(model.chat([{"role": "user", "content": "Hello"}]).content)`,
  ].join("\n");
  const runYourself = (
    <div>
      <button type="button" onClick={() => setOwn((o) => !o)} aria-expanded={own}
        className="inline-flex items-center gap-1.5 rounded-sm text-sm text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:outline-none">
        <ChevronDown className={cn("size-3.5 transition-transform duration-200", !own && "-rotate-90")} /> Run it yourself instead
      </button>
      {own && <CodeBlock label="With the downloaded adapter and the SDK" code={sdk} className="mt-2"
                         copied={copied === "sdk"} onCopy={() => copy(sdk, "sdk")} />}
    </div>
  );

  if (!dep) {
    return (
      <>
        <StationHeader title="Deploy"
          line="Serve this version at an OpenAI-compatible endpoint. Your agent keeps its code and changes only its base URL, key and model name."
          actions={secondary} />
        <div className="space-y-4 p-5">
          {revokeErr && <p className="text-sm text-destructive">Couldn't revoke: {revokeErr}</p>}
          {operator ? (
            <form onSubmit={(e) => { e.preventDefault(); void deploy(); }}
              className="flex flex-wrap items-end gap-3 border border-border bg-surface/40 p-4">
              <Field label="Model name" htmlFor="dep-name" className="min-w-56 flex-1"
                     hint="What your agent sends as model. Letters, digits, dots, dashes and underscores.">
                <Input id="dep-name" value={name} onChange={(e) => setName(e.target.value)} className="font-mono" />
              </Field>
              <Button type="submit" disabled={creating || !name.trim()} className="mb-5">
                {creating ? <LoaderCircle className="animate-spin" /> : <Rocket />} Deploy
              </Button>
              {actions.error && <p className="w-full text-sm text-destructive">{actions.error}</p>}
            </form>
          ) : (
            <p className="text-sm text-muted-foreground">{needs("operator")}</p>
          )}
          {runYourself}
        </div>
      </>
    );
  }

  const base = deploymentBaseUrl();
  const key = newKey || "YOUR_KEY";
  const python = [
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url="${base}", api_key="${key}")`,
    `reply = client.chat.completions.create(model="${dep.name}", messages=[{"role": "user", "content": "..."}])`,
    "print(reply.choices[0].message.content)",
  ].join("\n");
  const curl = [
    `curl ${base}/chat/completions \\`,
    `  -H "Authorization: Bearer ${key}" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{"model": "${dep.name}", "messages": [{"role": "user", "content": "Hello"}]}'`,
  ].join("\n");

  return (
    <>
      <StationHeader title="Deploy"
        line={<>Live as <span className="font-mono text-xs text-foreground">{dep.name}</span> · {dep.requests} {dep.requests === 1 ? "request" : "requests"} · last used {ago(dep.last_used)}</>}
        research="#deployments"
        actions={<>
          {operator && <Button variant="ghost" size="sm" onClick={() => setRevoking(true)} className="text-destructive hover:text-destructive">Revoke</Button>}
        </>} />
      <div className="space-y-4 p-5">
        {newKey && (
          <div className="border border-warning/40 bg-warning/5 p-4">
            <p className="flex items-center gap-1.5 text-sm font-medium text-warning">
              <TriangleAlert className="size-4" /> Copy this key now. It won't be shown again.
            </p>
            <div className="mt-2.5 flex items-center gap-2">
              <code className="min-w-0 flex-1 truncate border border-border bg-card px-2.5 py-1.5 font-mono text-xs select-all">{newKey}</code>
              <Button variant="outline" size="sm" onClick={() => copy(newKey, "key")}>
                {copied === "key" ? <Check /> : <Copy />} {copied === "key" ? "Copied" : "Copy key"}
              </Button>
            </div>
          </div>
        )}

        <dl className="grid gap-px overflow-hidden border border-border bg-border text-sm @2xl:grid-cols-3">
          {[
            { k: "Base URL", v: base, copyable: true },
            { k: "Model", v: dep.name, copyable: true },
            { k: "Key", v: `${dep.key_prefix}…`, copyable: false },
          ].map((row) => (
            <div key={row.k} className="flex min-w-0 items-center justify-between gap-2 bg-card px-4 py-3">
              <div className="min-w-0">
                <dt className="text-xs text-muted-foreground">{row.k}</dt>
                <dd className="mt-0.5 truncate font-mono text-xs" title={row.v}>
                  {row.k === "Key" && <KeyRound className="mr-1 inline size-3 text-muted-foreground" />}{row.v}
                </dd>
              </div>
              {row.copyable && (
                <Button variant="ghost" size="icon-sm" aria-label={`Copy ${row.k.toLowerCase()}`} onClick={() => copy(row.v, row.k)}>
                  {copied === row.k ? <Check /> : <Copy />}
                </Button>
              )}
            </div>
          ))}
        </dl>

        <div className="border border-border">
          <div className="flex items-center justify-between gap-3 border-b border-border px-2 py-1.5">
            <div role="tablist" aria-label="Snippet language" className="flex gap-0.5">
              {(["python", "curl"] as const).map((l) => (
                <button key={l} type="button" role="tab" aria-selected={lang === l} onClick={() => setLang(l)}
                  className={cn("h-7 rounded-md px-2.5 text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                    lang === l ? "bg-surface-2 text-foreground" : "text-muted-foreground hover:text-foreground")}>
                  {l === "python" ? "Python" : "curl"}
                </button>
              ))}
            </div>
            <Button variant="ghost" size="xs" onClick={() => copy(lang === "python" ? python : curl, "snippet")}>
              {copied === "snippet" ? <Check /> : <Copy />} {copied === "snippet" ? "Copied" : "Copy"}
            </Button>
          </div>
          <pre className="overflow-x-auto bg-surface/40 p-4 font-mono text-xs leading-relaxed">{lang === "python" ? python : curl}</pre>
        </div>
        <p className="max-w-[70ch] text-sm text-muted-foreground">
          Any OpenAI-compatible client works: point it at the base URL, use the key, and send <span className="font-mono text-xs text-foreground">{dep.name}</span> as the model.
          {!newKey && " The key was shown once when you deployed; revoke and deploy again for a new one."}
        </p>
        {revokeErr && <p className="text-sm text-destructive">Couldn't revoke: {revokeErr}</p>}
        <div className="flex flex-wrap gap-2">{secondary}</div>
        {runYourself}
      </div>
      <ConfirmDelete what={revoking ? dep.name : undefined} action="Revoke" onConfirm={revoke} onClose={() => setRevoking(false)}>
        The key stops working immediately, and any agent using it gets an error. The fine-tune itself stays; you can deploy it again with a new key.
      </ConfirmDelete>
    </>
  );
}
