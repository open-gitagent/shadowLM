// The examples a model learns from, every way they come in: written as a
// question/answer table, uploaded as JSONL, picked from the server's datasets,
// captured from an agent's live traffic to its frontier model, or imported
// from its OpenTelemetry traces. Which ways lead depends on the goal (an agent
// takeover leads with capture). The parent hears one normalized value: rows
// to save, or a dataset already on the server.
import {
  ArrowRightLeft, BookOpen, Cable, Check, ChevronDown, Copy, Database, FileJson,
  FileUp, ListChecks, LoaderCircle, PencilLine, Plus, Radio, Square, Trash2, TriangleAlert,
} from "lucide-react";
import { type ChangeEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";

import {
  captureBaseUrl, captureToDataset, closeCapture, createCapture, getCapture, getDatasets, importTraces,
  type CaptureInfo, type DatasetMeta, type FrontierInfo, type ProjectGoal,
} from "@/api";
import { EmptyState, Mono } from "@/components/common";
import { FrontierForm, useFrontier } from "@/components/frontier-settings";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export interface Pair { q: string; a: string }
export type Source = "write" | "upload" | "existing" | "agent" | "traces";

// The ways in, per goal: an agent's traffic and its traces lead for a takeover,
// and stay on offer for a task; knowledge is written down.
const sourceTabs: Record<Source, { icon: typeof PencilLine; label: string }> = {
  agent: { icon: Cable, label: "Connect your agent" },
  traces: { icon: FileJson, label: "Import traces" },
  write: { icon: PencilLine, label: "Write them" },
  upload: { icon: FileUp, label: "Upload a file" },
  existing: { icon: Database, label: "Use a dataset" },
};
export const sourcesFor = (g: ProjectGoal | null): Source[] =>
  g === "takeover" ? ["agent", "traces", "write", "upload", "existing"]
    : g === "task" ? ["write", "upload", "existing", "agent", "traces"]
    : ["write", "upload", "existing"];
const fewConversations = 20;

export const goals: { goal: ProjectGoal; icon: typeof BookOpen; title: string; body: string }[] = [
  { goal: "knowledge", icon: BookOpen, title: "Know our company's knowledge",
    body: "Facts it should answer exactly as you wrote them: an FAQ, product details, policies." },
  { goal: "task", icon: ListChecks, title: "Do a task from our examples",
    body: "Inputs and the outputs you want: classify tickets, draft replies, extract fields." },
  { goal: "takeover", icon: ArrowRightLeft, title: "Take over a task from a frontier model",
    body: "Something your agent sends to a rented model today, done by a model you own." },
];

export const minimum: Record<ProjectGoal, number> = { knowledge: 5, task: 20, takeover: 20 };
export const labels = (g: ProjectGoal | null) =>
  g === "knowledge" ? { q: "Question", a: "Answer", noun: "facts" } : { q: "Input", a: "Output", noun: "examples" };
const blank = (): Pair[] => [{ q: "", a: "" }, { q: "", a: "" }, { q: "", a: "" }];
const filled = (ps: Pair[]) => ps.filter((p) => p.q.trim() && p.a.trim());

// One JSONL line → a question/answer pair, from the shapes datasets come in:
// chat messages, prompt/completion, or instruction(+input)/output.
export function toPair(row: unknown): Pair | null {
  if (!row || typeof row !== "object") return null;
  const r = row as Record<string, unknown>;
  if (Array.isArray(r.messages)) {
    const msgs = r.messages as { role?: string; content?: unknown }[];
    const ai = msgs.map((m) => m.role).lastIndexOf("assistant");
    const user = msgs.slice(0, ai < 0 ? msgs.length : ai).reverse().find((m) => m.role === "user");
    if (ai < 0 || !user) return null;
    return { q: String(user.content ?? ""), a: String(msgs[ai].content ?? "") };
  }
  if (typeof r.prompt === "string" && typeof r.completion === "string") return { q: r.prompt, a: r.completion };
  if (typeof r.instruction === "string" && typeof r.output === "string")
    return { q: r.input ? `${r.instruction}\n\n${String(r.input)}` : r.instruction, a: r.output };
  return null;
}

function parseJsonl(text: string): { pairs: Pair[]; bad: number[] } {
  const pairs: Pair[] = [];
  const bad: number[] = [];
  text.split("\n").forEach((line, i) => {
    if (!line.trim()) return;
    try {
      const p = toPair(JSON.parse(line));
      if (p && p.q.trim() && p.a.trim()) pairs.push(p); else bad.push(i + 1);
    } catch {
      bad.push(i + 1);
    }
  });
  return { pairs, bad };
}

export const chatRow = (p: Pair) => ({ messages: [{ role: "user", content: p.q.trim() }, { role: "assistant", content: p.a.trim() }] });

export interface ExamplesValue {
  source: Source;
  pairs: Pair[];                // written or uploaded rows, ready to save
  dataset: DatasetMeta | null;  // or a dataset already on the server
  fileName: string;             // the uploaded file's name, for a suggested name
  count: number;
}

// ExamplesInput collects the examples; `name` names a capture or a saved
// dataset; onChange reports every change as one ExamplesValue.
export function ExamplesInput({ goal, name, onChange }: {
  goal: ProjectGoal; name: string; onChange: (v: ExamplesValue) => void;
}) {
  const [source, setSource] = useState<Source>(sourcesFor(goal)[0]);
  const [written, setWritten] = useState<Pair[]>(blank);
  const [uploaded, setUploaded] = useState<Pair[]>([]);
  const [fileName, setFileName] = useState("");
  const [uploadErr, setUploadErr] = useState("");
  const [datasets, setDatasets] = useState<DatasetMeta[] | null>(null);
  const [existing, setExisting] = useState<DatasetMeta | null>(null);
  const [agentDs, setAgentDs] = useState<DatasetMeta | null>(null);
  const [tracesDs, setTracesDs] = useState<DatasetMeta | null>(null);
  const tableRef = useRef<HTMLTableSectionElement>(null);
  const l = labels(goal);

  // a new goal brings its own ways in
  useEffect(() => { setSource(sourcesFor(goal)[0]); }, [goal]);
  useEffect(() => {
    getDatasets().then((d) => setDatasets(d.datasets)).catch(() => setDatasets([]));
  }, []);

  const pairs = source === "write" ? filled(written) : source === "upload" ? uploaded : [];
  const chosen = source === "existing" ? existing : source === "agent" ? agentDs : source === "traces" ? tracesDs : null;
  const fromDataset = source === "existing" || source === "agent" || source === "traces";
  const count = fromDataset ? (chosen?.rows ?? 0) : pairs.length;

  useEffect(() => {
    onChange({ source, pairs, dataset: fromDataset ? chosen : null, fileName: source === "upload" ? fileName : "", count });
    // pairs is derived each render; its content is what matters
  }, [source, JSON.stringify(pairs), chosen?.dataset_id, fileName, count]);  // eslint-disable-line react-hooks/exhaustive-deps

  async function onFile(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    setUploadErr("");
    const { pairs: ps, bad } = parseJsonl(await f.text());
    if (bad.length) {
      const shown = bad.slice(0, 5).join(", ") + (bad.length > 5 ? ` and ${bad.length - 5} more` : "");
      setUploadErr(`${bad.length === 1 ? "Line" : "Lines"} ${shown} ${bad.length === 1 ? "isn't" : "aren't"} a question and answer we can read. Each line needs {"messages": [...]} with a user and an assistant turn, {"prompt", "completion"}, or {"instruction", "output"}. Fix ${bad.length === 1 ? "it" : "them"} and upload again.`);
      return;
    }
    if (!ps.length) {
      setUploadErr("That file has no examples in it. Upload a .jsonl file with one example per line.");
      return;
    }
    setUploaded(ps);
    setFileName(f.name);
  }

  function editPair(i: number, key: keyof Pair, v: string) {
    setWritten((ps) => ps.map((p, j) => (j === i ? { ...p, [key]: v } : p)));
  }

  function addRow(focus = true) {
    setWritten((ps) => [...ps, { q: "", a: "" }]);
    if (focus) setTimeout(() => tableRef.current?.querySelector<HTMLInputElement>(`[data-cell="q-${written.length}"]`)?.focus(), 0);
  }

  function onAnswerKey(e: KeyboardEvent<HTMLInputElement>, i: number) {
    if (e.key === "Enter" && i === written.length - 1) {
      e.preventDefault();
      addRow();
    }
  }

  return (
    <div>
      <div role="tablist" aria-label="How to add examples" className="inline-flex flex-wrap rounded-lg bg-surface-2/70 p-0.5">
        {sourcesFor(goal).map((s) => {
          const { icon: Icon, label } = sourceTabs[s];
          return (
            <button key={s} type="button" role="tab" aria-selected={source === s} onClick={() => setSource(s)}
              className={cn("flex h-7 items-center gap-1.5 rounded-md px-3 text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                source === s ? "bg-card text-foreground shadow-paper" : "text-muted-foreground hover:text-foreground")}>
              <Icon className="size-3.5" strokeWidth={1.75} />{label}
            </button>
          );
        })}
      </div>

      {source === "write" && (
        <div className="mt-4 border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface/50 text-left text-xs text-muted-foreground">
                <th className="w-1/2 px-3 py-2 font-medium">{l.q}</th>
                <th className="px-3 py-2 font-medium">{l.a}</th>
                <th className="w-9" aria-label="Remove" />
              </tr>
            </thead>
            <tbody ref={tableRef}>
              {written.map((p, i) => (
                <tr key={i} className="border-b border-border last:border-b-0">
                  <td className="p-1.5">
                    <Input data-cell={`q-${i}`} aria-label={`${l.q} ${i + 1}`} value={p.q}
                      placeholder={i === 0 ? (goal === "knowledge" ? "What is Lyzr?" : "Customer can't log in after reset") : ""}
                      onChange={(e) => editPair(i, "q", e.target.value)} className="border-transparent bg-transparent shadow-none" />
                  </td>
                  <td className="p-1.5">
                    <Input aria-label={`${l.a} ${i + 1}`} value={p.a}
                      placeholder={i === 0 ? (goal === "knowledge" ? "Lyzr is an enterprise platform for AI agents." : "category: account-access") : ""}
                      onChange={(e) => editPair(i, "a", e.target.value)} onKeyDown={(e) => onAnswerKey(e, i)}
                      className="border-transparent bg-transparent shadow-none" />
                  </td>
                  <td className="pr-1.5 text-right">
                    <Button variant="ghost" size="icon-sm" aria-label={`Remove row ${i + 1}`} disabled={written.length === 1}
                      onClick={() => setWritten((ps) => ps.filter((_, j) => j !== i))}>
                      <Trash2 className="text-muted-foreground" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex items-center justify-between border-t border-border px-3 py-2">
            <Button variant="ghost" size="sm" onClick={() => addRow()}><Plus /> Add a row</Button>
            <span className="text-xs text-muted-foreground">Enter in the last {l.a.toLowerCase()} adds a row</span>
          </div>
        </div>
      )}

      {source === "upload" && (
        <div className="mt-4 space-y-3">
          <label className="flex cursor-pointer flex-col items-center gap-2 border border-dashed border-border-strong bg-surface/30 px-4 py-8 text-center transition-colors hover:bg-subtle focus-within:border-ring">
            <FileUp className="size-5 text-muted-foreground" strokeWidth={1.75} />
            <span className="text-sm font-medium">{fileName ? `Replace ${fileName}` : "Choose a .jsonl file"}</span>
            <span className="text-xs text-muted-foreground">One example per line: chat messages, prompt/completion, or instruction/output</span>
            <input type="file" accept=".jsonl,.json" className="sr-only" onChange={onFile} />
          </label>
          {uploadErr && <p role="alert" className="text-sm text-destructive">{uploadErr}</p>}
          {fileName && !uploadErr && (
            <p className="flex items-center gap-1.5 text-sm text-muted-foreground"><Check className="size-3.5 text-good" />{uploaded.length} {l.noun} read from {fileName}</p>
          )}
        </div>
      )}

      {source === "existing" && (
        <div className="mt-4">
          {datasets === null ? (
            <div className="space-y-2" aria-label="Loading datasets">
              {[0, 1, 2].map((i) => <div key={i} className="h-11 animate-pulse bg-surface/50" />)}
            </div>
          ) : datasets.length === 0 ? (
            <EmptyState icon={Database} title="No datasets on this server yet">Write your examples or upload a file instead.</EmptyState>
          ) : (
            <div role="radiogroup" aria-label="Datasets" className="max-h-80 divide-y divide-border overflow-y-auto border border-border">
              {[...datasets].sort((a, b) => Number(!!a.curated) - Number(!!b.curated)).map((d) => (
                <button key={d.dataset_id} type="button" role="radio" aria-checked={existing?.dataset_id === d.dataset_id}
                  onClick={() => setExisting(d)}
                  className={cn("flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors outline-none focus-visible:bg-subtle",
                    existing?.dataset_id === d.dataset_id ? "bg-primary/5" : "hover:bg-subtle")}>
                  <span className="min-w-0 flex-1">
                    <span className={cn("block truncate text-sm font-medium", existing?.dataset_id === d.dataset_id && "text-primary")}>{d.name}</span>
                    <Mono className="text-muted-foreground">{d.rows != null ? `${d.rows.toLocaleString()} rows` : "rows unknown"} · {d.format}</Mono>
                  </span>
                  {existing?.dataset_id === d.dataset_id && <Check className="size-4 shrink-0 text-primary" />}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {source === "agent" && <AgentSource name={name} saved={agentDs} onSaved={setAgentDs} />}
      {source === "traces" && <TracesSource saved={tracesDs} onSaved={setTracesDs} />}

      <Counter count={count} goal={goal} enough={count >= minimum[goal]} noun={l.noun} />
    </div>
  );
}

export function ReviewRow({ p }: { p: Pair }) {
  return (
    <div className="min-w-0 flex-1 text-sm">
      <p className="font-medium break-words">{p.q}</p>
      <p className="mt-0.5 break-words text-muted-foreground">{p.a}</p>
    </div>
  );
}

export function Counter({ count, goal, enough, noun }: { count: number; goal: ProjectGoal; enough: boolean; noun: string }) {
  if (count === 0) return <p className="mt-3 text-sm text-muted-foreground">No {noun} yet.</p>;
  return enough ? (
    <p className="mt-3 flex items-center gap-1.5 text-sm text-muted-foreground"><Check className="size-3.5 text-good" />{count} {noun}</p>
  ) : (
    <p className="mt-3 flex items-start gap-1.5 text-sm text-warning">
      <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
      <span>{count} {noun}. {goal === "knowledge"
        ? `Five or more facts give a clearer test of what it learned. You can still go ahead.`
        : `Fewer than ${minimum[goal]} examples rarely teach a task well. You can still go ahead, and add more later.`}</span>
    </p>
  );
}

// CopyField is a value to paste somewhere else, with a button that copies it.
export function CopyField({ value, label, multiline }: { value: string; label: string; multiline?: boolean }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard blocked (http, or a denied permission): the text stays selectable
    }
  }
  return (
    <div className="relative">
      <pre aria-label={label} className={cn("overflow-x-auto bg-surface py-2.5 pr-20 pl-3 font-mono text-xs leading-relaxed", !multiline && "whitespace-nowrap")}>{value}</pre>
      <Button type="button" variant="ghost" size="xs" onClick={copy} className="absolute top-1.5 right-1.5" aria-label={`Copy ${label}`}>
        {copied ? <><Check className="text-good" /> Copied</> : <><Copy /> Copy</>}
      </Button>
    </div>
  );
}

// AgentSource captures the agent's real traffic: it points its OpenAI base URL
// at a capture address, every call passes through to the frontier model and is
// recorded, and the conversations become the dataset.
function AgentSource({ name, saved, onSaved }: { name: string; saved: DatasetMeta | null; onSaved: (d: DatasetMeta) => void }) {
  const { frontier, loaded, refresh } = useFrontier();
  const [cap, setCap] = useState<CaptureInfo | null>(null);
  const [busy, setBusy] = useState<"" | "start" | "use" | "stop">("");
  const [err, setErr] = useState("");

  // Live count while the capture is open.
  useEffect(() => {
    if (!cap || cap.status !== "open") return;
    let alive = true;
    const tick = () => getCapture(cap.capture_id).then((c) => { if (alive) setCap(c); }).catch(() => {});
    const t = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(t); };
  }, [cap?.capture_id, cap?.status]);  // eslint-disable-line react-hooks/exhaustive-deps

  if (!loaded) return <div className="mt-4 h-24 animate-pulse bg-surface/50" aria-label="Loading" />;

  if (saved) {
    return (
      <p className="mt-4 flex items-center gap-1.5 text-sm text-muted-foreground">
        <Check className="size-3.5 text-good" />
        {saved.rows ?? 0} conversations saved as the dataset “{saved.name}”. Go on to Review.
      </p>
    );
  }

  if (!frontier) {
    return (
      <div className="mt-4 border border-border px-4 py-4">
        <p className="text-sm font-medium">First, the frontier model your agent uses today</p>
        <p className="mt-0.5 mb-4 max-w-prose text-sm text-muted-foreground">
          While capturing, your agent's calls go to this model through the studio, so it keeps answering exactly as
          before. The same model is the bar your fine-tune is measured against.
        </p>
        <FrontierForm current={null} compact onSaved={() => refresh()} />
      </div>
    );
  }

  async function start() {
    setBusy("start");
    setErr("");
    try {
      setCap(await createCapture(name.trim() || "Agent capture"));
    } catch (ex) {
      setErr(`${(ex as Error).message}. Try again in a moment.`);
    } finally {
      setBusy("");
    }
  }

  async function use() {
    if (!cap) return;
    setBusy("use");
    setErr("");
    try {
      const ds = await captureToDataset(cap.capture_id, name.trim() || cap.name);
      await closeCapture(cap.capture_id).catch(() => {});  // the dataset is saved; a still-open URL is harmless
      onSaved(ds);
    } catch (ex) {
      setErr(`${(ex as Error).message}. Your capture is still open; try again.`);
    } finally {
      setBusy("");
    }
  }

  async function stop() {
    if (!cap) return;
    setBusy("stop");
    setErr("");
    try {
      await closeCapture(cap.capture_id);
      setCap({ ...cap, status: "closed" });
    } catch (ex) {
      setErr(`${(ex as Error).message}. The capture is still running; try again.`);
    } finally {
      setBusy("");
    }
  }

  if (!cap) {
    return (
      <div className="mt-4 border border-border px-4 py-4">
        <p className="text-sm font-medium">Capture your agent's conversations</p>
        <p className="mt-0.5 max-w-prose text-sm text-muted-foreground">
          You'll get an address to use as your agent's OpenAI base URL. It keeps working exactly as before: each call
          goes to <span className="font-medium text-foreground">{frontier.model}</span> and is recorded. Nothing in the
          agent changes but that one setting.
        </p>
        {err && <p role="alert" className="mt-3 text-sm text-destructive">{err}</p>}
        <Button className="mt-4" onClick={start} disabled={busy !== ""}>
          {busy === "start" ? <LoaderCircle className="animate-spin" /> : <Radio />} Start capturing
        </Button>
      </div>
    );
  }

  return <CaptureLive cap={cap} frontier={frontier} busy={busy} err={err} onUse={use} onStop={stop} />;
}

function CaptureLive({ cap, frontier, busy, err, onUse, onStop }: {
  cap: CaptureInfo; frontier: FrontierInfo; busy: string; err: string; onUse: () => void; onStop: () => void;
}) {
  const url = captureBaseUrl(cap.capture_id);
  const convos = cap.episodes ?? 0;
  const open = cap.status === "open";
  return (
    <div className="mt-4 space-y-5 border border-border px-4 py-4">
      <div className="space-y-2">
        <p className="text-sm font-medium">Point your agent here</p>
        <CopyField label="capture base URL" value={url} />
        <p className="text-xs text-muted-foreground">
          Use it as the OpenAI base URL. Any model name and any API key work: every call goes to {frontier.model} with
          the key stored in the studio.
        </p>
      </div>

      <details className="group">
        <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground">
          <ChevronDown className="size-3.5 transition-transform group-open:rotate-180" /> Show the code
        </summary>
        <div className="mt-2 space-y-2">
          <CopyField label="Python snippet" multiline
            value={`from openai import OpenAI\n\nclient = OpenAI(base_url="${url}", api_key="unused")`} />
          <CopyField label="environment variable" value={`OPENAI_BASE_URL=${url}`} />
        </div>
      </details>

      <div aria-live="polite" className="border-t border-border pt-4">
        <p className="flex items-center gap-2 text-sm">
          {open
            ? <span className="relative flex size-2"><span className="absolute inline-flex size-full animate-ping rounded-full bg-good/60" /><span className="relative inline-flex size-2 rounded-full bg-good" /></span>
            : <Square className="size-3 text-muted-foreground" />}
          <span className="font-medium tabular-nums">{convos} {convos === 1 ? "conversation" : "conversations"} captured</span>
          <span className="text-muted-foreground tabular-nums">({cap.calls} {cap.calls === 1 ? "call" : "calls"})</span>
          {!open && <span className="text-muted-foreground">· stopped</span>}
        </p>
        {convos === 0 && open && (
          <p className="mt-1 text-sm text-muted-foreground">Waiting for the first call. Run your agent as you normally would.</p>
        )}
        {convos > 0 && convos < fewConversations && (
          <p className="mt-2 flex items-start gap-1.5 text-sm text-warning">
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
            <span>Fewer than {fewConversations} conversations rarely teach a task well. Let it run longer if you can.</span>
          </p>
        )}
        {cap.preview?.length ? (
          <ul className="mt-3 max-h-64 divide-y divide-border overflow-y-auto border border-border">
            {cap.preview.slice(0, 5).map((p, i) => (
              <li key={i} className="px-3 py-2.5">
                <ReviewRow p={{ q: p.question || "(no user message)", a: p.answer || "(no answer)" }} />
                {p.turns > 1 && <p className="mt-0.5 text-xs text-muted-foreground">{p.turns} turns</p>}
              </li>
            ))}
          </ul>
        ) : null}
      </div>

      {err && <p role="alert" className="text-sm text-destructive">{err}</p>}
      <div className="flex flex-wrap items-center justify-end gap-2">
        {open && (
          <Button variant="ghost" onClick={onStop} disabled={busy !== ""}>
            {busy === "stop" ? <LoaderCircle className="animate-spin" /> : <Square />} Stop capturing
          </Button>
        )}
        <Button onClick={onUse} disabled={busy !== "" || convos === 0}>
          {busy === "use" ? <LoaderCircle className="animate-spin" /> : <Check />} Use these conversations
        </Button>
      </div>
    </div>
  );
}

// TracesSource imports an agent's OpenTelemetry GenAI traces (OTLP JSON or an
// array of spans) as conversations.
function TracesSource({ saved, onSaved }: { saved: DatasetMeta | null; onSaved: (d: DatasetMeta) => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function onFile(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    setErr("");
    let parsed: unknown;
    try {
      parsed = JSON.parse(await f.text());
    } catch {
      setErr(`${f.name} isn't valid JSON. Export the traces as OTLP JSON (or a JSON array of spans) and upload that file.`);
      return;
    }
    setBusy(true);
    try {
      onSaved(await importTraces(f.name.replace(/\.json$/i, "") || "Imported traces", parsed));
    } catch (ex) {
      setErr(`${(ex as Error).message}. Check the file holds GenAI model-call spans, then upload it again.`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-4 space-y-3">
      <label className={cn("flex flex-col items-center gap-2 border border-dashed border-border-strong bg-surface/30 px-4 py-8 text-center transition-colors focus-within:border-ring",
        busy ? "cursor-wait opacity-70" : "cursor-pointer hover:bg-subtle")}>
        {busy ? <LoaderCircle className="size-5 animate-spin text-muted-foreground" /> : <FileJson className="size-5 text-muted-foreground" strokeWidth={1.75} />}
        <span className="text-sm font-medium">{busy ? "Reading the traces…" : saved ? "Import a different file" : "Choose a traces file"}</span>
        <span className="max-w-prose text-xs text-muted-foreground">
          OpenTelemetry GenAI spans, as OTLP JSON or an array of spans, from any instrumented agent. Each conversation
          becomes one example.
        </span>
        <input type="file" accept=".json,application/json" className="sr-only" onChange={onFile} disabled={busy} />
      </label>
      {err && <p role="alert" className="text-sm text-destructive">{err}</p>}
      {saved && !err && (
        <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <Check className="size-3.5 text-good" />{saved.rows ?? 0} conversations imported as the dataset “{saved.name}”
        </p>
      )}
    </div>
  );
}
