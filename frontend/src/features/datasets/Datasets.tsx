// Dataset library — upload JSONL, or reference a HuggingFace dataset (with a
// streamed preview before you add it). Both become trainable by reference.
import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Database, FileJson, LoaderCircle, Radio, Search, Sparkles, Trash2, Upload } from "lucide-react";
import {
  addHFDataset, cancelSynth, captureToDataset, createDataset, deleteCapture, deleteDataset, getCaptures, getDataset,
  getDatasets, getMethods, getSynthRun, hfInfo, importTraces, previewHF, startSynth,
} from "@/api";
import type { CaptureInfo, DatasetMeta, HFPreview, MethodInfo, SynthStatus } from "@/api";
import { useFrontier } from "@/components/frontier-settings";
import { ConfirmDelete, EmptyState, Field, Mono, PageHeader, SectionHeader } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { allows } from "@/lib/embed";

const FORMAT_COLORS: Record<string, string> = {
  chat: "border-primary/20 bg-primary/10 text-primary",
  sharegpt: "border-primary/20 bg-primary/10 text-primary",
  instruction: "border-warning/25 bg-warning/10 text-warning",
  preference: "border-good/30 bg-good/10 text-good",
  text: "text-muted-foreground",
};

function FormatBadge({ format }: { format: string }) {
  return (
    <Badge variant="outline" className={cn("font-mono font-normal", FORMAT_COLORS[format] ?? FORMAT_COLORS.text)}>
      {format}
    </Badge>
  );
}

interface PreviewState {
  title: string; source?: string; format: string;
  columns: string[]; total: number | null | undefined;
  rows: Record<string, unknown>[];
}

export default function Datasets() {
  const [list, setList] = useState<DatasetMeta[]>([]);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<"mine" | "explore">("mine");
  const [tab, setTab] = useState<"none" | "upload" | "hf" | "traces" | "synth">("none");
  const [captures, setCaptures] = useState<CaptureInfo[]>([]);
  const [rowPreview, setRowPreview] = useState<PreviewState | null>(null);
  const [previewing, setPreviewing] = useState<string | null>(null);

  const refresh = () => {
    getDatasets().then((d) => setList(d.datasets)).catch(() => {});
    getCaptures().then((c) => setCaptures(c.captures)).catch(() => {});
  };
  useEffect(() => { refresh(); }, []);

  async function previewRow(d: DatasetMeta) {
    setPreviewing(d.dataset_id);
    setRowPreview(null);
    try {
      if (d.source === "hf") {
        const p = await previewHF(d.repo!, d.subset ?? "default", d.split ?? "train");
        setRowPreview({ title: "Dataset Preview",
          source: `Hugging Face (${d.repo} / ${d.subset} / ${d.split})`,
          format: p.format, columns: p.columns, total: p.total, rows: p.preview });
      } else {
        const full = await getDataset(d.dataset_id);
        setRowPreview({ title: `${full.name} · first rows`, format: full.format,
          columns: Object.keys(full.preview?.[0] ?? {}), total: full.rows,
          rows: full.preview ?? [] });
      }
    } catch (e) {
      setRowPreview({ title: "Preview failed", format: "?", columns: [], total: null,
        rows: [{ error: (e as Error).message }] });
    } finally { setPreviewing(null); }
  }

  const counts = {
    mine: list.filter((d) => !d.curated).length,
    explore: list.filter((d) => d.curated).length,
  };
  const filtered = list.filter((d) =>
    (view === "explore" ? !!d.curated : !d.curated) &&
    d.name.toLowerCase().includes(search.toLowerCase()));

  return (
    <>
      <PageHeader
        title="Datasets"
        description="Upload JSONL, reference a Hugging Face dataset, or synthesize one from a task description. Chat, instruction, preference, or raw text — the format is auto-detected."
        actions={allows("operator") && (
          <>
            <Button variant="outline" onClick={() => setTab("synth")}>
              <Sparkles /> Synthesize
            </Button>
            <Button variant="outline" onClick={() => setTab("traces")}>
              <FileJson /> Import traces
            </Button>
            <Button variant="outline" onClick={() => setTab("hf")}>
              <Database /> Hugging Face
            </Button>
            <Button onClick={() => setTab("upload")}>
              <Upload /> Upload
            </Button>
          </>
        )
        }
      />

      <Dialog open={tab === "upload"} onOpenChange={(o) => !o && setTab("none")}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Upload dataset</DialogTitle>
            <DialogDescription>One JSON row per line — paste it, or pick a .jsonl file.</DialogDescription>
          </DialogHeader>
          <UploadForm onDone={() => { setTab("none"); refresh(); }} />
        </DialogContent>
      </Dialog>
      <Dialog open={tab === "hf"} onOpenChange={(o) => !o && setTab("none")}>
        <DialogContent className="sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>Add a Hugging Face dataset</DialogTitle>
            <DialogDescription>Streamed preview first; nothing is downloaded until you train on it.</DialogDescription>
          </DialogHeader>
          <HFForm onDone={() => { setTab("none"); refresh(); }} />
        </DialogContent>
      </Dialog>
      <Dialog open={tab === "synth"} onOpenChange={(o) => !o && setTab("none")}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Synthesize a dataset</DialogTitle>
            <DialogDescription>
              A teacher model writes training data from a task description, optionally grounded in a document.
              It runs on the server; the result lands here like any other dataset.
            </DialogDescription>
          </DialogHeader>
          <SynthForm onDone={() => { setTab("none"); refresh(); }} />
        </DialogContent>
      </Dialog>

      <Dialog open={tab === "traces"} onOpenChange={(o) => !o && setTab("none")}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Import traces</DialogTitle>
            <DialogDescription>
              OpenTelemetry GenAI traces from an agent that's already instrumented: an OTLP JSON export, or a JSON
              array of spans. Each conversation becomes one chat example.
            </DialogDescription>
          </DialogHeader>
          <TracesForm onDone={() => { setTab("none"); refresh(); }} />
        </DialogContent>
      </Dialog>

      <div className="mb-4 flex self-start border border-border">
        {(["mine", "explore"] as const).map((v) => (
          <button key={v} type="button" onClick={() => setView(v)}
            className={cn("h-9 px-4 text-sm", view === v ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted")}>
            {v === "mine" ? "My datasets" : "Explore"}
            <span className="ml-1.5 tabular-nums opacity-70">{counts[v]}</span>
          </button>
        ))}
      </div>
      {view === "explore" && (
        <p className="mb-4 text-sm text-muted-foreground">
          Popular open datasets — curated starting points. <span className="text-foreground">Use to train</span> picks one.
        </p>
      )}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] max-w-md flex-1">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input value={search} onChange={(e) => setSearch(e.target.value)}
                 placeholder="Search datasets…" className="pl-8" />
        </div>
        <div className="ml-auto text-xs text-muted-foreground tabular-nums">
          {filtered.length} of {list.length} datasets
        </div>
      </div>

      {filtered.length === 0 ? (
        <EmptyState icon={Database} title="No datasets yet">
          Upload one, or add a Hugging Face dataset.
        </EmptyState>
      ) : (
        <div className="@container rounded-xl border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead className="hidden @2xl:table-cell">Source</TableHead>
                <TableHead>Format</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((d) => (
                <TableRow key={d.dataset_id}>
                  <TableCell className="max-w-96">
                    <div className="flex min-w-0 items-center gap-2.5">
                      <Database className="size-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />
                      <div className="min-w-0">
                        <div className="truncate font-medium">{d.name}</div>
                        <div className="truncate font-mono text-[11px] text-muted-foreground">
                          {d.source === "hf"
                            ? `${d.subset}/${d.split}${d.eval_split ? ` · eval: ${d.eval_split}` : ""}`
                            : `${d.dataset_id}${d.rows != null ? ` · ${d.rows.toLocaleString()} rows` : ""}`}
                        </div>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell className="hidden text-sm text-muted-foreground @2xl:table-cell">
                    {d.source === "hf" ? "Hugging Face" : "Upload"}
                  </TableCell>
                  <TableCell><FormatBadge format={d.format} /></TableCell>
                  <TableCell>
                    <div className="flex justify-end gap-1.5">
                      <Button size="sm"
                        onClick={() => { sessionStorage.setItem("pick.dataset", d.dataset_id); window.location.hash = "#train"; }}>
                        Use to train
                      </Button>
                      <Button size="sm" variant="outline" disabled={previewing === d.dataset_id}
                        onClick={() => previewRow(d)}>
                        {previewing === d.dataset_id ? <><LoaderCircle className="animate-spin" /> Loading…</> : "Preview"}
                      </Button>
                      {allows("operator") && (
                        <Button size="sm" variant="ghost" className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                          onClick={() => deleteDataset(d.dataset_id).then(refresh, () => {})}>
                          Delete
                        </Button>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {view === "mine" && captures.length > 0 && <Captures captures={captures} onChange={refresh} />}

      <Dialog open={!!rowPreview} onOpenChange={(o) => !o && setRowPreview(null)}>
        <DialogContent className="gap-0 p-0 sm:max-w-3xl">
          {rowPreview && (
            <>
              <DialogHeader className="border-b border-border px-4 py-3.5">
                <DialogTitle>{rowPreview.title}</DialogTitle>
              </DialogHeader>
              <PreviewBody {...rowPreview} />
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

// TracesForm reads an OTel JSON file in the browser and sends it to be turned
// into a chat dataset on the server.
function TracesForm({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const file = useRef<HTMLInputElement>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    const f = file.current?.files?.[0];
    if (!f) return setErr("Pick a .json file of traces first.");
    let parsed: unknown;
    try {
      parsed = JSON.parse(await f.text());
    } catch {
      return setErr(`${f.name} isn't valid JSON. Export the traces as OTLP JSON (or a JSON array of spans) and try again.`);
    }
    setBusy(true);
    try {
      await importTraces(name.trim() || f.name.replace(/\.json$/i, ""), parsed);
      onDone();
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="grid gap-3">
      <Field label="Name" htmlFor="tr-name" hint="Defaults to the file's name">
        <Input id="tr-name" placeholder="support-agent-traces" value={name} onChange={(e) => setName(e.target.value)} />
      </Field>
      <div className="flex flex-wrap items-center gap-2.5">
        <input ref={file} type="file" accept=".json,application/json" aria-label="Traces file"
               className="text-xs text-muted-foreground file:mr-2 file:rounded-md file:border file:border-border file:bg-card file:px-2.5 file:py-1 file:text-xs file:text-foreground hover:file:bg-surface-2" />
        <Button type="submit" className="ml-auto" disabled={busy}>
          {busy && <LoaderCircle className="animate-spin" />} {busy ? "Importing…" : "Import"}
        </Button>
      </div>
      {err && <p className="text-sm text-destructive">{err}</p>}
    </form>
  );
}

// Captures are agent traffic recorded through the frontier model (a project's
// "Connect your agent" step starts them); saving one turns its conversations
// into a dataset.
function Captures({ captures, onChange }: { captures: CaptureInfo[]; onChange: () => void }) {
  const [saving, setSaving] = useState<string | null>(null);
  const [del, setDel] = useState<CaptureInfo | null>(null);
  const [err, setErr] = useState("");
  const operator = allows("operator");

  async function save(c: CaptureInfo) {
    setSaving(c.capture_id);
    setErr("");
    try {
      await captureToDataset(c.capture_id, c.name);
      onChange();
    } catch (ex) {
      setErr(`${c.name}: ${(ex as Error).message}`);
    } finally {
      setSaving(null);
    }
  }

  return (
    <section className="mt-8">
      <SectionHeader title="Agent captures"
        description="Conversations recorded while an agent talked to your frontier model through a capture URL. Save one as a dataset to fine-tune on it." />
      {err && <p className="mb-3 text-sm text-destructive">{err}</p>}
      <div className="@container rounded-xl border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Calls</TableHead>
              <TableHead className="hidden @2xl:table-cell">Started</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {captures.map((c) => (
              <TableRow key={c.capture_id}>
                <TableCell className="max-w-80">
                  <div className="flex min-w-0 items-center gap-2.5">
                    <Radio className="size-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />
                    <span className="truncate font-medium">{c.name}</span>
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className={cn("font-normal",
                    c.status === "open" ? "border-primary/25 bg-primary/10 text-primary" : "text-muted-foreground")}>
                    {c.status === "open" ? "Listening" : "Closed"}
                  </Badge>
                </TableCell>
                <TableCell className="text-right tabular-nums">{c.calls.toLocaleString()}</TableCell>
                <TableCell className="hidden text-sm text-muted-foreground @2xl:table-cell">
                  {new Date(c.created * 1000).toLocaleString()}
                </TableCell>
                <TableCell>
                  <div className="flex justify-end gap-1.5">
                    {operator && (
                      <Button size="sm" variant="outline" disabled={!c.calls || saving === c.capture_id}
                        title={c.calls ? undefined : "Nothing captured yet"} onClick={() => save(c)}>
                        {saving === c.capture_id ? <><LoaderCircle className="animate-spin" /> Saving…</> : "Save as dataset"}
                      </Button>
                    )}
                    {operator && (
                      <Button size="icon-sm" variant="ghost" aria-label={`Delete ${c.name}`}
                        className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive" onClick={() => setDel(c)}>
                        <Trash2 />
                      </Button>
                    )}
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <ConfirmDelete what={del?.name} onClose={() => setDel(null)}
        onConfirm={() => {
          const c = del;
          setDel(null);
          if (c) deleteCapture(c.capture_id).then(onChange).catch((ex) => setErr((ex as Error).message));
        }}>
        Its recorded calls are removed and its capture URL stops working. Datasets already saved from it stay.
      </ConfirmDelete>
    </section>
  );
}

function UploadForm({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [rows, setRows] = useState("");
  const [err, setErr] = useState("");
  const file = useRef<HTMLInputElement>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    try {
      let text = rows.trim();
      const f = file.current?.files?.[0];
      if (f) text = await f.text();
      const parsed = text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
      if (!parsed.length) throw new Error("no rows — paste JSONL or pick a file");
      await createDataset(name, parsed);
      onDone();
    } catch (ex) { setErr((ex as Error).message); }
  }

  return (
    <form onSubmit={submit} className="grid gap-3">
      <Field label="Name" htmlFor="ds-name">
        <Input id="ds-name" placeholder="support-tickets-v1" value={name} onChange={(e) => setName(e.target.value)} />
      </Field>
      <Field label="Rows" htmlFor="ds-rows">
        <Textarea id="ds-rows" rows={5} value={rows} onChange={(e) => setRows(e.target.value)}
          className="max-h-64 font-mono text-xs"
          placeholder={'one JSON row per line:\n{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}'} />
      </Field>
      <div className="flex flex-wrap items-center gap-2.5">
        <input ref={file} type="file" accept=".jsonl,.json"
               className="text-xs text-muted-foreground file:mr-2 file:rounded-md file:border file:border-border file:bg-card file:px-2.5 file:py-1 file:text-xs file:text-foreground hover:file:bg-surface-2" />
        <Button type="submit" className="ml-auto">Upload</Button>
      </div>
      {err && <p className="text-sm text-destructive">{err}</p>}
    </form>
  );
}

function HFForm({ onDone }: { onDone: () => void }) {
  const [repo, setRepo] = useState("");
  const [configs, setConfigs] = useState<string[]>([]);
  const [splits, setSplits] = useState<string[]>([]);
  const [subset, setSubset] = useState("");
  const [split, setSplit] = useState("");
  const [evalSplit, setEvalSplit] = useState("");  // "" = None
  const [advanced, setAdvanced] = useState(false);
  const [preview, setPreview] = useState<HFPreview | null>(null);
  const [loadingInfo, setLoadingInfo] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // debounce repo → fetch configs + splits, populate the dropdowns
  useEffect(() => {
    const r = repo.trim();
    if (!r || !r.includes("/")) { setConfigs([]); setSplits([]); return; }
    const t = setTimeout(async () => {
      setLoadingInfo(true); setErr(""); setPreview(null);
      try {
        const info = await hfInfo(r);
        setConfigs(info.configs);
        setSubset(info.subset ?? "");
        setSplits(info.splits);
        setSplit(info.splits.includes("train") ? "train" : (info.splits[0] ?? ""));
        setEvalSplit("");
      } catch (ex) { setConfigs([]); setSplits([]); setErr((ex as Error).message); }
      finally { setLoadingInfo(false); }
    }, 500);
    return () => clearTimeout(t);
  }, [repo]);

  // subset change → refetch its splits
  async function pickSubset(s: string) {
    setSubset(s); setPreview(null);
    try {
      const info = await hfInfo(repo.trim(), s);
      setSplits(info.splits);
      setSplit(info.splits.includes("train") ? "train" : (info.splits[0] ?? ""));
      setEvalSplit("");
    } catch (ex) { setErr((ex as Error).message); }
  }

  async function doPreview() {
    if (!repo.trim() || !split) return;
    setBusy(true); setErr(""); setPreview(null);
    try {
      setPreview(await previewHF(repo.trim(), subset, split));
    } catch (ex) { setErr((ex as Error).message); }
    finally { setBusy(false); }
  }
  async function add() {
    if (!preview) return;
    try {
      await addHFDataset(repo.trim(), subset, split, preview.format, evalSplit.trim());
      onDone();
    } catch (ex) { setErr((ex as Error).message); }
  }

  const ready = configs.length > 0 || splits.length > 0;

  return (
    <div className="grid min-w-0 gap-3">
      <Field label="Repo" htmlFor="hf-repo">
        <div className="relative">
          <Input id="hf-repo" placeholder="org/dataset (e.g. openai/gsm8k)" value={repo}
                 onChange={(e) => setRepo(e.target.value)} className="pr-24 font-mono" />
          {loadingInfo && (
            <span className="absolute top-1/2 right-2.5 flex -translate-y-1/2 items-center gap-1 text-xs text-muted-foreground">
              <LoaderCircle className="size-3 animate-spin" /> loading…
            </span>
          )}
        </div>
      </Field>
      <div className={cn("grid grid-cols-3 gap-2", !ready && "pointer-events-none opacity-40")}>
        <Field label="Subset">
          <select value={subset} onChange={(e) => pickSubset(e.target.value)}
                  className="w-full font-mono text-sm">
            {configs.length === 0 && <option value="">—</option>}
            {configs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Train split">
          <select value={split} onChange={(e) => setSplit(e.target.value)}
                  className="w-full font-mono text-sm">
            {splits.length === 0 && <option value="">—</option>}
            {splits.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="Eval split">
          <select value={evalSplit} onChange={(e) => setEvalSplit(e.target.value)}
                  className="w-full font-mono text-sm">
            <option value="">None</option>
            {splits.filter((s) => s !== split).map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </Field>
      </div>

      <button type="button" onClick={() => setAdvanced((v) => !v)}
              className="flex w-fit items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
        {advanced ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />} Advanced
      </button>
      {advanced && (
        <div className="border-l-2 border-border pl-3 text-xs text-muted-foreground">
          Format is auto-detected from the streamed rows. Need column mapping or a
          row range? Do it once via the SDK (<Mono className="text-foreground/80">Dataset.from_hf(...)</Mono>)
          and upload the result.
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button variant="outline" onClick={doPreview} disabled={busy || !split}>
          {busy ? <><LoaderCircle className="animate-spin" /> Loading…</> : "Preview"}
        </Button>
        {preview && <Button onClick={add}>Add to library</Button>}
        {err && <span className="text-sm text-destructive">{err}</span>}
      </div>

      {preview && (
        <PreviewCard
          title="Dataset preview"
          source={`Hugging Face (${repo} / ${subset} / ${split})`}
          format={preview.format}
          columns={preview.columns}
          total={preview.total}
          rows={preview.preview} />
      )}
    </div>
  );
}

// The preview body (meta grid + rows) — no outer chrome, drops into a dialog
// or a bordered card.
function PreviewBody({ source, format, columns, total, rows }: {
  source?: string; format: string; columns: string[];
  total: number | null | undefined; rows: Record<string, unknown>[];
}) {
  return (
    <>
      <div className="grid gap-x-8 gap-y-1.5 border-b border-border px-4 py-3 text-xs sm:grid-cols-2">
        {source && <Meta label="Source" value={source} />}
        <Meta label="Format" value={<FormatBadge format={format} />} />
        <Meta label="Total rows" value={total != null ? total.toLocaleString() : "—"} />
        <Meta label="Columns" value={columns.join(", ") || "—"} />
      </div>
      <div className="scrollbar-thin max-h-[55vh] space-y-1 overflow-auto p-4 font-mono text-[11px] text-foreground/80">
        {rows.map((r, i) => (
          <div key={i} className="border-b border-border/60 pb-1 break-words whitespace-pre-wrap">
            {JSON.stringify(r)}
          </div>
        ))}
      </div>
    </>
  );
}

// Bordered card with a title — used inline inside the HF add-flow.
function PreviewCard(props: {
  title: string; source?: string; format: string; columns: string[];
  total: number | null | undefined; rows: Record<string, unknown>[];
}) {
  return (
    <div className="rise min-w-0 overflow-hidden rounded-lg border border-border">
      <div className="border-b border-border bg-surface/40 px-4 py-2.5 text-sm font-medium">{props.title}</div>
      <PreviewBody {...props} />
    </div>
  );
}

function Meta({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 shrink-0 text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate font-mono text-foreground/90">{value}</span>
    </div>
  );
}

// ---- synthesize: make a dataset instead of bringing one --------------------------
// A teacher model writes the data from a task description (optionally grounded
// in a document). The run happens on the server; this polls it and lands the
// result in the library. The frontier model saved in settings is the default
// teacher, so its key never has to be pasted again.
function SynthForm({ onDone }: { onDone: () => void }) {
  const { frontier } = useFrontier();
  const [name, setName] = useState("");
  const [task, setTask] = useState("");
  const [doc, setDoc] = useState("");
  const [rows, setRows] = useState("100");
  const [method, setMethod] = useState("lora");
  const [minScore, setMinScore] = useState("0.6");
  const [kind, setKind] = useState<"frontier" | "openai" | "local">("openai");
  const [model, setModel] = useState("gpt-4o");
  const [baseUrl, setBaseUrl] = useState("");
  const [key, setKey] = useState("");
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [run, setRun] = useState<SynthStatus | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => { getMethods().then((m) => setMethods(m.methods)).catch(() => {}); }, []);
  useEffect(() => { if (frontier) setKind((k) => (k === "openai" ? "frontier" : k)); }, [frontier]);

  useEffect(() => {
    if (run?.status !== "running") return;
    const timer = setInterval(() => { getSynthRun(run.synth_id).then(setRun).catch(() => {}); }, 1000);
    return () => clearInterval(timer);
  }, [run?.synth_id, run?.status]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    const n = Number(rows), min = Number(minScore);
    if (!task.trim() && !doc.trim()) return setErr("Describe the task, or paste a document to ground it in.");
    if (!Number.isInteger(n) || n < 1) return setErr("Rows must be a whole number of 1 or more.");
    if (!(min >= 0 && min <= 1)) return setErr("Min score is between 0 and 1 (0 turns the judge gate off).");
    setBusy(true);
    try {
      const teacher = kind === "frontier" ? { kind: "frontier" as const, model: frontier?.model ?? "" }
        : { kind, model, base_url: baseUrl || undefined, api_key: key || undefined };
      const { synth_id } = await startSynth({
        name: name.trim(), n, method, min_score: min,
        task: task.trim() || undefined, document: doc.trim() || undefined,
        teacher,
      });
      setRun({ synth_id, name, status: "running", kept: 0, requested: n });
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (run) {
    // a batch in flight has no kept rows yet: follow the live phase counter
    const live = run.status === "running" && !!run.total;
    const pct = live
      ? Math.round((100 * (run.done ?? 0)) / Math.max(1, run.total!))
      : Math.round((100 * run.kept) / Math.max(1, run.requested));
    const phases: Record<string, string> = {
      starting: "Starting", planning: "Planning scenarios", generating: "Generating", judging: "Judging", kept: "Collecting",
    };
    const planning = run.phase === "planning" && live;
    return (
      <div className="grid gap-3">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="font-medium">
            {run.status === "running" ? phases[run.phase ?? "starting"] ?? run.phase
              : run.status === "succeeded" ? "Done" : run.status === "stopped" ? "Stopped" : "Failed"}
            {live && run.phase !== "planning" && <span className="text-muted-foreground tabular-nums"> · {run.done}/{run.total}</span>}
          </span>
          <span className="text-xs text-muted-foreground tabular-nums">
            {run.kept} of {run.requested} rows kept{run.tokens ? ` · ${run.tokens.toLocaleString()} tokens` : ""}
          </span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
          <div className={cn("h-full rounded-full bg-primary transition-all duration-300", planning && "motion-safe:animate-pulse")}
               style={{ width: `${planning ? 100 : pct}%` }} />
        </div>
        {run.error && <p className="text-sm text-destructive">{run.error}</p>}
        {!!run.logs?.length && (
          <pre className="max-h-56 overflow-auto bg-surface p-3 font-mono text-[11px] whitespace-pre-wrap scrollbar-thin">{run.logs.join("\n")}</pre>
        )}
        <div className="flex justify-end">
          {run.status === "running" ? (
            // stopping keeps the rows already generated, so this is safe to offer
            <Button variant="outline" disabled={cancelling}
                    onClick={() => { setCancelling(true); cancelSynth(run.synth_id).catch(() => {}); }}>
              {cancelling && <LoaderCircle className="animate-spin" />} {cancelling ? "Stopping…" : "Stop and keep what's done"}
            </Button>
          ) : (
            <Button onClick={onDone}>Done</Button>
          )}
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="grid gap-3">
      <Field label="Name" htmlFor="sy-name">
        <Input id="sy-name" placeholder="billing-triage-synth" value={name} onChange={(e) => setName(e.target.value)} />
      </Field>
      <Field label="Task" htmlFor="sy-task" hint="What the model should learn, in plain English.">
        <Textarea id="sy-task" rows={3} value={task} onChange={(e) => setTask(e.target.value)}
          placeholder="Triage customer billing emails: classify urgency, draft a reply, escalate refunds over $200." />
      </Field>
      <Field label="Ground it in a document" htmlFor="sy-doc" hint="Optional: answers must be supported by this text.">
        <Textarea id="sy-doc" rows={3} value={doc} onChange={(e) => setDoc(e.target.value)} placeholder="Paste reference material here" />
      </Field>
      <div className="grid gap-3 sm:grid-cols-3">
        <Field label="Rows" htmlFor="sy-rows">
          <Input id="sy-rows" type="number" min={1} value={rows} onChange={(e) => setRows(e.target.value)} />
        </Field>
        <Field label="For method" htmlFor="sy-method" hint="Picks the output shape.">
          <select id="sy-method" value={method} onChange={(e) => setMethod(e.target.value)}>
            {(methods.length ? methods.map((m) => m.name) : ["lora"]).map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </Field>
        <Field label="Min score" htmlFor="sy-min" hint="The judge's gate; 0 turns it off.">
          <Input id="sy-min" type="number" min={0} max={1} step={0.1} value={minScore} onChange={(e) => setMinScore(e.target.value)} />
        </Field>
      </div>
      <Field label="Teacher" htmlFor="sy-kind" hint="Who writes the data.">
        <select id="sy-kind" value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
          {frontier && <option value="frontier">Your frontier model · {frontier.model}</option>}
          <option value="openai">Another OpenAI-compatible API</option>
          <option value="local">A model on this machine</option>
        </select>
      </Field>
      {kind !== "frontier" && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Model" htmlFor="sy-model">
            <Input id="sy-model" placeholder={kind === "local" ? "Qwen/Qwen2.5-7B-Instruct" : "gpt-4o"} value={model} onChange={(e) => setModel(e.target.value)} />
          </Field>
          {kind === "openai" && (
            <Field label="Base URL" htmlFor="sy-url" hint="Blank uses api.openai.com.">
              <Input id="sy-url" placeholder="https://api.openai.com/v1" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
            </Field>
          )}
        </div>
      )}
      {kind === "openai" && (
        <Field label="API key" htmlFor="sy-key" hint="Used for this run only, never stored.">
          <Input id="sy-key" type="password" autoComplete="off" placeholder="sk-…" value={key} onChange={(e) => setKey(e.target.value)} />
        </Field>
      )}
      {err && <p className="text-sm text-destructive">{err}</p>}
      <div className="flex justify-end">
        <Button type="submit" disabled={busy}>
          {busy ? <LoaderCircle className="animate-spin" /> : <Sparkles />} Synthesize
        </Button>
      </div>
    </form>
  );
}
