// Dataset library — upload JSONL, or reference a HuggingFace dataset (with a
// streamed preview before you add it). Both become trainable by reference.
import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Database, LoaderCircle, Search, Upload } from "lucide-react";
import {
  addHFDataset, createDataset, deleteDataset, getDataset, getDatasets, hfInfo, previewHF,
} from "@/api";
import type { DatasetMeta, HFPreview } from "@/api";
import { EmptyState, Field, Mono, PageHeader } from "@/components/common";
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
  const [tab, setTab] = useState<"none" | "upload" | "hf">("none");
  const [rowPreview, setRowPreview] = useState<PreviewState | null>(null);
  const [previewing, setPreviewing] = useState<string | null>(null);

  const refresh = () => getDatasets().then((d) => setList(d.datasets)).catch(() => {});
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
        description="Upload JSONL, or reference a Hugging Face dataset. Chat, instruction, preference, or raw text — the format is auto-detected."
        actions={allows("operator") && (
          <>
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
