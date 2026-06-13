// Dataset library — upload JSONL, or reference a HuggingFace dataset (with a
// streamed preview before you add it). Both become trainable by reference.
import { useEffect, useRef, useState } from "react";
import { Database, Search, Upload } from "lucide-react";
import {
  addHFDataset, createDataset, deleteDataset, getDataset, getDatasets, previewHF,
} from "../api";
import type { DatasetMeta, HFPreview } from "../api";
import { PageHeader, btnGhost, btnPrimary } from "../ui";

const FORMAT_COLORS: Record<string, string> = {
  chat: "bg-primary/10 text-primary border-primary/30",
  sharegpt: "bg-primary/10 text-primary border-primary/30",
  instruction: "bg-warning/10 text-warning border-warning/30",
  preference: "bg-success/10 text-success border-success/30",
  text: "bg-muted text-muted-foreground border-border",
};

function Badge({ format }: { format: string }) {
  return (
    <span className={`inline-block px-2 py-0.5 text-[10px] font-mono rounded border ${
      FORMAT_COLORS[format] ?? FORMAT_COLORS.text}`}>{format}</span>
  );
}

export default function Datasets() {
  const [list, setList] = useState<DatasetMeta[]>([]);
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<"none" | "upload" | "hf">("none");
  const [localPreview, setLocalPreview] = useState<DatasetMeta | null>(null);

  const refresh = () => getDatasets().then((d) => setList(d.datasets)).catch(() => {});
  useEffect(() => { refresh(); }, []);

  const filtered = list.filter((d) => d.name.toLowerCase().includes(search.toLowerCase()));

  return (
    <>
      <PageHeader
        eyebrow="Library"
        title="Datasets"
        description="Upload JSONL, or reference a HuggingFace dataset. Chat, instruction, preference, or raw text — the format is auto-detected."
        actions={
          <>
            <button onClick={() => setTab(tab === "hf" ? "none" : "hf")}
              className={tab === "hf" ? btnPrimary : btnGhost}>
              <Database className="size-3.5" /> Hugging Face
            </button>
            <button onClick={() => setTab(tab === "upload" ? "none" : "upload")}
              className={tab === "upload" ? btnPrimary : btnGhost}>
              <Upload className="size-3.5" /> Upload
            </button>
          </>
        }
      />

      <div className="px-8 py-6 max-w-[1400px] space-y-4">
        {tab === "upload" && <UploadForm onDone={() => { setTab("none"); refresh(); }} />}
        {tab === "hf" && <HFForm onDone={() => { setTab("none"); refresh(); }} />}

        <div className="flex flex-wrap items-center gap-2">
          <div className="relative flex-1 min-w-[220px] max-w-md">
            <Search className="size-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input value={search} onChange={(e) => setSearch(e.target.value)}
                   placeholder="Search datasets…" className="w-full pl-9 pr-3 py-2 text-sm" />
          </div>
          <div className="ml-auto text-xs font-mono text-muted-foreground">
            {filtered.length} of {list.length} datasets
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card overflow-hidden">
          <div className="grid grid-cols-[1fr_110px_110px_90px_240px] px-4 py-2.5 border-b border-border text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            <div>Name</div><div>Source</div><div>Format</div>
            <div className="text-right">Rows</div><div className="text-right">Actions</div>
          </div>
          {filtered.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">
              no datasets yet — upload one, or add a HuggingFace dataset
            </div>
          )}
          {filtered.map((d) => (
            <div key={d.dataset_id}
                 className="grid grid-cols-[1fr_110px_110px_90px_240px] px-4 py-3 border-b border-border last:border-0 items-center text-sm hover:bg-accent/30 transition-colors">
              <div className="flex items-center gap-2.5 min-w-0">
                <Database className="size-4 text-muted-foreground shrink-0" />
                <div className="min-w-0">
                  <div className="font-medium truncate">{d.name}</div>
                  <div className="text-[10px] font-mono text-muted-foreground truncate">
                    {d.source === "hf" ? `${d.subset}/${d.split}` : d.dataset_id}
                  </div>
                </div>
              </div>
              <div className="text-[10px] font-mono text-muted-foreground uppercase">
                {d.source === "hf" ? "🤗 hub" : "upload"}
              </div>
              <div><Badge format={d.format} /></div>
              <div className="text-right font-mono text-xs text-muted-foreground">
                {d.rows != null ? d.rows.toLocaleString() : "—"}
              </div>
              <div className="flex justify-end gap-2">
                <button className={btnPrimary}
                  onClick={() => { sessionStorage.setItem("pick.dataset", d.dataset_id); window.location.hash = "#train"; }}>
                  Use to train
                </button>
                {d.source !== "hf" && (
                  <button className={btnGhost}
                    onClick={() => getDataset(d.dataset_id).then(setLocalPreview)}>
                    Preview
                  </button>
                )}
                <button
                  className="inline-flex items-center rounded-md border border-border bg-card px-2.5 py-2 text-xs text-destructive hover:bg-destructive/10 hover:border-destructive/30 transition-colors"
                  onClick={() => deleteDataset(d.dataset_id).then(refresh)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        {localPreview && (
          <PreviewCard
            title={`${localPreview.name} · first rows`}
            format={localPreview.format}
            columns={Object.keys(localPreview.preview?.[0] ?? {})}
            total={localPreview.rows}
            rows={localPreview.preview ?? []}
            onClose={() => setLocalPreview(null)} />
        )}
      </div>
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
    <form onSubmit={submit} className="rise rounded-lg border border-border bg-card p-4 grid gap-2.5 max-w-2xl">
      <div className="text-sm font-semibold">Upload dataset</div>
      <input placeholder="name (e.g. support-tickets-v1)" value={name} onChange={(e) => setName(e.target.value)} />
      <textarea rows={5} value={rows} onChange={(e) => setRows(e.target.value)}
        placeholder={'one JSON row per line:\n{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}'} />
      <div className="flex items-center gap-2.5">
        <input ref={file} type="file" accept=".jsonl,.json" className="text-[12px]" />
        <button className={btnPrimary}>upload ›</button>
        {err && <span className="text-xs text-destructive">{err}</span>}
      </div>
    </form>
  );
}

function HFForm({ onDone }: { onDone: () => void }) {
  const [repo, setRepo] = useState("");
  const [subset, setSubset] = useState("default");
  const [split, setSplit] = useState("train");
  const [advanced, setAdvanced] = useState(false);
  const [preview, setPreview] = useState<HFPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function doPreview() {
    if (!repo.trim()) return;
    setBusy(true); setErr(""); setPreview(null);
    try {
      setPreview(await previewHF(repo.trim(), subset, split));
    } catch (ex) { setErr((ex as Error).message); }
    finally { setBusy(false); }
  }
  async function add() {
    if (!preview) return;
    try {
      await addHFDataset(repo.trim(), subset, split, preview.format);
      onDone();
    } catch (ex) { setErr((ex as Error).message); }
  }

  return (
    <div className="rise rounded-lg border border-border bg-card p-4 grid gap-3 max-w-3xl">
      <div className="text-sm font-semibold">Hugging Face dataset</div>
      <div className="grid grid-cols-[1fr_140px_140px] gap-2">
        <label className="block">
          <div className="text-[11px] text-muted-foreground mb-1">Repo</div>
          <input placeholder="org/dataset (e.g. roneneldan/TinyStories)" value={repo}
                 onChange={(e) => setRepo(e.target.value)} className="w-full font-mono text-sm" />
        </label>
        <label className="block">
          <div className="text-[11px] text-muted-foreground mb-1">Subset</div>
          <input value={subset} onChange={(e) => setSubset(e.target.value)} className="w-full font-mono text-sm" />
        </label>
        <label className="block">
          <div className="text-[11px] text-muted-foreground mb-1">Split</div>
          <input value={split} onChange={(e) => setSplit(e.target.value)} className="w-full font-mono text-sm" />
        </label>
      </div>

      <button onClick={() => setAdvanced((v) => !v)}
              className="text-xs text-muted-foreground hover:text-foreground w-fit">
        {advanced ? "▾" : "▸"} Advanced
      </button>
      {advanced && (
        <div className="text-xs text-muted-foreground border-l-2 border-border pl-3">
          Format is auto-detected from the streamed rows. Need column mapping or a
          row range? Do it once via the SDK (<code className="text-foreground/80">Dataset.from_hf(...)</code>)
          and upload the result.
        </div>
      )}

      <div className="flex items-center gap-2.5">
        <button onClick={doPreview} disabled={busy || !repo.trim()} className={btnGhost}>
          {busy ? "loading…" : "Preview"}
        </button>
        {preview && <button onClick={add} className={btnPrimary}>Add to library ›</button>}
        {err && <span className="text-xs text-destructive">{err}</span>}
      </div>

      {preview && (
        <PreviewCard
          title="Dataset Preview"
          source={`Hugging Face (${repo} / ${subset} / ${split})`}
          format={preview.format}
          columns={preview.columns}
          total={preview.total}
          rows={preview.preview} />
      )}
    </div>
  );
}

function PreviewCard({ title, source, format, columns, total, rows, onClose }: {
  title: string; source?: string; format: string; columns: string[];
  total: number | null | undefined; rows: Record<string, unknown>[]; onClose?: () => void;
}) {
  return (
    <div className="rise rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-2.5 border-b border-border flex items-center justify-between">
        <div className="text-sm font-semibold">{title}</div>
        {onClose && <button className="text-xs text-primary" onClick={onClose}>close</button>}
      </div>
      <div className="px-4 py-3 grid sm:grid-cols-2 gap-x-8 gap-y-1.5 text-xs border-b border-border">
        {source && <Meta label="Source" value={source} />}
        <Meta label="Format" value={<Badge format={format} />} />
        <Meta label="Total rows" value={total != null ? total.toLocaleString() : "—"} />
        <Meta label="Columns" value={columns.join(", ") || "—"} />
      </div>
      <pre className="p-4 text-[11px] font-mono text-foreground/80 overflow-auto max-h-72 scrollbar-thin space-y-1">
        {rows.map((r, i) => (
          <div key={i} className="truncate hover:whitespace-normal">{JSON.stringify(r)}</div>
        ))}
      </pre>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="text-muted-foreground w-20 shrink-0">{label}:</span>
      <span className="font-mono text-foreground/90 min-w-0 truncate">{value}</span>
    </div>
  );
}
