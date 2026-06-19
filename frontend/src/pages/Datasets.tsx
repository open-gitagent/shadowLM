// Dataset library — upload JSONL, or reference a HuggingFace dataset (with a
// streamed preview before you add it). Both become trainable by reference.
import { useEffect, useRef, useState } from "react";
import { Database, Search, Upload } from "lucide-react";
import {
  addHFDataset, createDataset, deleteDataset, getDataset, getDatasets, hfInfo, previewHF,
} from "../api";
import type { DatasetMeta, HFPreview } from "../api";
import { Modal, ModalHeader, PageHeader, btnGhost, btnPrimary } from "../ui";

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

      {tab === "upload" && (
        <Modal onClose={() => setTab("none")}>
          <ModalHeader title="Upload dataset" onClose={() => setTab("none")} />
          <UploadForm onDone={() => { setTab("none"); refresh(); }} />
        </Modal>
      )}
      {tab === "hf" && (
        <Modal onClose={() => setTab("none")}>
          <ModalHeader title="Add a Hugging Face dataset" onClose={() => setTab("none")} />
          <HFForm onDone={() => { setTab("none"); refresh(); }} />
        </Modal>
      )}

      <div className="px-8 py-6 space-y-4">
        <div className="flex gap-1 p-1 bg-muted rounded-md w-fit">
          {(["mine", "explore"] as const).map((v) => (
            <button key={v} onClick={() => setView(v)}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                view === v ? "bg-card text-foreground shadow-sm" : "text-muted-foreground"}`}>
              {v === "mine" ? "My datasets" : "Explore"} ({counts[v]})
            </button>
          ))}
        </div>
        {view === "explore" && (
          <div className="text-xs text-muted-foreground">
            Popular open datasets — curated starting points. Click <b>Use to train</b> to pick one.
          </div>
        )}
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
          <div className="grid grid-cols-[1fr_90px_110px_300px] px-4 py-2.5 border-b border-border text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            <div>Name</div><div>Source</div><div>Format</div>
            <div className="text-right">Actions</div>
          </div>
          {filtered.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">
              no datasets yet — upload one, or add a HuggingFace dataset
            </div>
          )}
          {filtered.map((d) => (
            <div key={d.dataset_id}
                 className="grid grid-cols-[1fr_90px_110px_300px] px-4 py-3 border-b border-border last:border-0 items-center text-sm hover:bg-accent/30 transition-colors">
              <div className="flex items-center gap-2.5 min-w-0">
                <Database className="size-4 text-muted-foreground shrink-0" />
                <div className="min-w-0">
                  <div className="font-medium truncate">{d.name}</div>
                  <div className="text-[10px] font-mono text-muted-foreground truncate">
                    {d.source === "hf"
                      ? `${d.subset}/${d.split}${d.eval_split ? ` · eval: ${d.eval_split}` : ""}`
                      : `${d.dataset_id}${d.rows != null ? ` · ${d.rows.toLocaleString()} rows` : ""}`}
                  </div>
                </div>
              </div>
              <div className="text-[10px] font-mono text-muted-foreground uppercase">
                {d.source === "hf" ? "🤗 hub" : "upload"}
              </div>
              <div><Badge format={d.format} /></div>
              <div className="flex justify-end gap-2">
                <button className={btnPrimary}
                  onClick={() => { sessionStorage.setItem("pick.dataset", d.dataset_id); window.location.hash = "#train"; }}>
                  Use to train
                </button>
                <button className={btnGhost} disabled={previewing === d.dataset_id}
                  onClick={() => previewRow(d)}>
                  {previewing === d.dataset_id ? "loading…" : "Preview"}
                </button>
                <button
                  className="inline-flex items-center rounded-md border border-border bg-card px-2.5 py-2 text-xs text-destructive hover:bg-destructive/10 hover:border-destructive/30 transition-colors"
                  onClick={() => deleteDataset(d.dataset_id).then(refresh)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        {rowPreview && (
          <Modal onClose={() => setRowPreview(null)}>
            <ModalHeader title={rowPreview.title} onClose={() => setRowPreview(null)} />
            <PreviewBody {...rowPreview} />
          </Modal>
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
    <form onSubmit={submit} className="p-5 grid gap-2.5">
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
    <div className="p-5 grid gap-3">
      <label className="block">
        <div className="text-[11px] text-muted-foreground mb-1">Repo</div>
        <div className="relative">
          <input placeholder="org/dataset (e.g. openai/gsm8k)" value={repo}
                 onChange={(e) => setRepo(e.target.value)} className="w-full font-mono text-sm" />
          {loadingInfo && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[11px] text-muted-foreground">
              loading…</span>
          )}
        </div>
      </label>
      <div className={`grid grid-cols-3 gap-2 ${ready ? "" : "opacity-40 pointer-events-none"}`}>
        <label className="block">
          <div className="text-[11px] text-muted-foreground mb-1">Subset</div>
          <select value={subset} onChange={(e) => pickSubset(e.target.value)}
                  className="w-full font-mono text-sm">
            {configs.length === 0 && <option value="">—</option>}
            {configs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label className="block">
          <div className="text-[11px] text-muted-foreground mb-1">Train split</div>
          <select value={split} onChange={(e) => setSplit(e.target.value)}
                  className="w-full font-mono text-sm">
            {splits.length === 0 && <option value="">—</option>}
            {splits.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="block">
          <div className="text-[11px] text-muted-foreground mb-1">Eval split</div>
          <select value={evalSplit} onChange={(e) => setEvalSplit(e.target.value)}
                  className="w-full font-mono text-sm">
            <option value="">None</option>
            {splits.filter((s) => s !== split).map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
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
        <button onClick={doPreview} disabled={busy || !split} className={btnGhost}>
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

// The preview body (meta grid + rows) — no outer chrome, drops into a modal
// or a bordered card.
function PreviewBody({ source, format, columns, total, rows }: {
  source?: string; format: string; columns: string[];
  total: number | null | undefined; rows: Record<string, unknown>[];
}) {
  return (
    <>
      <div className="px-5 py-3 grid sm:grid-cols-2 gap-x-8 gap-y-1.5 text-xs border-b border-border">
        {source && <Meta label="Source" value={source} />}
        <Meta label="Format" value={<Badge format={format} />} />
        <Meta label="Total rows" value={total != null ? total.toLocaleString() : "—"} />
        <Meta label="Columns" value={columns.join(", ") || "—"} />
      </div>
      <pre className="p-5 text-[11px] font-mono text-foreground/80 overflow-auto max-h-[55vh] scrollbar-thin space-y-1">
        {rows.map((r, i) => (
          <div key={i} className="whitespace-pre-wrap break-words border-b border-border/40 pb-1">
            {JSON.stringify(r)}
          </div>
        ))}
      </pre>
    </>
  );
}

// Bordered card with a title — used inline inside the HF add-flow.
function PreviewCard(props: {
  title: string; source?: string; format: string; columns: string[];
  total: number | null | undefined; rows: Record<string, unknown>[];
}) {
  return (
    <div className="rise rounded-lg border border-border overflow-hidden">
      <div className="px-5 py-2.5 border-b border-border text-sm font-semibold">{props.title}</div>
      <PreviewBody {...props} />
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
