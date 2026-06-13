// Dataset library — table layout, format badges, real upload.
import { useEffect, useRef, useState } from "react";
import { Database, Search, Upload } from "lucide-react";
import { createDataset, deleteDataset, getDataset, getDatasets } from "../api";
import type { DatasetMeta } from "../api";
import { PageHeader, btnGhost, btnPrimary } from "../ui";

const FORMAT_COLORS: Record<string, string> = {
  chat: "bg-primary/10 text-primary border-primary/30",
  sharegpt: "bg-primary/10 text-primary border-primary/30",
  instruction: "bg-warning/10 text-warning border-warning/30",
  preference: "bg-success/10 text-success border-success/30",
  text: "bg-muted text-muted-foreground border-border",
};

export default function Datasets() {
  const [list, setList] = useState<DatasetMeta[]>([]);
  const [search, setSearch] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [preview, setPreview] = useState<DatasetMeta | null>(null);
  const [name, setName] = useState("");
  const [rows, setRows] = useState("");
  const [err, setErr] = useState("");
  const file = useRef<HTMLInputElement>(null);

  const refresh = () => getDatasets().then((d) => setList(d.datasets)).catch((e) => setErr(e.message));
  useEffect(() => { refresh(); }, []);

  const filtered = list.filter((d) => d.name.toLowerCase().includes(search.toLowerCase()));

  async function upload(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    try {
      let text = rows.trim();
      const f = file.current?.files?.[0];
      if (f) text = await f.text();
      const parsed = text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
      if (!parsed.length) throw new Error("no rows — paste JSONL or pick a file");
      await createDataset(name, parsed);
      setName(""); setRows(""); setShowUpload(false);
      if (file.current) file.current.value = "";
      refresh();
    } catch (ex) { setErr((ex as Error).message); }
  }

  return (
    <>
      <PageHeader
        eyebrow="Library"
        title="Datasets"
        description="Upload once, train many times. JSONL — chat, instruction, preference, or raw text; the format is auto-detected."
        actions={
          <button onClick={() => setShowUpload((v) => !v)} className={btnPrimary}>
            <Upload className="size-3.5" /> Upload dataset
          </button>
        }
      />

      <div className="px-8 py-6 max-w-[1400px] space-y-4">
        {showUpload && (
          <form onSubmit={upload}
                className="rise rounded-lg border border-border bg-card p-4 grid gap-2.5 max-w-2xl">
            <input placeholder="name (e.g. support-tickets-v1)" value={name}
                   onChange={(e) => setName(e.target.value)} />
            <textarea rows={6} value={rows} onChange={(e) => setRows(e.target.value)}
              placeholder={'one JSON row per line:\n{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}'} />
            <div className="flex items-center gap-2.5">
              <input ref={file} type="file" accept=".jsonl,.json" className="text-[12px]" />
              <button className={btnPrimary}>upload ›</button>
              {err && <span className="text-xs text-destructive">{err}</span>}
            </div>
          </form>
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
          <div className="grid grid-cols-[1fr_110px_110px_260px] px-4 py-2.5 border-b border-border text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            <div>Name</div><div>Format</div><div className="text-right">Rows</div>
            <div className="text-right">Actions</div>
          </div>
          {filtered.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">
              no datasets yet — upload one above
            </div>
          )}
          {filtered.map((d) => (
            <div key={d.dataset_id}
                 className="grid grid-cols-[1fr_110px_110px_260px] px-4 py-3 border-b border-border last:border-0 items-center text-sm hover:bg-accent/30 transition-colors">
              <div className="flex items-center gap-2.5 min-w-0">
                <Database className="size-4 text-muted-foreground shrink-0" />
                <div className="min-w-0">
                  <div className="font-medium truncate">{d.name}</div>
                  <div className="text-[10px] font-mono text-muted-foreground">{d.dataset_id}</div>
                </div>
              </div>
              <div>
                <span className={`inline-block px-2 py-0.5 text-[10px] font-mono rounded border ${
                  FORMAT_COLORS[d.format] ?? FORMAT_COLORS.text}`}>
                  {d.format}
                </span>
              </div>
              <div className="text-right font-mono text-xs text-muted-foreground">
                {d.rows.toLocaleString()}
              </div>
              <div className="flex justify-end gap-2">
                <button className={btnPrimary}
                  onClick={() => { sessionStorage.setItem("pick.dataset", d.dataset_id); window.location.hash = "#train"; }}>
                  Use to train
                </button>
                <button className={btnGhost}
                  onClick={() => getDataset(d.dataset_id).then(setPreview)}>
                  Preview
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

        {preview && (
          <div className="rise rounded-lg border border-border bg-card overflow-hidden">
            <div className="px-4 py-2.5 border-b border-border flex items-center justify-between">
              <div className="text-sm font-semibold">{preview.name} · first rows</div>
              <button className="text-xs text-primary" onClick={() => setPreview(null)}>close</button>
            </div>
            <pre className="p-4 text-[11px] font-mono text-foreground/80 overflow-auto max-h-72 scrollbar-thin">
              {preview.preview?.map((r) => JSON.stringify(r)).join("\n")}
            </pre>
          </div>
        )}
      </div>
    </>
  );
}
