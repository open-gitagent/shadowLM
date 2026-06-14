// Model library — catalog + recently trained, search and family filters.
import { useEffect, useMemo, useState } from "react";
import { Check, Cpu, Download, Loader2, Search } from "lucide-react";
import { downloadModel, getDownloads, getModels } from "../api";
import type { CatalogModel, DownloadStatus } from "../api";
import { PageHeader, btnGhost, btnPrimary } from "../ui";

const fmtGB = (b?: number) => (b ? `${(b / 1e9).toFixed(b < 1e9 ? 2 : 1)} GB` : "");

function pick(kind: "model" | "adapter", value: string, dest: string) {
  sessionStorage.setItem(`pick.${kind}`, value);
  window.location.hash = dest;
}

const familyOf = (id: string) => {
  const lower = id.toLowerCase();
  if (lower.includes("qwen")) return "qwen";
  if (lower.includes("llama")) return "llama";
  if (lower.includes("gemma")) return "gemma";
  if (lower.includes("smollm")) return "smollm";
  return "other";
};

export default function Models() {
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [recent, setRecent] = useState<string[]>([]);
  const [backend, setBackend] = useState("?");
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState("all");
  const [free, setFree] = useState("");
  const [downloads, setDownloads] = useState<Record<string, DownloadStatus>>({});

  useEffect(() => {
    getModels().then((m) => {
      setCatalog(m.catalog);
      setRecent(m.recent.filter((r) => !m.catalog.some((c) => c.id === r)));
      setBackend(m.server_backend);
    }).catch(() => {});
  }, []);

  // poll download progress while anything is in flight
  useEffect(() => {
    const tick = () => getDownloads().then((d) => setDownloads(d.downloads)).catch(() => {});
    tick();
    const t = setInterval(tick, 1500);
    return () => clearInterval(t);
  }, []);

  async function startDownload(id: string) {
    setDownloads((d) => ({ ...d, [id]: { state: "downloading", total: 0 } }));
    try {
      const st = await downloadModel(id);
      setDownloads((d) => ({ ...d, [id]: st }));
    } catch { /* the poller will pick up state */ }
  }

  const all: CatalogModel[] = useMemo(
    () => [...recent.map((id) => ({ id, note: "recently trained here" })), ...catalog],
    [recent, catalog]);
  const families = ["all", ...Array.from(new Set(all.map((m) => familyOf(m.id))))];
  const filtered = all.filter((m) =>
    m.id.toLowerCase().includes(search.toLowerCase()) &&
    (family === "all" || familyOf(m.id) === family));

  return (
    <>
      <PageHeader
        eyebrow="Library"
        title="Models"
        description={`Any open model on the HuggingFace hub works — these are good starting points. Server backend: ${backend}.`}
      />

      <div className="px-8 py-6 max-w-[1400px] space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative flex-1 min-w-[220px] max-w-md">
            <Search className="size-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input value={search} onChange={(e) => setSearch(e.target.value)}
                   placeholder="Search models…" className="w-full pl-9 pr-3 py-2 text-sm" />
          </div>
          <div className="flex gap-1 p-1 bg-muted rounded-md">
            {families.map((f) => (
              <button key={f} onClick={() => setFamily(f)}
                className={`px-2.5 py-1 text-[11px] font-mono rounded transition-colors ${
                  family === f ? "bg-card text-foreground shadow-sm" : "text-muted-foreground"}`}>
                {f}
              </button>
            ))}
          </div>
          <form className="ml-auto flex gap-2"
                onSubmit={(e) => { e.preventDefault(); if (free.trim()) pick("model", free.trim(), "#train"); }}>
            <input value={free} onChange={(e) => setFree(e.target.value)}
                   placeholder="org/model — any HF id" className="text-xs font-mono w-56" />
            <button className={btnPrimary}>use ›</button>
          </form>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {filtered.map((m) => {
            const dl = downloads[m.id];
            const onDisk = m.cached || dl?.state === "ready";
            const downloading = dl?.state === "downloading";
            return (
            <div key={m.id}
                 className="rounded-lg border border-border bg-card p-4 hover:border-primary/40 transition-colors">
              <div className="flex items-start justify-between mb-2">
                <div className="size-8 rounded-md bg-primary/10 border border-primary/20 grid place-items-center">
                  <Cpu className="size-4 text-primary" />
                </div>
                <span className="flex gap-1.5 items-center">
                  {onDisk && <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded border border-success/40 text-success inline-flex items-center gap-1"><Check className="size-2.5" />on disk</span>}
                  {m.dev && <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded border border-success/40 text-success">dev pick</span>}
                  {m.gated && <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded border border-warning/40 text-warning">HF token</span>}
                </span>
              </div>
              <div className="text-sm font-semibold truncate">{m.id.split("/").pop()}</div>
              <div className="text-xs text-muted-foreground font-mono mt-0.5 truncate">
                {m.id}{m.params ? ` · ${m.params}` : ""}
              </div>
              {m.note && <div className="text-xs text-muted-foreground mt-0.5">{m.note}</div>}

              {downloading && (
                <div className="mt-3">
                  <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                    <div className="h-full bg-primary transition-all"
                         style={{ width: `${dl.pct ?? 5}%` }} />
                  </div>
                  <div className="mt-1 text-[10px] font-mono text-muted-foreground flex items-center gap-1">
                    <Loader2 className="size-3 animate-spin" />
                    {dl.pct != null
                      ? `${dl.pct}% · ${fmtGB(dl.downloaded)} / ${fmtGB(dl.total)}`
                      : "downloading…"}
                  </div>
                </div>
              )}
              {dl?.state === "error" && (
                <div className="mt-2 text-[10px] font-mono text-destructive truncate" title={dl.error ?? ""}>
                  ⚠ {dl.error}
                </div>
              )}

              <div className="mt-4 flex gap-2">
                <button onClick={() => pick("model", m.id, "#train")}
                        className={`${btnPrimary} flex-1 justify-center`}>
                  Fine-tune
                </button>
                <button onClick={() => pick("model", m.id, "#playground")} className={btnGhost}>
                  Try
                </button>
                {!onDisk && !downloading && (
                  <button onClick={() => startDownload(m.id)} className={btnGhost} title="prefetch weights to disk">
                    <Download className="size-3.5" />
                  </button>
                )}
              </div>
            </div>
            );
          })}
        </div>
      </div>
    </>
  );
}
