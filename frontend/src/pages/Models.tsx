// Model library — catalog + recently trained, search and family filters.
import { useCallback, useEffect, useMemo, useState } from "react";
import { Box, Check, Download, LoaderCircle, Plus, Search, TriangleAlert, X } from "lucide-react";

import {
  addCustomModel, downloadModel, getDownloads, getModels, removeCustomModel,
} from "@/api";
import type { CatalogModel, DownloadStatus } from "@/api";
import { EmptyState, Mono, PageHeader } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

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

  const refresh = useCallback(() => {
    getModels().then((m) => {
      setCatalog(m.catalog);
      setRecent(m.recent.filter((r) => !m.catalog.some((c) => c.id === r)));
      setBackend(m.server_backend);
    }).catch(() => {});
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  async function addModel(id: string) {
    const v = id.trim();
    if (!v) return;
    try { await addCustomModel(v); setFree(""); refresh(); } catch { /* surfaced elsewhere */ }
  }
  async function removeModel(id: string) {
    try { await removeCustomModel(id); refresh(); } catch { /* ignore */ }
  }

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
        title="Models"
        description={`Any open model on the Hugging Face hub works; these are good starting points. Server backend: ${backend}.`}
        actions={
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); addModel(free); }}>
            <Input value={free} onChange={(e) => setFree(e.target.value)}
                   placeholder="org/model, any HF id" className="w-56 font-mono" />
            <Button type="submit" disabled={!free.trim()}><Plus /> Add</Button>
          </form>
        }
      />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative max-w-sm min-w-[220px] flex-1">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input value={search} onChange={(e) => setSearch(e.target.value)}
                 placeholder="Search models…" className="pl-8" />
        </div>
        <Tabs value={family} onValueChange={setFamily}>
          <TabsList>
            {families.map((f) => (
              <TabsTrigger key={f} value={f} className="px-2.5 capitalize">{f}</TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      {filtered.length === 0 ? (
        <EmptyState icon={Box} title="No models match">
          Clear the search, or add any Hugging Face model id above.
        </EmptyState>
      ) : (
        <div className="grid grid-cols-1 gap-3 @2xl:grid-cols-2 @5xl:grid-cols-3">
          {filtered.map((m) => {
            const dl = downloads[m.id];
            const onDisk = m.cached || dl?.state === "ready";
            const downloading = dl?.state === "downloading";
            return (
              <div key={m.id}
                   className="flex flex-col border border-border bg-card p-4 transition-colors hover:border-border-strong">
                <div className="mb-2 flex items-start justify-between gap-2">
                  <span className="grid size-7 shrink-0 place-items-center rounded-md border border-border bg-surface/60">
                    <Box className="size-3.5 text-muted-foreground" strokeWidth={1.75} />
                  </span>
                  <span className="flex flex-wrap items-center justify-end gap-1">
                    {onDisk && (
                      <Badge variant="outline" className="border-good/30 bg-good/10 font-normal text-good">
                        <Check /> On disk
                      </Badge>
                    )}
                    {m.dev && <Badge variant="secondary" className="font-normal">Dev pick</Badge>}
                    {m.gated && (
                      <Badge variant="outline" className="border-warning/30 bg-warning/10 font-normal text-warning">
                        HF token
                      </Badge>
                    )}
                    {m.custom && (
                      <Button variant="ghost" size="icon-xs" title="Remove from library" aria-label={`Remove ${m.id}`}
                              className="text-muted-foreground hover:text-destructive"
                              onClick={() => removeModel(m.id)}>
                        <X />
                      </Button>
                    )}
                  </span>
                </div>
                <div className="truncate text-sm font-semibold">{m.id.split("/").pop()}</div>
                <div className="mt-0.5 truncate text-muted-foreground">
                  <Mono>{m.id}</Mono>{m.params ? <span className="text-xs"> · {m.params}</span> : ""}
                </div>
                {m.note && <div className="mt-0.5 text-xs text-muted-foreground">{m.note}</div>}

                {downloading && (
                  <div className="mt-3">
                    <div className="h-1 overflow-hidden rounded-full bg-muted">
                      <div className="h-full bg-primary transition-all" style={{ width: `${dl.pct ?? 5}%` }} />
                    </div>
                    <div className="mt-1 flex items-center gap-1 text-xs text-muted-foreground tabular-nums">
                      <LoaderCircle className="size-3 animate-spin" />
                      {dl.pct != null
                        ? `${dl.pct}% · ${fmtGB(dl.downloaded)} / ${fmtGB(dl.total)}`
                        : "downloading…"}
                    </div>
                  </div>
                )}
                {dl?.state === "error" && (
                  <div className="mt-2 flex items-center gap-1 truncate text-xs text-destructive" title={dl.error ?? ""}>
                    <TriangleAlert className="size-3 shrink-0" /> {dl.error}
                  </div>
                )}

                <div className="mt-auto flex gap-2 pt-4">
                  <Button className="flex-1" onClick={() => pick("model", m.id, "#train")}>
                    Fine-tune
                  </Button>
                  <Button variant="outline" onClick={() => pick("model", m.id, "#playground")}>
                    Try
                  </Button>
                  {!onDisk && !downloading && (
                    <Button variant="outline" size="icon" title="Prefetch weights to disk"
                            aria-label={`Download ${m.id}`} onClick={() => startDownload(m.id)}>
                      <Download />
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
