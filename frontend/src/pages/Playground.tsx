// Playground — a clean chat. The model picker is a command palette: one
// search across base open models AND your shadows, grouped. Pick a shadow and a
// quiet "shadow mode" toggle appears — the shadow answers next to its base.
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Check, ChevronDown, ChevronRight, LoaderCircle, MessagesSquare, RotateCcw, Search, TriangleAlert } from "lucide-react";
import { chat, getCheckpoints, getJobs, getModels, prewarm } from "@/api";
import type { CatalogModel, Checkpoint, JobSummary } from "@/api";
import { Dots } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

type Msg = { role: "user" | "assistant"; content: string };

export default function Playground() {
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [model, setModel] = useState("mlx-community/Qwen2.5-0.5B-Instruct-4bit");
  const [adapter, setAdapter] = useState<string | null>(null);
  const [ckpts, setCkpts] = useState<Checkpoint[]>([]);
  const [ckptStep, setCkptStep] = useState<number | null>(null);  // null = final
  const [compare, setCompare] = useState(false);
  const [pop, setPop] = useState(false);
  const [q, setQ] = useState("");
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [base, setBase] = useState<Msg[]>([]);  // base-model replies in compare mode
  const [busy, setBusy] = useState(false);
  const [warming, setWarming] = useState(false);
  const [warmErr, setWarmErr] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Prewarm the picked model(s) so no chat request ever hits a cold load —
  // proxies (Cloudflare) cut requests around 100s, and a cold 8B takes longer.
  useEffect(() => {
    let stop = false;
    setWarmErr("");
    const targets: (string | null)[] = compare && adapter ? [adapter, null]
      : [adapter];
    const poll = async () => {
      try {
        const states = await Promise.all(
          targets.map((ad) => prewarm(model, ad, ad ? ckptStep : null)));
        if (stop) return;
        const failed = states.find((s) => s.error);
        if (failed) { setWarming(false); setWarmErr(failed.error!); return; }
        if (states.every((s) => s.ready)) { setWarming(false); return; }
        setWarming(true);
        setTimeout(poll, 3000);
      } catch { if (!stop) setWarming(false); }  // old server: no endpoint, no state
    };
    poll();
    return () => { stop = true; };
  }, [model, adapter, ckptStep, compare]);

  useEffect(() => {
    getModels().then((m) => {
      const ids = [...m.recent, ...m.catalog.map((c) => c.id)];
      setModels([...new Set(ids)].map((id) => m.catalog.find((c) => c.id === id) ?? { id }));
      // the built-in default is an mlx model — wrong on a torch/CUDA server
      if (m.server_backend !== "mlx" && !sessionStorage.getItem("pick.model")) {
        const fit = ids.find((id) => !id.startsWith("mlx-community/"));
        if (fit) setModel((cur) => cur.startsWith("mlx-community/") ? fit : cur);
      }
    }).catch(() => {});
    getJobs().then(({ jobs }) => {
      setJobs(jobs);
      const pa = sessionStorage.getItem("pick.adapter");
      const pm = sessionStorage.getItem("pick.model");
      if (pm) { setModel(pm); sessionStorage.removeItem("pick.model"); }
      if (pa) { setAdapter(pa); setCompare(true); sessionStorage.removeItem("pick.adapter"); }
    }).catch(() => {});
  }, []);
  useEffect(() => { inputRef.current?.focus(); });
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = 1e9; }, [msgs, base]);

  // a shadow's saved versions — only mid-run if it was trained with save_steps
  useEffect(() => {
    setCkpts([]); setCkptStep(null);
    if (!adapter) return;
    getCheckpoints(adapter).then(({ checkpoints }) => {
      setCkpts(checkpoints);
      const pc = sessionStorage.getItem("pick.checkpoint");  // deep-link from Runs
      if (pc && checkpoints.some((c) => !c.final && c.step === Number(pc))) setCkptStep(Number(pc));
      sessionStorage.removeItem("pick.checkpoint");
    }).catch(() => {});
  }, [adapter]);

  const done = jobs.filter((j) => j.status === "succeeded");
  const adapterJob = done.find((j) => j.job_id === adapter);
  const label = adapter
    ? (adapterJob?.name?.trim() || `${adapter.slice(0, 8)} · ${adapterJob?.method ?? "finetuned"}`)
    : model.split("/").pop();

  const hubRows = useMemo(() => {
    const ql = q.toLowerCase();
    const rows = models.filter((m) => m.id.toLowerCase().includes(ql));
    if (q && !models.some((m) => m.id.toLowerCase() === ql)) rows.push({ id: q.trim() });
    return rows;
  }, [models, q]);
  const tunedRows = done.filter((j) => {
    const ql = q.toLowerCase();
    return j.job_id.includes(q) || (j.name || "").toLowerCase().includes(ql)
      || (j.base_model || "").toLowerCase().includes(ql);
  });

  function pickHub(id: string) { setModel(id); setAdapter(null); setCompare(false); setPop(false); }
  function pickTuned(j: JobSummary) {
    setAdapter(j.job_id); setModel(j.base_model); setCompare(true); setPop(false);
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    const history = [...msgs, { role: "user", content: text } as Msg];
    setMsgs(history);
    const duet = compare && adapter;
    if (duet) setBase((b) => [...b, { role: "user", content: text }]);
    setBusy(true);
    // hide reasoning blocks (Qwen3 & co) — and keep them out of the history we
    // send back, which thinking models don't expect to see again
    const clean = (t: string) => {
      const s = t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
      return s || t.trim();  // reply was all think-block: show it rather than nothing
    };
    const ask = (ad: string | null, h: Msg[]) =>
      chat({ model, adapter: ad, checkpoint: ad ? ckptStep : null,
             messages: h, max_new_tokens: 256, temperature: 0.7, top_p: 0.95 })
        .then((o) => clean(o.text)).catch((e: Error) => `⚠ ${e.message}`);
    try {
      // fire shadow + base together; each pane fills in as its reply lands
      const shadowTurn = ask(adapter, history)
        .then((r) => setMsgs((m) => [...m, { role: "assistant", content: r }]));
      if (duet) {
        const baseTurn = ask(null, [...base, { role: "user", content: text }])
          .then((r) => setBase((m) => [...m, { role: "assistant", content: r }]));
        await Promise.all([shadowTurn, baseTurn]);
      } else {
        await shadowTurn;
      }
    } finally { setBusy(false); }
  }

  const empty = msgs.length === 0;

  return (
    <div className="flex min-h-0 flex-1 !shrink flex-col overflow-hidden border border-border bg-card">
      {/* model selector */}
      <div className="flex h-14 shrink-0 items-center gap-3 border-b border-border px-4">
        <MessagesSquare className="size-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />
        <button onClick={() => { setPop((v) => !v); setQ(""); }}
          className="flex min-w-0 items-center gap-2 rounded-md px-1.5 py-1 text-sm font-semibold transition-colors hover:bg-surface-2">
          <span className="text-xs font-normal text-muted-foreground">{adapter ? "Shadow" : "Base"}</span>
          <span className={cn("truncate", adapter && "text-primary")}>{label}</span>
          <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
        </button>
        {adapter && (
          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
            <input type="checkbox" checked={compare}
                   onChange={(e) => setCompare(e.target.checked)} />
            Shadow mode <span className="text-muted-foreground/70">(base ↔ shadow)</span>
          </label>
        )}
        {adapter && ckpts.length > 1 && (
          <div className="flex items-center gap-0.5 rounded-lg border border-border p-0.5 font-mono text-[11px]">
            <span className="px-1.5 text-muted-foreground select-none">ckpt</span>
            {ckpts.map((c) => {
              const on = c.final ? ckptStep === null : ckptStep === c.step;
              return (
                <button key={c.path} title={c.path}
                  onClick={() => setCkptStep(c.final ? null : c.step)}
                  className={cn("rounded-md px-2 py-0.5 transition-colors",
                    on ? "bg-primary/10 text-primary" : "text-muted-foreground hover:text-foreground")}>
                  {c.final ? "final" : c.step}
                </button>
              );
            })}
          </div>
        )}
        {msgs.length > 0 && (
          <Button variant="ghost" size="sm" className="ml-auto text-muted-foreground"
                  onClick={() => { setMsgs([]); setBase([]); }}>
            <RotateCcw /> Clear
          </Button>
        )}
      </div>

      {/* model picker — a command palette, not a dropdown: one search across
          base models AND your shadows, grouped. */}
      <Dialog open={pop} onOpenChange={setPop}>
        <DialogContent showCloseButton={false}
          className="top-[11vh] translate-y-0 gap-0 overflow-hidden p-0 sm:max-w-2xl data-open:zoom-in-100 data-closed:zoom-out-100">
          <DialogTitle className="sr-only">Pick a model</DialogTitle>
          <DialogDescription className="sr-only">Filter base models or your shadows, or paste any Hugging Face id.</DialogDescription>
          {/* prompt line */}
          <div className="flex items-center gap-2.5 border-b border-border px-4 py-3">
            <Search className="size-4 shrink-0 text-muted-foreground" />
            <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} data-slot="palette-input"
              onKeyDown={(e) => {
                if (e.key === "Escape") setPop(false);
                if (e.key === "Enter") {
                  if (hubRows[0] && (!tunedRows.length || q)) pickHub(hubRows[0].id);
                  else if (tunedRows[0]) pickTuned(tunedRows[0]);
                }
              }}
              placeholder="Filter base models or your shadows · paste any HF id"
              className="flex-1 border-0 bg-transparent p-0 text-sm outline-none placeholder:text-muted-foreground" />
            <kbd className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">esc</kbd>
          </div>

          <div className="max-h-[58vh] overflow-auto py-1.5 scrollbar-thin">
            {/* base models */}
            <div className="px-4 pt-2 pb-1 text-[11px] font-medium text-muted-foreground">Base · open models</div>
            {hubRows.map((m) => {
              const on = m.id === model && !adapter;
              return (
                <button key={m.id} onClick={() => pickHub(m.id)}
                  className="group flex w-full items-center gap-3 px-4 py-2 text-left text-sm hover:bg-subtle">
                  <span className="w-3.5 shrink-0">
                    {on ? <Check className="size-3.5 text-primary" />
                        : <ChevronRight className="size-3.5 text-muted-foreground/40 group-hover:text-primary" />}
                  </span>
                  <span className={cn("flex-1 truncate font-mono text-[13px]", on && "text-primary")}>{m.id}</span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {m.dev ? "dev pick" : m.gated ? "HF token" : m.params ?? m.note ?? ""}
                  </span>
                </button>
              );
            })}

            {/* shadows */}
            <div className="mt-1.5 border-t border-border px-4 pt-3 pb-1 text-[11px] font-medium text-muted-foreground">Shadows · your runs</div>
            {tunedRows.length ? tunedRows.map((j) => {
              const on = j.job_id === adapter;
              return (
                <button key={j.job_id} onClick={() => pickTuned(j)}
                  className="group flex w-full items-center gap-3 px-4 py-2 text-left text-sm hover:bg-subtle">
                  <span className="w-3.5 shrink-0">
                    {on ? <Check className="size-3.5 text-primary" />
                        : <ChevronRight className="size-3.5 text-muted-foreground/40 group-hover:text-primary" />}
                  </span>
                  <span className={cn("flex-1 truncate", on && "text-primary")}>
                    {j.name?.trim() || <span className="font-mono text-[13px]">{j.job_id.slice(0, 10)}</span>}
                    <span className="text-muted-foreground"> · {j.method}</span>
                  </span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    shadows {(j.base_model || "").split("/").pop()}
                  </span>
                </button>
              );
            }) : (
              <div className="px-4 py-3 text-sm text-muted-foreground">
                No shadows yet. <a href="#train" className="font-medium text-primary hover:underline">Train one</a> to see it here.
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* transcript or greeting */}
      <div ref={logRef} className="min-h-0 flex-1 overflow-auto px-6 scrollbar-thin"
           onClick={() => pop && setPop(false)}>
        {empty ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
            {/* The brain: red on the light theme, white on the dark one. */}
            <img src="/logo.png" alt=""
                 className="mb-2 size-28 dark:hidden [filter:drop-shadow(0_0_40px_#e5484d55)_drop-shadow(0_0_14px_#e5484d44)]" />
            <img src="/logo1.png" alt=""
                 className="mb-2 hidden size-28 dark:block [filter:drop-shadow(0_0_40px_#ffffff33)_drop-shadow(0_0_14px_#ffffff22)]" />
            <h1 className="text-xl font-semibold tracking-[-0.015em]">
              {adapter ? "Does it cast the same shadow?" : "Talk to a model"}
            </h1>
            <p className="text-[13px] text-muted-foreground">
              Base <span className="font-mono text-foreground">{model.split("/").pop()}</span>
              {adapter && <> · shadow <span className="font-mono text-primary">{adapter.slice(0, 8)}</span></>}
            </p>
          </div>
        ) : compare && adapter ? (
          <div className="mx-auto max-w-4xl space-y-4 py-6">
            {msgs.map((m, i) => m.role === "user" ? (
              <UserBubble key={i} text={m.content} />
            ) : (
              <div key={i} className="grid grid-cols-2 gap-3">
                <Pane tone="tuned" label="Shadow" text={m.content} />
                <Pane tone="base" label="Base" text={base[i]?.content} />
              </div>
            ))}
            {busy && <div className="grid grid-cols-2 gap-3"><Pane tone="tuned" label="Shadow" /><Pane tone="base" label="Base" /></div>}
          </div>
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-4 py-6">
            {msgs.map((m, i) => m.role === "user"
              ? <UserBubble key={i} text={m.content} />
              : <div key={i} className="text-[13px] leading-relaxed">
                  <div className="mb-1 font-mono text-[11px] text-muted-foreground">{label}</div>
                  <div className="whitespace-pre-wrap">{m.content}</div>
                </div>)}
            {busy && <Dots />}
          </div>
        )}
      </div>

      {/* input */}
      <div className="shrink-0 border-t border-border p-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea ref={inputRef} value={input} rows={1} data-slot="composer"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={warming ? "Warming up the model — first load can take a couple of minutes…"
                                 : "Say something to the shadow…"}
            className="field-sizing-content max-h-40 min-h-9 min-w-0 grow resize-none rounded-lg border border-input bg-transparent px-3 py-2 text-[13px] outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50" />
          <Button size="icon" className="size-9" aria-label="Send" onClick={send} disabled={!input.trim() || busy || warming}>
            {busy ? <LoaderCircle className="size-3.5 animate-spin" /> : <ArrowUp className="size-4" />}
          </Button>
        </div>
        <div className="mx-auto mt-1.5 max-w-3xl text-center text-[11px] text-muted-foreground">
          {warmErr ? <span className="inline-flex items-center gap-1 text-destructive"><TriangleAlert className="size-3" /> Can't load this model here: {warmErr}</span>
            : warming ? <span className="inline-flex items-center gap-1 text-primary"><LoaderCircle className="size-3 animate-spin" /> Loading weights onto the GPU — send unlocks when it's hot</span>
            : "Enter to send · Shift+Enter for a new line"}
        </div>
      </div>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] border border-border bg-subtle px-3 py-2 text-[13px] whitespace-pre-wrap">{text}</div>
    </div>
  );
}

function Pane({ tone, label, text }: { tone: "tuned" | "base"; label: string; text?: string }) {
  return (
    <div className={cn("border bg-card px-3.5 py-2.5 text-[13px] leading-relaxed",
      tone === "tuned" ? "border-primary/30" : "border-border")}>
      <div className={cn("mb-1.5 text-[11px] font-medium",
        tone === "tuned" ? "text-primary" : "text-muted-foreground")}>{label}</div>
      <span className="whitespace-pre-wrap">{text == null ? <Dots /> : text}</span>
    </div>
  );
}
