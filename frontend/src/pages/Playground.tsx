// Playground — a clean chat. The model picker is a command palette (slm❯): one
// search across base open models AND your shadows, grouped. Pick a shadow and a
// quiet "shadow mode" toggle appears — the shadow answers next to its base.
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, ChevronDown } from "lucide-react";
import { chat, getJobs, getModels } from "../api";
import type { CatalogModel, JobSummary } from "../api";
import { Dots } from "../ui";

type Msg = { role: "user" | "assistant"; content: string };

export default function Playground() {
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [model, setModel] = useState("mlx-community/Qwen2.5-0.5B-Instruct-4bit");
  const [adapter, setAdapter] = useState<string | null>(null);
  const [compare, setCompare] = useState(false);
  const [pop, setPop] = useState(false);
  const [q, setQ] = useState("");
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [base, setBase] = useState<Msg[]>([]);  // base-model replies in compare mode
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getModels().then((m) => {
      const ids = [...m.recent, ...m.catalog.map((c) => c.id)];
      setModels([...new Set(ids)].map((id) => m.catalog.find((c) => c.id === id) ?? { id }));
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
    const ask = (ad: string | null, h: Msg[]) =>
      chat({ model, adapter: ad, messages: h, max_new_tokens: 256, temperature: 0.7, top_p: 0.95 })
        .then((o) => o.text).catch((e: Error) => `⚠ ${e.message}`);
    try {
      const reply = await ask(adapter, history);
      setMsgs((m) => [...m, { role: "assistant", content: reply }]);
      if (duet) {
        const b = await ask(null, [...base, { role: "user", content: text }]);
        setBase((m) => [...m, { role: "assistant", content: b }]);
      }
    } finally { setBusy(false); }
  }

  const empty = msgs.length === 0;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* model selector */}
      <div className="relative flex items-center gap-3 px-6 py-4">
        <button onClick={() => { setPop((v) => !v); setQ(""); }}
          className="flex items-center gap-2 text-base font-semibold hover:text-primary transition-colors">
          <span className="font-mono text-primary">{adapter ? "shadow❯" : "base❯"}</span>
          <span className={adapter ? "text-primary" : ""}>{label}</span>
          <ChevronDown className="size-4 text-muted-foreground" />
        </button>
        {adapter && (
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer">
            <input type="checkbox" checked={compare} className="w-auto"
                   onChange={(e) => setCompare(e.target.checked)} />
            shadow mode <span className="text-muted-foreground/60">(base ↔ shadow)</span>
          </label>
        )}
        {msgs.length > 0 && (
          <button className="ml-auto text-xs text-muted-foreground hover:text-foreground"
                  onClick={() => { setMsgs([]); setBase([]); }}>clear</button>
        )}

      </div>

      {/* model picker — a command palette, not a dropdown: one search across
          base models AND your shadows, grouped, terminal-styled. */}
      {pop && (
        <div className="fixed inset-0 z-50 flex items-start justify-center bg-background/55 backdrop-blur-sm px-4"
             onMouseDown={() => setPop(false)}>
          <div onMouseDown={(e) => e.stopPropagation()}
               className="mt-[11vh] w-full max-w-2xl rounded-xl border border-border bg-card shadow-[0_32px_80px_#0004] overflow-hidden">
            {/* prompt line */}
            <div className="flex items-center gap-2.5 px-4 py-3.5 border-b border-border font-mono">
              <span className="text-primary font-bold select-none">slm❯</span>
              <input autoFocus value={q} onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Escape") setPop(false);
                  if (e.key === "Enter") {
                    if (hubRows[0] && (!tunedRows.length || q)) pickHub(hubRows[0].id);
                    else if (tunedRows[0]) pickTuned(tunedRows[0]);
                  }
                }}
                placeholder="filter base models or your shadows · paste any HF id"
                className="flex-1 border-0 bg-transparent p-0 text-sm focus:outline-none focus:ring-0" />
              <kbd className="text-[10px] text-muted-foreground border border-border rounded px-1.5 py-0.5">esc</kbd>
            </div>

            <div className="max-h-[58vh] overflow-auto scrollbar-thin py-1.5">
              {/* base models */}
              <div className="px-4 pt-2 pb-1 flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">base · open models</span>
                <span className="h-px flex-1 bg-border" />
              </div>
              {hubRows.map((m) => {
                const on = m.id === model && !adapter;
                return (
                  <button key={m.id} onClick={() => pickHub(m.id)}
                    className="group w-full flex items-center gap-3 px-4 py-2 text-sm hover:bg-accent/40 text-left font-mono">
                    <span className={`w-3 shrink-0 ${on ? "text-primary" : "text-muted-foreground/40 group-hover:text-primary"}`}>
                      {on ? "●" : "›"}
                    </span>
                    <span className={`truncate flex-1 ${on ? "text-primary" : ""}`}>{m.id}</span>
                    <span className="text-[11px] text-muted-foreground shrink-0">
                      {m.dev ? "dev pick" : m.gated ? "HF token" : m.params ?? m.note ?? ""}
                    </span>
                  </button>
                );
              })}

              {/* shadows */}
              <div className="px-4 pt-3 pb-1 flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-primary">shadow · your runs</span>
                <span className="h-px flex-1 bg-border" />
              </div>
              {tunedRows.length ? tunedRows.map((j) => {
                const on = j.job_id === adapter;
                return (
                  <button key={j.job_id} onClick={() => pickTuned(j)}
                    className="group w-full flex items-center gap-3 px-4 py-2 text-sm hover:bg-accent/40 text-left font-mono">
                    <span className={`w-3 shrink-0 ${on ? "text-primary" : "text-muted-foreground/40 group-hover:text-primary"}`}>
                      {on ? "●" : "›"}
                    </span>
                    <span className={`truncate flex-1 ${on ? "text-primary" : ""}`}>
                      {j.name?.trim() || j.job_id.slice(0, 10)} <span className="text-muted-foreground">· {j.method}</span>
                    </span>
                    <span className="text-[11px] text-muted-foreground shrink-0">
                      shadows {(j.base_model || "").split("/").pop()}
                    </span>
                  </button>
                );
              }) : (
                <div className="px-4 py-3 font-mono text-xs text-muted-foreground/70">
                  no shadows yet — <a href="#train" className="text-primary">train one</a> to see it here
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* transcript or greeting */}
      <div ref={logRef} className="flex-1 overflow-auto scrollbar-thin px-6"
           onClick={() => pop && setPop(false)}>
        {empty ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <div className="size-16 overflow-hidden rounded-2xl border border-primary/30 shadow-[0_0_56px_#e5484d44,0_0_16px_#e5484d33]">
              <img src="/logo.png" alt="" className="size-full" />
            </div>
            <div>
              <div className="mb-1.5 font-mono text-[11px] uppercase tracking-[0.25em] text-primary">slm♥ playground</div>
              <h1 className="text-2xl font-semibold tracking-tight">
                {adapter ? "Does it cast the same shadow?" : "Talk to a model"}
              </h1>
            </div>
            <p className="font-mono text-sm text-muted-foreground">
              base <span className="text-foreground">{model.split("/").pop()}</span>
              {adapter && <> &nbsp;·&nbsp; shadow <span className="text-primary">{adapter.slice(0, 8)}</span></>}
            </p>
          </div>
        ) : compare && adapter ? (
          <div className="mx-auto max-w-4xl py-6 space-y-4">
            {msgs.map((m, i) => m.role === "user" ? (
              <UserBubble key={i} text={m.content} />
            ) : (
              <div key={i} className="grid grid-cols-2 gap-3">
                <Pane tone="tuned" label="shadow ♥" text={m.content} />
                <Pane tone="base" label="base" text={base[i]?.content} />
              </div>
            ))}
            {busy && <div className="grid grid-cols-2 gap-3"><Pane tone="tuned" label="shadow ♥" /><Pane tone="base" label="base" /></div>}
          </div>
        ) : (
          <div className="mx-auto max-w-3xl py-6 space-y-4">
            {msgs.map((m, i) => m.role === "user"
              ? <UserBubble key={i} text={m.content} />
              : <div key={i} className="text-sm leading-relaxed whitespace-pre-wrap">
                  <span className="font-mono font-bold text-primary">slm♥ › </span>{m.content}
                </div>)}
            {busy && <Dots />}
          </div>
        )}
      </div>

      {/* input */}
      <div className="px-6 pb-6 pt-2">
        <div className="mx-auto flex max-w-3xl items-end gap-2.5 rounded-[26px] border border-border bg-card py-2.5 pl-5 pr-2.5 shadow-sm focus-within:border-primary/50 focus-within:shadow-[0_0_0_1px_#e5484d33] transition-all">
          <span className="select-none py-1.5 font-mono text-sm font-bold text-primary">you ›</span>
          <textarea ref={inputRef} value={input} rows={1}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder="say something to the shadow…"
            className="flex-1 resize-none border-0 bg-transparent py-1.5 text-sm focus:outline-none max-h-40" />
          <button onClick={send} disabled={!input.trim() || busy}
            className="grid size-9 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-colors">
            <ArrowUp className="size-4" />
          </button>
        </div>
        <div className="mx-auto mt-1.5 max-w-3xl text-center text-[10px] text-muted-foreground/70">
          runs on this server · nothing leaves it
        </div>
      </div>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl bg-primary px-4 py-2 text-sm text-primary-foreground whitespace-pre-wrap">{text}</div>
    </div>
  );
}

function Pane({ tone, label, text }: { tone: "tuned" | "base"; label: string; text?: string }) {
  return (
    <div className={`rounded-xl border bg-card px-3.5 py-2.5 text-sm ${
      tone === "tuned" ? "border-primary/40" : "border-border"}`}>
      <b className={`mb-1.5 block text-[10px] uppercase tracking-[0.12em] ${
        tone === "tuned" ? "text-primary" : "text-muted-foreground"}`}>{label}</b>
      <span className="whitespace-pre-wrap">{text == null ? <Dots /> : text}</span>
    </div>
  );
}
