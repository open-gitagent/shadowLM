// Playground — a clean chat. Pick a model from the "Select model" popover
// (Hub models / Fine-tuned tabs), then talk. Pick a finetuned run and a quiet
// "compare to base" toggle appears — the shadow answers next to the base.
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Check, ChevronDown, Search } from "lucide-react";
import { chat, getJobs, getModels } from "../api";
import type { CatalogModel, JobSummary } from "../api";
import { Dots } from "../ui";

type Msg = { role: "user" | "assistant"; content: string };

function greeting(): string {
  const h = new Date().getHours();
  return h < 5 ? "Working late" : h < 12 ? "Good morning"
    : h < 18 ? "Good afternoon" : "Good evening";
}

export default function Playground() {
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [model, setModel] = useState("mlx-community/Qwen2.5-0.5B-Instruct-4bit");
  const [adapter, setAdapter] = useState<string | null>(null);
  const [compare, setCompare] = useState(false);
  const [pop, setPop] = useState(false);
  const [tab, setTab] = useState<"hub" | "tuned">("hub");
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
    ? `${adapter.slice(0, 8)} · ${adapterJob?.method ?? "finetuned"}`
    : model.split("/").pop();

  const hubRows = useMemo(() => {
    const ql = q.toLowerCase();
    const rows = models.filter((m) => m.id.toLowerCase().includes(ql));
    if (q && !models.some((m) => m.id.toLowerCase() === ql)) rows.push({ id: q.trim() });
    return rows;
  }, [models, q]);
  const tunedRows = done.filter((j) =>
    j.job_id.includes(q) || (j.base_model || "").toLowerCase().includes(q.toLowerCase()));

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
          className="flex items-center gap-1.5 text-base font-semibold hover:text-primary transition-colors">
          {adapter ? <span className="text-primary">{label}</span> : label}
          <ChevronDown className="size-4 text-muted-foreground" />
        </button>
        {adapter && (
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer">
            <input type="checkbox" checked={compare} className="w-auto"
                   onChange={(e) => setCompare(e.target.checked)} />
            compare to base
          </label>
        )}
        {msgs.length > 0 && (
          <button className="ml-auto text-xs text-muted-foreground hover:text-foreground"
                  onClick={() => { setMsgs([]); setBase([]); }}>clear</button>
        )}

        {pop && (
          <div className="absolute left-6 top-14 z-20 w-[520px] rounded-2xl border border-border bg-card p-3 shadow-[0_24px_64px_#0003]">
            <div className="flex rounded-full bg-muted/40 p-1 mb-2.5">
              {(["hub", "tuned"] as const).map((t) => (
                <button key={t} onClick={() => setTab(t)}
                  className={`flex-1 rounded-full py-1.5 text-sm transition-colors ${
                    tab === t ? "bg-card shadow-sm text-foreground" : "text-muted-foreground"}`}>
                  {t === "hub" ? "Hub models" : "Fine-tuned"}
                </button>
              ))}
            </div>
            <div className="relative mb-1">
              <Search className="size-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input autoFocus value={q} onChange={(e) => setQ(e.target.value)}
                placeholder={tab === "hub" ? "Search models, or type any HF id" : "Search your runs"}
                className="w-full pl-9 pr-3 py-2 text-sm bg-background" />
            </div>
            <div className="max-h-80 overflow-auto scrollbar-thin">
              {tab === "hub" ? hubRows.map((m) => (
                <button key={m.id} onClick={() => pickHub(m.id)}
                  className="w-full flex items-center justify-between gap-3 px-3 py-2.5 rounded-lg text-sm hover:bg-accent/40 text-left">
                  <span className="truncate flex items-center gap-2">
                    {m.id === model && !adapter && <Check className="size-3.5 text-primary shrink-0" />}
                    {m.id}
                  </span>
                  <span className="text-[11px] text-muted-foreground shrink-0">
                    {m.dev ? "dev pick" : m.gated ? "HF token" : m.params ?? m.note ?? ""}
                  </span>
                </button>
              )) : tunedRows.length ? tunedRows.map((j) => (
                <button key={j.job_id} onClick={() => pickTuned(j)}
                  className="w-full flex items-center justify-between gap-3 px-3 py-2.5 rounded-lg text-sm hover:bg-accent/40 text-left">
                  <span className="truncate flex items-center gap-2">
                    {j.job_id === adapter && <Check className="size-3.5 text-primary shrink-0" />}
                    {j.job_id.slice(0, 10)} · {j.method}
                  </span>
                  <span className="text-[11px] text-muted-foreground shrink-0">
                    {(j.base_model || "").split("/").pop()}
                  </span>
                </button>
              )) : (
                <div className="px-3 py-6 text-center text-xs text-muted-foreground">
                  no finetuned runs yet — train one first
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* transcript or greeting */}
      <div ref={logRef} className="flex-1 overflow-auto scrollbar-thin px-6"
           onClick={() => pop && setPop(false)}>
        {empty ? (
          <div className="flex h-full flex-col items-center justify-center gap-3">
            <img src="/logo.png" alt="" className="size-12 rounded-xl border border-border" />
            <h1 className="text-3xl font-semibold tracking-tight">{greeting()}</h1>
            <p className="text-sm text-muted-foreground">
              Talk to <b className="text-foreground">{label}</b>
              {adapter ? " — your finetuned shadowLM" : ""}.
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
              : <div key={i} className="text-sm leading-relaxed whitespace-pre-wrap">{m.content}</div>)}
            {busy && <Dots />}
          </div>
        )}
      </div>

      {/* input */}
      <div className="px-6 pb-6 pt-2">
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-[26px] border border-border bg-card py-2.5 pl-5 pr-2.5 shadow-sm focus-within:border-primary/50 transition-colors">
          <textarea ref={inputRef} value={input} rows={1}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder="Ask anything…"
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
