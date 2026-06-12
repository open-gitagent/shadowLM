// The pairing stage. Not "select a model" — assemble a pair: base ↔ shadow.
// With both set, every prompt plays to the pair: the shadow answers first,
// the base follows, side by side.
import { useEffect, useRef, useState } from "react";
import { chat, getJobs, getModels } from "../api";
import type { CatalogModel, JobSummary } from "../api";
import { Dots, Pill } from "../ui";

interface Turn {
  role: "user" | "assistant";
  content?: string | null;
  compare?: boolean;
  tuned?: string | null;
  base?: string | null;
}

const DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit";
const primary = "rounded-lg bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-3.5 py-2 font-bold text-white";

export default function Playground() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [recent, setRecent] = useState<string[]>([]);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [adapter, setAdapter] = useState<string | null>(null);
  const [duet, setDuet] = useState(true);
  const [panel, setPanel] = useState<"base" | "shadow" | null>(null);
  const [gear, setGear] = useState(false);
  const [q, setQ] = useState("");
  const [sys, setSys] = useState("");
  const [temp, setTemp] = useState(0.7);
  const [maxNew, setMaxNew] = useState(256);
  const [msgs, setMsgs] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getJobs().then((j) => setJobs(j.jobs));
    getModels().then((m) => {
      setCatalog(m.catalog);
      setRecent(m.recent.filter((r) => !m.catalog.some((c) => c.id === r)));
    });
    const pm = sessionStorage.getItem("pick.model");
    const pa = sessionStorage.getItem("pick.adapter");
    if (pm) { setModel(pm); sessionStorage.removeItem("pick.model"); }
    if (pa) { setAdapter(pa); setDuet(true); sessionStorage.removeItem("pick.adapter"); }
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = 1e9;
  }, [msgs]);

  const shadowJob = jobs.find((j) => j.job_id === adapter);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    if (busy || !input.trim()) return;
    const text = input.trim();
    setInput("");
    const compare = Boolean(duet && adapter);
    const history: { role: string; content: string }[] = [];
    if (sys) history.push({ role: "system", content: sys });
    for (const m of [...msgs, { role: "user", content: text } as Turn]) {
      if (m.role === "user") history.push({ role: "user", content: m.content! });
      else history.push({ role: "assistant",
        content: m.compare ? (m.tuned ?? "") : (m.content ?? "") });
    }
    const turn: Turn = compare
      ? { role: "assistant", compare: true, tuned: null, base: null }
      : { role: "assistant", content: null };
    setMsgs((cur) => [...cur, { role: "user", content: text }, turn]);
    setBusy(true);
    const ask = (ad: string | null) =>
      chat({ model, adapter: ad, messages: history, max_new_tokens: maxNew,
             temperature: temp, top_p: 0.95 })
        .then((o) => o.text).catch((err: Error) => `⚠ ${err.message}`);
    try {
      if (compare) {
        const tuned = await ask(adapter);   // the shadow answers first
        setMsgs((cur) => cur.map((m) => (m === turn ? { ...turn, tuned } : m)));
        const base = await ask(null);
        setMsgs((cur) => cur.map((m) =>
          (m.compare && m.tuned === tuned && m.base == null) ? { ...m, base } : m));
      } else {
        const content = await ask(adapter);
        setMsgs((cur) => cur.map((m) => (m === turn ? { ...turn, content } : m)));
      }
    } finally { setBusy(false); }
  }

  const panelRows = panel === "base"
    ? [...recent.map((id) => ({ id, meta: "recently trained here", dev: false, gated: false })),
       ...catalog.map((m) => ({ id: m.id, meta: `${m.params ?? ""}${m.note ? ` · ${m.note}` : ""}`,
                                dev: !!m.dev, gated: !!m.gated }))]
        .filter((r) => r.id.toLowerCase().includes(q.toLowerCase()))
    : [];
  const panelRuns = panel === "shadow"
    ? jobs.filter((j) => j.status === "succeeded" &&
        (j.job_id.includes(q) || (j.base_model || "").toLowerCase().includes(q.toLowerCase())))
    : [];

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* the pairing bar */}
      <div className="flex flex-wrap items-center gap-2.5 pb-3.5">
        <button onClick={() => { setPanel(panel === "base" ? null : "base"); setQ(""); }}
          className="flex items-center gap-2 rounded-xl border border-seam bg-gradient-to-b from-[#272019] to-umbra px-4 py-2.5 hover:border-faded">
          <span className="text-[10.5px] uppercase tracking-[.14em] text-faded">base</span>
          {model.split("/").pop()} ⌄
        </button>
        <span className="text-faded">↔</span>
        <button onClick={() => { setPanel(panel === "shadow" ? null : "shadow"); setQ(""); }}
          className={`flex items-center gap-2 rounded-xl px-4 py-2.5
            ${adapter
              ? "border border-heart bg-gradient-to-b from-[#272019] to-umbra shadow-[0_0_0_1px_#e5484d55,0_6px_26px_#e5484d2e]"
              : "border border-dashed border-seam text-faded hover:border-faded"}`}>
          <span className="text-[10.5px] uppercase tracking-[.14em] text-faded">shadow</span>
          {adapter ? `${adapter.slice(0, 8)} · ${shadowJob?.method ?? ""}` : "none"} ⌄
        </button>
        {adapter && (
          <span className="flex overflow-hidden rounded-full border border-seam text-[12px]">
            <button onClick={() => setDuet(true)}
              className={duet ? "bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-3 py-1.5 text-white" : "px-3 py-1.5 text-faded"}>
              side by side</button>
            <button onClick={() => setDuet(false)}
              className={!duet ? "bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-3 py-1.5 text-white" : "px-3 py-1.5 text-faded"}>
              finetune only</button>
          </span>
        )}
        <span className="flex-1" />
        <button className="rounded-lg border border-seam px-3 py-1.5" onClick={() => setGear(!gear)}>⚙</button>
        <button className="rounded-lg border border-seam px-3 py-1.5" onClick={() => setMsgs([])}>clear</button>
      </div>

      {gear && (
        <div className="flex flex-wrap gap-2.5 pb-3">
          <input className="w-80" placeholder="system prompt" value={sys}
                 onChange={(e) => setSys(e.target.value)} />
          <input type="number" step={0.1} min={0} max={2} value={temp} title="temperature"
                 className="w-20" onChange={(e) => setTemp(parseFloat(e.target.value) || 0.7)} />
          <input type="number" min={1} value={maxNew} title="max new tokens"
                 className="w-24" onChange={(e) => setMaxNew(+e.target.value || 256)} />
        </div>
      )}

      {panel && (
        <div className="drop mb-3.5 rounded-2xl border border-seam bg-gradient-to-b from-panel to-[#241d17] p-3 shadow-[0_18px_44px_#0008]">
          <input className="search mb-2 w-full bg-ink" autoFocus value={q}
            placeholder={panel === "base" ? "search models, or type any HF id…" : "search your runs…"}
            onChange={(e) => setQ(e.target.value)} />
          <div className="max-h-72 overflow-y-auto">
            {panel === "base" && panelRows.map((r) => (
              <div key={r.id} onClick={() => { setModel(r.id); setAdapter(null); setPanel(null); }}
                   className="flex cursor-pointer items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-[13px] hover:bg-[#332a21]">
                <span>{r.id}</span>
                <span className="flex items-center gap-1.5 text-right text-[11px] text-faded">
                  {r.dev && <Pill tone="green">dev pick</Pill>}
                  {r.gated && <Pill tone="gold">HF token</Pill>}
                  {r.meta}
                </span>
              </div>
            ))}
            {panel === "base" && q && !panelRows.some((r) => r.id.toLowerCase() === q.toLowerCase()) && (
              <div onClick={() => { setModel(q); setAdapter(null); setPanel(null); }}
                   className="flex cursor-pointer items-center justify-between rounded-lg px-2.5 py-2 text-[13px] hover:bg-[#332a21]">
                <span>use “{q}”</span><span className="text-[11px] text-faded">any HF hub id</span>
              </div>
            )}
            {panel === "shadow" && (
              <div onClick={() => { setAdapter(null); setPanel(null); }}
                   className="flex cursor-pointer items-center justify-between rounded-lg px-2.5 py-2 text-[13px] hover:bg-[#332a21]">
                <span>none</span><span className="text-[11px] text-faded">base model only</span>
              </div>
            )}
            {panel === "shadow" && panelRuns.map((j) => (
              <div key={j.job_id}
                   onClick={() => { setAdapter(j.job_id); setModel(j.base_model); setDuet(true); setPanel(null); }}
                   className="flex cursor-pointer items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-[13px] hover:bg-[#332a21]">
                <span>{j.job_id.slice(0, 10)} <Pill tone="red">{j.method ?? ""}</Pill></span>
                <span className="text-right text-[11px] text-faded">
                  {(j.base_model || "").split("/").pop()}
                  {j.final_loss != null ? ` · loss ${j.final_loss.toFixed(3)}` : ""}
                </span>
              </div>
            ))}
            {panel === "shadow" && panelRuns.length === 0 && (
              <div className="px-2.5 py-2 text-[11px] text-faded">
                no finetuned runs yet — train one, then come back</div>
            )}
          </div>
        </div>
      )}

      {/* transcript / empty stage */}
      <div ref={logRef} className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto py-3">
        {msgs.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center">
            <div className="size-[104px] overflow-hidden rounded-[30px] border border-[#4a3a30]
                            shadow-[0_0_70px_#e5484d40,0_0_18px_#e5484d33,0_22px_50px_#000a]
                            animate-[breathe_4.5s_ease-in-out_infinite]">
              <img src="/logo.png" alt="" className="size-full" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight">
              Talk to what you trained<span className="text-heart">.</span>
            </h1>
            <div className="flex flex-wrap justify-center gap-2">
              <span className="rounded-full border border-seam bg-umbra/60 px-3.5 py-1 text-[12px] text-faded">
                base · {model.split("/").pop()}</span>
              {adapter ? (
                <span className="rounded-full border border-heart/40 px-3.5 py-1 text-[12px] text-heart">
                  shadow · {adapter.slice(0, 8)} — side by side</span>
              ) : (
                <span onClick={() => setPanel("shadow")}
                      className="cursor-pointer rounded-full border border-heart/40 px-3.5 py-1 text-[12px] text-heart hover:border-heart">
                  pick a finetuned run — does it cast the same shadow? ›</span>
              )}
              <span className="rounded-full border border-seam bg-umbra/60 px-3.5 py-1 text-[12px] text-faded">
                runs on this server · nothing leaves it</span>
            </div>
          </div>
        ) : (
          msgs.map((m, i) => {
            if (m.role === "user")
              return <div key={i} className="rise max-w-[75%] self-end whitespace-pre-wrap rounded-xl bg-panel px-3 py-2 text-[13px]">{m.content}</div>;
            if (m.compare)
              return (
                <div key={i} className="rise grid grid-cols-2 gap-2.5">
                  <div className="rounded-xl border border-heart/55 bg-gradient-to-b from-[#272019] to-[#211b15] px-3.5 py-2.5 text-[13px] shadow-[0_0_0_1px_#e5484d33,0_6px_24px_#e5484d1f]">
                    <b className="mb-1.5 block text-[10.5px] uppercase tracking-[.12em] text-heart">shadow ♥</b>
                    <span className="whitespace-pre-wrap">{m.tuned == null ? <Dots /> : m.tuned}</span>
                  </div>
                  <div className="rounded-xl border border-seam bg-gradient-to-b from-[#272019] to-[#211b15] px-3.5 py-2.5 text-[13px]">
                    <b className="mb-1.5 block text-[10.5px] uppercase tracking-[.12em] text-faded">base</b>
                    <span className="whitespace-pre-wrap">{m.base == null ? <Dots /> : m.base}</span>
                  </div>
                </div>
              );
            return (
              <div key={i} className="rise max-w-[75%] whitespace-pre-wrap rounded-xl border border-seam bg-umbra px-3 py-2 text-[13px]">
                <span className="font-bold text-heart">slm♥ </span>
                {m.content == null ? <Dots /> : m.content}
              </div>
            );
          })
        )}
      </div>

      <form onSubmit={send}
            className="mt-3 flex items-center gap-2.5 rounded-2xl border border-seam bg-gradient-to-b from-[#241d17] to-[#201a14] py-1.5 pl-4 pr-1.5 transition-shadow focus-within:border-heart/55 focus-within:shadow-[0_0_0_1px_#e5484d33,0_8px_30px_#e5484d14]">
        <span className="font-bold text-heart">you ›</span>
        <input className="flex-1 border-none bg-transparent px-0 py-2 focus:outline-none"
               placeholder={adapter && duet ? "one prompt, two answers…" : "say something…"}
               value={input} onChange={(e) => setInput(e.target.value)} autoFocus />
        <button className={primary}>›</button>
      </form>
    </div>
  );
}
