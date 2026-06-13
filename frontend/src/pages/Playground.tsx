// Playground — two panes, one prompt. Left: the base model. Right: your
// finetune ("the shadow"). Compare mode sends the same prompt to both, live.
import { useEffect, useRef, useState } from "react";
import { Columns2, RotateCcw, Send } from "lucide-react";
import { chat, getJobs, getModels } from "../api";
import type { CatalogModel, JobSummary } from "../api";
import { Dots, PageHeader, btnGhost } from "../ui";

type Msg = { role: "user" | "assistant"; content: string };

export default function Playground() {
  const [compare, setCompare] = useState(true);
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [baseModel, setBaseModel] = useState("mlx-community/Qwen2.5-0.5B-Instruct-4bit");
  const [adapter, setAdapter] = useState<string>("");
  const [leftMsgs, setLeftMsgs] = useState<Msg[]>([]);
  const [rightMsgs, setRightMsgs] = useState<Msg[]>([]);
  const [leftTyping, setLeftTyping] = useState(false);
  const [rightTyping, setRightTyping] = useState(false);
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const done = jobs.filter((j) => j.status === "succeeded");
  const adapterJob = done.find((j) => j.job_id === adapter);

  useEffect(() => {
    getModels().then((m) => {
      const ids = [...m.recent, ...m.catalog.map((c) => c.id)];
      setModels([...new Set(ids)].map((id) =>
        m.catalog.find((c) => c.id === id) ?? { id }));
    }).catch(() => {});
    getJobs().then(({ jobs }) => {
      setJobs(jobs);
      const pa = sessionStorage.getItem("pick.adapter");
      const pm = sessionStorage.getItem("pick.model");
      if (pm) { setBaseModel(pm); sessionStorage.removeItem("pick.model"); }
      if (pa) { setAdapter(pa); sessionStorage.removeItem("pick.adapter"); }
      else {
        const first = jobs.find((j) => j.status === "succeeded");
        if (first) setAdapter(first.job_id);
      }
    }).catch(() => {});
    inputRef.current?.focus();
  }, []);

  // picking a finetune locks the base pane to its base model in compare mode
  useEffect(() => {
    if (compare && adapterJob) setBaseModel(adapterJob.base_model);
  }, [adapter, compare]);  // eslint-disable-line react-hooks/exhaustive-deps

  const ask = (msgs: Msg[], ad: string | null) =>
    chat({ model: adapterJob && ad ? adapterJob.base_model : baseModel,
           adapter: ad, messages: msgs, max_new_tokens: 256,
           temperature: 0.7, top_p: 0.95 })
      .then((o) => o.text)
      .catch((e: Error) => `⚠ ${e.message}`);

  const send = async () => {
    const text = input.trim();
    if (!text || leftTyping || rightTyping) return;
    setInput("");
    inputRef.current?.focus();
    const userMsg: Msg = { role: "user", content: text };
    const newLeft = [...leftMsgs, userMsg];
    setLeftMsgs(newLeft); setLeftTyping(true);
    let newRight: Msg[] = [];
    if (compare && adapter) {
      newRight = [...rightMsgs, userMsg];
      setRightMsgs(newRight); setRightTyping(true);
    }
    // shadow first (it's the one being judged), then the base — one GPU slot
    if (compare && adapter) {
      const tuned = await ask(newRight, adapter);
      setRightMsgs((p) => [...p, { role: "assistant", content: tuned }]);
      setRightTyping(false);
    }
    const base = await ask(newLeft, null);
    setLeftMsgs((p) => [...p, { role: "assistant", content: base }]);
    setLeftTyping(false);
  };

  const clear = () => { setLeftMsgs([]); setRightMsgs([]); };

  return (
    <>
      <PageHeader
        eyebrow="Playground"
        title="Does it cast the same shadow?"
        description="Compare mode sends one prompt to the base model and your finetune, side by side. The shadow answers first."
        actions={
          <>
            <button onClick={clear} className={btnGhost}>
              <RotateCcw className="size-3.5" /> Clear
            </button>
            <button onClick={() => setCompare((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded-md px-3 py-2 text-xs transition-colors ${
                compare ? "bg-primary text-primary-foreground" : "border border-border bg-card hover:bg-accent"}`}>
              <Columns2 className="size-3.5" /> Compare
            </button>
          </>
        }
      />

      <div className="flex-1 flex flex-col min-h-0 px-8 py-6 gap-4 max-w-[1600px] w-full">
        <div className={`grid gap-4 flex-1 min-h-0 ${compare ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}>
          <ChatPane
            title="Base model" subtitle="untrained reference" tone="muted"
            picker={
              <select value={baseModel} onChange={(e) => { setBaseModel(e.target.value); }}
                      disabled={compare && Boolean(adapterJob)}
                      title={compare && adapterJob ? "locked to the finetune's base in compare mode" : undefined}
                      className="text-xs font-mono px-2 py-1.5 max-w-[60%] truncate">
                {[...new Set([baseModel, ...models.map((m) => m.id)])].map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            }
            messages={leftMsgs} typing={leftTyping} />
          {compare && (
            <ChatPane
              title="Finetuned" subtitle="your trained adapter — the shadow" tone="primary"
              picker={
                <select value={adapter} onChange={(e) => setAdapter(e.target.value)}
                        className="text-xs font-mono px-2 py-1.5 max-w-[60%] truncate">
                  {done.length === 0 && <option value="">no finetuned runs yet</option>}
                  {done.map((j) => (
                    <option key={j.job_id} value={j.job_id}>
                      {(j.base_model || "").split("/").pop()} · {j.method} · {j.job_id.slice(0, 8)}
                    </option>
                  ))}
                </select>
              }
              messages={rightMsgs} typing={rightTyping} />
          )}
        </div>

        <div className="rounded-lg border border-border bg-card p-3">
          <div className="flex gap-2 items-end">
            <textarea
              ref={inputRef} value={input} rows={2}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder={compare && adapter
                ? "Ask anything — both models answer the same prompt…"
                : "Say something to the base model…"}
              className="flex-1 resize-none bg-transparent text-sm focus:outline-none border-0 py-2 px-2" />
            <button onClick={send} disabled={!input.trim() || leftTyping || rightTyping}
              className="inline-flex items-center justify-center size-9 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0">
              <Send className="size-4" />
            </button>
          </div>
          <div className="px-2 mt-1 text-[10px] font-mono text-muted-foreground/70">
            Enter to send · Shift+Enter for newline · runs on this server, nothing leaves it
          </div>
        </div>
      </div>
    </>
  );
}

function ChatPane({ title, subtitle, tone, picker, messages, typing }: {
  title: string; subtitle: string; tone: "muted" | "primary";
  picker: React.ReactNode; messages: Msg[]; typing: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, typing]);

  return (
    <div className={`rounded-lg border bg-card flex flex-col min-h-0 overflow-hidden ${
      tone === "primary" ? "border-primary/40" : "border-border"}`}>
      <header className="px-4 py-3 border-b border-border flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className={`size-2 rounded-full ${tone === "primary" ? "bg-primary" : "bg-muted-foreground"}`} />
          <div className="min-w-0">
            <div className="text-sm font-semibold">{title}</div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">{subtitle}</div>
          </div>
        </div>
        {picker}
      </header>
      <div ref={scrollRef} className="flex-1 overflow-auto scrollbar-thin p-4 space-y-4 min-h-[320px]">
        {messages.length === 0 && !typing && (
          <div className="h-full flex items-center justify-center text-center text-xs text-muted-foreground/70 py-8">
            <div>
              <div className="font-mono uppercase tracking-wider mb-1">No messages</div>
              <div>Send a prompt below to see {title.toLowerCase()} respond.</div>
            </div>
          </div>
        )}
        {messages.map((m, i) => <Message key={i} msg={m} tone={tone} />)}
        {typing && <Dots />}
      </div>
    </div>
  );
}

function Message({ msg, tone }: { msg: Msg; tone: "muted" | "primary" }) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="rise max-w-[85%] rounded-lg bg-primary text-primary-foreground px-3.5 py-2 text-sm whitespace-pre-wrap">
          {msg.content}
        </div>
      </div>
    );
  }
  return (
    <div className="rise flex gap-3">
      <div className={`size-7 shrink-0 rounded-md grid place-items-center text-xs font-mono font-bold ${
        tone === "primary"
          ? "bg-primary/15 text-primary border border-primary/30"
          : "bg-muted text-muted-foreground border border-border"}`}>
        {tone === "primary" ? "♥" : "○"}
      </div>
      <div className="text-sm leading-relaxed whitespace-pre-wrap text-foreground/90 pt-0.5">
        {msg.content}
      </div>
    </div>
  );
}
