// Train — the guided flow, data first: Data → Model → Method → Tune.
// The data ranks the methods; the method shapes the form — each method exposes
// exactly its own hyperparameters (LoRA rank for adapters, beta for DPO, …).
import { useEffect, useMemo, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Play, Search } from "lucide-react";
import { getDatasets, getModels, submitFinetune } from "../api";
import type { CatalogModel, DatasetMeta, MethodInfo } from "../api";
import { Field, PageHeader, btnGhost, btnPrimary } from "../ui";

const STEPS = ["Data", "Model", "Method", "Tune"] as const;
const recommend = (format?: string): string[] =>
  format === "preference" ? ["dpo"]
  : format === "text" ? ["cpt"]
  : format === "prompt" ? ["grpo"]
  : ["lora", "qlora", "dora"];

// A tunable hyperparameter (key = the TrainConfig field it sets).
interface Param { key: string; label: string; kind: "int" | "float"; def: string; hint?: string }

// Common to every method.
const COMMON: Param[] = [
  { key: "max_steps", label: "Max steps", kind: "int", def: "60", hint: "total optimizer steps" },
  { key: "learning_rate", label: "Learning rate", kind: "float", def: "", hint: "blank = method default" },
  { key: "per_device_train_batch_size", label: "Batch size", kind: "int", def: "2" },
  { key: "gradient_accumulation_steps", label: "Grad accumulation", kind: "int", def: "4" },
  { key: "max_seq_length", label: "Context length", kind: "int", def: "2048" },
  { key: "weight_decay", label: "Weight decay", kind: "float", def: "0.01" },
];

const LORA: Param[] = [
  { key: "lora_r", label: "LoRA rank", kind: "int", def: "16" },
  { key: "lora_alpha", label: "LoRA alpha", kind: "int", def: "16" },
  { key: "lora_dropout", label: "LoRA dropout", kind: "float", def: "0" },
];

// Extra knobs by method name (on top of whatever the adapter kind contributes).
const EXTRA: Record<string, Param[]> = {
  more: [
    { key: "retrieval_k", label: "Retrieval k", kind: "int", def: "2", hint: "memories per token" },
    { key: "retrieval_layers", label: "Retrieval layers", kind: "int", def: "8" },
  ],
  dpo: [{ key: "beta", label: "Beta (KL)", kind: "float", def: "0.1", hint: "higher = stay closer to reference" }],
  grpo: [
    { key: "beta", label: "Beta (KL)", kind: "float", def: "0.1" },
    { key: "grpo_group_size", label: "Group size", kind: "int", def: "4", hint: "completions per prompt" },
    { key: "grpo_max_completion_length", label: "Max completion", kind: "int", def: "256" },
  ],
  prompt: [{ key: "num_virtual_tokens", label: "Virtual tokens", kind: "int", def: "16" }],
  ptuning: [{ key: "num_virtual_tokens", label: "Virtual tokens", kind: "int", def: "16" }],
};

// The adapter kind decides the adapter knobs — so cpt/dpo/grpo/qlora (all
// default-LoRA) get rank/alpha/dropout, bottleneck gets a width, bitfit/full/
// soft-prompts get none.
function adapterParams(adapter: string | undefined): Param[] {
  if (adapter === "lora" || adapter === "dora" || adapter === "more") return LORA;
  if (adapter === "bottleneck") return [{ key: "lora_r", label: "Adapter width (r)", kind: "int", def: "16" }];
  return [];
}

// Everything a method exposes beyond COMMON: its adapter knobs + its extras.
function methodParams(info?: MethodInfo): Param[] {
  if (!info) return [];
  return [...adapterParams(info.adapter), ...(EXTRA[info.name] ?? [])];
}

const paramsFor = (info?: MethodInfo): Param[] => [...COMMON, ...methodParams(info)];

export default function Train({ methods }: { methods: MethodInfo[] }) {
  const [step, setStep] = useState(0);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [recent, setRecent] = useState<string[]>([]);
  const [ds, setDs] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [method, setMethod] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [free, setFree] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [vals, setVals] = useState<Record<string, string>>(
    () => Object.fromEntries(COMMON.map((p) => [p.key, p.def])));
  const [lrTouched, setLrTouched] = useState(false);
  const [evalSplit, setEvalSplit] = useState(false);

  useEffect(() => {
    getDatasets().then((d) => setDatasets(d.datasets));
    getModels().then((m) => {
      setCatalog(m.catalog);
      setRecent(m.recent.filter((r) => !m.catalog.some((c) => c.id === r)));
    });
    const pd = sessionStorage.getItem("pick.dataset");
    const pm = sessionStorage.getItem("pick.model");
    const pme = sessionStorage.getItem("pick.method");
    const pst = sessionStorage.getItem("pick.steps");
    if (pd) { setDs(pd); setStep(1); sessionStorage.removeItem("pick.dataset"); }
    if (pm) { setModel(pm); setStep((s) => Math.max(s, pd ? 2 : 0)); sessionStorage.removeItem("pick.model"); }
    if (pme) { pickMethod(pme); sessionStorage.removeItem("pick.method"); }
    if (pst) { setVals((v) => ({ ...v, max_steps: pst })); sessionStorage.removeItem("pick.steps"); }
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const meta = datasets.find((d) => d.dataset_id === ds);
  const rec = recommend(meta?.format);
  const ordered = useMemo(() => [...methods].sort((a, b) =>
    Number(rec.includes(b.name)) - Number(rec.includes(a.name))), [methods, rec]);
  const methodInfo = methods.find((m) => m.name === method);
  const allModels: CatalogModel[] = useMemo(
    () => [...recent.map((id) => ({ id, note: "recently trained here" })), ...catalog],
    [recent, catalog]);
  const filteredModels = allModels.filter((m) =>
    m.id.toLowerCase().includes(search.toLowerCase()));

  const tuneParams = paramsFor(methodInfo);
  const extraParams = methodParams(methodInfo);
  const ready = Boolean(ds && model && method);
  const canNext = [Boolean(ds), Boolean(model), Boolean(method), true][step];

  function pickMethod(name: string) {
    setMethod(name);
    const info = methods.find((x) => x.name === name);
    setVals((v) => {
      const next = { ...v };
      for (const p of methodParams(info)) if (next[p.key] === undefined) next[p.key] = p.def;
      return next;
    });
    if (info && !lrTouched) setVals((v) => ({ ...v, learning_rate: String(info.default_lr) }));
  }
  const setVal = (k: string, val: string) => {
    if (k === "learning_rate") setLrTouched(true);
    setVals((v) => ({ ...v, [k]: val }));
  };

  function buildConfig(): Record<string, unknown> {
    const config: Record<string, unknown> = { method };
    for (const p of tuneParams) {
      const raw = (vals[p.key] ?? "").trim();
      if (raw === "") continue;
      config[p.key] = p.kind === "int" ? parseInt(raw) : parseFloat(raw);
    }
    return config;
  }

  async function start() {
    if (!ready || busy) return;
    setErr(""); setBusy(true);
    try {
      const out = await submitFinetune({
        base_model: model, config: buildConfig(), dataset_id: ds,
        eval_dataset: evalSplit ? "auto" : null,
        load_in_4bit: false, max_seq_length: parseInt(vals.max_seq_length || "2048") });
      window.location.hash = `#runs/${out.job_id}`;
    } catch (ex) { setErr((ex as Error).message); setBusy(false); }
  }

  // CLI preview: headline flags inline, the rest via --set field=value
  const cli = useMemo(() => {
    const headline: Record<string, string> = {
      max_steps: "--max-steps", learning_rate: "--lr",
      per_device_train_batch_size: "--batch-size", lora_r: "--lora-r" };
    const lines = [`shadowlm finetune <data.jsonl> \\`,
                   `  --model ${model ?? "<model>"} \\`,
                   `  --method ${method ?? "<method>"}`];
    for (const p of tuneParams) {
      const raw = (vals[p.key] ?? "").trim();
      if (raw === "" || raw === p.def && p.key !== "max_steps") continue;
      lines[lines.length - 1] += " \\";
      lines.push(headline[p.key] ? `  ${headline[p.key]} ${raw}` : `  --set ${p.key}=${raw}`);
    }
    if (evalSplit) { lines[lines.length - 1] += " \\"; lines.push("  --eval auto"); }
    return lines.join("\n");
  }, [model, method, vals, tuneParams, evalSplit]);

  const summaryVal = (k: string) => (vals[k] ?? "").trim();

  return (
    <>
      <PageHeader
        title="Configure and start training"
        description="Four decisions, in the order they depend on each other — the data ranks the methods, the method shapes the form."
      />

      <div className="px-8 py-6 grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-6 max-w-[1400px]">
        <div className="space-y-6 min-w-0">
          {/* stepper */}
          <ol className="flex items-center gap-2 rounded-lg border border-border bg-card p-2">
            {STEPS.map((s, i) => {
              const isActive = i === step, isDone = i < step;
              return (
                <li key={s} className="flex items-center gap-2 flex-1">
                  <button onClick={() => (isDone || isActive) && setStep(i)}
                    className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-sm flex-1 transition-colors ${
                      isActive ? "bg-primary/10 text-foreground border border-primary/30"
                               : "text-muted-foreground hover:bg-accent/40"}`}>
                    <span className={`size-6 grid place-items-center rounded-full text-[11px] font-mono ${
                      isActive ? "bg-primary text-primary-foreground"
                      : isDone ? "bg-success/20 text-success" : "bg-muted text-muted-foreground"}`}>
                      {isDone ? <Check className="size-3" /> : i + 1}
                    </span>
                    <span className="font-medium">{s}</span>
                  </button>
                  {i < STEPS.length - 1 && <ChevronRight className="size-4 text-muted-foreground/50 shrink-0" />}
                </li>
              );
            })}
          </ol>

          {/* step 1 — Data */}
          {step === 0 && (
            <section className="rounded-lg border border-border bg-card p-5">
              <div className="text-sm font-semibold mb-1">Dataset</div>
              <p className="text-xs text-muted-foreground mb-4">
                The data decides what training even means — formats are auto-detected
                and steer the method choice.</p>
              {datasets.length === 0 ? (
                <div className="text-sm text-muted-foreground py-6 text-center">
                  no datasets on this server yet —{" "}
                  <a href="#datasets" className="text-primary">upload one</a> first
                </div>
              ) : (
                <div className="border border-border rounded-md max-h-72 overflow-auto scrollbar-thin divide-y divide-border">
                  {datasets.map((d) => (
                    <button key={d.dataset_id} onClick={() => setDs(d.dataset_id)}
                      className={`w-full text-left px-3 py-2.5 flex items-center gap-3 hover:bg-accent/40 transition-colors ${
                        ds === d.dataset_id ? "bg-accent/60" : ""}`}>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium truncate">{d.name}</div>
                        <div className="text-xs text-muted-foreground font-mono">
                          {d.format}{d.rows != null ? ` · ${d.rows.toLocaleString()} rows` : ""}</div>
                      </div>
                      {ds === d.dataset_id &&
                        <div className="text-[10px] font-mono uppercase tracking-wider text-primary">Selected</div>}
                    </button>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* step 2 — Model */}
          {step === 1 && (
            <section className="rounded-lg border border-border bg-card p-5">
              <div className="text-sm font-semibold mb-1">Base model</div>
              <p className="text-xs text-muted-foreground mb-4">
                The catalog, models trained here before, or any HF hub id.</p>
              <div className="relative mb-2">
                <Search className="size-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                <input value={search} onChange={(e) => setSearch(e.target.value)}
                       placeholder="Search models…" className="w-full pl-9 pr-3 py-2 text-sm" />
              </div>
              <div className="border border-border rounded-md max-h-64 overflow-auto scrollbar-thin divide-y divide-border">
                {filteredModels.map((m) => (
                  <button key={m.id} onClick={() => setModel(m.id)}
                    className={`w-full text-left px-3 py-2.5 flex items-center gap-3 hover:bg-accent/40 transition-colors ${
                      model === m.id ? "bg-accent/60" : ""}`}>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">{m.id}</div>
                      <div className="text-xs text-muted-foreground font-mono">
                        {m.params ?? ""}{m.note ? ` · ${m.note}` : ""}
                        {m.gated ? " · needs HF token" : ""}</div>
                    </div>
                    {model === m.id &&
                      <div className="text-[10px] font-mono uppercase tracking-wider text-primary">Selected</div>}
                  </button>
                ))}
              </div>
              <form className="mt-3 flex gap-2"
                    onSubmit={(e) => { e.preventDefault(); if (free.trim()) setModel(free.trim()); }}>
                <input value={free} onChange={(e) => setFree(e.target.value)}
                       placeholder="org/model-name — any HF hub id" className="flex-1 text-sm font-mono" />
                <button className={btnGhost}>use custom</button>
              </form>
            </section>
          )}

          {/* step 3 — Method */}
          {step === 2 && (
            <section className="rounded-lg border border-border bg-card p-5">
              <div className="text-sm font-semibold mb-1">Method</div>
              <p className="text-xs text-muted-foreground mb-4">
                your dataset is <b className="text-foreground">{meta?.format ?? "?"}</b> — recommended methods first.
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
                {ordered.map((m) => (
                  <button key={m.name} onClick={() => pickMethod(m.name)}
                    title={m.description}
                    className={`text-left px-3 py-2 rounded-md border text-sm transition-colors ${
                      method === m.name
                        ? "border-primary bg-primary/10 text-foreground"
                        : "border-border bg-card hover:border-primary/40"}`}>
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{m.name}</span>
                      {rec.includes(m.name) &&
                        <span className="text-[9px] font-mono uppercase text-primary">rec</span>}
                    </div>
                    <div className="text-[10px] text-muted-foreground font-mono mt-0.5">
                      lr {m.default_lr} · {m.trainer}</div>
                  </button>
                ))}
              </div>
              {methodInfo && (
                <p className="mt-3 text-xs text-muted-foreground">{methodInfo.description}</p>
              )}
            </section>
          )}

          {/* step 4 — Tune (spec-driven: common + this method's own params) */}
          {step === 3 && (
            <section className="rounded-lg border border-border bg-card p-5 space-y-4">
              <div>
                <div className="text-sm font-semibold mb-1">Hyperparameters</div>
                <p className="text-xs text-muted-foreground">
                  {method ?? "this method"}'s knobs — defaults are sensible.
                  {extraParams.length > 0 &&
                    ` The ${method}-specific settings are split out below.`}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {COMMON.map((p) => (
                  <Field key={p.key} label={p.label}
                    hint={p.key === "learning_rate" && methodInfo
                      ? `default for ${method}: ${methodInfo.default_lr}` : p.hint}>
                    <input value={vals[p.key] ?? ""} placeholder={p.def || "method default"}
                           inputMode={p.kind === "int" ? "numeric" : "decimal"}
                           className="w-full font-mono text-sm"
                           onChange={(e) => setVal(p.key, e.target.value)} />
                  </Field>
                ))}
              </div>

              {extraParams.length > 0 && (
                <div className="pt-4 border-t border-border">
                  <div className="text-xs font-semibold text-primary mb-3 uppercase tracking-wider font-mono">
                    {method} settings
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {extraParams.map((p) => (
                      <Field key={p.key} label={p.label} hint={p.hint}>
                        <input value={vals[p.key] ?? ""} placeholder={p.def}
                               inputMode={p.kind === "int" ? "numeric" : "decimal"}
                               className="w-full font-mono text-sm"
                               onChange={(e) => setVal(p.key, e.target.value)} />
                      </Field>
                    ))}
                  </div>
                </div>
              )}

              <label className="flex items-center gap-2 text-sm text-muted-foreground pt-2 border-t border-border">
                <input type="checkbox" checked={evalSplit} className="w-auto"
                       onChange={(e) => setEvalSplit(e.target.checked)} />
                hold out 10% for eval — see overfitting, not just training loss
              </label>
            </section>
          )}

          {/* step nav */}
          <div className="flex items-center justify-between gap-3">
            <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}
                    className={btnGhost}>
              <ChevronLeft className="size-3.5" /> Back
            </button>
            {step < STEPS.length - 1 && (
              <button onClick={() => canNext && setStep(step + 1)} disabled={!canNext}
                      className={btnPrimary}>
                Next: {STEPS[step + 1]} <ChevronRight className="size-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* right rail — live summary + generated CLI */}
        <aside className="space-y-4">
          <div className="sticky top-4 space-y-4">
            <div className="rounded-lg border border-border bg-card overflow-hidden">
              <div className="px-4 py-3 border-b border-border">
                <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground">Run summary</div>
                <div className="text-sm font-semibold mt-1">
                  {ready ? "Ready to launch" : "Working through the steps…"}</div>
              </div>
              <div className="px-4 py-3 space-y-2 text-xs">
                <SummaryRow label="Dataset" value={meta ? (meta.rows != null ? `${meta.name} (${meta.rows})` : meta.name) : "—"} />
                <SummaryRow label="Model" value={model ? model.split("/").pop()! : "—"} />
                <SummaryRow label="Method" value={method ?? "—"} />
                <SummaryRow label="Steps" value={summaryVal("max_steps") || "60"} />
                <SummaryRow label="LR" value={summaryVal("learning_rate") || "method default"} />
                <SummaryRow label="Batch" value={summaryVal("per_device_train_batch_size") || "2"} />
                {extraParams.map((p) => (
                  <SummaryRow key={p.key} label={p.label}
                    value={summaryVal(p.key) || p.def} />
                ))}
                <SummaryRow label="Eval" value={evalSplit ? "10% held out" : "off"} />
              </div>
              <div className="px-4 pb-4 pt-2 space-y-2">
                <button onClick={start} disabled={!ready || busy}
                  className={`${btnPrimary} w-full justify-center py-2.5 text-sm`}>
                  <Play className="size-4" /> {busy ? "starting…" : "Start training"}
                </button>
                <div className="text-[10px] font-mono text-muted-foreground text-center">
                  this exact config runs — nothing hidden
                </div>
                {err && <div className="text-xs text-destructive">{err}</div>}
              </div>
            </div>

            <div className="rounded-lg border border-border bg-card p-4">
              <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground mb-2">
                The same run, from your shell</div>
              <pre className="text-[11px] font-mono text-foreground/80 leading-relaxed whitespace-pre-wrap break-all">{cli}</pre>
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono truncate text-right">{value}</span>
    </div>
  );
}
