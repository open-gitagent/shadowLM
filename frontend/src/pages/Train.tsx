// Train — the guided flow, data first: Data → Model → Method → Tune.
// The data ranks the methods; the method shapes the form — each method exposes
// exactly its own hyperparameters (LoRA rank for adapters, beta for DPO, …).
import { useEffect, useMemo, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Play, Search } from "lucide-react";
import { getDatasets, getModels, getWorkers, submitFinetune } from "../api";
import type { WorkerInfo } from "../api";
import type { CatalogModel, DatasetMeta, MethodInfo } from "../api";
import { Field, PageHeader, btnGhost, btnPrimary } from "../ui";

const STEPS = ["Data", "Model", "Method", "Tune"] as const;
const recommend = (format?: string): string[] =>
  format === "preference" ? ["dpo"]
  : format === "text" ? ["cpt"]
  : format === "prompt" ? ["grpo"]
  : ["lora", "qlora", "dora"];

// A tunable hyperparameter (key = the TrainConfig field it sets).
interface Param {
  key: string; label: string;
  kind: "int" | "float" | "bool" | "select";
  def: string; options?: string[]; hint?: string;
}

// Core knobs every method has.
const CORE: Param[] = [
  { key: "max_steps", label: "Max steps", kind: "int", def: "60", hint: "total optimizer steps" },
  { key: "num_train_epochs", label: "Epochs", kind: "int", def: "", hint: "overrides max steps when set" },
  { key: "learning_rate", label: "Learning rate", kind: "float", def: "", hint: "blank = method default" },
  { key: "per_device_train_batch_size", label: "Batch size", kind: "int", def: "2" },
  { key: "gradient_accumulation_steps", label: "Grad accumulation", kind: "int", def: "4" },
  { key: "max_seq_length", label: "Context length", kind: "int", def: "2048" },
  { key: "save_steps", label: "Checkpoint every", kind: "int", def: "",
    hint: "save a version every N steps — test any of them later (blank = final only)" },
];

// Optimizer / schedule — relevant to every method (Advanced).
const OPTIMIZER: Param[] = [
  { key: "warmup_steps", label: "Warmup steps", kind: "int", def: "5" },
  { key: "weight_decay", label: "Weight decay", kind: "float", def: "0.01" },
  { key: "lr_scheduler_type", label: "LR scheduler", kind: "select", def: "linear",
    options: ["linear", "cosine", "constant"] },
  { key: "max_grad_norm", label: "Grad clip", kind: "float", def: "", hint: "max grad norm (torch)" },
  { key: "optim", label: "Optimizer", kind: "select", def: "adamw_8bit",
    options: ["adamw_8bit", "adamw_torch", "adafactor", "sgd"], hint: "torch; mlx uses Adam" },
  { key: "seed", label: "Seed", kind: "int", def: "3407" },
];

// Data handling — supervised (sft) methods only (Advanced).
const DATA: Param[] = [
  { key: "packing", label: "Pack sequences", kind: "bool", def: "false", hint: "torch" },
  { key: "train_on_completions", label: "Train on completions only", kind: "bool", def: "false",
    hint: "mask the prompt (mlx)" },
];

const LORA: Param[] = [
  { key: "lora_r", label: "LoRA rank", kind: "int", def: "16" },
  { key: "lora_alpha", label: "LoRA alpha", kind: "int", def: "16" },
  { key: "lora_dropout", label: "LoRA dropout", kind: "float", def: "0" },
  { key: "target_modules", label: "Target modules", kind: "select", def: "all",
    options: ["all", "attention", "mlp"] },
  { key: "use_rslora", label: "Rank-stabilized (rsLoRA)", kind: "bool", def: "false", hint: "torch" },
];

// Extra knobs by method name (on top of the adapter knobs).
const EXTRA: Record<string, Param[]> = {
  more: [
    { key: "retrieval_k", label: "Retrieval k", kind: "int", def: "2", hint: "memories per token" },
    { key: "retrieval_layers", label: "Retrieval layers", kind: "int", def: "8" },
  ],
  more_plus: [
    { key: "more_plus_k", label: "Experts / query", kind: "int", def: "1", hint: "experts merged per prompt (>1 can interfere)" },
    { key: "more_plus_expert_steps", label: "Steps / expert", kind: "int", def: "0", hint: "0 = auto (scales with model size)" },
    { key: "more_plus_group_size", label: "Rows / expert", kind: "int", def: "1", hint: "dataset rows folded into one expert" },
    { key: "lora_r", label: "Expert LoRA rank", kind: "int", def: "4" },
    { key: "lora_alpha", label: "Expert LoRA alpha", kind: "int", def: "4" },
  ],
  dpo: [{ key: "beta", label: "Beta (KL)", kind: "float", def: "0.1", hint: "higher = stay closer to reference" }],
  grpo: [
    { key: "beta", label: "Beta (KL)", kind: "float", def: "0.1" },
    { key: "grpo_group_size", label: "Group size", kind: "int", def: "4", hint: "completions per prompt" },
    { key: "grpo_max_completion_length", label: "Max completion", kind: "int", def: "256" },
  ],
  sdft: [
    { key: "sdft_alpha", label: "Alpha (divergence)", kind: "float", def: "0",
      hint: "0 = forward KL, 1 = reverse KL, between = JSD" },
    { key: "sdft_max_completion_length", label: "Max completion", kind: "int", def: "512",
      hint: "on-policy tokens sampled per prompt" },
    { key: "sdft_temperature", label: "Rollout temperature", kind: "float", def: "1",
      hint: "0 = greedy" },
  ],
  sdpo: [
    { key: "sdpo_alpha", label: "Alpha (divergence)", kind: "float", def: "0.5",
      hint: "0 = forward KL, 1 = reverse KL, 0.5 = JSD" },
    { key: "sdpo_group_size", label: "Group size", kind: "int", def: "4",
      hint: "rollouts per prompt" },
    { key: "sdpo_max_completion_length", label: "Max completion", kind: "int", def: "256" },
    { key: "sdpo_temperature", label: "Rollout temperature", kind: "float", def: "1",
      hint: "0 = greedy" },
    { key: "sdpo_success_threshold", label: "Success threshold", kind: "float", def: "1",
      hint: "reward that makes a rollout a reusable solution" },
    { key: "sdpo_teacher_ema", label: "Teacher EMA", kind: "float", def: "0.05",
      hint: "0 = frozen teacher, 1 = live student" },
  ],
  prompt: [{ key: "num_virtual_tokens", label: "Virtual tokens", kind: "int", def: "16" }],
  ptuning: [{ key: "num_virtual_tokens", label: "Virtual tokens", kind: "int", def: "16" }],
};

// The adapter kind decides the adapter knobs — cpt/dpo/grpo/qlora (all
// default-LoRA) get the full LoRA set, bottleneck gets a width, bitfit/full/
// soft-prompts get none.
function adapterParams(adapter: string | undefined): Param[] {
  if (adapter === "lora" || adapter === "dora" || adapter === "more") return LORA;
  if (adapter === "bottleneck") return [{ key: "lora_r", label: "Adapter width (r)", kind: "int", def: "16" }];
  return [];
}

// The method's own section (adapter knobs + name extras).
function methodParams(info?: MethodInfo): Param[] {
  if (!info) return [];
  return [...adapterParams(info.adapter), ...(EXTRA[info.name] ?? [])];
}

// Advanced section depends on trainer: optimizer always; data only for sft.
function advancedParams(info?: MethodInfo): Param[] {
  return [...OPTIMIZER, ...(info?.trainer === "sft" ? DATA : [])];
}

// Everything this method exposes, for config building / defaults seeding.
function allParams(info?: MethodInfo): Param[] {
  return [...CORE, ...methodParams(info), ...advancedParams(info)];
}

// Group the methods into families so the picker reads as SFT / PEFT / RL / Memory.
const FAMILY: Record<string, string> = {
  lora: "peft", qlora: "peft", dora: "peft", adapter: "peft",
  bitfit: "peft", prompt: "peft", ptuning: "peft",
  full: "sft", cpt: "sft",
  dpo: "rl", grpo: "rl", sdft: "rl", sdpo: "rl",
  more: "memory",
  more_plus: "memory",
};
const FAMILY_LABEL: Record<string, string> = {
  peft: "PEFT · parameter-efficient",
  sft: "SFT · full & continued pretraining",
  rl: "Preference · RL · distillation",
  memory: "Memory · retrieval",
  other: "Other",
};
const FAMILY_ORDER = ["peft", "sft", "rl", "memory", "other"];
const familyOf = (name: string) => FAMILY[name] ?? "other";

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
    () => Object.fromEntries([...CORE, ...OPTIMIZER].map((p) => [p.key, p.def])));
  const [lrTouched, setLrTouched] = useState(false);
  const [evalSplit, setEvalSplit] = useState(false);
  const [evalPct, setEvalPct] = useState("10");
  const [advanced, setAdvanced] = useState(false);
  const [name, setName] = useState("");
  const [workers, setWorkers] = useState<WorkerInfo[]>([]);
  const [device, setDevice] = useState("");  // "" = train on this server

  useEffect(() => {
    getDatasets().then((d) => setDatasets(d.datasets));
    getWorkers().then((w) => setWorkers(w.workers)).catch(() => {});
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

  const extraParams = methodParams(methodInfo);
  const advParams = advancedParams(methodInfo);
  const configParams = allParams(methodInfo);
  // held-out eval only applies when it's meaningful and not already provided
  const noHoldoutTrainers = ["grpo", "sdft", "sdpo"];
  const useHoldout = evalSplit && !meta?.eval_split
    && !noHoldoutTrainers.includes(methodInfo?.trainer ?? "");
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
    for (const p of configParams) {
      const raw = (vals[p.key] ?? "").trim();
      if (p.kind === "bool") { if (raw === "true") config[p.key] = true; continue; }
      if (raw === "") continue;
      config[p.key] = p.kind === "int" ? parseInt(raw)
        : p.kind === "float" ? parseFloat(raw) : raw;  // select → string
    }
    return config;
  }

  async function start() {
    if (!ready || busy) return;
    setErr(""); setBusy(true);
    try {
      const out = await submitFinetune({
        base_model: model, name: name.trim(), config: buildConfig(), dataset_id: ds,
        eval_dataset: useHoldout ? `${evalPct}%` : null, worker: device || null,
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
    for (const p of configParams) {
      const raw = (vals[p.key] ?? "").trim();
      if (p.kind === "bool") {
        if (raw === "true") { lines[lines.length - 1] += " \\"; lines.push(`  --set ${p.key}=true`); }
        continue;
      }
      if (raw === "" || (raw === p.def && p.key !== "max_steps")) continue;
      lines[lines.length - 1] += " \\";
      lines.push(headline[p.key] ? `  ${headline[p.key]} ${raw}` : `  --set ${p.key}=${raw}`);
    }
    if (useHoldout) { lines[lines.length - 1] += " \\"; lines.push(`  --eval ${evalPct}%`); }
    return lines.join("\n");
  }, [model, method, vals, configParams, evalSplit, evalPct]);

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
              <div className="space-y-4">
                {FAMILY_ORDER.map((fam) => {
                  const items = ordered.filter((m) => familyOf(m.name) === fam);
                  if (!items.length) return null;
                  return (
                    <div key={fam}>
                      <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground mb-1.5">
                        {FAMILY_LABEL[fam]}
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
                        {items.map((m) => (
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
                    </div>
                  );
                })}
              </div>
              {methodInfo && (
                <p className="mt-3 text-xs text-muted-foreground">{methodInfo.description}</p>
              )}
            </section>
          )}

          {/* step 4 — Tune (every knob this method actually uses) */}
          {step === 3 && (
            <section className="rounded-lg border border-border bg-card p-5 space-y-5">
              <div>
                <label className="text-sm font-semibold mb-1 block">Name this shadow</label>
                <input value={name} onChange={(e) => setName(e.target.value)}
                  placeholder={`e.g. ${method ?? "support"}-${(model ?? "").split("/").pop()?.split("-")[0]?.toLowerCase() || "v1"}`}
                  className="w-full font-mono text-sm" />
                <p className="text-[11px] text-muted-foreground mt-1.5">
                  what it'll be called in Runs &amp; the playground — optional, the run id is the fallback.
                </p>
              </div>

              {workers.length > 0 && (
                <div className="pt-4 border-t border-border">
                  <label className="text-sm font-semibold mb-1 block">Train on</label>
                  <select value={device} onChange={(e) => setDevice(e.target.value)}
                          className="w-full font-mono text-sm">
                    <option value="">this server</option>
                    {workers.map((w) => (
                      <option key={w.name} value={w.name} disabled={!w.online}>
                        {w.name} — {w.backend} · {w.gpu_name || w.device}
                        {w.vram_gb ? ` · ${w.vram_gb} GB` : ""}{w.online ? "" : " (offline)"}
                      </option>
                    ))}
                  </select>
                  <p className="text-[11px] text-muted-foreground mt-1.5">
                    connected devices (`shadowlm worker`) — the run streams back here either way.
                  </p>
                </div>
              )}

              <div className="pt-4 border-t border-border">
                <div className="text-sm font-semibold mb-1">Hyperparameters</div>
                <p className="text-xs text-muted-foreground">
                  everything <b className="text-foreground">{method ?? "this method"}</b> uses —
                  defaults are sensible, the {method}-specific knobs are highlighted.
                </p>
              </div>

              <ParamGrid params={CORE} vals={vals} setVal={setVal} methodInfo={methodInfo} method={method} />

              {extraParams.length > 0 && (
                <div className="pt-4 border-t border-border">
                  <div className="text-xs font-semibold text-primary mb-3 uppercase tracking-wider font-mono">
                    {method} settings
                  </div>
                  <ParamGrid params={extraParams} vals={vals} setVal={setVal} />
                </div>
              )}

              <div className="pt-4 border-t border-border">
                <button onClick={() => setAdvanced((v) => !v)}
                        className="text-xs text-muted-foreground hover:text-foreground mb-3">
                  {advanced ? "▾" : "▸"} Advanced — optimizer{methodInfo?.trainer === "sft" ? " & data" : ""}
                </button>
                {advanced && <ParamGrid params={advParams} vals={vals} setVal={setVal} />}
              </div>

              {meta?.eval_split ? (
                <div className="pt-4 border-t border-border text-sm text-muted-foreground">
                  eval uses the dataset's own <b className="text-foreground">{meta.eval_split}</b> split
                </div>
              ) : noHoldoutTrainers.includes(methodInfo?.trainer ?? "") ? null : (
                <div className="pt-4 border-t border-border space-y-2">
                  <label className="flex items-center gap-2 text-sm text-muted-foreground">
                    <input type="checkbox" checked={evalSplit} className="w-auto"
                           onChange={(e) => setEvalSplit(e.target.checked)} />
                    hold out a slice for eval — watch for overfitting, not just training loss
                  </label>
                  {evalSplit && (
                    <div className="flex items-center gap-2 pl-6 text-sm text-muted-foreground">
                      hold out
                      <input type="number" min={1} max={50} value={evalPct}
                             onChange={(e) => setEvalPct(e.target.value)}
                             className="w-16 text-center font-mono" />
                      % of the data for evaluation
                    </div>
                  )}
                </div>
              )}
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
                <SummaryRow label="Eval" value={evalSplit ? `${evalPct}% held out` : "off"} />
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

function ParamGrid({ params, vals, setVal, methodInfo, method }: {
  params: Param[]; vals: Record<string, string>;
  setVal: (k: string, v: string) => void;
  methodInfo?: MethodInfo; method?: string | null;
}) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {params.map((p) => {
        const hint = p.key === "learning_rate" && methodInfo
          ? `default for ${method}: ${methodInfo.default_lr}` : p.hint;
        if (p.kind === "bool") {
          return (
            <label key={p.key} className="flex items-center gap-2 text-sm self-end pb-2">
              <input type="checkbox" checked={vals[p.key] === "true"} className="w-auto"
                     onChange={(e) => setVal(p.key, e.target.checked ? "true" : "false")} />
              <span>{p.label}{hint && <span className="text-muted-foreground"> · {hint}</span>}</span>
            </label>
          );
        }
        return (
          <Field key={p.key} label={p.label} hint={hint}>
            {p.kind === "select" ? (
              <select value={vals[p.key] ?? p.def} className="w-full font-mono text-sm"
                      onChange={(e) => setVal(p.key, e.target.value)}>
                {p.options!.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            ) : (
              <input value={vals[p.key] ?? ""} placeholder={p.def || "method default"}
                     inputMode={p.kind === "int" ? "numeric" : "decimal"}
                     className="w-full font-mono text-sm"
                     onChange={(e) => setVal(p.key, e.target.value)} />
            )}
          </Field>
        );
      })}
    </div>
  );
}
