// Train — the guided flow, data first: Data → Model → Method → Tune.
// The data ranks the methods; the method shapes the form — each method exposes
// exactly its own hyperparameters (LoRA rank for adapters, beta for DPO, …).
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Check, ChevronDown, ChevronLeft, ChevronRight, Database, LoaderCircle, Play, Search } from "lucide-react";
import { getDatasets, getModels, getWorkers, submitFinetune } from "@/api";
import type { WorkerInfo } from "@/api";
import type { CatalogModel, DatasetMeta, MethodInfo } from "@/api";
import { EmptyState, Field, Mono, PageHeader, SectionHeader } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

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
  dpo: "rl", grpo: "rl",
  more: "memory",
  more_plus: "memory",
};
const FAMILY_LABEL: Record<string, string> = {
  peft: "PEFT · parameter-efficient",
  sft: "SFT · full & continued pretraining",
  rl: "Preference & RL",
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
  const useHoldout = evalSplit && methodInfo?.trainer !== "grpo" && !meta?.eval_split;
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
    if (!ready || busy || !model || !ds) return;  // `ready` covers this; tsc can't see it
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

      <div className="grid max-w-[1400px] grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-6">
          {/* stepper — one band split by hairlines */}
          <ol className="grid grid-cols-2 gap-px overflow-hidden border border-border bg-border @2xl:grid-cols-4">
            {STEPS.map((s, i) => {
              const isActive = i === step, isDone = i < step;
              return (
                <li key={s} className="bg-card">
                  <button onClick={() => (isDone || isActive) && setStep(i)}
                    aria-current={isActive ? "step" : undefined}
                    disabled={!isDone && !isActive}
                    className={cn(
                      "flex w-full items-center gap-2.5 px-4 py-3 text-left text-sm transition-colors disabled:cursor-default",
                      isActive ? "bg-primary/10 text-primary" : isDone ? "text-foreground hover:bg-surface" : "text-muted-foreground")}>
                    <span className={cn(
                      "grid size-5 shrink-0 place-items-center rounded-full border text-[11px] tabular-nums",
                      isActive ? "border-primary/30 bg-primary/10"
                      : isDone ? "border-good/30 bg-good/10 text-good" : "border-border")}>
                      {isDone ? <Check className="size-3" /> : i + 1}
                    </span>
                    <span className="font-medium">{s}</span>
                  </button>
                </li>
              );
            })}
          </ol>

          {/* step 1 — Data */}
          {step === 0 && (
            <Panel>
              <SectionHeader title="Dataset"
                description="The data decides what training even means — formats are auto-detected and steer the method choice." />
              {datasets.length === 0 ? (
                <EmptyState icon={Database} title="No datasets on this server yet">
                  <a href="#datasets" className="text-primary hover:underline">Upload one</a> first.
                </EmptyState>
              ) : (
                <PickList>
                  {datasets.map((d) => (
                    <PickRow key={d.dataset_id} selected={ds === d.dataset_id} onClick={() => setDs(d.dataset_id)}
                      title={d.name}
                      sub={`${d.format}${d.rows != null ? ` · ${d.rows.toLocaleString()} rows` : ""}`} />
                  ))}
                </PickList>
              )}
            </Panel>
          )}

          {/* step 2 — Model */}
          {step === 1 && (
            <Panel>
              <SectionHeader title="Base model" description="The catalog, models trained here before, or any HF hub id." />
              <div className="relative mb-2">
                <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input value={search} onChange={(e) => setSearch(e.target.value)}
                       placeholder="Search models…" className="pl-8" />
              </div>
              <PickList className="max-h-64">
                {filteredModels.map((m) => (
                  <PickRow key={m.id} selected={model === m.id} onClick={() => setModel(m.id)}
                    title={m.id}
                    sub={`${m.params ?? ""}${m.note ? ` · ${m.note}` : ""}${m.gated ? " · needs HF token" : ""}`} />
                ))}
              </PickList>
              <form className="mt-3 flex gap-2"
                    onSubmit={(e) => { e.preventDefault(); if (free.trim()) setModel(free.trim()); }}>
                <Input value={free} onChange={(e) => setFree(e.target.value)}
                       placeholder="org/model-name — any HF hub id" className="flex-1 font-mono" />
                <Button type="submit" variant="outline">Use custom</Button>
              </form>
            </Panel>
          )}

          {/* step 3 — Method */}
          {step === 2 && (
            <Panel>
              <SectionHeader title="Method"
                description={<>Your dataset is <span className="font-medium text-foreground">{meta?.format ?? "?"}</span> — recommended methods first.</>} />
              <div className="space-y-5">
                {FAMILY_ORDER.map((fam) => {
                  const items = ordered.filter((m) => familyOf(m.name) === fam);
                  if (!items.length) return null;
                  return (
                    <div key={fam}>
                      <p className="mb-1.5 text-xs font-medium text-muted-foreground">{FAMILY_LABEL[fam]}</p>
                      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
                        {items.map((m) => (
                          <button key={m.name} onClick={() => pickMethod(m.name)}
                            title={m.description}
                            className={cn(
                              "rounded-md border px-3 py-2 text-left text-sm transition-colors",
                              method === m.name
                                ? "border-primary/40 bg-primary/5"
                                : "border-border bg-card hover:bg-surface")}>
                            <div className="flex items-center justify-between gap-2">
                              <span className={cn("font-medium", method === m.name && "text-primary")}>{m.name}</span>
                              {rec.includes(m.name) && <Badge variant="default" className="h-4 px-1.5 text-[10px]">rec</Badge>}
                            </div>
                            <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                              lr {m.default_lr} · {m.trainer}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
              {methodInfo && (
                <p className="mt-4 text-sm text-muted-foreground">{methodInfo.description}</p>
              )}
            </Panel>
          )}

          {/* step 4 — Tune (every knob this method actually uses) */}
          {step === 3 && (
            <Panel className="space-y-6">
              <Field label="Name this shadow" htmlFor="run-name"
                hint="What it'll be called in Runs & the playground — optional, the run id is the fallback.">
                <Input id="run-name" value={name} onChange={(e) => setName(e.target.value)}
                  placeholder={`e.g. ${method ?? "support"}-${(model ?? "").split("/").pop()?.split("-")[0]?.toLowerCase() || "v1"}`}
                  className="font-mono" />
              </Field>

              {workers.length > 0 && (
                <Field label="Train on" htmlFor="run-device"
                  hint="Connected devices (`shadowlm worker`) — the run streams back here either way.">
                  <select id="run-device" value={device} onChange={(e) => setDevice(e.target.value)}
                          className="w-full font-mono">
                    <option value="">this server</option>
                    {workers.map((w) => (
                      <option key={w.name} value={w.name} disabled={!w.online}>
                        {w.name} — {w.backend} · {w.gpu_name || w.device}
                        {w.vram_gb ? ` · ${w.vram_gb} GB` : ""}{w.online ? "" : " (offline)"}
                      </option>
                    ))}
                  </select>
                </Field>
              )}

              <div className="border-t border-border pt-5">
                <SectionHeader title="Hyperparameters"
                  description={<>Everything <span className="font-medium text-foreground">{method ?? "this method"}</span> uses — defaults are sensible, the {method}-specific knobs are set apart below.</>} />
                <ParamGrid params={CORE} vals={vals} setVal={setVal} methodInfo={methodInfo} method={method} />
              </div>

              {extraParams.length > 0 && (
                <div className="border-t border-border pt-5">
                  <p className="mb-3 text-xs font-medium text-primary">{method} settings</p>
                  <ParamGrid params={extraParams} vals={vals} setVal={setVal} />
                </div>
              )}

              <div className="border-t border-border pt-5">
                <button onClick={() => setAdvanced((v) => !v)} aria-expanded={advanced}
                        className="mb-3 inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground">
                  <ChevronDown className={cn("size-3.5 transition-transform", !advanced && "-rotate-90")} />
                  Advanced — optimizer{methodInfo?.trainer === "sft" ? " & data" : ""}
                </button>
                {advanced && <ParamGrid params={advParams} vals={vals} setVal={setVal} />}
              </div>

              {meta?.eval_split ? (
                <div className="border-t border-border pt-5 text-sm text-muted-foreground">
                  Eval uses the dataset's own <span className="font-medium text-foreground">{meta.eval_split}</span> split.
                </div>
              ) : methodInfo?.trainer === "grpo" ? null : (
                <div className="space-y-3 border-t border-border pt-5">
                  <div className="flex items-center gap-2.5">
                    <Switch id="eval-split" checked={evalSplit} onCheckedChange={setEvalSplit} />
                    <Label htmlFor="eval-split" className="font-normal text-muted-foreground">
                      Hold out a slice for eval — watch for overfitting, not just training loss
                    </Label>
                  </div>
                  {evalSplit && (
                    <div className="flex items-center gap-2 pl-11 text-sm text-muted-foreground">
                      Hold out
                      <Input type="number" min={1} max={50} value={evalPct}
                             onChange={(e) => setEvalPct(e.target.value)}
                             className="w-16 text-center font-mono" />
                      % of the data for evaluation
                    </div>
                  )}
                </div>
              )}
            </Panel>
          )}

          {/* step nav */}
          <div className="flex items-center justify-between gap-3">
            <Button variant="outline" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>
              <ChevronLeft /> Back
            </Button>
            {step < STEPS.length - 1 && (
              <Button onClick={() => canNext && setStep(step + 1)} disabled={!canNext}>
                Next: {STEPS[step + 1]} <ChevronRight />
              </Button>
            )}
          </div>
        </div>

        {/* right rail — live summary + generated CLI */}
        <aside>
          <div className="sticky top-0 space-y-4">
            <section className="border border-border bg-card">
              <header className="border-b border-border px-4 py-3">
                <p className="text-xs text-muted-foreground">Run summary</p>
                <p className="mt-0.5 text-sm font-semibold">
                  {ready ? "Ready to launch" : "Working through the steps…"}</p>
              </header>
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 px-4 py-3 text-xs">
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
              </dl>
              <div className="space-y-2 px-4 pt-1 pb-4">
                <Button size="lg" onClick={start} disabled={!ready || busy} className="h-10 w-full">
                  {busy ? <LoaderCircle className="animate-spin" /> : <Play />} {busy ? "Starting…" : "Start training"}
                </Button>
                <p className="text-center text-[11px] text-muted-foreground">
                  This exact config runs — nothing hidden.
                </p>
                {err && <p className="text-xs text-destructive">{err}</p>}
              </div>
            </section>

            <section className="border border-border bg-card p-4">
              <p className="mb-2 text-xs text-muted-foreground">The same run, from your shell</p>
              <pre className="rounded-md bg-surface p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-all text-foreground/80">{cli}</pre>
            </section>
          </div>
        </aside>
      </div>
    </>
  );
}

// A step's frame: hairline, card fill, room to breathe.
function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={cn("border border-border bg-card p-5", className)}>{children}</section>;
}

// A scrolling list of choices, one selected.
function PickList({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("max-h-72 divide-y divide-border overflow-auto border border-border scrollbar-thin", className)}>
      {children}
    </div>
  );
}

function PickRow({ selected, onClick, title, sub }: {
  selected: boolean; onClick: () => void; title: string; sub: string;
}) {
  return (
    <button onClick={onClick} aria-pressed={selected}
      className={cn("flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors",
        selected ? "bg-primary/5" : "hover:bg-subtle")}>
      <div className="min-w-0 flex-1">
        <div className={cn("truncate text-sm font-medium", selected && "text-primary")}>{title}</div>
        <Mono className="text-muted-foreground">{sub}</Mono>
      </div>
      {selected && <Check className="size-4 shrink-0 text-primary" />}
    </button>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="truncate text-right font-mono">{value}</dd>
    </>
  );
}

function ParamGrid({ params, vals, setVal, methodInfo, method }: {
  params: Param[]; vals: Record<string, string>;
  setVal: (k: string, v: string) => void;
  methodInfo?: MethodInfo; method?: string | null;
}) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {params.map((p) => {
        const hint = p.key === "learning_rate" && methodInfo
          ? `default for ${method}: ${methodInfo.default_lr}` : p.hint;
        const id = `param-${p.key}`;
        if (p.kind === "bool") {
          return (
            <div key={p.key} className="flex items-start gap-2.5 self-end pb-1.5">
              <Switch id={id} checked={vals[p.key] === "true"}
                      onCheckedChange={(on) => setVal(p.key, on ? "true" : "false")} className="mt-0.5" />
              <Label htmlFor={id} className="grid gap-0.5 font-normal">
                <span className="text-sm">{p.label}</span>
                {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
              </Label>
            </div>
          );
        }
        return (
          <Field key={p.key} label={p.label} hint={hint} htmlFor={id}>
            {p.kind === "select" ? (
              <select id={id} value={vals[p.key] ?? p.def} className="w-full font-mono"
                      onChange={(e) => setVal(p.key, e.target.value)}>
                {p.options!.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            ) : (
              <Input id={id} value={vals[p.key] ?? ""} placeholder={p.def || "method default"}
                     inputMode={p.kind === "int" ? "numeric" : "decimal"}
                     className="font-mono"
                     onChange={(e) => setVal(p.key, e.target.value)} />
            )}
          </Field>
        );
      })}
    </div>
  );
}
