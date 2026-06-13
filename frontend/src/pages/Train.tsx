// Train — the guided flow, data first: Data → Model → Method → Tune.
// The right rail is the always-live run summary + the generated CLI, and the
// launch button sits under the summary: this exact config runs, nothing hidden.
import { useEffect, useMemo, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Play, Search } from "lucide-react";
import { getDatasets, getModels, submitFinetune } from "../api";
import type { CatalogModel, DatasetMeta, MethodInfo } from "../api";
import { Field, PageHeader, btnGhost, btnPrimary } from "../ui";

const STEPS = ["Data", "Model", "Method", "Tune"] as const;
const LORA_FAMILY = ["lora", "qlora", "dora", "adapter", "more"];
const recommend = (format?: string): string[] =>
  format === "preference" ? ["dpo"]
  : format === "text" ? ["cpt"]
  : format === "prompt" ? ["grpo"]
  : ["lora", "qlora", "dora"];

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
  const [p, setP] = useState({ steps: 60, lorar: 16, lr: "", lrTouched: false,
                               batch: 2, ctx: 2048, evalSplit: false });

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
    if (pme) { setMethod(pme); sessionStorage.removeItem("pick.method"); }
    if (pst) { setP((q) => ({ ...q, steps: +pst || 60 })); sessionStorage.removeItem("pick.steps"); }
  }, []);

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

  const ready = Boolean(ds && model && method);
  const canNext = [Boolean(ds), Boolean(model), Boolean(method), true][step];

  function pickMethod(name: string) {
    setMethod(name);
    const m = methods.find((x) => x.name === name);
    if (m && !p.lrTouched) setP((q) => ({ ...q, lr: String(m.default_lr) }));
  }

  async function start() {
    if (!ready || busy) return;
    setErr(""); setBusy(true);
    const config: Record<string, unknown> = {
      method, max_steps: p.steps, per_device_train_batch_size: p.batch };
    if (LORA_FAMILY.includes(method!)) config.lora_r = p.lorar;
    if (p.lr) config.learning_rate = parseFloat(p.lr);
    try {
      const out = await submitFinetune({
        base_model: model, config, dataset_id: ds,
        eval_dataset: p.evalSplit ? "auto" : null,
        load_in_4bit: false, max_seq_length: p.ctx });
      window.location.hash = `#runs/${out.job_id}`;
    } catch (ex) { setErr((ex as Error).message); setBusy(false); }
  }

  const cli = [
    "shadowlm finetune <data.jsonl> \\",
    `  --model ${model ?? "<model>"} \\`,
    `  --method ${method ?? "<method>"} \\`,
    `  --max-steps ${p.steps}${p.lr ? ` \\\n  --lr ${p.lr}` : ""}` +
    (method && LORA_FAMILY.includes(method) ? ` \\\n  --lora-r ${p.lorar}` : "") +
    (p.evalSplit ? ` \\\n  --eval auto` : ""),
  ].join("\n");

  return (
    <>
      <PageHeader
        eyebrow="Fine-tuning Studio"
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
                          {d.format} · {d.rows.toLocaleString()} rows</div>
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

          {/* step 4 — Tune */}
          {step === 3 && (
            <section className="rounded-lg border border-border bg-card p-5 space-y-4">
              <div>
                <div className="text-sm font-semibold mb-1">Hyperparameters</div>
                <p className="text-xs text-muted-foreground">
                  Only the knobs {method ?? "this method"} actually has. Defaults are sensible.</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Max steps" hint="Total optimizer steps">
                  <input type="number" min={1} value={p.steps} className="w-full font-mono text-sm"
                         onChange={(e) => setP({ ...p, steps: +e.target.value || 60 })} />
                </Field>
                <Field label="Context length" hint="Max sequence length">
                  <input type="number" min={64} value={p.ctx} className="w-full font-mono text-sm"
                         onChange={(e) => setP({ ...p, ctx: +e.target.value || 2048 })} />
                </Field>
                <Field label="Learning rate" hint={methodInfo ? `Default for ${method}: ${methodInfo.default_lr}` : undefined}>
                  <input value={p.lr} placeholder="method default" className="w-full font-mono text-sm"
                         onChange={(e) => setP({ ...p, lr: e.target.value, lrTouched: true })} />
                </Field>
                <Field label="Batch size">
                  <input type="number" min={1} value={p.batch} className="w-full font-mono text-sm"
                         onChange={(e) => setP({ ...p, batch: +e.target.value || 2 })} />
                </Field>
                {method && LORA_FAMILY.includes(method) && (
                  <Field label={method === "adapter" ? "Adapter width (r)" : "LoRA rank"}>
                    <input type="number" min={1} value={p.lorar} className="w-full font-mono text-sm"
                           onChange={(e) => setP({ ...p, lorar: +e.target.value || 16 })} />
                  </Field>
                )}
              </div>
              <label className="flex items-center gap-2 text-sm text-muted-foreground">
                <input type="checkbox" checked={p.evalSplit} className="w-auto"
                       onChange={(e) => setP({ ...p, evalSplit: e.target.checked })} />
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
                <SummaryRow label="Dataset" value={meta ? `${meta.name} (${meta.rows})` : "—"} />
                <SummaryRow label="Model" value={model ? model.split("/").pop()! : "—"} />
                <SummaryRow label="Method" value={method ?? "—"} />
                <SummaryRow label="Steps" value={String(p.steps)} />
                <SummaryRow label="LR" value={p.lr || "method default"} />
                <SummaryRow label="Batch" value={String(p.batch)} />
                <SummaryRow label="Context" value={`${p.ctx} tok`} />
                {method && LORA_FAMILY.includes(method) &&
                  <SummaryRow label="Rank" value={`r=${p.lorar}`} />}
                <SummaryRow label="Eval" value={p.evalSplit ? "10% held out" : "off"} />
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
