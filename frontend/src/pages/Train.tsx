// The training wizard: Data → Model → Method → Tune → Launch.
// One decision per step; the data ranks the methods; the method shapes the form.
import { useEffect, useState } from "react";
import { getDatasets, getModels, submitFinetune } from "../api";
import type { CatalogModel, DatasetMeta, MethodInfo } from "../api";
import { Card, H2, Lead, Pill } from "../ui";

const STEPS = ["Data", "Model", "Method", "Tune", "Launch"];
const LORA_FAMILY = ["lora", "qlora", "dora", "adapter", "more"];
const recommend = (format?: string): string[] =>
  format === "preference" ? ["dpo"]
  : format === "text" ? ["cpt"]
  : format === "prompt" ? ["grpo"]
  : ["lora", "qlora", "dora"];

interface Params { steps: number; lorar: number; lr: string; lrTouched: boolean;
                   batch: number; evalSplit: boolean; }

export default function Train({ methods }: { methods: MethodInfo[] }) {
  const [step, setStep] = useState(0);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [recent, setRecent] = useState<string[]>([]);
  const [ds, setDs] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [method, setMethod] = useState<string | null>(null);
  const [free, setFree] = useState("");
  const [err, setErr] = useState("");
  const [p, setP] = useState<Params>({ steps: 60, lorar: 16, lr: "", lrTouched: false,
                                       batch: 2, evalSplit: false });

  useEffect(() => {
    getDatasets().then((d) => setDatasets(d.datasets));
    getModels().then((m) => {
      setCatalog(m.catalog);
      setRecent(m.recent.filter((r) => !m.catalog.some((c) => c.id === r)));
    });
    const pd = sessionStorage.getItem("pick.dataset");
    const pm = sessionStorage.getItem("pick.model");
    if (pd) { setDs(pd); sessionStorage.removeItem("pick.dataset"); setStep(1); }
    if (pm) { setModel(pm); sessionStorage.removeItem("pick.model"); setStep(2); }
  }, []);

  const meta = datasets.find((d) => d.dataset_id === ds);
  const rec = recommend(meta?.format);
  const ordered = [...methods].sort((a, b) =>
    Number(rec.includes(b.name)) - Number(rec.includes(a.name)));
  const canNext = [Boolean(ds), Boolean(model), Boolean(method), true, true][step];

  function pickMethod(name: string) {
    setMethod(name);
    const m = methods.find((x) => x.name === name);
    if (m && !p.lrTouched) setP((q) => ({ ...q, lr: String(m.default_lr) }));
  }

  async function start() {
    setErr("");
    const config: Record<string, unknown> = {
      method, max_steps: p.steps, per_device_train_batch_size: p.batch };
    if (LORA_FAMILY.includes(method!)) config.lora_r = p.lorar;
    if (p.lr) config.learning_rate = parseFloat(p.lr);
    try {
      const out = await submitFinetune({
        base_model: model, config, dataset_id: ds,
        eval_dataset: p.evalSplit ? "auto" : null,
        load_in_4bit: false, max_seq_length: 2048 });
      window.location.hash = `#runs/${out.job_id}`;
    } catch (ex) { setErr((ex as Error).message); }
  }

  const primary = "rounded-lg bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-4 py-2 font-bold text-white disabled:opacity-40";

  return (
    <div>
      <H2>New training</H2>
      <Lead>five decisions, in the order they depend on each other — everything
        else has defaults.</Lead>

      {/* stepper rail */}
      <div className="my-5 flex max-w-[860px]">
        {STEPS.map((s, i) => (
          <div key={s} onClick={() => i < step && setStep(i)}
               className={`relative flex-1 pt-[30px] text-center text-[12px]
                 ${i < step ? "cursor-pointer text-bone" : i === step ? "text-bone" : "text-faded"}`}>
            <span className={`absolute left-1/2 top-0 size-6 -translate-x-1/2 rounded-full border text-[11px] leading-[22px]
              ${i < step ? "border-okay text-okay bg-umbra"
                : i === step ? "border-heart bg-heart font-bold text-white"
                : "border-seam bg-umbra"}`}>
              {i < step ? "✓" : i + 1}
            </span>
            {i < STEPS.length - 1 &&
              <span className="absolute left-[calc(50%+14px)] top-3 h-px w-[calc(100%-28px)] bg-seam" />}
            {s}
          </div>
        ))}
      </div>

      <div className="max-w-[860px]">
        {step === 0 && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-3">
            {datasets.map((d) => (
              <Card key={d.dataset_id} selected={ds === d.dataset_id}
                    onClick={() => setDs(d.dataset_id)}>
                <div className="flex items-baseline justify-between gap-2">
                  <h4 className="text-[13.5px] font-bold">{d.name}</h4>
                  <Pill>{d.format}</Pill>
                </div>
                <div className="text-[11.5px] text-faded">{d.rows} rows</div>
              </Card>
            ))}
            <Card>
              <h4 className="text-[13.5px] font-bold">New dataset</h4>
              <div className="text-[11.5px] text-faded">paste or upload on the Datasets page</div>
              <button className="mt-2.5 rounded-lg border border-seam px-3 py-1.5 text-[13px]"
                      onClick={() => (window.location.hash = "#datasets")}>go to Datasets ›</button>
            </Card>
          </div>
        )}

        {step === 1 && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-3">
            {recent.map((id) => (
              <Card key={id} selected={model === id} onClick={() => setModel(id)}>
                <h4 className="text-[13.5px] font-bold">{id.split("/").pop()}</h4>
                <div className="text-[11.5px] text-faded">{id}</div>
                <div className="text-[11.5px] text-faded">recently trained here</div>
              </Card>
            ))}
            {catalog.map((m) => (
              <Card key={m.id} selected={model === m.id} onClick={() => setModel(m.id)}>
                <div className="flex items-baseline justify-between gap-2">
                  <h4 className="text-[13.5px] font-bold">{m.id.split("/").pop()}</h4>
                  <span className="flex gap-1.5">
                    {m.dev && <Pill tone="green">dev pick</Pill>}
                    {m.gated && <Pill tone="gold">HF token</Pill>}
                  </span>
                </div>
                <div className="text-[11.5px] text-faded">{m.id}</div>
                <div className="text-[11.5px] text-faded">{m.params}{m.note ? ` · ${m.note}` : ""}</div>
              </Card>
            ))}
            <Card>
              <h4 className="text-[13.5px] font-bold">Custom</h4>
              <form className="mt-2 flex gap-2"
                    onSubmit={(e) => { e.preventDefault(); if (free.trim()) setModel(free.trim()); }}>
                <input className="min-w-0 flex-1" placeholder="org/model-name" value={free}
                       onChange={(e) => setFree(e.target.value)} />
                <button className={primary}>use</button>
              </form>
              {model && !catalog.some((c) => c.id === model) && !recent.includes(model) &&
                <div className="mt-2 text-[11.5px] text-okay">✓ {model}</div>}
            </Card>
          </div>
        )}

        {step === 2 && (
          <>
            <p className="mb-3 text-[12px] text-faded">
              your dataset is <b className="text-bone">{meta?.format ?? "?"}</b> — recommended methods first.
            </p>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-3">
              {ordered.map((m) => (
                <Card key={m.name} selected={method === m.name} onClick={() => pickMethod(m.name)}>
                  <div className="flex items-baseline justify-between gap-2">
                    <h4 className="text-[13.5px] font-bold">{m.name}</h4>
                    <span className="flex gap-1.5">
                      {rec.includes(m.name) && <Pill tone="red">recommended</Pill>}
                      <Pill>{m.trainer}</Pill>
                    </span>
                  </div>
                  <div className="text-[11.5px] text-faded">{m.description}</div>
                  <div className="text-[11.5px] text-faded">default lr {m.default_lr}</div>
                </Card>
              ))}
            </div>
          </>
        )}

        {step === 3 && (
          <div className="grid max-w-[560px] gap-2.5">
            {[
              { label: "max steps", el: <input type="number" min={1} value={p.steps}
                  onChange={(e) => setP({ ...p, steps: +e.target.value || 60 })} /> },
              ...(LORA_FAMILY.includes(method!) ? [{
                label: method === "adapter" ? "adapter width (r)" : "lora r",
                el: <input type="number" min={1} value={p.lorar}
                    onChange={(e) => setP({ ...p, lorar: +e.target.value || 16 })} /> }] : []),
              { label: "learning rate", el: <input value={p.lr} placeholder="method default"
                  onChange={(e) => setP({ ...p, lr: e.target.value, lrTouched: true })} /> },
              { label: "batch size", el: <input type="number" min={1} value={p.batch}
                  onChange={(e) => setP({ ...p, batch: +e.target.value || 2 })} /> },
              { label: "held-out eval", el: (
                  <label className="flex items-center gap-2 text-[13px] text-faded">
                    <input type="checkbox" checked={p.evalSplit} className="w-auto"
                           onChange={(e) => setP({ ...p, evalSplit: e.target.checked })} />
                    hold out 10% — see overfitting, not just training loss
                  </label>) },
            ].map((row) => (
              <div key={row.label} className="grid grid-cols-[160px_1fr] items-center gap-2.5">
                <label className="text-[12px] text-faded">{row.label}</label>
                {row.el}
              </div>
            ))}
          </div>
        )}

        {step === 4 && (
          <>
            <table className="w-full max-w-[640px] border-collapse text-[13px]">
              <tbody>
                {[
                  ["dataset", `${meta?.name ?? ds} · ${meta?.rows} rows · ${meta?.format}`],
                  ["model", model],
                  ["method", method],
                  ["max steps", String(p.steps)],
                  ...(LORA_FAMILY.includes(method!) ? [["lora r", String(p.lorar)]] : []),
                  ["learning rate", p.lr || "(method default)"],
                  ["batch size", String(p.batch)],
                  ["held-out eval", p.evalSplit ? "10% held out" : "off"],
                ].map(([k, v]) => (
                  <tr key={k} className="border-b border-seam">
                    <td className="w-[180px] py-2 text-faded">{k}</td>
                    <td className="py-2">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-4 flex items-center gap-3">
              <button className={primary} onClick={start}>start training ›</button>
              <span className="text-[12px] text-faded">this exact config runs — nothing hidden</span>
            </div>
            {err && <div className="mt-2.5 text-heart">{err}</div>}
          </>
        )}
      </div>

      <div className="mt-5 flex max-w-[860px] gap-2.5">
        {step > 0 && (
          <button className="rounded-lg border border-seam px-4 py-2"
                  onClick={() => setStep(step - 1)}>‹ back</button>
        )}
        {step < 4 && (
          <button className={primary} disabled={!canNext}
                  onClick={() => setStep(step + 1)}>continue ›</button>
        )}
      </div>
    </div>
  );
}
