import { useEffect, useRef, useState } from "react";
import { createDataset, deleteDataset, getDataset, getDatasets } from "../api";
import type { DatasetMeta } from "../api";
import { Card, H2, Lead, Pill } from "../ui";

export default function Datasets() {
  const [list, setList] = useState<DatasetMeta[]>([]);
  const [preview, setPreview] = useState<Record<string, DatasetMeta>>({});
  const [name, setName] = useState("");
  const [rows, setRows] = useState("");
  const [err, setErr] = useState("");
  const file = useRef<HTMLInputElement>(null);

  const refresh = () => getDatasets().then((d) => setList(d.datasets)).catch((e) => setErr(e.message));
  useEffect(() => { refresh(); }, []);

  async function upload(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    try {
      let text = rows.trim();
      const f = file.current?.files?.[0];
      if (f) text = await f.text();
      const parsed = text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
      if (!parsed.length) throw new Error("no rows — paste JSONL or pick a file");
      await createDataset(name, parsed);
      setName(""); setRows(""); if (file.current) file.current.value = "";
      refresh();
    } catch (ex) { setErr((ex as Error).message); }
  }

  return (
    <div>
      <H2>Datasets</H2>
      <Lead>Upload once, train many times. JSONL — chat, instruction, preference,
        or raw text rows; the format is auto-detected.</Lead>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">
        {list.map((d) => (
          <Card key={d.dataset_id}>
            <div className="flex items-baseline justify-between gap-2">
              <h4 className="text-[13.5px] font-bold">{d.name}</h4>
              <Pill>{d.format}</Pill>
            </div>
            <div className="text-[11.5px] text-faded">{d.rows} rows · {d.dataset_id}</div>
            <div className="mt-2.5 flex flex-wrap gap-2">
              <button className="rounded-lg bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-3 py-1.5 text-[13px] font-bold text-white"
                onClick={() => { sessionStorage.setItem("pick.dataset", d.dataset_id); window.location.hash = "#train"; }}>
                train on this ›
              </button>
              <button className="rounded-lg border border-seam px-3 py-1.5 text-[13px]"
                onClick={() => getDataset(d.dataset_id).then((full) =>
                  setPreview((p) => ({ ...p, [d.dataset_id]: full })))}>
                preview
              </button>
              <button className="rounded-lg px-3 py-1.5 text-[13px] text-heart"
                onClick={() => deleteDataset(d.dataset_id).then(refresh)}>
                delete
              </button>
            </div>
            {preview[d.dataset_id] && (
              <pre className="drop mt-2 max-h-44 overflow-auto rounded-lg border border-seam bg-ink p-2.5 text-[11px] text-faded">
                {preview[d.dataset_id].preview?.map((r) => JSON.stringify(r)).join("\n")}
              </pre>
            )}
          </Card>
        ))}
        <Card>
          <h4 className="text-[13.5px] font-bold">New dataset</h4>
          <form onSubmit={upload} className="mt-2 grid gap-2.5">
            <input placeholder="name (e.g. support-tickets-v1)" value={name}
                   onChange={(e) => setName(e.target.value)} />
            <textarea rows={6} value={rows} onChange={(e) => setRows(e.target.value)}
              placeholder={'one JSON row per line:\n{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}'} />
            <div className="flex items-center gap-2.5">
              <input ref={file} type="file" accept=".jsonl,.json" className="text-[12px]" />
              <button className="rounded-lg bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-3.5 py-2 font-bold text-white">
                upload ›
              </button>
            </div>
            {err && <div className="text-[12.5px] text-heart">{err}</div>}
          </form>
        </Card>
      </div>
    </div>
  );
}
