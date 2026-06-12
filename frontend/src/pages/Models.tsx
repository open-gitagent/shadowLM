import { useEffect, useState } from "react";
import { getModels } from "../api";
import type { CatalogModel } from "../api";
import { Card, H2, Lead, Pill } from "../ui";

function pick(kind: "model" | "adapter", value: string, dest: string) {
  sessionStorage.setItem(`pick.${kind}`, value);
  window.location.hash = dest;
}

export default function Models() {
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [recent, setRecent] = useState<string[]>([]);
  const [backend, setBackend] = useState("?");
  const [free, setFree] = useState("");

  useEffect(() => {
    getModels().then((m) => {
      setCatalog(m.catalog);
      setRecent(m.recent.filter((r) => !m.catalog.some((c) => c.id === r)));
      setBackend(m.server_backend);
    }).catch(() => {});
  }, []);

  const card = (m: CatalogModel) => (
    <Card key={m.id}>
      <div className="flex items-baseline justify-between gap-2">
        <h4 className="text-[13.5px] font-bold">{m.id.split("/").pop()}</h4>
        <span className="flex gap-1.5">
          {m.dev && <Pill tone="green">dev pick</Pill>}
          {m.gated && <Pill tone="gold">HF token</Pill>}
        </span>
      </div>
      <div className="text-[11.5px] text-faded">{m.id}</div>
      <div className="text-[11.5px] text-faded">{m.params}{m.note ? ` · ${m.note}` : ""}</div>
      <div className="mt-2.5 flex gap-2">
        <button className="rounded-lg bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-3 py-1.5 text-[13px] font-bold text-white"
                onClick={() => pick("model", m.id, "#train")}>train this ›</button>
        <button className="rounded-lg border border-seam px-3 py-1.5 text-[13px]"
                onClick={() => pick("model", m.id, "#playground")}>playground</button>
      </div>
    </Card>
  );

  return (
    <div>
      <H2>Models</H2>
      <Lead>Any open model on the HuggingFace hub works — these are good starting
        points. Server backend: <b className="text-bone">{backend}</b>.</Lead>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">
        {recent.map((id) => card({ id, note: "recently trained here" }))}
        {catalog.map(card)}
        <Card>
          <h4 className="text-[13.5px] font-bold">Custom</h4>
          <div className="text-[11.5px] text-faded">any HF hub id</div>
          <form className="mt-2.5 flex gap-2"
                onSubmit={(e) => { e.preventDefault(); if (free.trim()) pick("model", free.trim(), "#train"); }}>
            <input className="flex-1" placeholder="org/model-name" value={free}
                   onChange={(e) => setFree(e.target.value)} />
            <button className="rounded-lg bg-gradient-to-br from-[#f05a5f] to-[#c73a3f] px-3.5 font-bold text-white">use</button>
          </form>
        </Card>
      </div>
    </div>
  );
}
