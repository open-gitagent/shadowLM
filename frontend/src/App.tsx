// The studio shell: brain-brand sidebar + hash router.
import { useEffect, useState } from "react";
import { LayoutGrid, Database, Cpu, FlaskConical, GitBranch, MessageSquare } from "lucide-react";
import { apiKey, getHealth, getMethods } from "./api";
import type { MethodInfo } from "./api";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import Models from "./pages/Models";
import Train from "./pages/Train";
import Runs from "./pages/Runs";
import RunDetail from "./pages/RunDetail";
import Playground from "./pages/Playground";

function useHash(): string {
  const [h, setH] = useState(window.location.hash);
  useEffect(() => {
    const f = () => setH(window.location.hash);
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return h.replace(/^#/, "");
}

const NAV = [
  { hash: "dashboard", label: "Dashboard", Icon: LayoutGrid },
  { hash: "datasets", label: "Datasets", Icon: Database },
  { hash: "models", label: "Models", Icon: Cpu },
  { hash: "train", label: "Trainings", Icon: FlaskConical },
  { hash: "runs", label: "Runs", Icon: GitBranch },
  { hash: "playground", label: "Playground", Icon: MessageSquare },
];

export default function App() {
  const hash = useHash() || "dashboard";
  const [section, arg] = hash.split("/");
  const [health, setHealth] = useState("connecting…");
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [key, setKey] = useState(apiKey.get());

  useEffect(() => {
    getHealth()
      .then((h) => setHealth(`backend=${h.backend} · v${h.version}`))
      .catch((e) => setHealth(`⚠ ${e.message}`));
    getMethods().then((m) => setMethods(m.methods)).catch(() => {});
  }, []);

  const page =
    section === "datasets" ? <Datasets /> :
    section === "models" ? <Models /> :
    section === "train" ? <Train methods={methods} /> :
    section === "runs" && arg ? <RunDetail id={arg} /> :
    section === "runs" ? <Runs /> :
    section === "playground" ? <Playground /> :
    <Dashboard />;

  return (
    <div className="flex h-full">
      <aside className="flex w-[212px] min-w-[212px] flex-col gap-0.5 border-r border-seam p-4 pt-4">
        <div className="mb-3 flex items-center gap-2.5 px-3 font-bold">
          <img src="/logo.png" alt="" className="size-9 rounded-lg border border-seam" />
          <div>
            ShadowLM
            <div className="text-[11px] font-normal text-faded">
              <span className="text-heart">slm♥</span> trainer
            </div>
          </div>
        </div>
        {NAV.map(({ hash: h, label, Icon }) => (
          <a key={h} href={`#${h}`}
             className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px] no-underline
               ${section === h
                 ? "bg-umbra text-bone shadow-[inset_2px_0_0_#e5484d]"
                 : "text-faded hover:text-bone"}`}>
            <Icon className={`size-4 ${section === h ? "text-heart" : ""}`} />
            {label}
          </a>
        ))}
        <div className="flex-1" />
        <div className="grid gap-2 px-3 pb-1">
          <span className="text-[11px] text-faded">{health}</span>
          <input type="password" placeholder="API key" value={key}
                 title="Sent as Bearer auth; stored in this browser only"
                 onChange={(e) => { setKey(e.target.value); apiKey.set(e.target.value); }} />
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto p-6 px-8">{page}</main>
    </div>
  );
}
