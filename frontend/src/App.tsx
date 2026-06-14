// The studio shell — cream sidebar, lucide icons, hash router.
import { useEffect, useState } from "react";
import {
  Box, Cpu, Database, ExternalLink, History, LayoutDashboard, MessagesSquare,
} from "lucide-react";
import { apiKey, getHealth, getMethods, getSettings, setHfToken } from "./api";
import type { MethodInfo } from "./api";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import Models from "./pages/Models";
import Train from "./pages/Train";
import Runs from "./pages/Runs";
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
  { hash: "playground", label: "Playground", icon: MessagesSquare },
  { hash: "", label: "Dashboard", icon: LayoutDashboard },
  { hash: "datasets", label: "Datasets", icon: Database },
  { hash: "models", label: "Models", icon: Box },
  { hash: "train", label: "Train", icon: Cpu },
  { hash: "runs", label: "Runs", icon: History },
] as const;

export default function App() {
  const hash = useHash();
  const [section, arg] = hash.split("/");
  const [health, setHealth] = useState("connecting…");
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [key, setKey] = useState(apiKey.get());
  const [hf, setHf] = useState("");
  const [hfSet, setHfSet] = useState(false);

  useEffect(() => {
    getHealth()
      .then((h) => setHealth(`backend=${h.backend} · v${h.version}`))
      .catch((e) => setHealth(`⚠ ${e.message}`));
    getMethods().then((m) => setMethods(m.methods)).catch(() => {});
    getSettings().then((s) => setHfSet(s.hf_token_set)).catch(() => {});
  }, []);

  async function saveHfToken() {
    try { setHfSet((await setHfToken(hf)).hf_token_set); setHf(""); } catch { /* ignore */ }
  }

  const page =
    section === "models" ? <Models /> :
    section === "datasets" ? <Datasets /> :
    section === "train" ? <Train methods={methods} /> :
    section === "playground" ? <Playground /> :
    section === "runs" ? <Runs initialId={arg} /> :
    <Dashboard />;

  return (
    <div className="flex min-h-screen w-full">
      <aside className="w-60 shrink-0 border-r border-sidebar-border bg-sidebar flex flex-col sticky top-0 h-screen">
        <div className="px-5 py-5 border-b border-sidebar-border">
          <a href="#" className="flex items-center gap-2.5">
            <img src="/logo.png" alt="" className="size-8 rounded-md border border-primary/30" />
            <div className="leading-tight">
              <div className="text-sm font-semibold tracking-tight">ShadowLM</div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Studio</div>
            </div>
          </a>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {NAV.map(({ hash: to, label, icon: Icon }) => {
            const active = to === "" ? section === "" : section === to;
            return (
              <a key={to} href={`#${to}`}
                 className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors no-underline ${
                   active
                     ? "bg-sidebar-accent text-foreground"
                     : "text-foreground/60 hover:bg-sidebar-accent/40 hover:text-foreground"
                 }`}>
                <Icon className="size-4" />
                <span>{label}</span>
                {active && <span className="ml-auto size-1.5 rounded-full bg-primary" />}
              </a>
            );
          })}
        </nav>
        <div className="px-3 py-3 border-t border-sidebar-border space-y-2">
          <div className="px-3 text-[10px] font-mono text-muted-foreground">{health}</div>
          <input type="password" placeholder="API key" value={key}
                 title="Sent as Bearer auth; stored in this browser only"
                 className="w-full text-xs"
                 onChange={(e) => { setKey(e.target.value); apiKey.set(e.target.value); }} />
          <div className="flex gap-1.5">
            <input type="password" value={hf}
                   placeholder={hfSet ? "HF token ✓ set — replace" : "HF token (gated models)"}
                   title="Hugging Face token for gated/private models; stored on the server"
                   className="w-full text-xs"
                   onKeyDown={(e) => { if (e.key === "Enter") saveHfToken(); }}
                   onChange={(e) => setHf(e.target.value)} />
            <button onClick={saveHfToken} disabled={!hf.trim()}
                    className="text-[11px] px-2 rounded-md border border-sidebar-border text-muted-foreground hover:text-foreground disabled:opacity-40">
              save
            </button>
          </div>
          <a href="https://github.com/open-gitagent/shadowLM" target="_blank" rel="noreferrer"
             className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-xs text-muted-foreground hover:bg-sidebar-accent/40 transition-colors no-underline">
            <ExternalLink className="size-3.5" />
            <span>GitHub</span>
            <span className="ml-auto font-mono text-[10px] text-primary">slm♥</span>
          </a>
        </div>
      </aside>
      <main className="flex-1 min-w-0 flex flex-col">{page}</main>
    </div>
  );
}
