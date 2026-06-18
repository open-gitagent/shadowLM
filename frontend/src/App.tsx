// The studio shell — cream sidebar, lucide icons, hash router, login gate.
import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  Box, Cpu, Database, ExternalLink, History, LayoutDashboard, LogOut,
  MessagesSquare, Zap,
} from "lucide-react";
import {
  apiKey, clearVram, getAuthInfo, getHealth, getMethods, getSettings, getVram,
  login, logout, setHfToken,
} from "./api";
import type { AuthInfo, MethodInfo } from "./api";
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

// ---- the login gate ---------------------------------------------------------
export default function App() {
  const [auth, setAuth] = useState<AuthInfo | null>(null);
  const [token, setToken] = useState(apiKey.get());

  useEffect(() => {
    getAuthInfo()
      .then(setAuth)
      .catch(() => setAuth({ auth_required: false, mode: "none" }));
  }, []);

  useEffect(() => {
    const onUnauth = () => setToken("");
    window.addEventListener("slm-unauthorized", onUnauth);
    return () => window.removeEventListener("slm-unauthorized", onUnauth);
  }, []);

  if (auth === null) {
    return (
      <div className="min-h-screen grid place-items-center text-sm text-muted-foreground">
        connecting…
      </div>
    );
  }
  if (auth.auth_required && !token) {
    return <Login mode={auth.mode} onAuthed={() => setToken(apiKey.get())} />;
  }
  return (
    <Studio
      authEnabled={auth.auth_required}
      onSignOut={() => { logout(); setToken(""); }}
    />
  );
}

function Login({ mode, onAuthed }: { mode: string; onAuthed: () => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [key, setKey] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const apikeyMode = mode === "apikey";

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      if (apikeyMode) {
        apiKey.set(key.trim());
        await getHealth(); // 401 throws → invalid key
      } else {
        await login(username.trim(), password);
      }
      onAuthed();
    } catch (e2) {
      apiKey.clear();
      setErr(apikeyMode ? "Invalid API key" : (e2 as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen grid place-items-center bg-background px-4">
      <form onSubmit={submit}
            className="w-full max-w-xs rounded-xl border border-sidebar-border bg-sidebar p-6 space-y-4">
        <div className="flex items-center gap-2.5">
          <img src="/lyzr-mark.png" alt="" className="size-8 rounded-lg" />
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-tight">ShadowLM</div>
            <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              Studio
            </div>
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          {apikeyMode ? "Enter the API key to continue." : "Sign in to continue."}
        </div>
        {apikeyMode ? (
          <input type="password" autoFocus value={key} placeholder="API key"
                 className="w-full text-sm"
                 onChange={(e) => setKey(e.target.value)} />
        ) : (
          <>
            <input type="text" autoFocus value={username} placeholder="Username"
                   autoComplete="username" className="w-full text-sm"
                   onChange={(e) => setUsername(e.target.value)} />
            <input type="password" value={password} placeholder="Password"
                   autoComplete="current-password" className="w-full text-sm"
                   onChange={(e) => setPassword(e.target.value)} />
          </>
        )}
        {err && <div className="text-xs text-red-500">{err}</div>}
        <button type="submit" disabled={busy}
                className="w-full rounded-md bg-primary text-primary-foreground text-sm py-2 disabled:opacity-50">
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}

// ---- the authenticated studio shell ----------------------------------------
function Studio({ authEnabled, onSignOut }: { authEnabled: boolean; onSignOut: () => void }) {
  const hash = useHash();
  const [section, arg] = hash.split("/");
  const [health, setHealth] = useState("connecting…");
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [hf, setHf] = useState("");
  const [hfSet, setHfSet] = useState(false);
  const [vram, setVram] = useState("");

  const showVram = (used: number | null | undefined) =>
    setVram(used != null ? `VRAM ${(used / 1024).toFixed(1)} GB used` : "");

  useEffect(() => {
    getHealth()
      .then((h) => setHealth(`backend=${h.backend} · v${h.version}`))
      .catch((e) => setHealth(`⚠ ${e.message}`));
    getMethods().then((m) => setMethods(m.methods)).catch(() => {});
    getSettings().then((s) => setHfSet(s.hf_token_set)).catch(() => {});
    getVram().then((v) => showVram(v.used_mb)).catch(() => {});
  }, []);

  async function saveHfToken() {
    try { setHfSet((await setHfToken(hf)).hf_token_set); setHf(""); } catch { /* ignore */ }
  }

  async function cleanVram() {
    setVram("clearing…");
    try { const r = await clearVram(); showVram(r.after_mb); }
    catch { setVram("clear failed"); }
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
            <img src="/lyzr-mark.png" alt="" className="size-8 rounded-lg" />
            <div className="leading-tight">
              <div className="text-sm font-semibold tracking-tight">ShadowLM</div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Studio</div>
            </div>
          </a>
          <div className="mt-2.5 text-[10px] text-muted-foreground">from Lyzr Research Labs</div>
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
          <button onClick={cleanVram} title="Unload cached models + free GPU memory"
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-xs text-muted-foreground hover:bg-sidebar-accent/40 transition-colors">
            <Zap className="size-3.5" />
            <span>Clean VRAM</span>
            {vram && <span className="ml-auto font-mono text-[10px]">{vram}</span>}
          </button>
          {authEnabled && (
            <button onClick={onSignOut}
                    className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-xs text-muted-foreground hover:bg-sidebar-accent/40 transition-colors">
              <LogOut className="size-3.5" />
              <span>Sign out</span>
            </button>
          )}
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
