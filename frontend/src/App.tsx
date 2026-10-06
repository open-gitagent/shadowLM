// The studio shell, in the opencontroller console's language: a floating
// side panel (the "island") that folds to a rail, the page beside it, a
// hash router, and the sign-in gate. Embedded in a host console
// (lib/embed.ts), the host's menu and sign-in replace both.
import {
  ArrowRight, BookOpen, Box, Cpu, Database, ExternalLink, History, KeyRound, LayoutDashboard,
  LoaderCircle, type LucideIcon, LogOut, MessagesSquare, MonitorSmartphone, Moon,
  PanelLeftClose, PanelLeftOpen, Sun, Zap,
} from "lucide-react";
import { AnimatePresence, motion, MotionConfig } from "motion/react";
import { ThemeProvider, useTheme } from "next-themes";
import { type CSSProperties, type FormEvent, type ReactNode, useEffect, useState } from "react";

import {
  apiKey, clearVram, getAuthInfo, getHealth, getMethods, getSettings, getVram,
  login, logout, setHfToken,
} from "@/api";
import type { AuthInfo, MethodInfo } from "@/api";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  allows, embedded, hashToPage, type MenuItem, onHostAction, reportMenu, reportRoute, reportTitle, useEmbedTheme,
} from "@/lib/embed";
import { cn } from "@/lib/utils";
import Dashboard from "@/pages/Dashboard";
import Datasets from "@/pages/Datasets";
import Machines from "@/pages/Machines";
import Models from "@/pages/Models";
import Playground from "@/pages/Playground";
import Runs from "@/pages/Runs";
import Train from "@/pages/Train";

function useHash(): string {
  const [h, setH] = useState(window.location.hash);
  useEffect(() => {
    const f = () => setH(window.location.hash);
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return h.replace(/^#/, "");
}

interface NavItem { hash: string; label: string; icon: LucideIcon }
type Section = { title?: string; items: NavItem[] };

// The navigation leads with the Playground, where you talk to what you own,
// then follows the shadowing loop: bring data and a base model, train, watch
// the run. Machines, where training runs, is setup, so it sits apart at the
// foot.
const sections: Section[] = [
  {
    items: [
      { hash: "playground", label: "Playground", icon: MessagesSquare },
      { hash: "", label: "Overview", icon: LayoutDashboard },
    ],
  },
  {
    title: "Build",
    items: [
      { hash: "datasets", label: "Datasets", icon: Database },
      { hash: "models", label: "Models", icon: Box },
    ],
  },
  {
    title: "Train",
    items: [
      { hash: "train", label: "New run", icon: Cpu },
      { hash: "runs", label: "Runs", icon: History },
    ],
  },
];
const machinesItem: NavItem = { hash: "machines", label: "Machines", icon: MonitorSmartphone };
const allItems = [...sections.flatMap((s) => s.items), machinesItem];

// glyphs are the menu's icons by name, for a host that draws the menu itself
// (HostMenu): lucide's names, which the host knows.
const glyphs = new Map<LucideIcon, string>([
  [LayoutDashboard, "layout-dashboard"], [MessagesSquare, "messages-square"], [Database, "database"],
  [Box, "box"], [Cpu, "cpu"], [History, "history"], [MonitorSmartphone, "monitor-smartphone"],
]);

// The repository; its README is the documentation.
const repo = "https://github.com/open-gitagent/shadowLM";

const openWidth = 240;
const railWidth = 48;
const inset = 12;
const gap = 24;
const pageRight = 32;
const spring = { type: "spring", stiffness: 380, damping: 36 } as const;
const storageKey = "of-island-open";

function readOpen(): boolean {
  try {
    return localStorage.getItem(storageKey) !== "false";
  } catch {
    return true;
  }
}

// ---- the root: theme, then the gate ----------------------------------------
export default function App() {
  // Embedded, the host's theme wins; otherwise the person's own choice.
  const hostTheme = useEmbedTheme();
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange
                   forcedTheme={embedded ? hostTheme : undefined}>
      {embedded ? <Studio embeddedIn /> : <Gate />}
    </ThemeProvider>
  );
}

function Gate() {
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
      <div className="grid h-full place-items-center text-sm text-muted-foreground">
        <LoaderCircle className="size-4 animate-spin" />
      </div>
    );
  }
  if (auth.auth_required && !token) {
    return <SignIn apikeyMode={auth.mode === "apikey"} onAuthed={() => setToken(apiKey.get())} />;
  }
  return <Studio onSignOut={auth.auth_required ? () => { logout(); setToken(""); } : undefined} />;
}

// ---- sign in ------------------------------------------------------------------
// Outside the shell, on the canvas, with the studio's own mark.
function SignIn({ apikeyMode, onAuthed }: { apikeyMode: boolean; onAuthed: () => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [key, setKey] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      if (apikeyMode) {
        apiKey.set(key.trim());
        await getHealth();
      } else {
        await login(username, password);
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
    <main className="grid min-h-svh place-items-center bg-canvas px-4 py-10">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex items-center gap-2.5">
          <Monogram />
          <span className="grid leading-tight">
            <span className="text-sm font-semibold tracking-[-0.01em]">openfinetuner</span>
            <span className="text-[11px] text-muted-foreground">formerly ShadowLM</span>
          </span>
        </div>
        <section className="grid gap-5 border border-border bg-background p-6">
          <header className="grid gap-1.5">
            <h1 className="text-lg font-semibold tracking-[-0.01em]">Sign in to openfinetuner</h1>
            <p className="text-sm text-muted-foreground">
              {apikeyMode ? "Use the API key this server was started with." : "Use the username and password this server was started with."}
            </p>
          </header>
          <form onSubmit={submit} className="grid gap-3">
            {apikeyMode ? (
              <div className="grid gap-1.5">
                <Label htmlFor="key">API key</Label>
                <Input id="key" type="password" autoFocus value={key} onChange={(e) => setKey(e.target.value)} />
              </div>
            ) : (
              <>
                <div className="grid gap-1.5">
                  <Label htmlFor="username">Username</Label>
                  <Input id="username" autoComplete="username" autoFocus value={username}
                         onChange={(e) => setUsername(e.target.value)} />
                </div>
                <div className="grid gap-1.5">
                  <Label htmlFor="password">Password</Label>
                  <Input id="password" type="password" autoComplete="current-password" value={password}
                         onChange={(e) => setPassword(e.target.value)} />
                </div>
              </>
            )}
            {err && <p className="text-sm text-destructive">{err}</p>}
            <Button type="submit" size="lg" disabled={busy} className="h-10 w-full justify-between px-3">
              {busy ? "Signing in…" : "Sign in"}
              {busy ? <LoaderCircle className="animate-spin" /> : <ArrowRight />}
            </Button>
          </form>
        </section>
        <p className="mt-4 font-mono text-[11px] text-muted-foreground">from Lyzr Research Labs</p>
      </div>
    </main>
  );
}

function Monogram() {
  return (
    <span className="grid size-7 shrink-0 place-items-center rounded-md border border-primary/20 bg-primary/10 font-mono text-[10px] font-semibold text-primary">
      of
    </span>
  );
}

// ---- the studio ---------------------------------------------------------------
function Studio({ onSignOut, embeddedIn }: { onSignOut?: () => void; embeddedIn?: boolean }) {
  const hash = useHash();
  const [section, arg] = hash.split("/");
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [open, setOpen] = useState(readOpen);

  useEffect(() => {
    getMethods().then((m) => setMethods(m.methods)).catch(() => {});
  }, []);

  // Embedded, the host keeps the studio's page in its own address and shows
  // what the page is as its subtitle.
  useEffect(() => {
    if (!embeddedIn) return;
    reportRoute(hashToPage(hash));
    reportTitle(allItems.find((i) => i.hash === section)?.label ?? "Overview");
  }, [embeddedIn, hash, section]);

  const toggle = () =>
    setOpen((o) => {
      try {
        localStorage.setItem(storageKey, String(!o));
      } catch {
        // private window: the island just won't remember
      }
      return !o;
    });

  const page =
    section === "models" ? <Models /> :
    section === "datasets" ? <Datasets /> :
    section === "train" ? <Train methods={methods} /> :
    section === "playground" ? <Playground /> :
    section === "runs" ? <Runs initialId={arg} /> :
    section === "machines" ? <Machines /> :
    <Dashboard />;

  if (embeddedIn) {
    return (
      <main className="@container flex h-full min-h-0 flex-col overflow-y-auto bg-canvas px-6 pt-6 pb-8 *:shrink-0">
        {page}
        <HostMenu />
      </main>
    );
  }

  return (
    <MotionConfig reducedMotion="user">
      <div className="h-full" style={{ "--nav-right": `${inset + (open ? openWidth : railWidth)}px` } as CSSProperties}>
        <Island open={open} onToggle={toggle} section={section} onSignOut={onSignOut} />
        <motion.main
          initial={false}
          animate={{ paddingLeft: inset + (open ? openWidth : railWidth) + gap, paddingRight: pageRight }}
          transition={spring}
          className="@container flex h-full min-h-0 flex-col overflow-y-auto pt-6 pb-8 *:shrink-0"
        >
          {page}
        </motion.main>
      </div>
    </MotionConfig>
  );
}

const fade = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.12 },
};

function Island({ open, onToggle, section, onSignOut }: {
  open: boolean; onToggle: () => void; section: string; onSignOut?: () => void;
}) {
  const [backend, setBackend] = useState<string>();
  const [version, setVersion] = useState<string>();
  const [live, setLive] = useState(false);

  useEffect(() => {
    let alive = true;
    const check = () =>
      getHealth()
        .then((h) => { if (alive) { setLive(true); setBackend(h.backend); setVersion(h.version); } })
        .catch(() => alive && setLive(false));
    check();
    const t = setInterval(check, 15_000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <motion.nav
      aria-label="Navigation"
      initial={false}
      animate={{ width: open ? openWidth : railWidth }}
      transition={spring}
      style={{ left: inset, top: inset, bottom: inset }}
      className="fixed z-50 flex flex-col overflow-clip rounded-xl border border-border bg-sidebar p-1 text-foreground shadow-paper"
    >
      <a href="#" title="openfinetuner, formerly ShadowLM" className="flex h-12 shrink-0 items-center gap-2.5 px-1.5 outline-none focus-visible:ring-2 focus-visible:ring-ring/40">
        <Monogram />
        <span className="grid min-w-0 leading-tight">
          <NavLabel open={open} className="font-heading text-sm font-semibold tracking-[-0.01em]">openfinetuner</NavLabel>
          <NavLabel open={open} className="text-[11px] text-muted-foreground">formerly ShadowLM</NavLabel>
        </span>
      </a>

      <div className="mt-2 flex min-h-0 flex-col gap-0.5 overflow-y-auto px-1.5 [scrollbar-width:none]">
        {sections.map((sec, i) => (
          <div key={sec.title ?? i} className="flex flex-col gap-0.5">
            {sec.title && <SectionTitle open={open}>{sec.title}</SectionTitle>}
            {sec.items.map((item) => (
              <Item key={item.hash} item={item} open={open} active={section === item.hash} />
            ))}
          </div>
        ))}
      </div>

      <div className="flex-1" />

      <div className="flex flex-col gap-0.5 px-1.5 pb-2">
        <Item item={machinesItem} open={open} active={section === machinesItem.hash} />
        <HfTokenButton open={open} />
        <VramButton open={open} />
        <ActionRow open={open} icon={BookOpen} label="Docs" title="The README: SDK, CLI, methods, studio"
                   onClick={() => window.open(`${repo}#readme`, "_blank", "noreferrer")} />
        <ActionRow open={open} icon={ExternalLink} label="GitHub" title="github.com/open-gitagent/shadowLM"
                   onClick={() => window.open(repo, "_blank", "noreferrer")} />
      </div>

      <div className={cn("mx-1.5 flex gap-2.5 border-t border-border pt-3 pb-1.5", open ? "flex-col" : "flex-col items-center")}>
        <AnimatePresence initial={false} mode="popLayout">
          {open ? (
            <motion.div key="status" {...fade} className="grid min-w-0 gap-1.5 text-[11px] whitespace-nowrap text-muted-foreground">
              {backend && (
                <span className="grid min-w-0 gap-0.5">
                  <span className="truncate text-xs font-medium text-foreground">This machine</span>
                  <span className="text-[11px] text-muted-foreground">backend {backend}</span>
                </span>
              )}
              <span className="flex items-center gap-1.5">
                <LiveDot live={live} label />
                {version && (
                  <>
                    <span aria-hidden>·</span>
                    <span className="tabular-nums">v{version}</span>
                  </>
                )}
              </span>
            </motion.div>
          ) : (
            <motion.div key="dot" {...fade} className="grid justify-items-center gap-2 py-1">
              <LiveDot live={live} />
            </motion.div>
          )}
        </AnimatePresence>
        <div className={cn("flex gap-0.5", open ? "-mx-1" : "flex-col")}>
          <ThemeToggle />
          {onSignOut && (
            <IconButton label="Sign out" onClick={onSignOut}>
              <LogOut className="size-3.5" />
            </IconButton>
          )}
          <IconButton label={open ? "Collapse navigation" : "Expand navigation"} onClick={onToggle}>
            {open ? <PanelLeftClose className="size-3.5" /> : <PanelLeftOpen className="size-3.5" />}
          </IconButton>
        </div>
      </div>
    </motion.nav>
  );
}

const row = (active: boolean) =>
  cn(
    "group flex h-8 shrink-0 items-center gap-3 rounded-md px-1.5 text-[13px] font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
    active ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-surface-2/70 hover:text-foreground",
  );

function Item({ item, open, active }: { item: NavItem; open: boolean; active: boolean }) {
  return (
    <a href={`#${item.hash}`} aria-current={active ? "page" : undefined} title={open ? undefined : item.label} className={row(active)}>
      <item.icon className="size-4 shrink-0 opacity-70 group-aria-[current=page]:opacity-100" strokeWidth={1.75} aria-hidden />
      <NavLabel open={open} className="min-w-0 flex-1">{item.label}</NavLabel>
    </a>
  );
}

// ActionRow is a nav-styled row that does something rather than going
// somewhere: its label says what, and hint, on the right, its state.
function ActionRow({ open, icon: Icon, label, hint, title, onClick }: {
  open: boolean; icon: LucideIcon; label: string; hint?: ReactNode; title?: string; onClick: () => void;
}) {
  return (
    <button type="button" onClick={onClick} title={open ? title : (title ?? label)} className={cn(row(false), "w-full text-left")}>
      <Icon className="size-4 shrink-0 opacity-70" strokeWidth={1.75} aria-hidden />
      <NavLabel open={open} className="min-w-0 flex-1">{label}</NavLabel>
      {open && hint && <span className="shrink-0 text-[11px] font-normal text-muted-foreground tabular-nums">{hint}</span>}
    </button>
  );
}

function SectionTitle({ open, children }: { open: boolean; children: ReactNode }) {
  return (
    <div className="relative mt-4 mb-1 h-4 shrink-0">
      <motion.span
        initial={false}
        animate={{ opacity: open ? 0 : 1 }}
        transition={{ duration: 0.12 }}
        className="absolute inset-x-1.5 top-1/2 h-px bg-border"
        aria-hidden
      />
      <div className="absolute inset-x-0 -top-0.5 flex h-5 items-center px-1.5">
        <NavLabel open={open} className="text-[11px] font-medium text-muted-foreground/80">{children}</NavLabel>
      </div>
    </div>
  );
}

// NavLabel is text that is there when the island is open and gone, not
// squashed, when it folds.
function NavLabel({ open, className, children }: { open: boolean; className?: string; children: ReactNode }) {
  return (
    <motion.span
      initial={false}
      animate={{ opacity: open ? 1 : 0 }}
      transition={{ duration: open ? 0.2 : 0.08, delay: open ? 0.08 : 0 }}
      className={cn("truncate whitespace-nowrap", className)}
      aria-hidden={!open}
    >
      {children}
    </motion.span>
  );
}

// LiveDot says whether the API is answering.
function LiveDot({ live, label }: { live: boolean; label?: boolean }) {
  return (
    <span title={live ? "API reachable" : "the API is not answering; what is shown may be out of date"} className="flex items-center gap-1.5">
      <span className={cn("size-1.5 shrink-0 rounded-full", live ? "bg-good" : "bg-muted-foreground/40")} />
      {label && (live ? "live" : "offline")}
    </span>
  );
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="grid size-7 place-items-center rounded-md text-muted-foreground outline-none hover:bg-surface-2 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40"
    >
      {children}
    </button>
  );
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  return (
    <IconButton label="Toggle theme" onClick={() => setTheme(dark ? "light" : "dark")}>
      {dark ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
    </IconButton>
  );
}

// useHfToken is the Hugging Face token the server uses for gated and private
// models: whether one is set, and its dialog, opened from the island
// (HfTokenButton) or, embedded, from the host's menu (HostMenu).
function useHfToken() {
  const [open, setOpen] = useState(false);
  const [isSet, setIsSet] = useState(false);
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    getSettings().then((s) => setIsSet(s.hf_token_set)).catch(() => {});
  }, []);

  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      setIsSet((await setHfToken(value)).hf_token_set);
      setValue("");
      setOpen(false);
    } catch (e2) {
      setErr((e2 as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const dialog = (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-md">
        <form onSubmit={save} className="grid gap-4">
          <DialogHeader>
            <DialogTitle>Hugging Face token</DialogTitle>
            <DialogDescription>
              For gated and private models. It is stored on this server, never in the browser.
              {isSet && " A token is set; saving replaces it."}
            </DialogDescription>
          </DialogHeader>
          <Input type="password" autoFocus placeholder="hf_…" value={value} onChange={(e) => setValue(e.target.value)} />
          {err && <p className="text-sm text-destructive">{err}</p>}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
            <Button type="submit" disabled={!value.trim() || busy}>{busy ? "Saving…" : "Save token"}</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
  return { isSet, setOpen, dialog };
}

function HfTokenButton({ open: islandOpen }: { open: boolean }) {
  const hf = useHfToken();
  return (
    <>
      <ActionRow open={islandOpen} icon={KeyRound} label="Hugging Face token"
                 hint={hf.isSet ? "set" : "not set"}
                 title="The token the server uses for gated and private models"
                 onClick={() => hf.setOpen(true)} />
      {hf.dialog}
    </>
  );
}

// VramButton unloads cached models and frees GPU memory; beside it, how
// much is in use, and after a clean, what it came down to.
// useVram is the GPU memory cached models hold, and freeing it.
function useVram() {
  const [hint, setHint] = useState("");
  const [busy, setBusy] = useState(false);
  const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`;

  useEffect(() => {
    getVram().then((v) => v.used_mb != null && setHint(`${gb(v.used_mb)} used`)).catch(() => {});
  }, []);

  async function clean() {
    setBusy(true);
    try {
      const r = await clearVram();
      setHint(r.after_mb != null ? `freed · ${gb(r.after_mb)}` : "freed");
    } catch {
      setHint("failed");
    } finally {
      setBusy(false);
    }
  }
  return { hint: busy ? "clearing…" : hint, busy, clean };
}

function VramButton({ open }: { open: boolean }) {
  const vram = useVram();
  return (
    <ActionRow open={open} icon={vram.busy ? LoaderCircle : Zap} label="Clean VRAM"
               hint={vram.hint}
               title="Unload cached models and free GPU memory"
               onClick={vram.clean} />
  );
}

// HostMenu is the studio's menu for a host console that shows its own in
// place of the island (lib/embed.ts): the same sections, the island's
// actions the person's role in the host allows, and its links. The host
// says when an action is chosen; it runs here, where its dialog is.
function HostMenu() {
  const hf = useHfToken();
  const vram = useVram();
  const admin = allows("admin");
  const operator = allows("operator");
  const { setOpen } = hf;
  const { clean } = vram;

  useEffect(() => {
    const item = (i: NavItem): MenuItem => ({ label: i.label, icon: glyphs.get(i.icon) ?? "", path: hashToPage(i.hash) });
    const foot: MenuItem[] = [item(machinesItem)];
    if (admin) foot.push({ label: "Hugging Face token", icon: "key-round", hint: hf.isSet ? "set" : "not set", action: "hf-token" });
    if (operator) foot.push({ label: "Clean VRAM", icon: vram.busy ? "loader-circle" : "zap", hint: vram.hint || undefined, action: "clear-vram" });
    foot.push({ label: "Docs", icon: "book-open", href: `${repo}#readme` }, { label: "GitHub", icon: "external-link", href: repo });
    reportMenu({ groups: sections.map((s) => ({ title: s.title, items: s.items.map(item) })), foot });
  }, [admin, operator, hf.isSet, vram.busy, vram.hint]);

  useEffect(() => onHostAction((id) => {
    if (id === "hf-token" && admin) setOpen(true);
    if (id === "clear-vram" && operator) void clean();
  }), [admin, operator, setOpen, clean]);

  return hf.dialog;
}
