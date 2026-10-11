// The cockpit's left pane: the copilot's conversation about one project. The
// transcript is derived from the loop's state (copilot.narrate), so it is always
// true; what the person types, and the copilot's replies, run as a log after
// it. The newest proposal is the live action card; approving it lights its
// station on the loop map, and every message that speaks about a station is
// linked to it both ways (hover highlights it, the rail scrolls the thread).
import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft, ArrowRight, ArrowUp, ChevronDown, LoaderCircle, MessagesSquare, Pencil, Play, Rocket, Sparkles, TriangleAlert,
} from "lucide-react";
import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import { Mono } from "@/components/common";
import { FrontierDialog, useFrontier } from "@/components/frontier-settings";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { allows, needs } from "@/lib/embed";
import { breakable, examples, methodLabel, modelParts } from "@/lib/format";
import { useMode } from "@/lib/mode";
import { goalLabel, RECIPE_METHODS, type Recipe, withMethod } from "@/lib/recipe";
import { cn } from "@/lib/utils";

import { type CockpitActions, type Loop, useCockpitLink } from "./Cockpit";
import { askCtrl } from "@/api";

import { ctrlState, fromAction, interpret, type Message, narrate, type Proposal, suggestions } from "./copilot";
import { STATIONS, type StationId } from "./model";

const stationLabel = (s: StationId) => STATIONS.find((x) => x.id === s)?.label ?? s;

// ---- the typed log, kept per project in this browser ---------------------------
// by: the model that answered (Claude), unset when the rules did
interface Typed extends Message { spent?: boolean; by?: string }
const MAX_LOG = 50;
const logKey = (pid: string) => `of.thread.${pid}`;

function readLog(pid: string): Typed[] {
  try {
    const v = JSON.parse(localStorage.getItem(logKey(pid)) ?? "[]");
    return Array.isArray(v) ? v.slice(-MAX_LOG) : [];
  } catch {
    return [];
  }
}

function writeLog(pid: string, log: Typed[]): void {
  try {
    localStorage.setItem(logKey(pid), JSON.stringify(log.slice(-MAX_LOG)));
  } catch {
    // storage blocked: the exchange holds for this visit only
  }
}

const reducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

// ---- the pane --------------------------------------------------------------------
export function Conversation({ loop, actions }: { loop: Loop; actions: CockpitActions }) {
  const project = loop.project!;
  const pid = project.project_id;
  const link = useCockpitLink();
  const [log, setLog] = useState<Typed[]>(() => readLog(pid));
  const [flash, setFlash] = useState<string | null>(null);
  const [thinking, setThinking] = useState(false);  // Claude is answering
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setLog(readLog(pid)); }, [pid]);

  const derived = narrate(loop);
  // the narration's pending decision stays at the bottom of the thread, under
  // any chat since, unless the latest typed reply is itself an open proposal
  const messages: Typed[] = useMemo(() => {
    const last = derived.at(-1);
    const typedLast = log.at(-1);
    const typedOpen = !!typedLast?.proposal && !typedLast.spent;
    return last?.proposal && log.length && !typedOpen
      ? [...derived.slice(0, -1), ...log, last]
      : [...derived, ...log];
  }, [derived, log]);
  const liveIndex = messages.findLastIndex((m) => m.proposal && !m.spent);

  // new messages rise in; ones present on first paint don't
  const seen = useRef<Set<string> | null>(null);
  useEffect(() => {
    seen.current = new Set(messages.map((m) => m.id));
  });

  // stay at the newest message as the thread grows
  const count = messages.length + (thinking ? 1 : 0);
  useEffect(() => {
    const el = threadRef.current;
    if (el && !link.stacked) el.scrollTo({ top: el.scrollHeight, behavior: reducedMotion() ? "auto" : "smooth" });
  }, [count, link.stacked]);

  // stacked, the thread opens on the newest exchange; earlier ones fold away
  const [showAll, setShowAll] = useState(false);
  const EARLIER = 3;
  const folded = link.stacked && !showAll && messages.length > EARLIER;
  // fold up to the live proposal, so the decision is what a phone opens on
  const firstShown = folded ? Math.max(0, liveIndex >= 0 ? liveIndex : messages.length - EARLIER) : 0;

  // the rail asked for a station: show its latest message and mark it briefly
  useEffect(() => {
    if (!link.scrollTo) return;
    const target = messages.findLast((m) => m.station === link.scrollTo!.station);
    if (!target) return;
    const el = threadRef.current?.querySelector<HTMLElement>(`[data-msg="${CSS.escape(target.id)}"]`);
    el?.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
    if (reducedMotion()) return;
    setFlash(target.id);
    const t = setTimeout(() => setFlash(null), 400);
    return () => clearTimeout(t);
    // only when the rail asks, not when the thread changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [link.scrollTo?.tick]);

  function append(entries: Typed[]) {
    setLog((prev) => {
      const next = [...prev, ...entries].slice(-MAX_LOG);
      writeLog(pid, next);
      return next;
    });
  }

  // Claude answers when the server has a model for Ctrl agent; the rules
  // answer otherwise, and whenever Claude can't (the reply says so)
  async function send(text: string) {
    const said = text.trim();
    if (!said || thinking) return;
    const now = Date.now();
    const you: Typed = { id: `you-${now}`, from: "you", text: said, at: now };
    append([you]);
    let reply: Typed;
    if (loop.ctrl) {
      setThinking(true);
      try {
        const thread = messages.slice(-16).map((m) => ({ from: m.from, text: m.text }));
        const r = await askCtrl(pid, { message: said, state: ctrlState(loop), thread });
        const a = fromAction(r.action, loop);
        reply = { id: `re-${now}`, from: "copilot", by: loop.ctrl.model, at: Date.now(),
                  text: r.reply || (a.proposal ? "Here's what I'd do:" : "I don't have an answer for that."), ...a };
      } catch (e) {
        const u = interpret(said, loop);
        reply = { id: `re-${now}`, from: "copilot", text: u.reply, station: u.station, proposal: u.proposal, at: Date.now(),
                  detail: [`Claude couldn't answer (${(e as Error).message}), so this is the built-in reply.`] };
      } finally {
        setThinking(false);
      }
    } else {
      const u = interpret(said, loop);
      reply = { id: `re-${now}`, from: "copilot", text: u.reply, station: u.station, proposal: u.proposal, at: now };
    }
    append([reply]);
    if (reply.station) link.select(reply.station, "thread");
  }

  // a typed proposal that ran is spent, so a reload doesn't offer it again
  function spend(id: string) {
    setLog((prev) => {
      if (!prev.some((m) => m.id === id)) return prev;
      const next = prev.map((m) => (m.id === id ? { ...m, spent: true } : m));
      writeLog(pid, next);
      return next;
    });
  }

  return (
    <>
      <Header loop={loop} />
      <div ref={threadRef} className={link.stacked ? "space-y-4 px-5 py-5" : "min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-5 scrollbar-thin"} aria-live="polite">
        {folded && firstShown > 0 && (
          <Button variant="ghost" size="sm" className="-ml-2 text-muted-foreground" onClick={() => setShowAll(true)}>
            Show {firstShown} earlier {firstShown === 1 ? "message" : "messages"}
          </Button>
        )}
        {messages.map((m, i) => i < firstShown ? null : (
          <MessageRow key={m.id} m={m} isNew={!!seen.current && !seen.current.has(m.id)} flashing={flash === m.id}>
            {m.proposal && (i === liveIndex
              ? <ActionCard proposal={m.proposal} loop={loop} actions={actions} onRan={() => spend(m.id)} />
              : <SpentProposal proposal={m.proposal} />)}
          </MessageRow>
        ))}
        {thinking && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground" role="status">
            <LoaderCircle className="size-3.5 animate-spin" /> Ctrl agent is thinking…
          </div>
        )}
      </div>
      {/* stacked, the composer stays pinned to the bottom of the screen */}
      <div className={link.stacked ? "sticky -bottom-8 z-10 bg-card pb-8" : "contents"}>
        <LiveLine loop={loop} />
        <Composer loop={loop} onSend={(t) => void send(t)} busy={thinking} oneRow={link.stacked} />
      </div>
    </>
  );
}

// ---- header ----------------------------------------------------------------------
function Header({ loop }: { loop: Loop }) {
  const mode = useMode();
  const p = loop.project!;
  const s = loop.stations[loop.next];
  return (
    <header className="flex shrink-0 items-start justify-between gap-3 border-b border-border px-5 py-3.5">
      <div className="min-w-0">
        <h1 className="text-base leading-tight font-semibold tracking-[-0.01em] text-balance">{p.name}</h1>
        <p className="mt-0.5 text-xs text-muted-foreground">{goalLabel[p.goal]}</p>
        <p className="mt-1 text-xs">
          <span className={cn("font-medium", s.tone === "warning" ? "text-warning" : s.tone === "good" ? "text-good" : "text-foreground/80")}>{s.label}</span>
          <span className="text-muted-foreground"> · {s.tone === "warning" ? "not there yet · " : s.tone === "good" ? "ready · " : ""}{s.line}</span>
        </p>
        {mode === "research" && p.run_id && <Mono className="mt-1 block text-[11px] text-muted-foreground">run {p.run_id}</Mono>}
      </div>
      <Button variant="ghost" size="sm" asChild className="shrink-0 text-muted-foreground">
        <a href="#projects" aria-label="All projects" title="All projects">
          <ArrowLeft /><span className="hidden @md:inline">All projects</span>
        </a>
      </Button>
    </header>
  );
}

// ---- one message -------------------------------------------------------------------
function MessageRow({ m, isNew, flashing, children }: {
  m: Typed; isNew: boolean; flashing: boolean; children?: React.ReactNode;
}) {
  const link = useCockpitLink();
  const mode = useMode();
  const linked = !!m.station;
  const lit = linked && link.hovered === m.station;
  const rise = isNew ? "motion-safe:[animation:rise_0.2s_ease-out]" : "";

  if (m.from === "you") {
    return (
      <div data-msg={m.id} className={cn("flex justify-end", rise)}>
        <p className="max-w-[85%] border border-border bg-subtle px-3.5 py-2 text-sm whitespace-pre-wrap">{m.text}</p>
      </div>
    );
  }

  return (
    <div
      data-msg={m.id}
      onMouseEnter={linked ? () => link.hover(m.station!) : undefined}
      onMouseLeave={linked ? () => link.hover(null) : undefined}
      className={cn("-mx-2 rounded-md px-2 py-1.5 transition-colors duration-200 ease-out", rise,
        flashing ? "bg-primary/10" : lit ? "bg-primary/5" : "")}
    >
      <div className="mb-1 flex items-center gap-2 text-[11px] text-muted-foreground">
        <MessagesSquare className="size-3" strokeWidth={1.75} aria-hidden />
        <span>Ctrl agent</span>
        {m.by && mode === "research" && <span className="font-mono text-[10px]">· {m.by}</span>}
        {linked && (
          <button type="button" onClick={() => link.select(m.station!, "thread")}
            className={cn("ml-auto rounded-sm px-1.5 py-px text-[11px] transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
              link.selected === m.station ? "text-primary" : "text-muted-foreground hover:text-foreground")}>
            {stationLabel(m.station!)}
          </button>
        )}
      </div>
      <p className="text-sm leading-relaxed">{m.text}</p>
      {m.detail && m.detail.length > 0 && (!m.proposal || m.from === "copilot" && m.id.startsWith("re-")) && (
        <ul className="mt-1.5 space-y-0.5 text-sm text-muted-foreground">
          {m.detail.map((d) => <li key={d} className="flex gap-2"><span aria-hidden>·</span><span>{d}</span></li>)}
        </ul>
      )}
      {children && <div className="mt-2.5">{children}</div>}
    </div>
  );
}

// ---- proposals ----------------------------------------------------------------------
const proposalTitle: Record<Proposal["kind"], string> = {
  finetune: "Fine-tune",
  evaluate: "Evaluate",
  deploy: "Deploy",
  "add-frontier": "Frontier model",
  playground: "Playground",
  synthesize: "Write more examples",
};

// An earlier proposal: what was offered, without buttons.
function SpentProposal({ proposal }: { proposal: Proposal }) {
  const what =
    proposal.kind === "finetune" ? `Version ${proposal.version}: ${proposal.recipe.method} on ${proposal.recipe.base_model.split("/").pop()}, ${proposal.recipe.config.max_steps} steps` :
    proposal.kind === "evaluate" ? (proposal.withFrontier ? "Against the base and frontier models" : "Against the base model") :
    proposal.kind === "deploy" ? "Deploy behind an OpenAI-compatible endpoint" :
    proposal.kind === "add-frontier" ? "Add a frontier model" :
    proposal.kind === "synthesize" ? `${proposal.n} more examples, written by the frontier model` : "Try it in the Playground";
  return (
    <p className="border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
      Proposed · {proposalTitle[proposal.kind]}: {what}
    </p>
  );
}

function withEdits(r: Recipe, steps: number, base: string): Recipe {
  const cli = r.cli
    .replace(/--max-steps \d+/, `--max-steps ${steps}`)
    .replace(/--model \S+/, `--model ${base}`);
  return { ...r, base_model: base, config: { ...r.config, max_steps: steps }, cli };
}

function toPlayground(loop: Loop) {
  if (!loop.project?.run_id || !loop.job) return;
  try {
    sessionStorage.setItem("pick.adapter", loop.project.run_id);
    sessionStorage.setItem("pick.model", loop.job.base_model);
    sessionStorage.removeItem("pick.checkpoint");
  } catch {
    // storage blocked: the Playground opens on its default
  }
  window.location.hash = "#playground";
}

function ActionCard({ proposal, loop, actions, onRan }: {
  proposal: Proposal; loop: Loop; actions: CockpitActions; onRan: () => void;
}) {
  const link = useCockpitLink();
  const operator = allows("operator");
  const busy = actions.busy !== null;

  return (
    <div className="border border-border bg-card px-4 py-3.5 shadow-paper">
      <p className="text-sm font-semibold tracking-[-0.01em]">{proposalTitle[proposal.kind]}</p>
      {proposal.kind === "finetune" && (
        <FinetuneBody proposal={proposal} loop={loop} actions={actions} operator={operator}
          onStart={async (recipe) => {
            await actions.run({ kind: "finetune", recipe });
            onRan();
            link.select("finetune");
          }} />
      )}
      {proposal.kind === "evaluate" && (
        <>
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-muted-foreground">Compares</dt>
            <dd>Your fine-tune, the base model{proposal.withFrontier && loop.frontier ? `, ${loop.frontier.model}` : ""}</dd>
            <dt className="text-muted-foreground">Scored by</dt>
            <dd>{proposal.withFrontier ? `${loop.frontier?.model ?? "the frontier model"} judging each answer` : "whether each answer contains yours"}</dd>
          </dl>
          <CardActions operator={operator}>
            <Button disabled={busy} onClick={async () => {
              await actions.run({ kind: "evaluate", withFrontier: proposal.withFrontier });
              onRan();
              link.select("evaluate");
            }}>
              {actions.busy === "evaluate" ? <LoaderCircle className="animate-spin" /> : <Play />} Run evaluation
            </Button>
          </CardActions>
        </>
      )}
      {proposal.kind === "deploy" && (
        <>
          <p className="mt-1 text-sm text-muted-foreground">
            Serve it behind an OpenAI-compatible endpoint. Your agent changes only its base URL, key and model name.
          </p>
          <CardActions operator={operator}>
            <Button onClick={() => link.select("deploy")}><Rocket /> Set up deployment</Button>
            <Button variant="ghost" onClick={() => toPlayground(loop)}>Try in the Playground <ArrowRight /></Button>
          </CardActions>
        </>
      )}
      {proposal.kind === "add-frontier" && <FrontierBody />}
      {proposal.kind === "synthesize" && (
        <SynthesizeBody proposal={proposal} loop={loop} actions={actions} operator={operator}
          onStart={async (n) => {
            await actions.run({ kind: "synthesize", n });
            onRan();
            link.select("data");
          }} />
      )}
      {proposal.kind === "playground" && (
        <CardActions operator>
          <Button variant="outline" onClick={() => toPlayground(loop)} disabled={!loop.project?.run_id || !loop.job}>
            Open the Playground <ArrowRight />
          </Button>
        </CardActions>
      )}
      {actions.error && (
        <p className="mt-3 flex gap-1.5 text-sm text-destructive">
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
          <span>{actions.error}. Check the station on the right, then try again.</span>
        </p>
      )}
    </div>
  );
}

function CardActions({ operator, children }: { operator: boolean; children: React.ReactNode }) {
  if (!operator) return <p className="mt-3 text-xs text-muted-foreground">{needs("operator")}</p>;
  return <div className="mt-3 flex flex-wrap items-center gap-2">{children}</div>;
}

function FinetuneBody({ proposal, loop, actions, operator, onStart }: {
  proposal: Extract<Proposal, { kind: "finetune" }>; loop: Loop; actions: CockpitActions;
  operator: boolean; onStart: (r: Recipe) => void;
}) {
  const mode = useMode();
  const r = proposal.recipe;
  const [editing, setEditing] = useState(false);
  const [steps, setSteps] = useState(String(r.config.max_steps ?? ""));
  const [base, setBase] = useState(r.base_model);
  const [method, setMethod] = useState(r.method);
  const stepsN = Number(steps);
  const valid = Number.isInteger(stepsN) && stepsN >= 1 && stepsN <= 100_000 && base.trim().length > 0;
  const chosen = withMethod(r, method);
  const recipe = valid ? withEdits(chosen, stepsN, base.trim()) : chosen;
  // the methods offered here train on plain examples; a recipe's own (from a
  // typed request) stays offered even if it isn't one of them
  const methods = RECIPE_METHODS.some((m) => m.id === r.method) ? RECIPE_METHODS : [{ id: r.method, why: "" }, ...RECIPE_METHODS];
  const busy = actions.busy !== null;
  void loop;

  return (
    <>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted-foreground">Version</dt><dd className="tabular-nums">{proposal.version}</dd>
        <dt className="text-muted-foreground">Base model</dt>
        <dd className="min-w-0 font-mono text-xs leading-5" title={recipe.base_model}>
          <span>{breakable(modelParts(recipe.base_model).name)}</span>
          {modelParts(recipe.base_model).org && <span className="block text-[11px] text-muted-foreground">{modelParts(recipe.base_model).org}</span>}
        </dd>
        <dt className="text-muted-foreground">Method</dt><dd>{methodLabel(recipe.method)}</dd>
        <dt className="text-muted-foreground">Steps</dt><dd className="tabular-nums">{String(recipe.config.max_steps)}</dd>
        {mode === "research" && Object.entries(recipe.config).filter(([k]) => k !== "method" && k !== "max_steps").map(([k, v]) => (
          <div key={k} className="contents"><dt className="font-mono text-xs text-muted-foreground">{k}</dt><dd className="font-mono text-xs">{String(v)}</dd></div>
        ))}
      </dl>
      {recipe.why.length > 0 && (
        <ul className="mt-2.5 space-y-0.5 text-sm text-muted-foreground">
          {recipe.why.map((w) => <li key={w} className="flex gap-2"><span aria-hidden>·</span><span>{w}</span></li>)}
        </ul>
      )}
      {editing && (
        <div className="mt-3 grid gap-2 @md:grid-cols-[8rem_1fr]">
          <label className="grid gap-1 text-xs text-muted-foreground @md:col-span-2">
            Method
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              {methods.map((m) => <option key={m.id} value={m.id}>{methodLabel(m.id)}{m.why ? ` — ${m.why.split(": ")[1]}` : ""}</option>)}
            </select>
          </label>
          <label className="grid gap-1 text-xs text-muted-foreground">
            Steps
            <Input type="number" min={1} value={steps} onChange={(e) => setSteps(e.target.value)} className="tabular-nums" />
          </label>
          <label className="grid gap-1 text-xs text-muted-foreground">
            Base model
            <Input value={base} onChange={(e) => setBase(e.target.value)} className="font-mono text-xs" />
          </label>
          {!valid && <p className="text-xs text-destructive @md:col-span-2">Steps must be a whole number from 1 to 100,000, and the base model can't be empty.</p>}
        </div>
      )}
      <details className="group mt-2.5">
        <summary className="flex cursor-pointer list-none items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
          <ChevronDown className="size-3 transition-transform group-open:rotate-180" /> Same run from your shell
        </summary>
        <pre className="mt-1.5 overflow-x-auto rounded-md bg-surface px-3 py-2 font-mono text-[11px] leading-relaxed">{recipe.cli}</pre>
      </details>
      <CardActions operator={operator}>
        <Button disabled={busy || !valid} onClick={() => onStart(recipe)}>
          {actions.busy === "finetune" ? <LoaderCircle className="animate-spin" /> : <Play />} Start version {proposal.version}
        </Button>
        <Button variant="ghost" disabled={busy} onClick={() => setEditing((e) => !e)} aria-expanded={editing}>
          <Pencil /> {editing ? "Done editing" : "Edit"}
        </Button>
      </CardActions>
    </>
  );
}

function SynthesizeBody({ proposal, loop, actions, operator, onStart }: {
  proposal: Extract<Proposal, { kind: "synthesize" }>; loop: Loop; actions: CockpitActions;
  operator: boolean; onStart: (n: number) => void;
}) {
  const [n, setN] = useState(String(proposal.n));
  const nN = Number(n);
  const valid = Number.isInteger(nN) && nN >= 1 && nN <= 1000;
  const rows = loop.dataset?.rows ?? 0;
  const teacher = loop.frontier?.model;
  return (
    <>
      <dl className="mt-2 grid grid-cols-[auto_1fr] items-center gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted-foreground">Written by</dt>
        <dd className="min-w-0 truncate font-mono text-xs" title={teacher}>{teacher ?? "no frontier model yet"}</dd>
        <dt className="text-muted-foreground">New examples</dt>
        <dd><Input type="number" min={1} max={1000} value={n} onChange={(e) => setN(e.target.value)}
                   aria-label="New examples" className="h-7 w-20 tabular-nums" /></dd>
        <dt className="text-muted-foreground">Keeps</dt><dd>Your {examples(rows)}, unchanged</dd>
        <dt className="text-muted-foreground">Checked by</dt><dd>{teacher ?? "The frontier model"} scoring each one; weak ones are dropped</dd>
      </dl>
      {!valid && <p className="mt-2 text-xs text-destructive">Ask for 1 to 1,000 examples.</p>}
      <CardActions operator={operator}>
        <Button disabled={actions.busy !== null || !valid || !teacher} onClick={() => onStart(nN)}>
          {actions.busy === "synthesize" ? <LoaderCircle className="animate-spin" /> : <Sparkles />} Write {valid ? nN : ""} examples
        </Button>
      </CardActions>
    </>
  );
}

function FrontierBody() {
  const qc = useQueryClient();
  const { frontier, refresh } = useFrontier();
  const [open, setOpen] = useState(false);
  return (
    <>
      <p className="mt-1 text-sm text-muted-foreground">
        Any OpenAI-compatible model, by base URL, model name and key. The key stays on the server.
      </p>
      <CardActions operator={allows("operator")}>
        <Button variant="outline" onClick={() => setOpen(true)}>{frontier ? `Change ${frontier.model}` : "Add frontier model"}</Button>
      </CardActions>
      <FrontierDialog open={open} onOpenChange={setOpen} current={frontier}
        onSaved={() => { refresh(); void qc.invalidateQueries({ queryKey: ["settings"] }); }} />
    </>
  );
}

// ---- the live line while a fine-tune runs ---------------------------------------------
function LiveLine({ loop }: { loop: Loop }) {
  const job = loop.job;
  if (!job || (job.status !== "running" && job.status !== "pending")) return null;
  const n = loop.versions.length || 1;
  return (
    <div className="flex shrink-0 items-center gap-3 border-t border-border bg-primary/5 px-5 py-2 text-xs text-primary">
      <span aria-hidden className="h-0.5 w-10 shrink-0 rounded-full bg-[repeating-linear-gradient(90deg,currentColor_0_8px,transparent_8px_16px)] motion-safe:[animation:of-dashflow_0.6s_linear_infinite]" />
      <span className="truncate">
        {job.status === "pending" ? `Version ${n} is queued for the GPU` : `Fine-tuning version ${n} · ${loop.steps.length} steps`}
      </span>
    </div>
  );
}

// ---- composer -----------------------------------------------------------------------------
function Composer({ loop, onSend, busy = false, oneRow = false }: { loop: Loop; onSend: (text: string) => void; busy?: boolean; oneRow?: boolean }) {
  const [text, setText] = useState("");
  const chips = suggestions(loop);

  function submit() {
    if (!text.trim() || busy) return;
    onSend(text);
    setText("");
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="shrink-0 border-t border-border px-4 pt-3 pb-4">
      {chips.length > 0 && (
        <div className={oneRow
          ? "-mx-4 mb-2.5 flex gap-1.5 overflow-x-auto px-4 [scrollbar-width:none] *:shrink-0"
          : "mb-2.5 flex flex-wrap gap-1.5"}>
          {chips.map((c) => (
            <Button key={c} type="button" variant="outline" size="xs" disabled={busy} onClick={() => onSend(c)} className="font-normal">
              {c}
            </Button>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2">
        <textarea
          value={text} rows={1} data-slot="composer" aria-label="Message Ctrl agent"
          onChange={(e) => setText(e.target.value)} onKeyDown={onKey}
          placeholder={loop.ctrl ? "Ask anything about this model, or say what to do next" : "Ask, or say what to do next: “train again with 300 steps”"}
          className="field-sizing-content max-h-32 min-h-9 min-w-0 grow resize-none rounded-lg border border-input bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        />
        <Button size="icon" className="size-9 shrink-0" aria-label="Send" onClick={submit} disabled={!text.trim() || busy}>
          <ArrowUp className="size-4" />
        </Button>
      </div>
    </div>
  );
}
