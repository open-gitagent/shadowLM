// The cockpit before the project exists: the same two panes, so the first
// minute already feels like the instrument. The copilot asks what the model
// should do (answer with a choice or in your own words), the Data station on
// the right collects the examples, and the plan arrives as an action card.
// Approving it creates the project, saves the examples and starts the
// fine-tune; the full cockpit takes over with the run already live.
import { ArrowRight, Check, ChevronDown, CornerDownLeft, LoaderCircle, MessagesSquare, Sparkles, TriangleAlert } from "lucide-react";
import { type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import {
  createDataset, createProject, deleteProject, getHealth,
  type Health, type Project, type ProjectGoal,
} from "@/api";
import { EmptyState, Field } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { allows, needs } from "@/lib/embed";
import { pickRecipe, startProjectFinetune } from "@/lib/recipe";
import { cn } from "@/lib/utils";

import { chatRow, type ExamplesValue, ExamplesInput, goals, labels, minimum } from "./start/ExamplesInput";

// ---- reading what someone types ---------------------------------------------
const knowledgeWords = ["faq", "knowledge", "fact", "policy", "policies", "docs", "documentation", "answer questions about", "handbook", "product details", "wiki"];
const takeoverWords = ["agent", "gpt", "openai", "frontier", "replace", "take over", "takeover", "cheaper", "claude", "gemini", "rented"];

function classify(text: string): { goal: ProjectGoal; strong: boolean } {
  const t = text.toLowerCase();
  if (takeoverWords.some((w) => t.includes(w))) return { goal: "takeover", strong: true };
  if (knowledgeWords.some((w) => t.includes(w))) return { goal: "knowledge", strong: true };
  return { goal: "task", strong: false };
}

const reading: Record<ProjectGoal, string> = {
  knowledge: "Sounds like a knowledge model: it should answer your questions exactly as you wrote the answers.",
  task: "Sounds like a task model: it should learn to turn inputs like yours into the outputs you want.",
  takeover: "Sounds like taking over a task from a frontier model: it learns from your agent's real conversations.",
};

const askFor: Record<ProjectGoal, string> = {
  knowledge: "Now give it the facts, on the Data station: write them as questions and answers, upload a file, or pick a dataset.",
  task: "Now give it examples, on the Data station: inputs with the outputs you want. Write them, upload a file, pick a dataset, or capture your agent.",
  takeover: "Now connect your agent on the Data station: it keeps working as before while its conversations are recorded. Traces or a file work too.",
};

interface Turn { id: number; from: "you" | "copilot"; text: string; choose?: boolean }

export default function CockpitStart() {
  const [goal, setGoal] = useState<ProjectGoal | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [examples, setExamples] = useState<ExamplesValue | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [draft, setDraft] = useState("");
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [phase, setPhase] = useState("");
  const [err, setErr] = useState("");
  const nextId = useRef(1);
  const threadEnd = useRef<HTMLDivElement>(null);

  useEffect(() => { getHealth().then(setHealth).catch(() => setHealth(null)); }, []);

  const count = examples?.count ?? 0;
  const recipe = useMemo(() => (goal ? pickRecipe(goal, Math.max(count, 1), health) : null), [goal, count, health]);
  const suggested = examples?.dataset ? examples.dataset.name
    : examples?.fileName ? examples.fileName.replace(/\.jsonl?$/i, "")
    : goal === "knowledge" ? "My knowledge model" : "My task model";
  const projectName = nameTouched ? name : suggested;
  const l = labels(goal);
  const busy = phase !== "";

  // keep the newest message in view
  useEffect(() => { threadEnd.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }, [turns.length, goal, count > 0]);

  const say = (...ts: Omit<Turn, "id">[]) =>
    setTurns((prev) => [...prev, ...ts.map((t) => ({ ...t, id: nextId.current++ }))]);

  function choose(g: ProjectGoal, said?: string) {
    setGoal(g);
    say({ from: "you", text: said ?? goals.find((x) => x.goal === g)!.title },
        { from: "copilot", text: reading[g] });
  }

  function send(e?: FormEvent) {
    e?.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    if (!goal) {
      choose(classify(text).goal, text);
      return;
    }
    const c = classify(text);
    const t = text.toLowerCase();
    if (c.strong && c.goal !== goal) {
      setGoal(c.goal);
      say({ from: "you", text }, { from: "copilot", text: `${reading[c.goal]} I've switched the Data station to match.` });
    } else if (/(how many|what examples|which examples|enough|need)/.test(t)) {
      say({ from: "you", text }, { from: "copilot", text: goal === "knowledge"
        ? `Every fact you want it to know, one question and answer each. ${minimum.knowledge} or more give a clear test of what it learned.`
        : `Real inputs with the outputs you want, ideally ${minimum[goal]} or more. A few hundred teach a task well.` });
    } else {
      say({ from: "you", text }, { from: "copilot", text: count
        ? "When the examples look right, check the plan below and start fine-tuning."
        : askFor[goal] });
    }
  }

  async function start() {
    if (!goal || !recipe || busy || !examples || count === 0) return;
    const finalName = projectName.trim();
    if (!finalName) {
      setErr("Give the model a name, so you can find it later.");
      return;
    }
    setErr("");
    let project: Project | null = null;
    let savedAs = "";
    try {
      setPhase("Creating the project…");
      project = await createProject(finalName, goal);
      let datasetId = examples.dataset?.dataset_id ?? "";
      if (!datasetId) {
        setPhase(`Saving your ${count} ${l.noun}…`);
        const ds = await createDataset(finalName, examples.pairs.map(chatRow));
        datasetId = ds.dataset_id;
        savedAs = ds.name;
      }
      setPhase("Starting the fine-tune…");
      await startProjectFinetune(project, datasetId, recipe);
      window.location.hash = `#projects/${project.project_id}`;
    } catch (ex) {
      // a project without its fine-tune would sit half-made in the list, so
      // take it back; the examples, once saved, stay as a dataset
      if (project) await deleteProject(project.project_id).catch(() => {});
      setErr(`${(ex as Error).message}. Nothing was started${savedAs ? `; your examples are saved as the dataset “${savedAs}”` : ""}. Try again, or check the server is running.`);
      setPhase("");
    }
  }

  if (!allows("operator")) {
    return (
      <div className="max-w-2xl">
        <EmptyState icon={TriangleAlert} title="You can't start a model here">{needs("operator")}</EmptyState>
      </div>
    );
  }

  const chips: string[] = !goal ? goals.map((g) => g.title) : count ? ["What examples does it need?"] : ["What examples does it need?", "How many do I need?"];

  return (
    <div className="grid min-h-0 flex-1 !shrink gap-4 @4xl:grid-cols-12">
      {/* Ctrl agent, on the right beside the loop (first when stacked) */}
      <section aria-label="Ctrl agent" className="flex min-h-[560px] flex-col overflow-hidden border border-border bg-card @4xl:order-2 @4xl:col-span-5 @4xl:min-h-0">
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-border px-4">
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold tracking-[-0.01em]">New model</h1>
            <p className="truncate text-xs text-muted-foreground">
              {!goal ? "What it does" : count ? `Data · ${count} ${l.noun}` : "Data · waiting for examples"}
            </p>
          </div>
        </header>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4 scrollbar-thin" aria-live="polite">
          <Copilot>
            <p>What should your model do? Pick one, or say it in your own words below.</p>
            <div role="radiogroup" aria-label="What it does" className="mt-3 grid gap-1.5">
              {goals.map((g) => (
                <button key={g.goal} type="button" role="radio" aria-checked={goal === g.goal}
                  onClick={() => goal !== g.goal && choose(g.goal)} disabled={busy}
                  className={cn("flex items-start gap-2.5 border px-3 py-2.5 text-left transition-colors outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-60",
                    goal === g.goal ? "border-primary/40 bg-primary/5" : "border-border bg-card hover:bg-subtle")}>
                  <g.icon className={cn("mt-0.5 size-4 shrink-0", goal === g.goal ? "text-primary" : "text-muted-foreground")} strokeWidth={1.75} />
                  <span className="min-w-0 flex-1">
                    <span className={cn("block text-sm font-medium", goal === g.goal && "text-primary")}>{g.title}</span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">{g.body}</span>
                  </span>
                  {goal === g.goal && <Check className="mt-0.5 size-4 shrink-0 text-primary" />}
                </button>
              ))}
            </div>
          </Copilot>

          {turns.map((t) => t.from === "you"
            ? <You key={t.id}>{t.text}</You>
            : <Copilot key={t.id}><p>{t.text}</p></Copilot>)}

          {goal && (
            <Copilot>
              <p>{askFor[goal]}</p>
              {count > 0 && (
                <p className={cn("mt-2 flex items-start gap-1.5", count < minimum[goal] ? "text-warning" : "text-muted-foreground")}>
                  {count < minimum[goal] ? <TriangleAlert className="mt-0.5 size-3.5 shrink-0" /> : <Check className="mt-0.5 size-3.5 shrink-0 text-good" />}
                  <span>
                    {count} {l.noun} so far{examples?.dataset ? `, from ${examples.dataset.name}` : ""}.
                    {count < minimum[goal] ? (goal === "knowledge" ? " Five or more give a clearer test. You can still go ahead." : ` Fewer than ${minimum[goal]} rarely teach a task well. You can still go ahead.`) : ""}
                  </span>
                </p>
              )}
            </Copilot>
          )}

          {goal && recipe && count > 0 && (
            <div className="border border-border bg-card">
              <div className="px-4 pt-3.5 pb-3">
                <p className="flex items-center gap-1.5 text-sm font-medium"><Sparkles className="size-3.5 text-primary" />The plan for version 1</p>
                <ul className="mt-2 space-y-1.5 text-sm">
                  {recipe.why.map((w) => (
                    <li key={w} className="flex gap-2"><Check className="mt-0.5 size-3.5 shrink-0 text-primary" /><span>{w}</span></li>
                  ))}
                </ul>
                <details className="group mt-2.5">
                  <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground">
                    <ChevronDown className="size-3.5 transition-transform group-open:rotate-180" /> Same run from your shell
                  </summary>
                  <pre className="mt-2 overflow-x-auto bg-surface px-3 py-2.5 font-mono text-xs leading-relaxed">{recipe.cli}</pre>
                </details>
                <Field label="Name" htmlFor="cs-name" className="mt-3.5">
                  <Input id="cs-name" value={projectName} maxLength={80} disabled={busy}
                    onChange={(e) => { setNameTouched(true); setName(e.target.value); }} />
                </Field>
                {err && <p role="alert" className="mt-3 text-sm text-destructive">{err}</p>}
              </div>
              <div className="flex items-center justify-end gap-2 border-t border-border bg-subtle px-4 py-2.5">
                <Button onClick={start} disabled={busy || !projectName.trim()}>
                  {busy ? <><LoaderCircle className="animate-spin" /> {phase}</> : <>Create and start fine-tuning <ArrowRight /></>}
                </Button>
              </div>
            </div>
          )}
          <div ref={threadEnd} />
        </div>

        <form onSubmit={send} className="shrink-0 border-t border-border px-3 pt-2.5 pb-3">
          <div className="mb-2 flex flex-wrap gap-1.5">
            {chips.map((c) => (
              <button key={c} type="button" disabled={busy}
                onClick={() => {
                  const g = goals.find((x) => x.title === c);
                  if (g) choose(g.goal);
                  else { say({ from: "you", text: c }); setDraft(""); replyTo(c); }
                }}
                className="rounded-full border border-border bg-card px-2.5 py-1 text-xs text-muted-foreground transition-colors outline-none hover:border-primary/30 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40 disabled:opacity-50">
                {c}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <Input value={draft} onChange={(e) => setDraft(e.target.value)} disabled={busy}
              placeholder={goal ? "Ask Ctrl agent…" : "Say what your model should do…"} aria-label="Message Ctrl agent" />
            <Button type="submit" size="icon" aria-label="Send" disabled={!draft.trim() || busy}><CornerDownLeft /></Button>
          </div>
        </form>
      </section>

      {/* the loop */}
      <section aria-label="The loop" className="flex min-h-0 flex-col gap-4 @4xl:order-1 @4xl:col-span-7">
        <StartRail goal={goal} count={count} noun={l.noun} />
        <div className="min-h-0 flex-1 overflow-y-auto border border-border bg-card scrollbar-thin">
          <div className="border-b border-border px-5 py-3.5">
            <h2 className="text-sm font-semibold tracking-[-0.01em]">Data</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">The examples your model learns from, and the questions its evaluation asks.</p>
          </div>
          <div className="p-5">
            {goal ? (
              <ExamplesInput goal={goal} name={projectName} onChange={setExamples} />
            ) : (
              <EmptyState icon={Sparkles} title="Examples arrive here">
                Tell Ctrl agent what your model should do. Then add what it learns from: written questions and answers,
                a file, a dataset, or your agent's real conversations.
              </EmptyState>
            )}
          </div>
        </div>
      </section>
    </div>
  );

  // a chip that asks a question is answered like typed text
  function replyTo(text: string) {
    if (!goal) return;
    say({ from: "copilot", text: goal === "knowledge"
      ? `Every fact you want it to know, one question and answer each. ${minimum.knowledge} or more give a clear test of what it learned.`
      : `Real inputs with the outputs you want, ideally ${minimum[goal]} or more. A few hundred teach a task well.` });
    void text;
  }
}

// One of Ctrl agent's messages: the same author line as the cockpit's thread.
function Copilot({ children }: { children: ReactNode }) {
  return (
    <div>
      <div className="mb-1 flex items-center gap-2 text-[11px] text-muted-foreground">
        <MessagesSquare className="size-3" strokeWidth={1.75} aria-hidden />
        <span>Ctrl agent</span>
      </div>
      <div className="text-sm leading-relaxed">{children}</div>
    </div>
  );
}

function You({ children }: { children: ReactNode }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] border border-border bg-subtle px-3 py-2 text-sm break-words">{children}</p>
    </div>
  );
}

// The loop before it exists: Data is where the work is; the rest wait their turn.
function StartRail({ goal, count, noun }: { goal: ProjectGoal | null; count: number; noun: string }) {
  const stations = [
    { label: "Data", line: !goal ? "Waits for what it does" : count ? `${count} ${noun}` : "Add examples", active: true },
    { label: "Fine-tune", line: "After the data", active: false },
    { label: "Evaluate", line: "After the fine-tune", active: false },
    { label: "Deploy", line: "Once it's proven", active: false },
  ];
  return (
    <ol aria-label="The loop" className="relative grid grid-cols-2 gap-2 @2xl:grid-cols-4">
      <span aria-hidden className="absolute top-1/2 right-6 left-6 hidden h-px bg-border-strong @2xl:block" />
      {stations.map((s) => (
        <li key={s.label} aria-current={s.active ? "step" : undefined}
          className={cn("relative border px-3.5 py-3",
            s.active ? "border-primary/40 bg-primary/5" : "border-border bg-card")}>
          <p className={cn("flex items-center gap-1.5 text-sm font-medium", s.active ? "text-primary" : "text-muted-foreground")}>
            <span className={cn("size-1.5 rounded-full", s.active ? "bg-primary" : "bg-border-strong")} />
            {s.label}
          </p>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">{s.line}</p>
        </li>
      ))}
    </ol>
  );
}
