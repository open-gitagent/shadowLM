// The copilot: what the studio says about a project, and what it proposes next.
// The transcript is derived from the loop's state, one message per milestone,
// so it is always true and survives a reload; the last message carries the
// pending decision as an action card. What the person types is understood by
// plain rules (no model call): intents like "evaluate", "train again with 300
// steps", "deploy", "how's it going". Every proposal shows what it will run.
import { verdict } from "@/components/scorecard";
import type { Recipe } from "@/lib/recipe";
import { examples, methodLabel } from "@/lib/format";
import { goalLabel, withMethod } from "@/lib/recipe";

import { type Loop, moreExamples, nextRecipe, firstRecipe, type StationId, synthPhase } from "./model";

export type Proposal =
  | { kind: "finetune"; recipe: Recipe; version: number }
  | { kind: "evaluate"; withFrontier: boolean }
  | { kind: "deploy" }
  | { kind: "add-frontier" }
  | { kind: "playground" }
  | { kind: "synthesize"; n: number };

export interface Message {
  id: string;
  from: "copilot" | "you";
  station?: StationId;
  text: string;
  detail?: string[];  // short supporting lines (the plan's reasons, the verdict)
  proposal?: Proposal;
  at?: number;
}

const fmt = (n: number) => n.toLocaleString();
// an error's gist for the thread: no exception name, up to its first break
const firstClause = (e: string) => e.replace(/^\w+Error:\s*/, "").split(/ — |\. |\n/)[0];

// narrate: the project so far, oldest first, ending on what to do next.
export function narrate(loop: Loop): Message[] {
  const p = loop.project;
  if (!p) return [];
  const out: Message[] = [];
  out.push({ id: "goal", from: "copilot", text: `${p.name}: a model that ${goalLabel[p.goal].toLowerCase()}.`, at: p.created });

  const s = loop.synth;
  const writingLine = () => s
    ? `${synthPhase[s.phase ?? "starting"] ?? "Writing"}: ${s.kept} of ${s.requested} examples kept so far${s.tokens ? ` · ${fmt(s.tokens)} tokens` : ""}.`
    : "";
  if (!loop.dataset && loop.writing) {
    out.push({ id: "writing", from: "copilot", station: "data",
               text: `Writing its examples from your description. ${writingLine()} I'll plan the fine-tune when they're in.` });
    return out;
  }
  if (!loop.dataset && s && s.status !== "succeeded") {
    out.push({ id: "write-failed", from: "copilot", station: "data",
               text: `Writing the examples ${s.status === "stopped" ? "was stopped" : "failed"}${s.error ? `: ${firstClause(s.error)}` : ""}. The Data station has the details. Check the frontier model in the side panel, or make the model again with a fuller description or examples of your own.` });
    return out;
  }
  if (!loop.dataset) {
    out.push({ id: "need-data", from: "copilot", station: "data",
               text: "First it needs examples. Add them on the Data station: write a few, upload a file, use a dataset, or connect your agent." });
    return out;
  }
  const rows = loop.dataset.rows ?? 0;
  out.push({ id: "data", from: "copilot", station: "data",
             text: `It has ${examples(rows)} from ${loop.dataset.name}.` });
  // writing more didn't work: say so once, then carry on with what it has
  if (!loop.writing && s && s.status === "failed" && s.dataset_id !== p.dataset_id) {
    out.push({ id: `write-failed-${s.synth_id}`, from: "copilot", station: "data",
               text: `Writing more examples failed${s.error ? `: ${firstClause(s.error)}` : ""}. It still has its ${examples(rows)}; the Data station has the details.` });
  }
  if (loop.writing) {
    out.push({ id: "writing-more", from: "copilot", station: "data",
               text: `Writing more examples like yours with ${loop.frontier?.model ?? "the frontier model"}. ${writingLine()} I'll propose the next version when they're in.` });
    return out;
  }

  // earlier versions, each with how it scored
  for (const v of loop.versions.filter((x) => !x.current)) {
    const ev = v.evaluation;
    const r = ev?.results[0];
    out.push({ id: `v${v.n}`, from: "copilot", station: "evaluate",
               text: r ? `Version ${v.n} answered ${Math.round(r.score * r.n)} of ${r.n} correctly${ev && verdict(ev) ? ` (${verdict(ev)!.title.toLowerCase()})` : ""}.`
                       : `Version ${v.n} was replaced before it was evaluated.` });
  }

  const n = loop.versions.length || 1;
  const job = loop.job;
  if (!p.run_id || !job) {
    if (rows < 20 && loop.frontier) {
      out.push({ id: "small-start", from: "copilot", station: "data",
                 text: `${examples(rows)} is a small start. It can train on them as they are, or say “write more examples” and ${loop.frontier.model} will write more in their style.` });
    }
    const r = firstRecipe(loop);
    if (r) out.push({ id: "plan", from: "copilot", station: "finetune",
                      text: "Here's the plan for the first version. Start it when you're ready, or edit any step.",
                      detail: r.why, proposal: { kind: "finetune", recipe: r, version: 1 } });
    return out;
  }

  if (job.status === "pending" || job.status === "running") {
    out.push({ id: `run${n}`, from: "copilot", station: "finetune",
               text: `Fine-tuning version ${n}: ${methodLabel(job.method)} on ${job.base_model.split("/").pop()}. ${loop.steps.length ? `${loop.steps.length} steps done, loss ${loop.steps.at(-1)!.loss.toFixed(3)}.` : "Loading the base model."}` });
    return out;
  }
  if (job.status !== "succeeded") {
    const r = nextRecipe(loop) ?? firstRecipe(loop);
    out.push({ id: `fail${n}`, from: "copilot", station: "finetune",
               text: `Version ${n} ${job.status === "stopped" ? "was stopped" : "failed"}${job.error ? `: ${job.error.split("\n")[0].slice(0, 160)}` : ""}.`,
               ...(r ? { detail: ["Try again with the same plan, or edit it first."], proposal: { kind: "finetune" as const, recipe: r, version: n + 1 } } : {}) });
    return out;
  }
  out.push({ id: `done${n}`, from: "copilot", station: "finetune",
             text: `Version ${n} finished: ${fmt(job.steps)} steps${job.final_loss != null ? `, final loss ${job.final_loss.toFixed(3)}` : ""}.` });

  const ev = loop.evaluation;
  if (!p.eval_id || !ev) {
    const withFrontier = !!loop.frontier;
    out.push({ id: `ask-eval${n}`, from: "copilot", station: "evaluate",
               text: withFrontier
                 ? `Next, prove it: I'll ask version ${n}, the base model and ${loop.frontier!.model} your questions, and ${loop.frontier!.model} judges the answers.`
                 : `Next, prove it: I'll ask version ${n} and the base model your questions and check which answers match yours.`,
               proposal: { kind: "evaluate", withFrontier } });
    return out;
  }
  if (ev.status === "pending" || ev.status === "running") {
    out.push({ id: `eval${n}`, from: "copilot", station: "evaluate", text: "Asking your questions now. The first answers are slow while the models load." });
    return out;
  }
  if (ev.status === "failed") {
    out.push({ id: `evalfail${n}`, from: "copilot", station: "evaluate", text: `The evaluation stopped: ${ev.error ?? "unknown error"}`,
               proposal: { kind: "evaluate", withFrontier: !!loop.frontier } });
    return out;
  }
  const v = verdict(ev);
  out.push({ id: `verdict${n}`, from: "copilot", station: "evaluate",
             text: v ? `${v.title}. ${v.detail}` : "The evaluation finished." });

  if (loop.deployment) {
    out.push({ id: "live", from: "copilot", station: "deploy",
               text: `It's live as ${loop.deployment.name}: ${fmt(loop.deployment.requests)} requests so far. Your agent only changes its base URL, key and model name.` });
    return out;
  }
  // the frontier hint comes before the decision, so the decision stays the
  // live card at the bottom of the thread
  if (!loop.frontier) {
    out.push({ id: "frontier-hint", from: "copilot", station: "evaluate",
               text: "To measure it against the model your agent uses today, add a frontier model from the side panel; it will also judge answers, so a correct paraphrase counts." });
  }
  if (v?.tone !== "good" && loop.dataChanged) {
    const r = nextRecipe(loop);
    if (r) out.push({ id: `more-data${n}`, from: "copilot", station: "finetune",
                      text: `The examples grew to ${fmt(rows)}. I'd train version ${n + 1} on all of them:`, detail: r.why,
                      proposal: { kind: "finetune", recipe: r, version: n + 1 } });
    return out;
  }
  // thin data that hasn't been added to yet: more examples beat more steps
  if (v?.tone !== "good" && rows < 50 && loop.frontier) {
    const more = moreExamples(rows);
    out.push({ id: `write${n}`, from: "copilot", station: "data",
               text: `Version ${n} isn't there yet, and ${examples(rows)} is little to learn from. I'd have ${loop.frontier.model} write ${more} more like yours, then train version ${n + 1} on all of them.`,
               detail: [`${loop.frontier.model} judges every example it writes; weak ones are dropped`,
                        `Your ${examples(rows)} stay in, unchanged`,
                        "Or say “train again” to retrain on these as they are"],
               proposal: { kind: "synthesize", n: more } });
    return out;
  }
  if (v?.tone === "good") {
    out.push({ id: `ship${n}`, from: "copilot", station: "deploy",
               text: "It's ready. Deploy it behind an OpenAI-compatible endpoint, or try it in the Playground first.",
               proposal: { kind: "deploy" } });
  } else {
    const r = nextRecipe(loop);
    if (r) out.push({ id: `next${n}`, from: "copilot", station: "finetune",
                      text: `I'd train version ${n + 1} like this:`, detail: r.why,
                      proposal: { kind: "finetune", recipe: r, version: n + 1 } });
  }
  return out;
}

// ---- understanding what the person types ------------------------------------
export interface Understood { reply: string; station?: StationId; proposal?: Proposal }

const has = (t: string, ...words: string[]) => words.some((w) => t.includes(w));

export function interpret(text: string, loop: Loop): Understood {
  const t = text.toLowerCase();
  const n = loop.versions.length || 1;
  const job = loop.job;

  if (has(t, "status", "how is", "how's", "where", "progress", "what now", "next")) {
    const s = loop.stations[loop.next];
    return { station: loop.next, reply: `Right now: ${s.label.toLowerCase()} — ${s.line.toLowerCase()}.` };
  }
  if (has(t, "deploy", "ship", "endpoint", "serve", "go live", "api")) {
    if (job?.status !== "succeeded") return { station: "finetune", reply: "It has to finish fine-tuning before it can be deployed." };
    return { station: "deploy", reply: "Name the deployment on the Deploy station; that name becomes the model name your agent asks for.", proposal: { kind: "deploy" } };
  }
  if (has(t, "evaluat", "test", "score", "prove", "check", "how good", "accuracy")) {
    if (job?.status !== "succeeded") return { station: "finetune", reply: "I'll evaluate it once version " + n + " finishes training." };
    return { station: "evaluate", reply: "Here's the evaluation I'd run:", proposal: { kind: "evaluate", withFrontier: !!loop.frontier && !has(t, "base only", "without frontier") } };
  }
  if (has(t, "compare", "versions", "better", "worse", "history")) {
    const lines = loop.versions.map((v) => {
      const r = v.evaluation?.results[0];
      return `Version ${v.n}: ${r ? `${Math.round(r.score * r.n)} of ${r.n}` : "not evaluated"}${v.current ? " (current)" : ""}`;
    });
    return { station: "evaluate", reply: lines.length ? lines.join(" · ") : "There's only one version so far." };
  }
  if (has(t, "more examples", "more data", "write", "generate", "synthes", "amplif", "augment")) {
    if (!loop.frontier) return { station: "data", reply: "I write examples with your frontier model. Add one, then ask again.", proposal: { kind: "add-frontier" } };
    if (!loop.dataset) return { station: "data", reply: "It needs a few examples first, so the new ones can follow their style." };
    if (loop.writing) return { station: "data", reply: "I'm already writing more; they'll land on the Data station." };
    const m = t.match(/(\d{1,4})\s*(?:more\s*)?(?:examples?|rows?)/);
    const more = m ? Math.max(1, Math.min(1000, Number(m[1]))) : moreExamples(loop.dataset.rows ?? 0);
    return { station: "data", reply: `${loop.frontier.model} would write ${more} more like yours:`, proposal: { kind: "synthesize", n: more } };
  }
  // a method by name or by what it's for: SDFT keeps the base model's general skills
  if (has(t, "sdft", "self-distill", "general skill", "forget", "keep what it knows") || /\blora\b/.test(t)) {
    const base = nextRecipe(loop) ?? firstRecipe(loop);
    if (!base) return { station: "data", reply: "It needs examples first: add them on the Data station." };
    const method = /\blora\b/.test(t) && !has(t, "sdft") ? "lora" : "sdft";
    return { station: "finetune", reply: `Version ${job ? n + 1 : 1} with ${methodLabel(method)}:`,
             proposal: { kind: "finetune", recipe: withMethod(base, method), version: job ? n + 1 : 1 } };
  }
  if (has(t, "train", "again", "retrain", "fine-tune", "finetune", "more steps", "longer", "bigger", "larger", "steps")) {
    const base = nextRecipe(loop) ?? firstRecipe(loop);
    if (!base) return { station: "data", reply: "It needs examples first: add them on the Data station." };
    const m = t.match(/(\d{2,5})\s*steps?/) ?? t.match(/steps?\s*(?:to|=|of)?\s*(\d{2,5})/);
    const recipe: Recipe = m ? { ...base, config: { ...base.config, max_steps: Number(m[1]) },
                                 why: [`${m[1]} steps, as you asked`, ...base.why.filter((w) => !/steps/.test(w))],
                                 cli: base.cli.replace(/--max-steps \d+/, `--max-steps ${m[1]}`) } : base;
    return { station: "finetune", reply: `Version ${job ? n + 1 : 1} would run like this:`, proposal: { kind: "finetune", recipe, version: job ? n + 1 : 1 } };
  }
  if (has(t, "example", "data", "dataset", "add", "upload", "capture", "agent")) {
    return { station: "data", reply: "The Data station shows the examples it learns from. Adding more makes a new version worth training." };
  }
  if (has(t, "try", "chat", "playground", "talk")) {
    return { reply: "Open it in the Playground to talk to it side by side with its base model.", proposal: { kind: "playground" } };
  }
  if (has(t, "frontier", "openai", "gpt", "judge", "key")) {
    return { station: "evaluate", reply: "Add your frontier model and I'll measure against it and use it as the judge.", proposal: { kind: "add-frontier" } };
  }
  return { reply: "I can fine-tune a new version (\"train again with 300 steps\", \"use SDFT\"), write more examples, evaluate it, compare versions, deploy it, or tell you where things stand." };
}

// The quick replies offered under the composer for this moment.
export function suggestions(loop: Loop): string[] {
  const job = loop.job;
  if (loop.writing) return ["How's it going?"];
  if (!loop.dataset) return ["What examples does it need?"];
  const canWrite = !!loop.frontier;
  if (!job) return ["Start with 200 steps", ...(canWrite && (loop.dataset.rows ?? 0) < 50 ? ["Write more examples"] : []), "Use SDFT"];
  if (job.status === "pending" || job.status === "running") return ["How's it going?"];
  if (job.status !== "succeeded") return ["Train again", "Train again with more steps"];
  if (!loop.evaluation) return ["Evaluate it", "Try it in the Playground"];
  const ready = verdict(loop.evaluation)?.tone === "good";
  if (!ready && !loop.deployment) return [canWrite ? "Write more examples" : "Train again with more steps", "Use SDFT", "Compare versions"];
  return [loop.deployment ? "How's it going?" : "Deploy it", "Train another version", "Compare versions"];
}
