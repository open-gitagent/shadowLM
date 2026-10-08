// The copilot: what the studio says about a project, and what it proposes next.
// The transcript is derived from the loop's state, one message per milestone,
// so it is always true and survives a reload; the last message carries the
// pending decision as an action card. What the person types is understood by
// plain rules (no model call): intents like "evaluate", "train again with 300
// steps", "deploy", "how's it going". Every proposal shows what it will run.
import { verdict } from "@/components/scorecard";
import type { Recipe } from "@/lib/recipe";
import { methodLabel } from "@/lib/format";
import { goalLabel } from "@/lib/recipe";

import { type Loop, nextRecipe, firstRecipe, type StationId } from "./model";

export type Proposal =
  | { kind: "finetune"; recipe: Recipe; version: number }
  | { kind: "evaluate"; withFrontier: boolean }
  | { kind: "deploy" }
  | { kind: "add-frontier" }
  | { kind: "playground" };

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

// narrate: the project so far, oldest first, ending on what to do next.
export function narrate(loop: Loop): Message[] {
  const p = loop.project;
  if (!p) return [];
  const out: Message[] = [];
  out.push({ id: "goal", from: "copilot", text: `${p.name}: a model that ${goalLabel[p.goal].toLowerCase()}.`, at: p.created });

  if (!loop.dataset) {
    out.push({ id: "need-data", from: "copilot", station: "data",
               text: "First it needs examples. Add them on the Data station: write a few, upload a file, use a dataset, or connect your agent." });
    return out;
  }
  out.push({ id: "data", from: "copilot", station: "data",
             text: `It has ${fmt(loop.dataset.rows ?? 0)} examples from ${loop.dataset.name}.` });

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
  return { reply: "I can fine-tune a new version (\"train again with 300 steps\"), evaluate it, compare versions, deploy it, or tell you where things stand." };
}

// The quick replies offered under the composer for this moment.
export function suggestions(loop: Loop): string[] {
  const job = loop.job;
  if (!loop.dataset) return ["What examples does it need?"];
  if (!job) return ["Start with 200 steps", "What will it do?"];
  if (job.status === "pending" || job.status === "running") return ["How's it going?"];
  if (job.status !== "succeeded") return ["Train again", "Train again with more steps"];
  if (!loop.evaluation) return ["Evaluate it", "Try it in the Playground"];
  const s = [loop.deployment ? "How's it going?" : "Deploy it", "Train another version", "Compare versions"];
  return s;
}
