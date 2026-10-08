// Asked once, on a first visit: which way into the studio. The answer sets the
// mode (lib/mode.ts); the side panel switches it any time after.
import { ArrowRight, FlaskConical, Target } from "lucide-react";

import { type Mode, setMode } from "@/lib/mode";

const choices: { mode: Mode; icon: typeof Target; title: string; body: string; then: string }[] = [
  {
    mode: "business", icon: Target,
    title: "Make a model for a job",
    body: "Bring your knowledge or examples, and get a fine-tuned model proven on your own questions. The studio picks the model and settings, and shows you what it picked.",
    then: "Projects, one per model you're making",
  },
  {
    mode: "research", icon: FlaskConical,
    title: "Train and research",
    body: "Choose base models and any of 13 methods, tune every setting, read loss curves and logs, compare checkpoints and runs, and reproduce any run from the CLI.",
    then: "Datasets, models, runs and evaluations",
  },
];

export function ModeChooser() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center py-10">
      <h1 className="text-xl leading-tight font-semibold tracking-[-0.015em]">What brings you here?</h1>
      <p className="mt-1 max-w-[64ch] text-[13px] text-muted-foreground">
        Both work on the same data and models, and you can switch any time from the side panel.
      </p>
      <div className="mt-7 grid gap-3 @2xl:grid-cols-2">
        {choices.map((c) => (
          <button key={c.mode} type="button" onClick={() => setMode(c.mode)}
            className="group flex flex-col items-start gap-3 border border-border bg-card p-5 text-left transition-colors hover:border-primary/40 hover:bg-primary/5 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none">
            <c.icon className="size-5 text-primary" strokeWidth={1.75} />
            <span className="text-base font-semibold tracking-[-0.01em]">{c.title}</span>
            <span className="text-sm text-muted-foreground">{c.body}</span>
            <span className="mt-auto flex items-center gap-1.5 pt-2 text-xs font-medium text-primary">
              {c.then} <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
