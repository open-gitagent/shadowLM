// Business mode's fine-tune choices, made for the user and shown to them.
// A recipe is the base model, method and settings picked from what the project
// is for and the hardware this server has; `why` says each choice in plain
// words and `cli` is the same run from a shell. Nothing here is hidden: the run
// it starts opens in Research mode with every setting visible.
import {
  startEval, submitFinetune, updateProject,
  type Health, type Project, type ProjectGoal,
} from "@/api";

export interface Recipe {
  base_model: string;
  method: string;
  config: Record<string, unknown>;
  eval_dataset: string | null;  // a holdout like "10%", or null to train on every row
  why: string[];
  cli: string;
}

const clamp = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n));

// A CUDA box trains the small instruct model; without a GPU this is the
// Apple-Silicon dev loop, where the 4-bit mlx build trains in seconds.
function baseFor(health: Health | null): { model: string; why: string } {
  return health && health.gpus > 0
    ? { model: "Qwen/Qwen2.5-1.5B-Instruct", why: "Qwen2.5-1.5B-Instruct: small and fast, with plenty of room on this GPU" }
    : { model: "mlx-community/Qwen2.5-0.5B-Instruct-4bit", why: "Qwen2.5-0.5B (4-bit): the fast local model for a machine without a CUDA GPU" };
}

export function pickRecipe(goal: ProjectGoal, rows: number, health: Health | null): Recipe {
  const base = baseFor(health);
  if (goal === "knowledge") {
    const steps = clamp(rows * 20, 100, 400);
    return finish(base.model, "more", { method: "more", max_steps: steps }, null, [
      "MoRE: each fact becomes a retrieval expert, so answers come back exactly as written",
      base.why,
      rows === 1 ? "The example trains, and the evaluation asks its question: does it know it?"
        : `All ${rows} examples train, and the evaluation asks those same questions: does it know them?`,
      `${steps} steps`,
    ]);
  }
  const steps = clamp(rows * 3, 60, 600);
  const holdout = rows >= 50 ? "10%" : null;
  return finish(base.model, "lora", { method: "lora", max_steps: steps }, holdout, [
    "LoRA: learns the task's pattern from your examples without retraining the whole model",
    base.why,
    holdout
      ? "10% of your examples are held back, so the evaluation tests questions it never saw"
      : `${rows === 1 ? "The one example trains" : `All ${rows} examples train`}; with fewer than 50, holding some back would leave too little to learn from`,
    `${steps} steps`,
  ]);
}

function finish(base_model: string, method: string, config: Record<string, unknown>,
                eval_dataset: string | null, why: string[]): Recipe {
  const cli = [
    "shadowlm finetune <data.jsonl> \\",
    `  --model ${base_model} \\`,
    `  --method ${method} \\`,
    `  --max-steps ${config.max_steps}${eval_dataset ? " \\" : ""}`,
    ...(eval_dataset ? [`  --eval ${eval_dataset}`] : []),
  ].join("\n");
  return { base_model, method, config, eval_dataset, why, cli };
}

// The same recipe with another method, its reason said in the first line. The
// cockpit offers the methods that train on plain examples; the rest stay in
// Research mode's Train page.
export const RECIPE_METHODS: { id: string; why: string }[] = [
  { id: "lora", why: "LoRA: learns the task's pattern from your examples without retraining the whole model" },
  { id: "sdft", why: "SDFT: learns from your examples on its own samples, so it keeps more of its general skills" },
  { id: "more", why: "MoRE: each fact becomes a retrieval expert, so answers come back exactly as written" },
];
export function withMethod(r: Recipe, method: string): Recipe {
  if (method === r.method) return r;
  const why = RECIPE_METHODS.find((m) => m.id === method)?.why ?? method;
  return {
    ...r, method, config: { ...r.config, method },
    why: [why, ...r.why.filter((w) => !RECIPE_METHODS.some((m) => w === m.why) && !/^(LoRA|SDFT|MoRE):/.test(w))],
    cli: r.cli.replace(/--method \S+/, `--method ${method}`),
  };
}

// Start the project's fine-tune and link the run (and its data) to it.
export async function startProjectFinetune(project: Project, datasetId: string, recipe: Recipe): Promise<Project> {
  const { job_id } = await submitFinetune({
    base_model: recipe.base_model, name: project.name, config: recipe.config,
    dataset_id: datasetId, eval_dataset: recipe.eval_dataset,
    load_in_4bit: false, max_seq_length: 2048,
  });
  return updateProject(project.project_id, { dataset_id: datasetId, run_id: job_id });
}

// Evaluate the project's fine-tune against its base on the project's data, and
// against the frontier model too when one is set up (then the frontier model
// also judges the answers, so a correct paraphrase counts).
export async function startProjectEval(project: Project, baseModel: string, withFrontier = false): Promise<Project> {
  if (!project.dataset_id || !project.run_id) throw new Error("fine-tune the project before evaluating it");
  const ev = await startEval({
    name: project.name, dataset_id: project.dataset_id,
    metric: withFrontier ? "judge" : "contains",
    targets: [
      { label: "Your fine-tune", model: baseModel, adapter: project.run_id },
      { label: "Base model", model: baseModel },
      ...(withFrontier ? [{ kind: "frontier" as const, label: "Frontier model" }] : []),
    ],
  });
  return updateProject(project.project_id, { eval_id: ev.eval_id });
}

// What a project's goal is called where a user reads it.
export const goalLabel: Record<ProjectGoal, string> = {
  knowledge: "Knows your knowledge",
  task: "Does a task from examples",
  takeover: "Takes over a frontier-model task",
};
