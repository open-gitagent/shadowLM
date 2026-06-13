// Pre-configured training setups. Each recipe pre-fills the Train wizard —
// the user supplies the dataset, the recipe wires model + method + steps.
export interface Recipe {
  id: string;
  name: string;
  tagline: string;
  method: string;
  model: string;
  steps: number;
  use: string;
  emoji: string;
}

export const RECIPES: Recipe[] = [
  {
    id: "support-distill", name: "Customer Support Distill",
    tagline: "Shadow the frontier model on support traffic",
    method: "qlora", model: "Qwen/Qwen2.5-1.5B-Instruct", steps: 200,
    use: "Capture your agent's traffic, train a small in-house model, switch when the evals hold.",
    emoji: "◆",
  },
  {
    id: "tone-dpo", name: "Tone Transfer",
    tagline: "Teach a model your brand voice with DPO",
    method: "dpo", model: "Qwen/Qwen2.5-1.5B-Instruct", steps: 150,
    use: "Preference pairs (chosen vs rejected) for style and tone alignment.",
    emoji: "◇",
  },
  {
    id: "fact-memorizer", name: "Fact Memorizer",
    tagline: "MoRE — facts with near-zero hallucination",
    method: "more", model: "Qwen/Qwen2.5-0.5B-Instruct", steps: 100,
    use: "Internal docs, product specs, runbooks — retrieval fused into attention.",
    emoji: "◈",
  },
  {
    id: "verifier-grpo", name: "Reward-Driven Reasoner",
    tagline: "GRPO from a programmable verifier",
    method: "grpo", model: "Qwen/Qwen2.5-1.5B-Instruct", steps: 300,
    use: "Reward correct final answers; the model learns the steps. (Local backend — reward fns don't serialize.)",
    emoji: "△",
  },
  {
    id: "bitfit-light", name: "Ultra-Light BitFit",
    tagline: "Adapt with ~0.1% of parameters",
    method: "bitfit", model: "Qwen/Qwen2.5-0.5B-Instruct", steps: 80,
    use: "Quick personalization with minimal memory — bias terms only.",
    emoji: "○",
  },
  {
    id: "domain-cpt", name: "Domain Pretrain",
    tagline: "Continued pretraining on raw text",
    method: "cpt", model: "Qwen/Qwen2.5-0.5B-Instruct", steps: 200,
    use: "A corpus of domain text, no chat template — vocabulary and style soak.",
    emoji: "▣",
  },
];
