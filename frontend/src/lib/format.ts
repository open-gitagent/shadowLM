// How the studio names things a person reads: a training method by its
// registry label ("more" → "MoRE"), a model by its repository name with the
// organisation kept apart, so the part being approved is never the part cut.
const METHOD_LABELS: Record<string, string> = {
  lora: "LoRA", qlora: "QLoRA", dora: "DoRA", full: "Full", cpt: "CPT", dpo: "DPO", grpo: "GRPO",
  more: "MoRE", more_plus: "MoRE+", bitfit: "BitFit", prompt: "Prompt tuning", ptuning: "P-tuning",
  adapter: "Adapter", sdft: "SDFT", sdpo: "SDPO",
};

// "1 example", "12 examples"
export const examples = (n: number) => `${n.toLocaleString()} ${n === 1 ? "example" : "examples"}`;
export const methodLabel = (m?: string | null) => (m ? METHOD_LABELS[m] ?? m : "");

// "mlx-community/Qwen2.5-0.5B-Instruct-4bit" → { org: "mlx-community", name: "Qwen2.5-0.5B-Instruct-4bit" }
export function modelParts(id: string): { org: string | null; name: string } {
  const i = id.lastIndexOf("/");
  return i < 0 ? { org: null, name: id } : { org: id.slice(0, i), name: id.slice(i + 1) };
}

// A long model id, breakable only at its own seams ("-", ".", "/"), never mid-word.
export function breakable(id: string): string {
  return id.replace(/([-./])/g, "$1\u200b");
}
