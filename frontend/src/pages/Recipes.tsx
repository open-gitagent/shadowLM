// Recipes — tested starting points. "Use this recipe" pre-fills the Train
// wizard; you bring the dataset.
import { ArrowRight } from "lucide-react";
import type { MethodInfo } from "../api";
import { RECIPES } from "../recipes";
import { PageHeader } from "../ui";

export default function Recipes({ methods }: { methods: MethodInfo[] }) {
  function use(model: string, method: string, steps: number) {
    sessionStorage.setItem("pick.model", model);
    sessionStorage.setItem("pick.method", method);
    sessionStorage.setItem("pick.steps", String(steps));
    window.location.hash = "#train";
  }

  return (
    <>
      <PageHeader
        eyebrow="Recipes"
        title="Pre-configured training setups"
        description="Start from a tested configuration. Each recipe wires a model, method, and step budget for a specific use case — you bring the dataset."
      />
      <div className="px-8 py-6 max-w-[1400px]">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {RECIPES.map((r) => {
            const m = methods.find((x) => x.name === r.method);
            return (
              <div key={r.id}
                   className="rounded-lg border border-border bg-card p-5 flex flex-col hover:border-primary/40 transition-colors">
                <div className="flex items-start gap-3 mb-3">
                  <div className="size-10 rounded-md bg-primary/10 border border-primary/30 grid place-items-center text-primary text-lg">
                    {r.emoji}
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="text-base font-semibold truncate">{r.name}</h3>
                    <p className="text-xs text-muted-foreground mt-0.5">{r.tagline}</p>
                  </div>
                </div>

                <p className="text-sm text-foreground/80 mb-4 leading-relaxed">{r.use}</p>

                <div className="space-y-1.5 text-xs font-mono mb-5">
                  <Row label="model" value={r.model.split("/").pop()!} />
                  <Row label="method" value={r.method} />
                  <Row label="steps" value={String(r.steps)} />
                  <Row label="lr" value={m ? String(m.default_lr) : "method default"} />
                </div>

                <button onClick={() => use(r.model, r.method, r.steps)}
                  className="mt-auto inline-flex items-center justify-between rounded-md border border-border bg-background/40 px-3 py-2 text-xs hover:bg-primary hover:text-primary-foreground hover:border-primary transition-colors group">
                  <span>Use this recipe</span>
                  <ArrowRight className="size-3.5 group-hover:translate-x-0.5 transition-transform" />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-muted-foreground">
      <span>{label}</span>
      <span className="truncate text-foreground/80">{value}</span>
    </div>
  );
}
