// Data: the examples this model learns from, readable as questions and answers.
import { Database } from "lucide-react";

import { EmptyState } from "@/components/common";
import { Skeleton } from "@/components/ui/skeleton";
import { useMode } from "@/lib/mode";

import type { Loop } from "../model";
import { rowPair, StationHeader } from "./shared";

export function DataStation({ loop }: { loop: Loop }) {
  const mode = useMode();
  const p = loop.project!;
  const d = loop.dataset;

  if (!p.dataset_id) {
    return (
      <>
        <StationHeader title="Data" line="No examples yet." />
        <div className="p-5">
          <EmptyState icon={Database} title="It learns from your examples">
            A model gets its examples when it's made: written as questions and answers, uploaded as JSONL, picked from a dataset, or captured from your agent.{" "}
            <a className="text-primary underline-offset-4 hover:underline" href={mode === "research" ? "#datasets" : "#projects/new"}>
              {mode === "research" ? "Go to Datasets" : "Make a new model"}
            </a>{" "}to bring them in.
          </EmptyState>
        </div>
      </>
    );
  }

  if (!d) {
    return (
      <>
        <StationHeader title="Data" line={<Skeleton className="h-4 w-56" />} />
        <div className="space-y-2 p-5">
          {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-10 w-full" />)}
        </div>
      </>
    );
  }

  const rows = (d.preview ?? []).slice(0, 8);
  const source = d.source === "hf" ? `Hugging Face · ${d.repo ?? ""}` : "Uploaded to this server";
  return (
    <>
      <StationHeader title="Data"
        line={<>{d.name} · {d.rows ?? "?"} {d.rows === 1 ? "example" : "examples"} · {d.format} format</>}
        research="#datasets" />
      <div className="space-y-3 p-5">
        <p className="text-xs text-muted-foreground">
          {source}{mode === "research" && <> · <span className="font-mono">{d.dataset_id}</span></>}
          {rows.length > 0 && (d.rows ?? 0) > rows.length && <> · showing {rows.length} of {d.rows}</>}
        </p>
        {rows.length > 0 ? (
          <ul className="divide-y divide-border border border-border">
            {rows.map((r, i) => {
              const { q, a } = rowPair(r);
              return (
                <li key={i} className="grid gap-1 px-4 py-3 text-sm @2xl:grid-cols-2 @2xl:gap-5">
                  <span className="font-medium">{q}</span>
                  <span className="text-muted-foreground">{a || "—"}</span>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground">This dataset has no preview rows; open it in Research to read it.</p>
        )}
      </div>
    </>
  );
}
