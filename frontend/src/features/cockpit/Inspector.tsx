// The inspector: the selected station of the loop, in full. One station at a
// time; the rail above chooses which, and the copilot's proposals land here.
import type { CockpitActions } from "./Cockpit";
import type { Loop, StationId } from "./model";
import { DataStation } from "./stations/DataStation";
import { DeployStation } from "./stations/DeployStation";
import { EvaluateStation } from "./stations/EvaluateStation";
import { FinetuneStation } from "./stations/FinetuneStation";

export function Inspector({ loop, station, actions }: { loop: Loop; station: StationId; actions: CockpitActions }) {
  if (!loop.project) return null;
  return (
    <div key={station} className="motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200">
      {station === "data" ? <DataStation loop={loop} />
        : station === "finetune" ? <FinetuneStation loop={loop} actions={actions} />
          : station === "evaluate" ? <EvaluateStation loop={loop} actions={actions} />
            : <DeployStation loop={loop} actions={actions} />}
    </div>
  );
}
