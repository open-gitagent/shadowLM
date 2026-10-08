// The fine-tuning cockpit: Ctrl agent's conversation on the right drives a live
// map of the whole loop on the left. Approving a proposal lights its station;
// hovering a message highlights the station it speaks about; selecting a
// station opens it in the inspector and scrolls the conversation to its latest
// message. One project per cockpit; a new project starts in CockpitStart.
import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";

import { ErrorState } from "@/components/common";
import { Skeleton } from "@/components/ui/skeleton";

import { Conversation } from "./Conversation";
import { Inspector } from "./Inspector";
import { LoopRail } from "./LoopRail";
import { type Loop, type StationId, useActions, useLoop } from "./model";

// ---- the link between the two panes ------------------------------------------
export interface CockpitLink {
  selected: StationId;
  select: (s: StationId, from?: "rail" | "thread") => void;
  hovered: StationId | null;
  hover: (s: StationId | null) => void;
  // bumps when a station is chosen on the rail, so the thread scrolls to it
  scrollTo: { station: StationId; tick: number } | null;
  // the two panes are stacked (narrow): the page is the only scroller then
  stacked: boolean;
}

const Link = createContext<CockpitLink | null>(null);
export function useCockpitLink(): CockpitLink {
  const v = useContext(Link);
  if (!v) throw new Error("useCockpitLink outside the cockpit");
  return v;
}

export type CockpitActions = ReturnType<typeof useActions>;

export default function Cockpit({ id }: { id: string }) {
  const loop = useLoop(id);
  const actions = useActions(loop);
  const [selected, setSelected] = useState<StationId | null>(null);
  const [hovered, setHovered] = useState<StationId | null>(null);
  const [scrollTo, setScrollTo] = useState<CockpitLink["scrollTo"]>(null);
  const stacked = useStacked();

  // follow the work: until the person picks a station, show where attention belongs
  const shown = selected ?? loop.next;
  useEffect(() => { setSelected(null); }, [id]);

  const link = useMemo<CockpitLink>(() => ({
    selected: shown,
    select: (s, from) => {
      setSelected(s);
      if (from === "rail") setScrollTo((p) => ({ station: s, tick: (p?.tick ?? 0) + 1 }));
    },
    hovered, hover: setHovered, scrollTo, stacked: stacked.value,
  }), [shown, hovered, scrollTo, stacked.value]);

  if (loop.missing) return <ErrorState title="No such project" error="It may have been deleted. Pick another from Projects." />;
  if (loop.loading || !loop.project) return <CockpitSkeleton />;

  return (
    <Link.Provider value={link}>
      {/* side by side, each pane scrolls inside the viewport; stacked (narrow),
          the panes flow at their natural height and the page is the only scroller */}
      {/* grid-cols-1 is minmax(0, 1fr): a scrolling row inside never widens the page */}
      <div ref={stacked.ref} className="grid min-h-0 flex-1 !shrink grid-cols-1 gap-4 @4xl:grid-cols-12">
        {/* Ctrl agent sits on the right beside the loop; stacked, it comes first,
            since the decision it holds is what a phone should open on */}
        <section aria-label="Ctrl agent" className={stacked.value
          ? "flex min-w-0 flex-col border border-border bg-card"
          : "order-2 col-span-5 flex min-h-0 min-w-0 flex-col overflow-hidden border border-border bg-card"}>
          <Conversation loop={loop} actions={actions} />
        </section>
        <section aria-label="The loop" className={stacked.value ? "flex min-w-0 flex-col gap-4" : "order-1 col-span-7 flex min-h-0 min-w-0 flex-col gap-4"}>
          <LoopRail loop={loop} />
          <div className={stacked.value ? "border border-border bg-card" : "min-h-0 flex-1 overflow-y-auto border border-border bg-card scrollbar-thin"}>
            <Inspector loop={loop} station={shown} actions={actions} />
          </div>
        </section>
      </div>
    </Link.Provider>
  );
}

// Whether the cockpit is too narrow for its two panes side by side: the same
// threshold as its @4xl container query (56rem), read from its own width.
function useStacked() {
  const ref = useRef<HTMLDivElement>(null);
  const [value, setValue] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setValue(e.contentRect.width < 896));
    ro.observe(el);
    return () => ro.disconnect();
  });
  return { ref, value };
}

function CockpitSkeleton() {
  return (
    <div className="grid flex-1 gap-4 @4xl:grid-cols-12">
      <div className="space-y-3 border border-border bg-card p-5 @4xl:col-span-5">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
      <div className="space-y-4 @4xl:col-span-7">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-80 w-full" />
      </div>
    </div>
  );
}

export type { Loop };
