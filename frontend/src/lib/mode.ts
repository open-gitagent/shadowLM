// The studio's two modes over the same datasets, runs and evaluations:
// Business walks a project from data to a proven fine-tune; Research opens every
// object and setting. The choice is asked once, remembered in this browser, and
// switched from the side panel. Embedded in a host console, it starts in
// Business unless the person already chose.
import { useSyncExternalStore } from "react";

import { embedded } from "@/lib/embed";

export type Mode = "business" | "research";

const key = "of-mode";
const listeners = new Set<() => void>();

function read(): Mode | null {
  try {
    const v = localStorage.getItem(key);
    if (v === "business" || v === "research") return v;
  } catch {
    // storage blocked: fall through to the default below
  }
  return embedded ? "business" : null;
}

let current: Mode | null = read();

export function setMode(m: Mode): void {
  current = m;
  try {
    localStorage.setItem(key, m);
  } catch {
    // private window: the mode holds for this visit only
  }
  listeners.forEach((l) => l());
}

// useMode is the current mode, or null before the person has chosen one.
export function useMode(): Mode | null {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => current,
  );
}

// openIn switches mode and goes to a page, e.g. "Open in Research" on a run.
export function openIn(m: Mode, hash: string): void {
  setMode(m);
  window.location.hash = hash;
}
