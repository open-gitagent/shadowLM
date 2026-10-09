// The studio's router: one table of routes over the URL hash, in the format
// the embed bridge already speaks ("#runs/<id>" ↔ host path "/runs/<id>",
// lib/embed.ts). Pages ask useRoute() where they are and navigate() to move;
// nothing else reads window.location.
import { useSyncExternalStore } from "react";

export type RouteName =
  | "home" | "projects" | "project" | "new-project" | "cockpit"
  | "playground" | "datasets" | "models" | "train" | "runs" | "evaluate"
  | "deployments" | "machines";

export interface Route { name: RouteName; id?: string; hash: string }

// section → route, with the id (the part after the slash) where one applies
function parse(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  const [section, id] = h.split("/");
  const known: Record<string, RouteName> = {
    "": "home", playground: "playground", datasets: "datasets", models: "models",
    train: "train", runs: "runs", evaluate: "evaluate", deployments: "deployments",
    machines: "machines", cockpit: "cockpit",
  };
  if (section === "projects") {
    if (id === "new") return { name: "new-project", hash: h };
    return id ? { name: "project", id, hash: h } : { name: "projects", hash: h };
  }
  return { name: known[section] ?? "home", id: id || undefined, hash: h };
}

const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());
if (typeof window !== "undefined") window.addEventListener("hashchange", notify);

let snapshot = { raw: "", route: parse("") };
function current(): Route {
  const raw = typeof window === "undefined" ? "" : window.location.hash;
  if (raw !== snapshot.raw) snapshot = { raw, route: parse(raw) };
  return snapshot.route;
}

export function useRoute(): Route {
  return useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, current);
}

// to() builds the hash for a route, e.g. to("project", id) → "#projects/<id>".
export function to(name: RouteName, id?: string): string {
  const path: Record<RouteName, string> = {
    home: "", projects: "projects", project: `projects/${id ?? ""}`, "new-project": "projects/new",
    cockpit: `cockpit${id ? `/${id}` : ""}`, playground: "playground", datasets: "datasets",
    models: "models", train: "train", runs: `runs${id ? `/${id}` : ""}`,
    evaluate: `evaluate${id ? `/${id}` : ""}`, deployments: "deployments", machines: "machines",
  };
  return `#${path[name]}`;
}

export function navigate(name: RouteName, id?: string): void {
  window.location.hash = to(name, id);
}
