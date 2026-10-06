// Embedded mode: the studio shown inside a host console, such as
// opencontroller, in a frame. It is on only when the page is framed by an
// origin the server lists: opencontroller on studio-dev, localhost on any
// port, and whatever SHADOWLM_FRAME_ANCESTORS adds (the same list sets the
// response's frame-ancestors; shadowlm/serve.py). Opened on its own, nothing
// here runs, and the studio is as it always is.
//
// Embedded, the studio holds no session of its own. The host gives it a
// short-lived pass and the API address to call, both over postMessage
// (protocol "oc-embed/1", the same bridge Rekori's console speaks), and
// renews the pass while the page is open; the host's proxy checks every
// call. The host's menu replaces the studio's sidebar.
//
//   host → frame   init {pass, role, path, api_base, api_url} · pass {pass} · theme {theme} · navigate {path}
//                  · action {id}, taken from the menu the studio sent
//   frame → host   ready · pass (one now, please) · route {path} · title {title} · toast {level, message}
//                  · menu {menu}, the studio's menu for the host to show in place of its own

import { useSyncExternalStore } from "react";

const protocol = "oc-embed/1";

interface Pass {
  token: string;
  expires_at: string;
}

export type EmbedRole = "viewer" | "operator" | "admin";
export type EmbedTheme = "light" | "dark";

// parents are the origins the server lets frame the studio, as
// frame-ancestors writes them: "https://host", or "http://host:*" for any
// port.
function readParents(): string[] {
  if (typeof document === "undefined") return [];
  const meta = document.querySelector('meta[name="shadowlm-embed-parents"]');
  return (meta?.getAttribute("content") ?? "").split(" ").filter((o) => /^https?:\/\/[^/\s:]+(:(\d+|\*))?$/.test(o));
}

const parents = readParents();

// allowed reports whether origin is one of parents.
const allowed = (origin: string) =>
  parents.some((p) => p === origin || (p.endsWith(":*") && origin.startsWith(p.slice(0, -1)) && /^\d+$/.test(origin.slice(p.length - 1))));

// embedded reports whether the studio runs inside a host console.
export const embedded = typeof window !== "undefined" && window.self !== window.top && parents.length > 0;

const state: { parent?: string; pass?: Pass; apiBase?: string; role: EmbedRole; theme: EmbedTheme } = { role: "viewer", theme: "light" };
const themeListeners = new Set<() => void>();
let passWaiters: (() => void)[] = [];

// A page of the studio, as the host sees it: "/" or "/<section>[/<id>]",
// the studio's own #<section>/<id> as a path. The only places the host may
// move it to.
const sections = ["playground", "datasets", "models", "train", "runs", "machines"];
export const isPage = (p: unknown): p is string =>
  typeof p === "string" && p.length <= 512 &&
  (p === "/" || new RegExp(`^/(${sections.join("|")})(/[A-Za-z0-9._-]+)?$`).test(p));

// The studio's hash for a page path, and back.
export const pageToHash = (p: string) => p.slice(1);
export const hashToPage = (h: string) => "/" + h.replace(/^#/, "");

// An API path on this origin: where a host's proxy may be served beside
// the studio. Never another origin, so the pass goes nowhere else.
const apiPath = (p: unknown): p is string => typeof p === "string" && /^\/[A-Za-z0-9._~/-]*$/.test(p) && !/\/\/|\/\.\.?(\/|$)/.test(p);

// apiURL is the host's proxy on the host's own origin, which framed the
// studio, so the two can be deployed apart: an address on that origin only,
// so the pass goes nowhere else.
function apiURL(u: unknown, host: string): string | undefined {
  if (typeof u !== "string") return undefined;
  try {
    const url = new URL(u);
    return url.origin === host && !url.search && !url.hash && apiPath(url.pathname) ? url.origin + url.pathname.replace(/\/$/, "") : undefined;
  } catch {
    return undefined;
  }
}

const isPass = (p: unknown): p is Pass =>
  !!p && typeof p === "object" && typeof (p as Pass).token === "string" && typeof (p as Pass).expires_at === "string";

function post(m: Record<string, unknown>) {
  if (state.parent) window.parent.postMessage({ protocol, ...m }, state.parent);
}

function setPass(p: Pass) {
  state.pass = p;
  const waiters = passWaiters;
  passWaiters = [];
  waiters.forEach((w) => w());
}

// startEmbed listens to the host and says the studio is ready. It resolves
// with the page to open once the host has sent its first pass; navigate
// moves the studio when the host asks.
export function startEmbed(navigate: (path: string) => void): Promise<string> {
  return new Promise((resolve) => {
    window.addEventListener("message", (e: MessageEvent) => {
      // Only the frame's own parent, from a listed origin, speaking our protocol.
      if (e.source !== window.parent || !allowed(e.origin)) return;
      if (state.parent && e.origin !== state.parent) return;
      const m = e.data as Record<string, unknown> | null;
      if (!m || typeof m !== "object" || m.protocol !== protocol) return;
      switch (m.type) {
        case "init": {
          // The host's own origin when it says, else the path beside the studio.
          const base = apiURL(m.api_url, e.origin) ?? (apiPath(m.api_base) ? m.api_base.replace(/\/$/, "") : undefined);
          if (state.pass || !isPass(m.pass) || !base) return;
          state.parent = e.origin;
          state.apiBase = base;
          state.role = m.role === "admin" || m.role === "operator" ? m.role : "viewer";
          setPass(m.pass);
          resolve(isPage(m.path) ? m.path : "/");
          break;
        }
        case "pass":
          if (isPass(m.pass)) setPass(m.pass);
          break;
        case "theme":
          if (m.theme === "light" || m.theme === "dark") {
            state.theme = m.theme;
            themeListeners.forEach((l) => l());
          }
          break;
        case "navigate":
          if (isPage(m.path)) navigate(m.path);
          break;
        case "action":
          if (typeof m.id === "string") actions.forEach((a) => a(m.id as string));
          break;
      }
    });
    // The parent's origin is not known yet, and may be any port of a listed
    // host: "ready" carries nothing, and only a listed origin can frame the
    // page at all (frame-ancestors). Everything after goes to the origin
    // that answered.
    window.parent.postMessage({ protocol, type: "ready" }, "*");
  });
}

// embedToken is the pass to send as the bearer.
export const embedToken = () => state.pass?.token;

// embedApiBase is where API calls go when embedded.
export const embedApiBase = () => state.apiBase;

// embedRole is the person's role in the host, for showing controls; the
// host's proxy decides what is allowed.
export const embedRole = () => state.role;

const ranks: EmbedRole[] = ["viewer", "operator", "admin"];

// allows reports whether to show a control whose change needs the role
// need: on its own, the studio's sign-in decides, so always; embedded, by
// the person's role in the host. Through opencontroller, operators add
// datasets and models, train and use the playground, and administrators
// mint machine tokens. Showing only: the host's proxy decides.
export const allows = (need: EmbedRole) => !embedded || ranks.indexOf(state.role) >= ranks.indexOf(need);

// needs says why a control is not there, for a page that would be empty
// without it.
export const needs = (need: EmbedRole) => `This needs the ${need} role in the console the studio is shown in.`;

// renewPass asks the host for a pass now, after a refusal, and waits for
// it, or five seconds.
export function renewPass(): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, 5000);
    passWaiters.push(() => {
      clearTimeout(t);
      resolve();
    });
    post({ type: "pass" });
  });
}

// reportRoute tells the host where the studio went.
export const reportRoute = (path: string) => isPage(path) && post({ type: "route", path });

// reportTitle tells the host what the page shows, as its subtitle.
export const reportTitle = (title: string) => post({ type: "title", title: title.slice(0, 120) });

// A menu item, as the host shows it: a page of the studio, an action the
// studio takes when the host says it was chosen, or a link elsewhere (https
// only). Icons are lucide's, by name.
export interface MenuItem {
  label: string;
  icon: string;
  hint?: string;
  path?: string;
  action?: string;
  href?: string;
}

export interface Menu {
  groups: { title?: string; items: MenuItem[] }[];
  foot: MenuItem[];
}

// reportMenu gives the host the studio's menu, to show in place of its own.
export const reportMenu = (menu: Menu) => post({ type: "menu", menu });

const actions = new Set<(id: string) => void>();

// onHostAction runs a when the host says an action of the studio's menu was
// chosen; it returns how to stop.
export function onHostAction(a: (id: string) => void): () => void {
  actions.add(a);
  return () => {
    actions.delete(a);
  };
}

// toast asks the host to show a notification.
export const hostToast = (level: "success" | "error" | "info", message: string) =>
  post({ type: "toast", level, message: message.slice(0, 300) });

// useEmbedTheme is the host's theme.
export function useEmbedTheme(): EmbedTheme {
  return useSyncExternalStore(
    (l) => {
      themeListeners.add(l);
      return () => themeListeners.delete(l);
    },
    () => state.theme,
  );
}
