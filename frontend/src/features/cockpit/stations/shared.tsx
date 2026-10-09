// Pieces every station in the inspector shares: its header (title, state line,
// actions), the "Open in Research" link, copy-to-clipboard, a code block, and
// reading a dataset row as a question and its answer.
import { Check, Copy, FlaskConical } from "lucide-react";
import { type ReactNode, useState } from "react";

import { Button } from "@/components/ui/button";
import { openIn } from "@/lib/mode";
import { cn } from "@/lib/utils";

export function StationHeader({ title, line, actions, research }: {
  title: string; line?: ReactNode; actions?: ReactNode; research?: string;
}) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-3 border-b border-border px-5 pt-4 pb-3.5">
      <div className="min-w-0">
        <h2 className="text-base font-semibold tracking-[-0.01em]">{title}</h2>
        {line && <div className="mt-0.5 text-sm text-muted-foreground">{line}</div>}
        {research && <div className="mt-1.5"><ResearchLink hash={research} /></div>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

// A quiet text link, so it sits with the title's words rather than as a
// ragged extra button beside the station's real action.
export function ResearchLink({ hash }: { hash: string }) {
  return (
    <button type="button" onClick={() => openIn("research", hash)}
      className="inline-flex items-center gap-1 rounded-sm text-xs font-medium text-primary underline-offset-4 outline-none hover:underline focus-visible:ring-2 focus-visible:ring-ring/40">
      <FlaskConical className="size-3.5" /> Open in Research
    </button>
  );
}

export function useCopy() {
  const [copied, setCopied] = useState("");
  async function copy(text: string, what: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(what);
      setTimeout(() => setCopied((c) => (c === what ? "" : c)), 1600);
    } catch {
      // clipboard blocked: the text stays selectable
    }
  }
  return { copied, copy };
}

export function CodeBlock({ label, code, copied, onCopy, className }: {
  label: string; code: string; copied: boolean; onCopy: () => void; className?: string;
}) {
  return (
    <div className={cn("border border-border", className)}>
      <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2">
        <span className="text-xs text-muted-foreground">{label}</span>
        <Button variant="ghost" size="xs" onClick={onCopy}>{copied ? <Check /> : <Copy />} {copied ? "Copied" : "Copy"}</Button>
      </div>
      <pre className="overflow-x-auto bg-surface/40 p-4 font-mono text-xs leading-relaxed">{code}</pre>
    </div>
  );
}

// A row's question and answer, whatever its format.
export function rowPair(row: Record<string, unknown>): { q: string; a: string } {
  const msgs = row.messages as { role: string; content: string }[] | undefined;
  if (Array.isArray(msgs)) {
    const q = [...msgs].reverse().find((m) => m.role === "user")?.content ?? "";
    const a = [...msgs].reverse().find((m) => m.role === "assistant")?.content ?? "";
    return { q, a };
  }
  const pick = (...keys: string[]) => keys.map((k) => row[k]).find((v) => typeof v === "string") as string | undefined;
  return {
    q: pick("instruction", "prompt", "question", "input", "text") ?? JSON.stringify(row).slice(0, 160),
    a: pick("output", "response", "answer", "completion", "chosen") ?? "",
  };
}

export const ago = (t: number | null | undefined) => {
  if (!t) return "never";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago`
    : s < 86400 ? `${Math.floor(s / 3600)} h ago` : `${Math.floor(s / 86400)} d ago`;
};

// The API model name: the project's name in lowercase letters, digits and dashes.
export const slug = (name: string) =>
  name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "my-model";
