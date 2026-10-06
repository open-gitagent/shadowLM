// Shared page pieces, in the opencontroller console's language: a plain
// masthead, hairline stat strips, quiet empty/error states, status as a
// tinted outline badge.
import { CircleAlert, CircleCheck, CircleDot, CircleX, LoaderCircle, PauseCircle } from "lucide-react";
import type { ReactNode } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import type { JobSummary } from "@/api";
import { cn } from "@/lib/utils";

// PageHeader is the view's masthead: its title, one quiet line of what the
// view is for, and at most its one main action. Space sets it off from the
// content, not a rule.
export function PageHeader({ title, description, actions }: {
  title: string; description?: ReactNode; actions?: ReactNode;
}) {
  return (
    <header className="mb-7 flex shrink-0 flex-wrap items-end justify-between gap-4 pt-1">
      <div>
        <h1 className="text-xl leading-tight font-semibold tracking-[-0.015em]">{title}</h1>
        {description && <p className="mt-1 max-w-[64ch] text-[13px] text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}

// SectionHeader heads one part of a page.
export function SectionHeader({ title, description, actions }: {
  title: string; description?: ReactNode; actions?: ReactNode;
}) {
  return (
    <header className="mb-4 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h2 className="text-base font-semibold tracking-[-0.01em]">{title}</h2>
        {description && <p className="mt-1 max-w-[70ch] text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}

export function ErrorState({ error, title = "Could not load" }: { error: unknown; title?: string }) {
  return (
    <Alert variant="destructive">
      <CircleAlert />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{error instanceof Error ? error.message : String(error)}</AlertDescription>
    </Alert>
  );
}

export function EmptyState({ icon: Icon, title, children }: {
  icon: typeof CircleCheck; title: string; children?: ReactNode;
}) {
  return (
    <div className="flex items-start gap-3 border border-border bg-surface/40 px-4 py-3.5">
      <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />
      <div className="min-w-0">
        <p className="text-sm font-medium">{title}</p>
        {children && <div className="mt-0.5 max-w-prose text-sm text-muted-foreground">{children}</div>}
      </div>
    </div>
  );
}

// StatStrip is a row of headline numbers read across as one band, split by
// hairlines, rather than a row of separate cards.
export function StatStrip({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("grid grid-cols-2 gap-px overflow-hidden border border-border bg-border @2xl:grid-cols-4", className)}>
      {children}
    </div>
  );
}

// Stat is one number in a StatStrip, a link where it leads somewhere; sub
// is a line beneath on what the number is made of.
export function Stat({ label, value, sub, loading, href }: {
  label: string; value?: ReactNode; sub?: ReactNode; loading?: boolean; href?: string;
}) {
  const body = (
    <>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1.5 text-2xl font-semibold tracking-[-0.01em] tabular-nums">
        {loading ? <Skeleton className="h-7 w-16" /> : (value ?? "—")}
      </div>
      {sub && !loading && <div className="mt-0.5 truncate text-xs text-muted-foreground">{sub}</div>}
    </>
  );
  return href ? (
    <a href={href} className="block bg-card px-4 py-3.5 transition-colors hover:bg-surface">{body}</a>
  ) : (
    <div className="bg-card px-4 py-3.5">{body}</div>
  );
}

// Mono is for identifiers: run ids, model ids, paths.
export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("font-mono text-xs break-all", className)}>{children}</span>;
}

// Field is a labelled control with an optional hint beneath.
export function Field({ label, hint, htmlFor, children, className }: {
  label: string; hint?: ReactNode; htmlFor?: string; children: ReactNode; className?: string;
}) {
  return (
    <div className={cn("grid gap-1.5", className)}>
      <Label htmlFor={htmlFor} className="text-xs text-muted-foreground">{label}</Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

// A run's status, as the console shows a scan's: a tinted outline badge
// with its icon.
const runStatus: Record<string, { label: string; icon: typeof CircleCheck; className: string }> = {
  succeeded: { label: "Succeeded", icon: CircleCheck, className: "border-good/30 bg-good/10 text-good" },
  running: { label: "Running", icon: LoaderCircle, className: "border-primary/25 bg-primary/10 text-primary [&>svg]:animate-spin" },
  pending: { label: "Pending", icon: CircleDot, className: "text-muted-foreground" },
  failed: { label: "Failed", icon: CircleX, className: "border-destructive/25 bg-destructive/10 text-destructive" },
  stopped: { label: "Stopped", icon: PauseCircle, className: "text-muted-foreground" },
};

export function StatusBadge({ status }: { status: JobSummary["status"] }) {
  const s = runStatus[status] ?? runStatus.stopped;
  return (
    <Badge variant="outline" className={cn("gap-1 font-normal", s.className)}>
      <s.icon className="size-3" />
      {s.label}
    </Badge>
  );
}

// Dots is the "thinking…" indicator while a model answers.
export function Dots() {
  return (
    <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
      <span className="size-1.5 animate-pulse rounded-full bg-current" />
      <span className="size-1.5 animate-pulse rounded-full bg-current [animation-delay:0.15s]" />
      <span className="size-1.5 animate-pulse rounded-full bg-current [animation-delay:0.3s]" />
      <span className="ml-1">thinking…</span>
    </span>
  );
}

// ConfirmDelete asks before deleting what (open while what is set), saying
// what follows from it.
export function ConfirmDelete({ what, action = "Delete", onConfirm, onClose, children }: {
  what?: string; action?: string; onConfirm: () => void; onClose: () => void; children: ReactNode;
}) {
  return (
    <AlertDialog open={what !== undefined} onOpenChange={(o) => !o && onClose()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{action} {what}?</AlertDialogTitle>
          <AlertDialogDescription>{children}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={onConfirm}>{action}</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
