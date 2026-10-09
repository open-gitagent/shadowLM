// The user's frontier model: any OpenAI-compatible API, by base URL, model name
// and key. It is the evaluation baseline ("does the fine-tune match it?"), the
// judge, and the upstream an agent's captured traffic passes through. The key
// is stored on the server and never shown again; this form only sets or clears.
import { LoaderCircle } from "lucide-react";
import { type FormEvent, useCallback, useEffect, useState } from "react";

import { type FrontierInfo, getSettings, setFrontier } from "@/api";
import { Field } from "@/components/common";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

// useFrontier is the configured frontier model (null when none), refreshed on
// demand after a change.
export function useFrontier() {
  const [frontier, setInfo] = useState<FrontierInfo | null>(null);
  const [loaded, setLoaded] = useState(false);
  const refresh = useCallback(() => {
    getSettings().then((s) => { setInfo(s.frontier); setLoaded(true); }).catch(() => setLoaded(true));
  }, []);
  useEffect(refresh, [refresh]);
  return { frontier, loaded, refresh };
}

export function FrontierForm({ current, onSaved, compact }: {
  current: FrontierInfo | null; onSaved: (f: FrontierInfo | null) => void; compact?: boolean;
}) {
  const [baseUrl, setBaseUrl] = useState(current?.base_url ?? "https://api.openai.com/v1");
  const [model, setModel] = useState(current?.model ?? "");
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState<"save" | "remove" | null>(null);
  const [err, setErr] = useState("");

  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy("save");
    setErr("");
    try {
      onSaved((await setFrontier({ base_url: baseUrl.trim(), api_key: key.trim(), model: model.trim() })).frontier);
      setKey("");
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function remove() {
    setBusy("remove");
    setErr("");
    try {
      onSaved((await setFrontier(null)).frontier);
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <form onSubmit={save} className="grid gap-3">
      <div className={compact ? "grid gap-3 @xl:grid-cols-2" : "grid gap-3"}>
        <Field label="API base URL" htmlFor="fm-url" hint="OpenAI, Azure OpenAI, or any OpenAI-compatible endpoint">
          <Input id="fm-url" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://api.openai.com/v1" />
        </Field>
        <Field label="Model" htmlFor="fm-model" hint="the model your agent uses today">
          <Input id="fm-model" value={model} onChange={(e) => setModel(e.target.value)} placeholder="gpt-4o-mini" />
        </Field>
      </div>
      <Field label="API key" htmlFor="fm-key"
             hint={current ? "A key is stored. Enter it again to change these settings." : "Stored on this server, never shown again."}>
        <Input id="fm-key" type="password" autoComplete="off" value={key} onChange={(e) => setKey(e.target.value)} placeholder="sk-…" />
      </Field>
      {err && <p className="text-sm text-destructive">{err}</p>}
      <div className="flex items-center justify-end gap-2">
        {current && (
          <Button type="button" variant="ghost" onClick={remove} disabled={busy !== null}>
            {busy === "remove" && <LoaderCircle className="animate-spin" />} Remove
          </Button>
        )}
        <Button type="submit" disabled={busy !== null || !baseUrl.trim() || !model.trim() || !key.trim()}>
          {busy === "save" && <LoaderCircle className="animate-spin" />} Save frontier model
        </Button>
      </div>
    </form>
  );
}

export function FrontierDialog({ open, onOpenChange, current, onSaved }: {
  open: boolean; onOpenChange: (o: boolean) => void; current: FrontierInfo | null; onSaved: (f: FrontierInfo | null) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Frontier model</DialogTitle>
          <DialogDescription>
            The model your fine-tunes are measured against. It also judges answers in evaluations, and it's
            what an agent's traffic passes through while you capture it.
          </DialogDescription>
        </DialogHeader>
        <FrontierForm current={current} onSaved={(f) => { onSaved(f); onOpenChange(false); }} />
      </DialogContent>
    </Dialog>
  );
}
