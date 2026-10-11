// The ShadowLM remote protocol, typed. Same endpoints the SDK speaks.
import { embedApiBase, embedded, embedToken, hostToast, renewPass } from "@/lib/embed";

export interface DatasetMeta {
  dataset_id: string;
  name: string;
  format: string;
  rows: number | null;
  created: number;
  source?: "upload" | "hf";
  curated?: boolean;  // bundled starter catalog → shown under Explore
  repo?: string;
  subset?: string;
  split?: string;
  eval_split?: string | null;
  preview?: Record<string, unknown>[];
}

export interface CatalogModel {
  id: string;
  params?: string;
  note?: string;
  gated?: boolean;
  dev?: boolean;
  cached?: boolean;
  custom?: boolean;  // user-added repo (removable)
}

export interface DownloadStatus {
  state: "downloading" | "ready" | "error";
  total?: number;
  downloaded?: number;
  pct?: number | null;
  error?: string | null;
}

export interface MethodInfo {
  name: string;
  description: string;
  default_lr: number;
  trainer: string;
  // lora | dora | more | more_plus | bottleneck | bitfit | prompt | ptuning | none
  adapter: string;
}

export interface JobSummary {
  job_id: string;
  base_model: string;
  name?: string;
  status: "pending" | "running" | "succeeded" | "failed" | "stopped";
  error: string | null;
  final_loss: number | null;
  steps: number;
  method: string | null;
}

export interface Checkpoint {
  step: number;
  path: string;
  final: boolean;
  label: string;
}

export interface JobDetail {
  status: JobSummary["status"];
  error: string | null;
  checkpoint: string | null;
  final_loss: number | null;
}

export interface StepMetric {
  step: number;
  loss: number;
  lr: number;
  tokens_per_s?: number | null;
}

export const apiKey = {
  get: () => localStorage.getItem("slm_api_key") || "",
  set: (v: string) => localStorage.setItem("slm_api_key", v),
  clear: () => localStorage.removeItem("slm_api_key"),
};

// apiUrl is where a call goes: this server, or embedded in a host console
// (lib/embed.ts), the host's proxy for it.
export const apiUrl = (path: string) => (embedded ? `${embedApiBase() ?? ""}${path}` : path);

// authHeaders is the bearer for a call: the host's pass when embedded, else
// this studio's own key or login token.
export function authHeaders(): Record<string, string> {
  const t = embedded ? embedToken() : apiKey.get();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

// apiFetch is fetch against the API with its credential. Embedded, a 401
// asks the host for a fresh pass and tries once more; no cookie is sent.
export async function apiFetch(path: string, opts: RequestInit = {}): Promise<Response> {
  const go = () => fetch(apiUrl(path), {
    ...opts,
    headers: { ...(opts.headers as Record<string, string> | undefined), ...authHeaders() },
    credentials: embedded ? "omit" : "same-origin",
  });
  let r = await go();
  if (r.status === 401 && embedded) {
    await renewPass();
    r = await go();
  } else if (r.status === 401) {
    // token missing/expired — drop it and bounce back to the login gate
    apiKey.clear();
    window.dispatchEvent(new Event("slm-unauthorized"));
  }
  return r;
}

// failure is why a call failed, as its answer says: the studio's own
// {error}, or the problem a host's proxy answers ({detail}, {title}, RFC
// 9457). Embedded, a change the host refuses (its roles, not the studio's)
// is also said in the host, so it is seen even where a page drops errors.
export async function failure(r: Response): Promise<Error> {
  const b = await r.json().catch(() => ({} as { error?: string; detail?: string; title?: string }));
  const message = b.error || b.detail || b.title || r.statusText || `HTTP ${r.status}`;
  if (embedded && r.status === 403) hostToast("error", message);
  return new Error(message);
}

export async function api<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const r = await apiFetch(path, {
    ...opts, headers: { "Content-Type": "application/json", ...(opts.headers as Record<string, string> | undefined) },
  });
  if (!r.ok) throw await failure(r);
  return r.json() as Promise<T>;
}

// ---- auth: login/password gate ---------------------------------------------
export interface AuthInfo { auth_required: boolean; mode: "password" | "apikey" | "none"; }
export const getAuthInfo = () =>
  fetch("/v1/auth").then((r) => r.json() as Promise<AuthInfo>);

export interface LoginResponse { token: string; user: string; expires: number; }

export async function login(username: string, password: string): Promise<void> {
  const r = await fetch("/v1/login", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({} as { error?: string }));
    throw new Error(d.error || "login failed");
  }
  const { token } = (await r.json()) as LoginResponse;
  apiKey.set(token);
}

export const logout = () => apiKey.clear();

// the server spreads capacity() into this — fleet depth, not just liveness
export interface Health {
  ok: boolean;
  backend: string;
  version: string;
  gpus: number;
  running: number;
  pending: number;
  workers: number;
}
export const getHealth = () => api<Health>("/v1/health");
export const getDatasets = () =>
  api<{ datasets: DatasetMeta[] }>("/v1/datasets");
export const getDataset = (id: string) => api<DatasetMeta>(`/v1/datasets/${id}`);
export const createDataset = (name: string, rows: unknown[]) =>
  api<DatasetMeta>("/v1/datasets", { method: "POST", body: JSON.stringify({ name, rows }) });

export interface HFInfo {
  configs: string[];
  subset: string | null;
  splits: string[];
}
export const hfInfo = (repo: string, subset?: string) =>
  api<HFInfo>("/v1/datasets/hf-info", {
    method: "POST", body: JSON.stringify({ repo, subset }) });

export interface HFPreview {
  format: string;
  columns: string[];
  total: number | null;
  preview: Record<string, unknown>[];
}
export const previewHF = (repo: string, subset: string, split: string) =>
  api<HFPreview>("/v1/datasets/preview", {
    method: "POST", body: JSON.stringify({ repo, subset, split, limit: 8 }) });
export const addHFDataset = (
  repo: string, subset: string, split: string, format: string, evalSplit = "",
) =>
  api<DatasetMeta>("/v1/datasets", {
    method: "POST",
    body: JSON.stringify({ source: "hf", repo, subset, split, format,
                           eval_split: evalSplit || null }) });
export const deleteDataset = (id: string) =>
  api<{ ok: boolean }>(`/v1/datasets/${id}`, { method: "DELETE" });

// ---- synthesis: make a dataset instead of bringing one ----------------------
export interface SynthStatus {
  synth_id: string;
  name: string;
  status: "running" | "succeeded" | "failed" | "stopped";
  kept: number;
  requested: number;
  tokens?: number;   // what the provider billed — zero for a local teacher
  // live phase counters — a round's generating/judging batches tick as each
  // job lands, so the bar moves instead of waiting for the whole round
  phase?: "starting" | "planning" | "generating" | "judging" | "kept";
  done?: number;
  total?: number;
  dataset_id?: string;
  error?: string;
  logs?: string[];
}
export interface SynthRequest {
  name: string;
  n: number;
  method?: string;
  min_score?: number;
  task?: string;
  document?: string;
  dataset_id?: string;
  include_seed?: boolean;  // the saved dataset also carries the dataset_id rows it was written from
  project_id?: string;     // shows on that project while it writes; the result becomes its data
  // "frontier": the frontier model saved in settings (its stored key)
  teacher: { kind: "openai" | "local" | "frontier"; model: string; base_url?: string; api_key?: string };
}
export const startSynth = (body: SynthRequest) =>
  api<{ synth_id: string }>("/v1/synth", { method: "POST", body: JSON.stringify(body) });
export const getSynthRun = (id: string) => api<SynthStatus>(`/v1/synth/${id}`);
export const cancelSynth = (id: string) =>
  api<{ ok: boolean }>(`/v1/synth/${id}/cancel`, { method: "POST" });
export const getModels = () =>
  api<{ catalog: CatalogModel[]; recent: string[]; server_backend: string }>("/v1/models");
export const getDownloads = () =>
  api<{ downloads: Record<string, DownloadStatus> }>("/v1/models/downloads");
export const downloadModel = (model: string) =>
  api<DownloadStatus>("/v1/models/download", { method: "POST", body: JSON.stringify({ model }) });
export const addCustomModel = (model: string) =>
  api<{ custom: CatalogModel[] }>("/v1/models/custom", { method: "POST", body: JSON.stringify({ model }) });
export const removeCustomModel = (model: string) =>
  api<{ custom: CatalogModel[] }>("/v1/models/custom", { method: "POST", body: JSON.stringify({ model, remove: true }) });
export interface FrontierInfo { base_url: string; model: string }
// ctrl: the model answering the cockpit's conversation (the server's
// ANTHROPIC_API_KEY), or null when Ctrl agent answers by its rules
export interface Settings { hf_token_set: boolean; frontier: FrontierInfo | null; ctrl?: { model: string } | null }

// ---- Ctrl agent: a typed message answered by Claude ---------------------------------
// It may propose one action; the cockpit shows it as a card, and nothing runs
// until the person approves it there.
export type CtrlActionKind = "finetune" | "evaluate" | "synthesize" | "deploy" | "playground" | "add-frontier";
export interface CtrlReply { reply: string; action: { kind: CtrlActionKind; args: Record<string, unknown> } | null }
export const askCtrl = (projectId: string, body: { message: string; state: unknown; thread: { from: string; text: string }[] }) =>
  api<CtrlReply>(`/v1/projects/${projectId}/ctrl`, { method: "POST", body: JSON.stringify(body) });
export const getSettings = () => api<Settings>("/v1/settings");
// The user's frontier model (any OpenAI-compatible API): an evaluation baseline,
// the judge, and the upstream an agent's captured traffic passes through. The
// key is stored on the server and never comes back. null removes it.
export const setFrontier = (frontier: { base_url: string; api_key: string; model: string } | null) =>
  api<Settings>("/v1/settings", { method: "POST", body: JSON.stringify({ frontier }) });
export const setHfToken = (hf_token: string) =>
  api<{ hf_token_set: boolean }>("/v1/settings", { method: "POST", body: JSON.stringify({ hf_token }) });
export const getVram = () =>
  api<{ used_mb: number | null; cached_models: number }>("/v1/vram");
export const clearVram = () =>
  api<{ unloaded: number; before_mb: number | null; after_mb: number | null }>(
    "/v1/vram/clear", { method: "POST" });
export const getMethods = () => api<{ methods: MethodInfo[] }>("/v1/methods");
export interface WorkerInfo {
  name: string; backend: string; device: string; gpus: number;
  gpu_name: string; vram_gb: number; ram_gb: number; cores: number;
  models: { id: string; size_gb: number }[];
  last_seen: number; online: boolean; queued: number;
}
export const getWorkers = () => api<{ workers: WorkerInfo[] }>("/v1/workers");
export interface MachineToken { name: string; created: number }
export const getTokens = () => api<{ tokens: MachineToken[] }>("/v1/tokens");
export const createToken = (name: string) =>
  api<{ name: string; token: string }>("/v1/tokens", {
    method: "POST", body: JSON.stringify({ name }) });
export const revokeToken = (name: string) =>
  api<{ ok: boolean }>(`/v1/tokens/${encodeURIComponent(name)}`, { method: "DELETE" });
export const getJobs = () => api<{ jobs: JobSummary[] }>("/v1/finetunes");
export const getJob = (id: string) => api<JobDetail>(`/v1/finetunes/${id}`);
export const getMetrics = (id: string) =>
  api<{ steps: StepMetric[]; evals: StepMetric[] }>(`/v1/finetunes/${id}/metrics`);
export const getLogs = (id: string) =>
  api<{ logs: string[] }>(`/v1/finetunes/${id}/logs`);
export const getCheckpoints = (id: string) =>
  api<{ checkpoints: Checkpoint[] }>(`/v1/finetunes/${id}/checkpoints`);
export const cancelJob = (id: string) =>
  api<{ ok: boolean }>(`/v1/finetunes/${id}/cancel`, { method: "POST" });
export interface DatasetPayload { rows: unknown[]; format?: string }

// what POST /v1/finetunes accepts — either `dataset` rows or a `dataset_id`.
// eval_dataset is rows, a holdout ("20%" or a 0–1 fraction), or null for none.
export interface FinetuneRequest {
  base_model: string;
  config: Record<string, unknown>;
  name?: string;
  dataset?: DatasetPayload | null;
  dataset_id?: string | null;
  eval_dataset?: DatasetPayload | string | number | null;
  worker?: string | null;
  load_in_4bit?: boolean;
  max_seq_length?: number;
}
export const submitFinetune = (body: FinetuneRequest) =>
  api<{ job_id: string }>("/v1/finetunes", { method: "POST", body: JSON.stringify(body) });
export const prewarm = (model: string, adapter: string | null, checkpoint: number | null = null) =>
  api<{ ready: boolean; error?: string }>("/v1/prewarm", {
    method: "POST", body: JSON.stringify({ model, adapter, checkpoint }) });
export const chat = (body: object) =>
  api<{ text: string }>("/v1/chat", { method: "POST", body: JSON.stringify(body) });

// ---- chat as a background task ------------------------------------------------
// The first answer from a cold model can take longer than the proxy in front of
// the studio holds a request, so a chat is started, then polled until it lands.
export interface ChatTask { task_id: string; status: "running" | "succeeded" | "failed"; text: string | null; error: string | null }
export const startChat = (body: object) =>
  api<{ task_id: string }>("/v1/tasks/chat", { method: "POST", body: JSON.stringify(body) });
export const getChatTask = (id: string) => api<ChatTask>(`/v1/tasks/${id}`);
export async function chatAsync(body: object, { every = 1200, timeoutMs = 15 * 60_000 } = {}): Promise<{ text: string }> {
  const { task_id } = await startChat(body);
  const end = Date.now() + timeoutMs;
  for (;;) {
    const t = await getChatTask(task_id);
    if (t.status === "succeeded") return { text: t.text ?? "" };
    if (t.status === "failed") throw new Error(t.error || "the model couldn't answer");
    if (Date.now() > end) throw new Error("no answer after 15 minutes");
    await new Promise((r) => setTimeout(r, every));
  }
}

// ---- projects: one fine-tune, as Business mode walks it -----------------------
// goal: "knowledge" (teach it facts) · "task" (teach it a job from examples) ·
// "takeover" (move a task off a frontier model, from captured agent traffic).
// The stage is read off the links: no dataset → Data; a run → Fine-tune; an
// eval → Evaluate. Deploy comes with the serving endpoint.
export type ProjectGoal = "knowledge" | "task" | "takeover";
export interface Project {
  project_id: string;
  name: string;
  goal: ProjectGoal;
  dataset_id: string | null;
  run_id: string | null;
  eval_id: string | null;
  synth_id?: string | null;  // examples being written for it (the synthesizer)
  run_dataset_id?: string | null;  // what the current version trained on
  // earlier versions, oldest first: each fine-tune this project replaced
  history: { run_id: string; eval_id: string | null; dataset_id: string | null; replaced: number }[];
  created: number;
  updated: number;
}
export const getProjects = () => api<{ projects: Project[] }>("/v1/projects");
export const getProject = (id: string) => api<Project>(`/v1/projects/${id}`);
export const createProject = (name: string, goal: ProjectGoal) =>
  api<Project>("/v1/projects", { method: "POST", body: JSON.stringify({ name, goal }) });
export const updateProject = (id: string, patch: Partial<Pick<Project, "name" | "dataset_id" | "run_id" | "eval_id" | "synth_id">>) =>
  api<Project>(`/v1/projects/${id}`, { method: "PATCH", body: JSON.stringify(patch) });
export const deleteProject = (id: string) =>
  api<{ ok: boolean }>(`/v1/projects/${id}`, { method: "DELETE" });

// ---- evaluations: task quality, as a background job ---------------------------
// Several targets (a base model, a fine-tune at a checkpoint) answer the same
// questions; each gets a score and per-question detail. "contains": the
// expected answer appears in the output · "exact": it is the output.
export type EvalMetric = "contains" | "exact" | "judge";  // judge: the frontier model scores
// kind "frontier" is the configured frontier model (the server fills in model)
export interface EvalTarget { label: string; model: string; adapter?: string | null; checkpoint?: number | null; kind?: "frontier" }
export interface EvalExample { input: string; output: string; expected: string; score: number }
export interface EvalTargetResult { score: number; n: number; examples?: EvalExample[] }
export interface Evaluation {
  eval_id: string;
  name: string;
  dataset_id: string;
  metric: EvalMetric;
  targets: EvalTarget[];
  sample: number | null;
  status: "pending" | "running" | "succeeded" | "failed";
  error: string | null;
  results: EvalTargetResult[];  // in target order; fills in as each finishes
  created: number;
  finished: number | null;
}
export const getEvals = () => api<{ evals: Evaluation[] }>("/v1/evals");
export const getEval = (id: string) => api<Evaluation>(`/v1/evals/${id}`);
export const startEval = (body: { dataset_id: string; targets: (EvalTarget | { kind: "frontier"; label?: string })[]; metric?: EvalMetric; sample?: number; name?: string }) =>
  api<Evaluation>("/v1/evals", { method: "POST", body: JSON.stringify(body) });
export const deleteEval = (id: string) =>
  api<{ ok: boolean }>(`/v1/evals/${id}`, { method: "DELETE" });

// ---- captures: an agent's traffic to its frontier model, recorded -------------
// Point the agent's OpenAI base URL at `${origin}/v1/capture/<capture_id>`; each
// call passes through to the frontier model and is recorded, then the episodes
// become a dataset. The id is the URL's only secret.
export interface CaptureInfo {
  capture_id: string;
  name: string;
  project_id: string | null;
  status: "open" | "closed";
  created: number;
  calls: number;
  episodes?: number;
  preview?: { question: string; answer: string; turns: number }[];
}
export const captureBaseUrl = (id: string) => `${window.location.origin}/v1/capture/${id}`;
export const getCaptures = () => api<{ captures: CaptureInfo[] }>("/v1/captures");
export const getCapture = (id: string) => api<CaptureInfo>(`/v1/captures/${id}`);
export const createCapture = (name: string, project_id?: string | null) =>
  api<CaptureInfo>("/v1/captures", { method: "POST", body: JSON.stringify({ name, project_id: project_id ?? null }) });
export const closeCapture = (id: string) =>
  api<{ ok: boolean }>(`/v1/captures/${id}/close`, { method: "POST" });
export const captureToDataset = (id: string, name: string) =>
  api<DatasetMeta>(`/v1/captures/${id}/dataset`, { method: "POST", body: JSON.stringify({ name }) });
export const deleteCapture = (id: string) =>
  api<{ ok: boolean }>(`/v1/captures/${id}`, { method: "DELETE" });

// OpenTelemetry GenAI traces (OTLP JSON, or a list of spans) → a chat dataset.
export const importTraces = (name: string, traces: unknown) =>
  api<DatasetMeta>("/v1/datasets/traces", { method: "POST", body: JSON.stringify({ name, traces }) });

// ---- deployments: a fine-tune behind an OpenAI-compatible endpoint ------------
// An agent uses it with base URL `${origin}/openai/v1`, the deployment's key and
// its name as the model. The key is returned once, at creation.
export interface Deployment {
  deployment_id: string;
  name: string;
  run_id: string;
  base_model: string;
  checkpoint: number | null;
  project_id: string | null;
  key_prefix: string;
  created: number;
  requests: number;
  last_used: number | null;
}
export const deploymentBaseUrl = () => `${window.location.origin}/openai/v1`;
export const getDeployments = () => api<{ deployments: Deployment[] }>("/v1/deployments");
export const createDeployment = (body: { name: string; run_id: string; checkpoint?: number | null; project_id?: string | null }) =>
  api<{ deployment: Deployment; key: string }>("/v1/deployments", { method: "POST", body: JSON.stringify(body) });
export const deleteDeployment = (id: string) =>
  api<{ ok: boolean }>(`/v1/deployments/${id}`, { method: "DELETE" });
