// The ShadowLM remote protocol, typed. Same endpoints the SDK speaks.

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
  adapter: string;  // lora | dora | more | bottleneck | bitfit | prompt | ptuning | none
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

export async function api<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const key = apiKey.get();
  if (key) headers["Authorization"] = `Bearer ${key}`;
  const r = await fetch(path, { ...opts, headers });
  if (r.status === 401) {
    // token missing/expired — drop it and bounce back to the login gate
    apiKey.clear();
    window.dispatchEvent(new Event("slm-unauthorized"));
  }
  if (!r.ok) {
    const detail = await r.json().catch(() => ({} as { error?: string }));
    throw new Error(detail.error || r.statusText);
  }
  return r.json() as Promise<T>;
}

// ---- auth: login/password gate ---------------------------------------------
export interface AuthInfo { auth_required: boolean; mode: "password" | "apikey" | "none"; }
export const getAuthInfo = () =>
  fetch("/v1/auth").then((r) => r.json() as Promise<AuthInfo>);

export async function login(username: string, password: string): Promise<void> {
  const r = await fetch("/v1/login", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({} as { error?: string }));
    throw new Error(d.error || "login failed");
  }
  const { token } = (await r.json()) as { token: string };
  apiKey.set(token);
}

export const logout = () => apiKey.clear();

export const getHealth = () =>
  api<{ ok: boolean; backend: string; version: string }>("/v1/health");
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
export const getSettings = () => api<{ hf_token_set: boolean }>("/v1/settings");
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
export const submitFinetune = (body: object) =>
  api<{ job_id: string }>("/v1/finetunes", { method: "POST", body: JSON.stringify(body) });
export const prewarm = (model: string, adapter: string | null, checkpoint: number | null = null) =>
  api<{ ready: boolean; error?: string }>("/v1/prewarm", {
    method: "POST", body: JSON.stringify({ model, adapter, checkpoint }) });
export const chat = (body: object) =>
  api<{ text: string }>("/v1/chat", { method: "POST", body: JSON.stringify(body) });
