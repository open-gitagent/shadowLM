// The ShadowLM remote protocol, typed. Same endpoints the SDK speaks.

export interface DatasetMeta {
  dataset_id: string;
  name: string;
  format: string;
  rows: number | null;
  created: number;
  source?: "upload" | "hf";
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
};

export async function api<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const key = apiKey.get();
  if (key) headers["Authorization"] = `Bearer ${key}`;
  const r = await fetch(path, { ...opts, headers });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({} as { error?: string }));
    throw new Error(detail.error || r.statusText);
  }
  return r.json() as Promise<T>;
}

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
export const getMethods = () => api<{ methods: MethodInfo[] }>("/v1/methods");
export const getJobs = () => api<{ jobs: JobSummary[] }>("/v1/finetunes");
export const getJob = (id: string) => api<JobDetail>(`/v1/finetunes/${id}`);
export const getMetrics = (id: string) =>
  api<{ steps: StepMetric[]; evals: StepMetric[] }>(`/v1/finetunes/${id}/metrics`);
export const getLogs = (id: string) =>
  api<{ logs: string[] }>(`/v1/finetunes/${id}/logs`);
export const cancelJob = (id: string) =>
  api<{ ok: boolean }>(`/v1/finetunes/${id}/cancel`, { method: "POST" });
export const submitFinetune = (body: object) =>
  api<{ job_id: string }>("/v1/finetunes", { method: "POST", body: JSON.stringify(body) });
export const chat = (body: object) =>
  api<{ text: string }>("/v1/chat", { method: "POST", body: JSON.stringify(body) });
