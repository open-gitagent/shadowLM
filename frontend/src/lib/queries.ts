// The data layer: one QueryClient and a hook per server resource, so pages
// share a cache, poll only while something is moving, and refetch after a
// change instead of hand-rolling intervals. Long jobs (fine-tunes,
// evaluations) poll while running and stop when they settle.
import { QueryClient, useQuery } from "@tanstack/react-query";

import {
  getCheckpoints, getDataset, getDatasets, getDeployments, getEval, getHealth, getJob, getJobs,
  getLogs, getMetrics, getProject, getProjects, getSettings,
} from "@/api";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 2_000, retry: 1, refetchOnWindowFocus: true } },
});

const moving = (s?: string) => s === "pending" || s === "running";

export const useHealthQ = () => useQuery({ queryKey: ["health"], queryFn: getHealth, refetchInterval: 15_000 });
export const useSettingsQ = () => useQuery({ queryKey: ["settings"], queryFn: getSettings });
export const useProjectsQ = () => useQuery({ queryKey: ["projects"], queryFn: () => getProjects().then((r) => r.projects), refetchInterval: 5_000 });
export const useProjectQ = (id?: string) =>
  useQuery({ queryKey: ["project", id], queryFn: () => getProject(id!), enabled: !!id });
export const useDatasetsQ = () => useQuery({ queryKey: ["datasets"], queryFn: () => getDatasets().then((r) => r.datasets) });
export const useDatasetQ = (id?: string | null) =>
  useQuery({ queryKey: ["dataset", id], queryFn: () => getDataset(id!), enabled: !!id });
export const useJobsQ = () =>
  useQuery({ queryKey: ["jobs"], queryFn: () => getJobs().then((r) => r.jobs),
             refetchInterval: (q) => (q.state.data?.some((j) => moving(j.status)) ? 3_000 : 15_000) });
export const useJobQ = (id?: string | null) =>
  useQuery({ queryKey: ["job", id], queryFn: () => getJob(id!), enabled: !!id,
             refetchInterval: (q) => (moving(q.state.data?.status) ? 3_000 : false) });
export const useMetricsQ = (id?: string | null, live = false) =>
  useQuery({ queryKey: ["metrics", id], queryFn: () => getMetrics(id!), enabled: !!id,
             refetchInterval: live ? 3_000 : false });
export const useLogsQ = (id?: string | null, live = false) =>
  useQuery({ queryKey: ["logs", id], queryFn: () => getLogs(id!), enabled: !!id,
             refetchInterval: live ? 3_000 : false });
export const useCheckpointsQ = (id?: string | null) =>
  useQuery({ queryKey: ["checkpoints", id], queryFn: () => getCheckpoints(id!).then((r) => r.checkpoints), enabled: !!id });
export const useEvalQ = (id?: string | null) =>
  useQuery({ queryKey: ["eval", id], queryFn: () => getEval(id!), enabled: !!id,
             refetchInterval: (q) => (moving(q.state.data?.status) ? 2_000 : false) });
export const useDeploymentsQ = () =>
  useQuery({ queryKey: ["deployments"], queryFn: () => getDeployments().then((r) => r.deployments), refetchInterval: 15_000 });
