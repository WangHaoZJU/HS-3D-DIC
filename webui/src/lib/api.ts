/** Typed client for the StereoDIC server (stereodic_server, /api). */
import axios, { AxiosError } from 'axios'

export const http = axios.create({ baseURL: '/api', timeout: 60_000 })

export function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg)
    return error.message
  }
  return error instanceof Error ? error.message : String(error)
}

// ------------------------------------------------------------------ types
export type StageKey =
  | 'calibrate'
  | 'stereo'
  | 'temporal'
  | 'reconstruct'
  | 'rbm'
  | 'analyze'
  | 'uncertainty'

export type StageState =
  'missing' | 'done' | 'stale' | 'partial' | 'unknown' | 'running'

export type StageRecord = {
  state: StageState
  detail: string
  done_utc: string | null
  fraction: number | null
}

export type ExperimentStatus = {
  stages: Record<StageKey, StageRecord>
  busy: boolean
  running_job: string | null
}

export type ExperimentSummary = {
  name: string
  description: string
  modified_utc: string
  valid: boolean
  error?: string
  frame_rate_hz?: number
}

export type ExperimentConfig = Record<string, unknown>

export type ExperimentDetail = {
  name: string
  path: string
  yaml: string
  config: ExperimentConfig
}

export type ValidationError = { loc: string[]; msg: string; type: string }

export type ValidationResult = {
  valid: boolean
  errors: ValidationError[]
  config: ExperimentConfig | null
  invalidates: StageKey[]
}

export type JobState =
  'queued' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled'

export type Job = {
  id: string
  experiment: string
  stage: string
  params: Record<string, unknown>
  force: boolean
  state: JobState
  phase: string
  done: number
  total: number
  fraction: number | null
  message: string
  error: string
  created_utc: string
  started_utc: string | null
  finished_utc: string | null
}

export type FsEntry = {
  name: string
  kind: 'dir' | 'file'
  size: number | null
  modified_utc: string
  ext: string
}

export type FsListing = {
  root: string
  root_path: string
  path: string
  abs_path: string
  parent: string | null
  entries: FsEntry[]
  n_images: number
  truncated: boolean
}

export type SystemInfo = {
  environment: Record<string, string>
  git: { commit: string | null; dirty: boolean | null }
  backends: {
    available?: string[]
    auto?: string
    numba?: Record<string, unknown>
    cuda?: Record<string, unknown>
    error?: string
  }
  gpus: {
    index: number
    name: string
    memory_used_mb: number
    memory_total_mb: number
    utilization_pct: number
    temperature_c: number
  }[]
  roots: Record<string, string>
  disks: Record<
    string,
    { path: string; total_gb: number; free_gb: number; used_gb: number } | null
  >
  static_ui: boolean
}

export type JsonSchema = {
  type?: string | string[]
  title?: string
  description?: string
  default?: unknown
  enum?: unknown[]
  const?: unknown
  format?: string
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  items?: JsonSchema
  properties?: Record<string, JsonSchema>
  required?: string[]
  anyOf?: JsonSchema[]
  $ref?: string
  $defs?: Record<string, JsonSchema>
}

// ------------------------------------------------------------------ calls
const get = <T>(url: string, params?: object) =>
  http.get<T>(url, { params }).then((r) => r.data)

export const api = {
  system: (refresh = false) => get<SystemInfo>('/system', { refresh }),
  fs: (root: string, path: string) => get<FsListing>('/fs', { root, path }),
  schema: () => get<JsonSchema>('/schema/experiment'),

  experiments: () => get<ExperimentSummary[]>('/experiments'),
  experiment: (name: string) =>
    get<ExperimentDetail>(`/experiments/${encodeURIComponent(name)}`),
  status: (name: string) =>
    get<ExperimentStatus>(`/experiments/${encodeURIComponent(name)}/status`),
  createExperiment: (body: {
    name: string
    copy_from?: string | null
    description?: string
  }) => http.post<ExperimentDetail>('/experiments', body).then((r) => r.data),
  deleteExperiment: (name: string) =>
    http
      .delete<{ moved_to: string }>(`/experiments/${encodeURIComponent(name)}`)
      .then((r) => r.data),
  validate: (name: string, body: { config?: object; yaml?: string }) =>
    http
      .post<ValidationResult>(
        `/experiments/${encodeURIComponent(name)}/validate`,
        body
      )
      .then((r) => r.data),
  saveConfig: (name: string, body: { config?: object; yaml?: string }) =>
    http
      .put<ExperimentDetail>(
        `/experiments/${encodeURIComponent(name)}/config`,
        body
      )
      .then((r) => r.data),
  exportUrl: (name: string) =>
    `/api/experiments/${encodeURIComponent(name)}/export.yaml`,

  jobs: (params?: { experiment?: string; active?: boolean; limit?: number }) =>
    get<Job[]>('/jobs', params),
  job: (id: string) => get<Job>(`/jobs/${id}`),
  submit: (
    experiment: string,
    stage: string,
    params: Record<string, unknown> = {},
    force = false
  ) =>
    http
      .post<Job>('/jobs', { experiment, stage, params, force })
      .then((r) => r.data),
  cancel: (id: string) =>
    http.post<Job>(`/jobs/${id}/cancel`).then((r) => r.data),
  kill: (id: string) => http.post<Job>(`/jobs/${id}/kill`).then((r) => r.data),
  retry: (id: string) =>
    http.post<Job>(`/jobs/${id}/retry`).then((r) => r.data),
  log: (id: string, offset = 0) =>
    get<{ text: string; offset: number; size: number }>(`/jobs/${id}/log`, {
      offset,
    }),
  summaries: (id: string, since = 0) =>
    get<Record<string, unknown>[]>(`/jobs/${id}/summaries`, { since }),
}

export const qk = {
  system: ['system'] as const,
  schema: ['schema'] as const,
  experiments: ['experiments'] as const,
  experiment: (name: string) => ['experiment', name] as const,
  status: (name: string) => ['status', name] as const,
  jobs: ['jobs'] as const,
  fs: (root: string, path: string) => ['fs', root, path] as const,
}
