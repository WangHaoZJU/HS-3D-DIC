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

export type ProbeResult = {
  path: string
  pattern: string
  reader: string | null
  n_frames: number
  shape: [number, number]
  dtype: string
  container_bits: number
  effective_bits: number
  msb_aligned_shift: number
  data_bits: number
  value_min: number
  value_max: number
  full_scale: number
  first_name: string
  last_name: string
  total_bytes: number | null
  numbering: {
    numbered: boolean
    first: number | null
    last: number | null
    gaps: { after: number; missing: number }[]
    n_missing: number
    duplicates: number
  }
  timestamp_candidates: string[]
  warnings: string[]
}

export type CameraSequence = Partial<ProbeResult> & {
  path: string
  pattern: string
  frame_offset: number
  timestamp_path: string | null
  raw_frames?: number
  error?: string
}

export type Sequences = {
  cameras: { left: CameraSequence; right: CameraSequence }
  frame_rate_hz: number
  n_frames: number
  length_mismatch: boolean
  reference_frame: number
  frames: { start: number; stop: number | null; step: number }
}

export type FrameReport = {
  frame: number
  name: string
  raw_index: number
  full_scale: number
  min: number
  max: number
  mean: number
  std: number
  p01: number
  p999: number
  saturation_fraction: number
  dark_fraction: number
  contrast: number
  mig: number
  speckle_px: number
  sharpness: number
  histogram: { counts: number[]; edges: number[] }
}

export type Timing = {
  n: number
  duration_s?: number
  rate_hz: number | null
  jitter_us: number | null
  n_gaps: number
  gaps: { after_frame: number; dt_s: number; missing: number }[]
  rate_error_ppm?: number | null
}

export type SyncResult = {
  n_left: number
  n_right: number
  n_pairs: number
  index_aligned: boolean
  delta_us: { max: number | null; mean: number | null; p99: number | null }
  unpaired_left: number[]
  unpaired_right: number[]
  n_unpaired_left: number
  n_unpaired_right: number
  left: Timing
  right: Timing
  source: { left: 'file' | 'frame_rate'; right: 'file' | 'frame_rate' }
  max_sync_delta_us: number | null
}

export type Sharpness = {
  frames: number[]
  n_frames: number
  left: number[]
  right: number[]
}

export type CalibrationReportCamera = {
  K: number[][]
  dist: number[]
  image_size: [number, number]
  mono_rms_px: number
  views_used: number
  views_total: number
  view_rms_px: (number | null)[]
  std_fx_fy_cx_cy?: number[]
}

export type CalibrationReport = {
  stereo_rms_px: number
  left: CalibrationReportCamera
  right: CalibrationReportCamera
  baseline_mm: number
  optical_axis_angle_deg: number
  pairs: number[]
  residual_left_rms_px: number | null
  residual_right_rms_px: number | null
  residual_left_max_px: number | null
  residual_right_max_px: number | null
  n_corners: number
  excluded_views?: number[]
  views?: [string, string][]
}

export type CalibrationView = {
  index: number
  name_left: string
  name_right: string
  n_left: number
  n_right: number
  n_common: number
  excluded: boolean
}

export type CalibrationOverview = {
  busy: boolean
  board: Record<string, unknown>
  board_size_mm: [number, number]
  n_corners_board: number
  exclude_views: number[]
  status: StageRecord
  images: { left: number; right: number; shape: [number, number] } | null
  images_error?: string
  detections: {
    up_to_date: boolean
    origins: [number, number][]
    image_shapes: [number, number][]
    views: CalibrationView[]
  } | null
  report: CalibrationReport | null
  pairs: {
    view: number
    rms_left: number
    rms_right: number
    n: number
    distance_mm: number
  }[]
  residuals?: { left: [number, number][]; right: [number, number][] }
}

export type CalibrationViewDetail = {
  index: number
  names: [string, string]
  left: { ids: number[]; points: [number, number][] } | null
  right: { ids: number[]; points: [number, number][] } | null
  pair: {
    ids: number[]
    rms_left: number
    rms_right: number
    left: { points: [number, number][]; residuals: [number, number][] }
    right: { points: [number, number][]; residuals: [number, number][] }
  } | null
}

export type CalibrationPoses = {
  cameras: {
    name: string
    center: number[]
    R: number[][]
    K: number[][]
    image_size: [number, number]
  }[]
  boards: { view: number; corners: number[][]; rms: number }[]
  board_size_mm: [number, number]
  baseline_mm: number
}

export type MaskShape = { op: 'add' | 'subtract'; points: [number, number][] }

export type MasksInfo = {
  roi: [number, number, number, number] | null
  image_shape: [number, number]
  reference_frame: number
  subset_size: number
  step: number
  poi: {
    path: string | null
    shapes: MaskShape[]
    drawn: boolean
    exists?: boolean
  }
  rbm: {
    path: string | null
    shapes: MaskShape[]
    drawn: boolean
    exists?: boolean
  }
}

export type ReferenceResult = {
  n: number
  frame_index: number
  input: Record<string, string>
  points_left: (number | null)[]
  points_right: (number | null)[]
  status: number[]
  status_counts: Record<string, number>
  znssd: (number | null)[]
  reprojection_error: (number | null)[]
  points_3d: (number | null)[]
  connectivity: number[]
  mean_axes: Partial<Record<'c' | 's' | 'n', number[]>>
  local_frames: (number | null)[]
  depth_range: [number, number] | null
  span_direction: number[]
}

export type DicSummary = {
  n_frames: number
  n_points: number
  frames_done: number
  frame_ids: [number, number][] | null
  timestamps: (number | null)[]
  valid_left: (number | null)[]
  valid_right: (number | null)[]
  znssd_median: (number | null)[]
  iterations_mean: (number | null)[]
  status_counts: Record<'left' | 'right', Record<string, number>>
  reference_frame: number | null
}

export type StatusMap = {
  valid: number[][]
  worst: number[][]
  t_edges: number[]
  n_frames: number
  n_points: number
}

export type DicPoint = {
  i: number
  timestamps: (number | null)[]
  left: {
    u: (number | null)[]
    v: (number | null)[]
    status: number[]
    znssd: (number | null)[]
  }
  right: {
    u: (number | null)[]
    v: (number | null)[]
    status: number[]
    znssd: (number | null)[]
  }
}

export type DisplacementInfo = {
  n_frames: number
  n_points: number
  kinds: string[]
  timestamps: (number | null)[]
  ranges: Record<string, { abs: number[]; mag: number }>
  valid_fraction: number | null
  rbm: boolean
}

export type DisplacementFrame = {
  t: number
  kind: string
  time: number
  d: (number | null)[][]
  status: number[]
}

export type PointSeries = {
  t: (number | null)[]
  c: (number | null)[]
  s: (number | null)[]
  n: (number | null)[]
}
export type DisplacementPoint = {
  i: number
  n_frames: number
  local?: PointSeries
  rbm_local?: PointSeries
}

export type RbmSeries = {
  t: (number | null)[]
  tx: (number | null)[]
  ty: (number | null)[]
  tz: (number | null)[]
  rx: (number | null)[]
  ry: (number | null)[]
  rz: (number | null)[]
  residual_rms: (number | null)[]
  reference_points: number | null
}

export type AnalyzeOptions = {
  direction: 0 | 1 | 2
  use_rbm: boolean
  min_frequency: number
  max_frequency: number | null
  max_peaks: number
  nperseg: number | null
  frequencies: number[]
  fit_peaks: number
  refine: boolean
  n_harmonics: number
  fdd: boolean
  n_modes: number
  force_file: string | null
}

export type Vibration = {
  options: Partial<AnalyzeOptions>
  sample_rate_hz: number
  n_frames: number
  duration_s: number
  spectrum: { frequencies: (number | null)[]; amplitude: (number | null)[][] }
  peaks: {
    frequency_hz: number
    amplitude_mm: number
    damping: number | null
    damping_limited: boolean
    left_hz: number
    right_hz: number
  }[]
  damping_resolution_hz: number | null
  harmonic: {
    key: string
    frequency_hz: number
    requested_hz: number | null
    direction: number
    use_rbm: boolean
    max_amplitude_mm: (number | null)[]
  }[]
  fdd: {
    frequencies: (number | null)[]
    singular_values: (number | null)[][]
    peak_frequencies: (number | null)[]
  } | null
  frf: {
    frequencies: (number | null)[]
    H1_mean_abs: (number | null)[]
    coherence_mean: (number | null)[]
  } | null
  noise?: {
    experiment: string
    frequencies: (number | null)[]
    amplitude: (number | null)[][]
  }
}

export type Harmonic = {
  key: string
  frequency_hz: number
  amplitude: (number | null)[][]
  phase: (number | null)[][]
  re: (number | null)[][]
  im: (number | null)[][]
  residual_rms: (number | null)[][]
  significant: number[][] | null
  noise_from?: string
}

export type FddMode = {
  p: number
  frequency_hz: number
  direction: number
  re: (number | null)[]
  im: (number | null)[]
}
export type MacResult = { labels: string[]; mac: (number | null)[][] }

export type Quality = {
  n_frames: number | null
  valid_fraction?: number
  noise_floor: {
    std: (number | null)[][]
    median_std_mm: number[]
    p95_std_mm: number[]
    n_frames: number
    use_rbm: boolean
    spectrum: { frequencies: (number | null)[]; amplitude: (number | null)[][] }
  } | null
  uncertainty: {
    key: string
    pixel_sigma: number
    std: (number | null)[][]
    median_std_mm: number[]
    monte_carlo_median_std_mm?: number[]
    monte_carlo?: number
  }[]
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

  probe: (path: string, pattern: string) =>
    http
      .post<ProbeResult>('/sequences/probe', { path, pattern })
      .then((r) => r.data),
  sequences: (name: string) =>
    get<Sequences>(`/experiments/${encodeURIComponent(name)}/sequences`),
  frameUrl: (
    name: string,
    cam: 'left' | 'right',
    k: number,
    opts: {
      lo?: number
      hi?: number
      overlay?: 'saturation'
      scale?: number
      v?: string
    } = {}
  ) => {
    const q = new URLSearchParams()
    for (const [key, val] of Object.entries(opts))
      if (val !== undefined && val !== null) q.set(key, String(val))
    const qs = q.toString()
    return `/api/experiments/${encodeURIComponent(name)}/frames/${cam}/${k}.png${qs ? `?${qs}` : ''}`
  },
  frameReport: (name: string, cam: 'left' | 'right', k: number) =>
    get<FrameReport>(
      `/experiments/${encodeURIComponent(name)}/frames/${cam}/${k}/report`
    ),
  sync: (name: string) =>
    get<SyncResult>(`/experiments/${encodeURIComponent(name)}/sync`),
  calibration: (name: string) =>
    get<CalibrationOverview>(
      `/experiments/${encodeURIComponent(name)}/calibration`
    ),
  calibrationView: (name: string, i: number) =>
    get<CalibrationViewDetail>(
      `/experiments/${encodeURIComponent(name)}/calibration/views/${i}`
    ),
  calibrationImageUrl: (
    name: string,
    i: number,
    cam: 'left' | 'right',
    scale = 1
  ) =>
    `/api/experiments/${encodeURIComponent(name)}/calibration/views/${i}/${cam}.png?scale=${scale}`,
  boardUrl: (name: string, board: Record<string, unknown>) => {
    const q = new URLSearchParams()
    for (const [k, v] of Object.entries(board))
      if (v !== undefined && v !== null && v !== '') q.set(k, String(v))
    return `/api/experiments/${encodeURIComponent(name)}/calibration/board.png?${q}`
  },
  calibrationPoses: (name: string) =>
    get<CalibrationPoses>(
      `/experiments/${encodeURIComponent(name)}/calibration/poses`
    ),
  masks: (name: string) =>
    get<MasksInfo>(`/experiments/${encodeURIComponent(name)}/masks`),
  saveMasks: (
    name: string,
    body: {
      roi: number[] | null
      poi: MaskShape[]
      rbm: MaskShape[]
      subset_size: number
      step: number
    }
  ) =>
    http
      .put<MasksInfo>(`/experiments/${encodeURIComponent(name)}/masks`, body)
      .then((r) => r.data),
  poiPreview: (
    name: string,
    body: {
      roi: number[] | null
      poi: MaskShape[] | null
      subset_size: number
      step: number
    }
  ) =>
    http
      .post<{ n: number; points: number[]; error?: string }>(
        `/experiments/${encodeURIComponent(name)}/poi-preview`,
        body
      )
      .then((r) => r.data),
  reference: (name: string) =>
    get<ReferenceResult>(`/experiments/${encodeURIComponent(name)}/reference`),
  dicSummary: (name: string) =>
    get<DicSummary>(`/experiments/${encodeURIComponent(name)}/dic/summary`),
  statusMap: (
    name: string,
    side: 'left' | 'right' | 'recon',
    tBins = 240,
    nBins = 200
  ) =>
    get<StatusMap>(`/experiments/${encodeURIComponent(name)}/dic/status-map`, {
      side,
      t_bins: tBins,
      n_bins: nBins,
    }),
  dicPoint: (name: string, i: number) =>
    get<DicPoint>(`/experiments/${encodeURIComponent(name)}/dic/point/${i}`),
  displacementInfo: (name: string) =>
    get<DisplacementInfo>(
      `/experiments/${encodeURIComponent(name)}/displacement/info`
    ),
  displacementFrame: (name: string, t: number, kind: string) =>
    get<DisplacementFrame>(
      `/experiments/${encodeURIComponent(name)}/displacement/frame/${t}`,
      { kind }
    ),
  displacementPoint: (name: string, i: number) =>
    get<DisplacementPoint>(
      `/experiments/${encodeURIComponent(name)}/displacement/point/${i}`
    ),
  rbm: (name: string) =>
    get<RbmSeries>(`/experiments/${encodeURIComponent(name)}/rbm`),
  vibration: (name: string, noiseFrom?: string | null) =>
    get<Vibration>(
      `/experiments/${encodeURIComponent(name)}/vibration`,
      noiseFrom ? { noise_from: noiseFrom } : undefined
    ),
  harmonic: (
    name: string,
    key: string,
    noiseFrom?: string | null,
    kSigma = 5
  ) =>
    get<Harmonic>(
      `/experiments/${encodeURIComponent(name)}/vibration/harmonic/${key}`,
      {
        ...(noiseFrom ? { noise_from: noiseFrom } : {}),
        k_sigma: kSigma,
      }
    ),
  fddMode: (name: string, p: number) =>
    get<FddMode>(`/experiments/${encodeURIComponent(name)}/vibration/fdd/${p}`),
  mac: (name: string) =>
    get<MacResult>(`/experiments/${encodeURIComponent(name)}/vibration/mac`),
  quality: (name: string) =>
    get<Quality>(`/experiments/${encodeURIComponent(name)}/quality`),
  reportUrl: (name: string) =>
    `/api/experiments/${encodeURIComponent(name)}/report.html`,
  exportUrl2: (name: string, path: string) =>
    `/api/experiments/${encodeURIComponent(name)}/export/${path}`,
  sharpness: (name: string, points = 120) =>
    get<Sharpness>(`/experiments/${encodeURIComponent(name)}/sharpness`, {
      points,
    }),
}

export const qk = {
  system: ['system'] as const,
  schema: ['schema'] as const,
  experiments: ['experiments'] as const,
  experiment: (name: string) => ['experiment', name] as const,
  status: (name: string) => ['status', name] as const,
  jobs: ['jobs'] as const,
  fs: (root: string, path: string) => ['fs', root, path] as const,
  sequences: (name: string) => ['sequences', name] as const,
  sync: (name: string) => ['sync', name] as const,
  sharpness: (name: string) => ['sharpness', name] as const,
  calibration: (name: string) => ['calibration', name] as const,
  masks: (name: string) => ['masks', name] as const,
  reference: (name: string) => ['reference', name] as const,
  results: (name: string) => ['results', name] as const,
  frameReport: (name: string, cam: string, k: number) =>
    ['frame-report', name, cam, k] as const,
}
