import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, FolderInput, ScanSearch, Save } from 'lucide-react'
import { toast } from 'sonner'
import {
  api,
  errorMessage,
  qk,
  type ExperimentConfig,
  type ProbeResult,
  type Sequences,
} from '@/lib/api'
import { bytes } from '@/lib/format'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { PathPicker } from '@/components/path-picker'
import { FrameChecks, SharpnessChart, SyncCard } from './checks'
import { FramePairViewer } from './frame-viewer'
import { Metric } from './metric'

type Cam = 'left' | 'right'
type CamDraft = {
  path: string
  pattern: string
  frame_offset: number
  timestamp_path: string | null
  sensor_origin: [number, number]
}
type Draft = { left: CamDraft; right: CamDraft; frame_rate_hz: number }

function draftFrom(cfg: ExperimentConfig): Draft {
  const cams = cfg.cameras as Record<string, Record<string, unknown>> & {
    frame_rate_hz: number
  }
  const cam = (c: Cam): CamDraft => ({
    path: String(cams[c].path ?? ''),
    pattern: String(cams[c].pattern ?? '*.tif'),
    frame_offset: Number(cams[c].frame_offset ?? 0),
    timestamp_path: (cams[c].timestamp_path as string | null) ?? null,
    sensor_origin: (cams[c].sensor_origin as [number, number]) ?? [0, 0],
  })
  return {
    left: cam('left'),
    right: cam('right'),
    frame_rate_hz: Number(cams.frame_rate_hz),
  }
}

export function ImportPage({ name }: { name: string }) {
  const detail = useQuery({
    queryKey: qk.experiment(name),
    queryFn: () => api.experiment(name),
  })
  if (!detail.data) return <Skeleton className='h-96' />
  return (
    <ImportBody
      key={`${name}:${detail.dataUpdatedAt}`}
      name={name}
      config={detail.data.config}
    />
  )
}

function ImportBody({
  name,
  config,
}: {
  name: string
  config: ExperimentConfig
}) {
  const qc = useQueryClient()
  const initial = useMemo(() => draftFrom(config), [config])
  const [draft, setDraft] = useState<Draft>(initial)
  const dirty = JSON.stringify(draft) !== JSON.stringify(initial)
  const seqs = useQuery({
    queryKey: qk.sequences(name),
    queryFn: () => api.sequences(name),
    meta: { silent: true },
  })
  const [frame, setFrame] = useState<number>(
    Number(config.reference_frame ?? 0)
  )

  const refresh = () => {
    for (const k of [
      qk.experiment(name),
      qk.sequences(name),
      qk.sync(name),
      qk.sharpness(name),
      qk.status(name),
    ])
      qc.invalidateQueries({ queryKey: k })
    qc.removeQueries({ queryKey: ['frame-report', name] })
  }
  const save = useMutation({
    mutationFn: (patch: (c: ExperimentConfig) => ExperimentConfig) =>
      api.saveConfig(name, { config: patch(structuredClone(config)) }),
    onSuccess: () => {
      toast.success('已写入实验配置')
      refresh()
    },
  })
  const saveDraft = () =>
    save.mutate((c) => {
      const cams = c.cameras as Record<string, unknown>
      for (const cam of ['left', 'right'] as const) {
        cams[cam] = { ...(cams[cam] as object), ...draft[cam] }
      }
      cams.frame_rate_hz = draft.frame_rate_hz
      return c
    })

  const s = seqs.data
  const n = s?.n_frames ?? 0
  const k = Math.min(frame, Math.max(0, n - 1))
  const ready = !!s && n > 0 && !s.cameras.left.error && !s.cameras.right.error

  return (
    <div className='space-y-6'>
      <Card>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <FolderInput className='size-4 text-brand' /> 图像序列
          </CardTitle>
          <CardDescription>
            选择从相机软件导出的左右序列目录（16 位 TIFF / PNG /
            BMP）。两台相机预触发长度不同时，用“帧偏移”跳过多出来的开头帧，使第
            k 帧是同一时刻。
          </CardDescription>
        </CardHeader>
        <CardContent className='space-y-5'>
          <div className='grid gap-5 xl:grid-cols-2'>
            {(['left', 'right'] as const).map((cam) => (
              <CameraEditor
                key={cam}
                cam={cam}
                value={draft[cam]}
                onChange={(v) => setDraft((d) => ({ ...d, [cam]: v }))}
                current={s?.cameras[cam]}
              />
            ))}
          </div>
          <div className='flex flex-wrap items-end gap-4 border-t pt-4'>
            <div className='space-y-1.5'>
              <Label htmlFor='fps'>帧率 fps</Label>
              <Input
                id='fps'
                inputMode='decimal'
                className='w-36 font-mono'
                value={draft.frame_rate_hz}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    frame_rate_hz: Number(e.target.value),
                  }))
                }
              />
            </div>
            <p className='max-w-xl text-xs text-muted-foreground'>
              没有时间戳文件时按帧率生成时间轴。定频驻留且帧率受限时不要恰好取激振频率的整数分之一（规划计算器可检查混叠）。
            </p>
            <div className='ms-auto flex gap-2'>
              <Button
                variant='outline'
                disabled={!dirty}
                onClick={() => setDraft(initial)}
              >
                还原
              </Button>
              <Button disabled={!dirty || save.isPending} onClick={saveDraft}>
                <Save /> 写入配置
              </Button>
            </div>
          </div>
          {dirty && (
            <p className='text-xs text-warn'>
              有未写入的修改；下方的预览和检查仍按已保存的配置显示。
            </p>
          )}
        </CardContent>
      </Card>

      {seqs.isLoading && <Skeleton className='h-80' />}
      {s && (
        <PairStatus
          s={s}
          onLimit={(stop) =>
            save.mutate((c) => ({
              ...c,
              frames: { ...(c.frames as object), stop },
            }))
          }
        />
      )}

      {ready && (
        <>
          <FramePairViewer
            name={name}
            nFrames={n}
            frame={k}
            onFrame={setFrame}
            fullScale={{
              left: s.cameras.left.full_scale,
              right: s.cameras.right.full_scale,
            }}
            referenceFrame={s.reference_frame}
          />
          <div className='grid gap-4 xl:grid-cols-2'>
            <FrameChecks name={name} cam='left' frame={k} />
            <FrameChecks name={name} cam='right' frame={k} />
          </div>
          <SharpnessChart
            name={name}
            frame={k}
            referenceFrame={s.reference_frame}
            onFrame={setFrame}
            saving={save.isPending}
            onSetReference={(f) =>
              save.mutate((c) => ({ ...c, reference_frame: f }))
            }
          />
          <SyncCard name={name} />
        </>
      )}
    </div>
  )
}

function CameraEditor({
  cam,
  value,
  onChange,
  current,
}: {
  cam: Cam
  value: CamDraft
  onChange: (v: CamDraft) => void
  current?: Sequences['cameras']['left']
}) {
  const [probe, setProbe] = useState<ProbeResult | null>(null)
  const run = useMutation({
    mutationFn: () => api.probe(value.path, value.pattern),
    onSuccess: setProbe,
    onError: (e) => {
      setProbe(null)
      toast.error(errorMessage(e))
    },
  })
  const info: Partial<ProbeResult> | undefined = probe ?? current
  const set = (patch: Partial<CamDraft>) => onChange({ ...value, ...patch })
  return (
    <div className='space-y-3 rounded-lg border bg-muted/10 p-4'>
      <div className='flex items-center justify-between'>
        <span className='font-medium'>
          {cam === 'left' ? '左相机' : '右相机'}
        </span>
        <Button
          size='sm'
          variant='outline'
          disabled={!value.path || run.isPending}
          onClick={() => run.mutate()}
        >
          <ScanSearch /> 探测
        </Button>
      </div>
      <div className='space-y-1.5'>
        <Label className='text-xs text-muted-foreground'>图像目录</Label>
        <PathPicker
          mode='dir'
          value={value.path}
          onChange={(v) => set({ path: v ?? '' })}
        />
      </div>
      <div className='grid grid-cols-[1fr_7rem] gap-3'>
        <div className='space-y-1.5'>
          <Label className='text-xs text-muted-foreground'>文件匹配</Label>
          <Input
            className='font-mono'
            value={value.pattern}
            onChange={(e) => set({ pattern: e.target.value })}
          />
        </div>
        <div className='space-y-1.5'>
          <Label className='text-xs text-muted-foreground'>帧偏移</Label>
          <Input
            className='font-mono'
            inputMode='numeric'
            value={value.frame_offset}
            onChange={(e) =>
              set({
                frame_offset: Math.max(
                  0,
                  Math.floor(Number(e.target.value) || 0)
                ),
              })
            }
          />
        </div>
      </div>
      <div className='space-y-1.5'>
        <Label className='text-xs text-muted-foreground'>
          时间戳文件（.csv / .txt / .npy，单位 s）
        </Label>
        <PathPicker
          mode='file'
          value={value.timestamp_path}
          onChange={(v) => set({ timestamp_path: v })}
        />
        {probe &&
          probe.timestamp_candidates.length > 0 &&
          !value.timestamp_path && (
            <div className='flex flex-wrap items-center gap-1 text-xs'>
              <span className='text-muted-foreground'>发现：</span>
              {probe.timestamp_candidates.slice(0, 3).map((p) => (
                <button
                  key={p}
                  className='rounded border px-1.5 py-0.5 font-mono hover:bg-accent'
                  onClick={() => set({ timestamp_path: p })}
                >
                  {p.split('/').pop()}
                </button>
              ))}
            </div>
          )}
      </div>
      {current?.error && !probe && (
        <p className='text-sm text-bad'>{current.error}</p>
      )}
      {info?.n_frames != null && (
        <div className='grid grid-cols-2 gap-2 sm:grid-cols-4'>
          <Metric
            label='帧数'
            value={
              probe ? probe.n_frames : (current?.raw_frames ?? info.n_frames)
            }
          />
          <Metric
            label='尺寸'
            value={info.shape ? `${info.shape[1]}×${info.shape[0]}` : '—'}
          />
          <Metric
            label='位深'
            value={`${info.data_bits} / ${info.container_bits}`}
            unit='bit'
            verdict={(info.data_bits ?? 0) > 8 ? 'ok' : 'warn'}
            rule='有效数据位数 / 存储位数。8 位导出会丢精度，应导出相机原始位深（例如 12 位存在 16 位 TIFF 中）'
          />
          <Metric
            label='编号缺口'
            value={info.numbering?.numbered ? info.numbering.n_missing : '—'}
            verdict={
              !info.numbering?.numbered
                ? 'info'
                : info.numbering.n_missing
                  ? 'bad'
                  : 'ok'
            }
            rule='文件名中的帧号是否连续；缺号通常意味着丢帧或导出不完整'
          />
        </div>
      )}
      {probe && (
        <div className='space-y-1 text-xs text-muted-foreground'>
          <div className='font-mono'>
            {probe.first_name} … {probe.last_name} · {probe.reader} ·{' '}
            {bytes(probe.total_bytes)} · 灰度 {probe.value_min}–
            {probe.value_max}
          </div>
          {probe.warnings.map((w) => (
            <div key={w} className='flex items-start gap-1 text-warn'>
              <AlertTriangle className='mt-0.5 size-3 shrink-0' /> {w}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function PairStatus({
  s,
  onLimit,
}: {
  s: Sequences
  onLimit: (stop: number) => void
}) {
  const l = s.cameras.left
  const r = s.cameras.right
  const nl = l.n_frames ?? 0
  const nr = r.n_frames ?? 0
  const limited = s.frames.stop != null && s.frames.stop <= s.n_frames
  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border px-4 py-3 text-sm',
        s.length_mismatch && !limited ? 'border-warn/40 bg-warn/10' : 'bg-card'
      )}
    >
      <span>
        左 <span className='font-mono'>{nl}</span> 帧（偏移 {l.frame_offset}）
      </span>
      <span>
        右 <span className='font-mono'>{nr}</span> 帧（偏移 {r.frame_offset}）
      </span>
      <span>
        共同 <span className='font-mono'>{s.n_frames}</span> 帧 · 参考帧{' '}
        <span className='font-mono'>{s.reference_frame}</span>
      </span>
      {s.length_mismatch && !limited && (
        <span className='flex items-center gap-2 text-warn'>
          <AlertTriangle className='size-4' />
          左右帧数不同：确认帧偏移，或只处理共同部分
          <Button
            size='sm'
            variant='outline'
            onClick={() => onLimit(s.n_frames)}
          >
            结束帧设为 {s.n_frames}
          </Button>
        </span>
      )}
    </div>
  )
}
