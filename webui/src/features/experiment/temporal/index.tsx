import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  Grid3x3,
  MousePointerClick,
  Play,
  RotateCcw,
} from 'lucide-react'
import { toast } from 'sonner'
import { useLive } from '@/stores/live'
import {
  api,
  qk,
  type DicSummary,
  type ExperimentConfig,
  type ReferenceResult,
} from '@/lib/api'
import { STATUS_COLORS, statusByName, viridisCss } from '@/lib/colormap'
import { duration } from '@/lib/format'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { ProgressBar } from '@/components/progress-bar'
import { TimeChart } from '@/components/time-chart'
import { Metric } from '../import/metric'

export function TemporalPage({ name }: { name: string }) {
  const qc = useQueryClient()
  const status = useQuery({
    queryKey: qk.status(name),
    queryFn: () => api.status(name),
  })
  const detail = useQuery({
    queryKey: qk.experiment(name),
    queryFn: () => api.experiment(name),
  })
  const jobs = useQuery({
    queryKey: [...qk.jobs, 'active'],
    queryFn: () => api.jobs({ active: true }),
    refetchInterval: 10_000,
  })
  const liveJobs = useLive((s) => s.jobs)
  const running = (jobs.data ?? [])
    .map((j) => liveJobs[j.id] ?? j)
    .find(
      (j) =>
        j.experiment === name &&
        (j.state === 'running' || j.state === 'cancelling')
    )
  const progress = useLive((s) =>
    running ? s.progress[running.id] : undefined
  )
  const liveTemporal =
    !!running &&
    (progress?.stage === 'temporal' || running.stage === 'temporal')
  const busy = !!status.data?.busy
  const summary = useQuery({
    queryKey: [...qk.results(name), 'dic-summary'],
    queryFn: () => api.dicSummary(name),
    enabled: !busy,
    retry: false,
    meta: { silent: true },
  })
  const submit = useMutation({
    mutationFn: (force: boolean) => api.submit(name, 'temporal', {}, force),
    onSuccess: () => {
      toast.success('已提交时序 DIC')
      qc.invalidateQueries({ queryKey: qk.jobs })
    },
  })
  const rec = status.data?.stages.temporal
  const cfg = detail.data?.config
  return (
    <div className='space-y-6'>
      <Card>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <Grid3x3 className='size-4 text-brand' /> 固定参考帧时序 DIC
          </CardTitle>
          <CardDescription>
            左右相机分别相对各自参考帧求解，前一帧的解只作初值；帧级断点续算，取消后可从中断处继续（技术设计
            §7.6）。
          </CardDescription>
          <CardAction className='flex gap-2'>
            {rec?.state === 'done' && (
              <Button
                size='sm'
                variant='outline'
                disabled={busy || submit.isPending}
                onClick={() =>
                  confirm('强制重算时序 DIC？已有结果会被覆盖。') &&
                  submit.mutate(true)
                }
              >
                <RotateCcw /> 强制重算
              </Button>
            )}
            <Button
              size='sm'
              disabled={busy || submit.isPending || rec?.state === 'done'}
              onClick={() => submit.mutate(false)}
            >
              <Play />{' '}
              {rec?.state === 'partial'
                ? `续算（已完成 ${Math.round((rec.fraction ?? 0) * 100)}%）`
                : rec?.state === 'done'
                  ? '已是最新'
                  : '运行'}
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent>{cfg && <Settings cfg={cfg} />}</CardContent>
      </Card>

      {liveTemporal && running && <LivePanel jobId={running.id} />}
      {busy && !liveTemporal && (
        <p className='text-sm text-muted-foreground'>
          该实验有任务正在写结果文件，完成后显示结果。
        </p>
      )}
      {!busy && summary.isError && (
        <p className='py-8 text-center text-sm text-muted-foreground'>
          还没有时序 DIC 结果。
        </p>
      )}
      {!busy && summary.isLoading && <Skeleton className='h-64' />}
      {!busy && summary.data && <Results name={name} s={summary.data} />}
    </div>
  )
}

function Settings({ cfg }: { cfg: ExperimentConfig }) {
  const dic = cfg.dic as Record<string, Record<string, unknown>> & {
    backend: string
  }
  const fr = cfg.frames as { start: number; stop: number | null; step: number }
  return (
    <div className='grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6'>
      <Metric label='计算后端' value={String(dic.backend)} />
      <Metric label='形函数阶数' value={String(dic.temporal.shape_order)} />
      <Metric
        label='子区 / 步长'
        value={`${dic.poi.subset_size} / ${dic.poi.step}`}
        unit='px'
      />
      <Metric label='插值' value={`${dic.interpolation.order} 次样条`} />
      <Metric
        label='帧范围'
        value={`${fr.start}–${fr.stop ?? '末尾'} / ${fr.step}`}
      />
      <Metric label='ZNSSD 上限' value={String(dic.temporal.max_znssd)} />
    </div>
  )
}

function LivePanel({ jobId }: { jobId: string }) {
  const live = useLive((s) => s.summaries[jobId])
  const p = useLive((s) => s.progress[jobId])
  const backfill = useQuery({
    queryKey: ['summaries', jobId],
    queryFn: () => api.summaries(jobId),
    staleTime: Infinity,
  })
  const rows = useMemo(() => {
    const all = [...(backfill.data ?? []), ...(live ?? [])].filter(
      (r) => 'valid_left' in r
    ) as {
      row: number
      valid_left: number
      valid_right: number
      znssd_median: number | null
    }[]
    const m = new Map<number, (typeof all)[number]>()
    for (const r of all) m.set(r.row, r)
    return [...m.values()].sort((a, b) => a.row - b.row)
  }, [backfill.data, live])
  const x = rows.map((r) => r.row)
  return (
    <Card className='border-brand/30'>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <Activity className='size-4 animate-pulse text-brand' /> 运行中
        </CardTitle>
        <CardDescription className='font-mono'>
          {p ? `${p.done}/${p.total} 帧` : ''}
          {p?.rate ? ` · ${p.rate.toFixed(1)} 帧/s` : ''}
          {p?.eta_s != null ? ` · 剩余 ${duration(p.eta_s)}` : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className='space-y-3'>
        <ProgressBar value={p?.total ? p.done / p.total : null} />
        {rows.length > 1 ? (
          <TimeChart
            x={x}
            height={220}
            xLabel='帧（行）'
            yLabel='有效比例 %'
            yRange={[0, 101]}
            y2Label='ZNSSD 中位数'
            series={[
              {
                label: '左有效',
                color: '#34d399',
                values: rows.map((r) => r.valid_left * 100),
              },
              {
                label: '右有效',
                color: '#a78bfa',
                values: rows.map((r) => r.valid_right * 100),
              },
              {
                label: 'ZNSSD',
                color: '#f59e0b',
                values: rows.map((r) => r.znssd_median),
                scale: 'y2',
                width: 1,
              },
            ]}
          />
        ) : (
          <p className='text-sm text-muted-foreground'>等待第一批帧…</p>
        )}
      </CardContent>
    </Card>
  )
}

function Results({ name, s }: { name: string; s: DicSummary }) {
  const x = useMemo(() => s.timestamps.map((t, i) => (t == null ? i : t)), [s])
  const minValid = Math.min(
    ...(s.valid_left.filter((v) => v != null) as number[]),
    ...(s.valid_right.filter((v) => v != null) as number[])
  )
  return (
    <div className='space-y-6'>
      <Card>
        <CardHeader>
          <CardTitle className='text-base'>逐帧有效比例与 ZNSSD</CardTitle>
          <CardDescription>
            {s.n_points.toLocaleString()} 个测点 × {s.n_frames} 帧（完成{' '}
            {s.frames_done}
            ）。有效比例突然下降通常对应运动模糊、反光或大位移；ZNSSD
            中位数整体抬高说明散斑去相关或照明变化。
          </CardDescription>
        </CardHeader>
        <CardContent className='space-y-4'>
          <div className='grid grid-cols-2 gap-2 sm:grid-cols-4'>
            <Metric
              label='最低有效比例'
              value={`${(minValid * 100).toFixed(1)}%`}
              verdict={
                minValid >= 0.95 ? 'ok' : minValid >= 0.8 ? 'warn' : 'bad'
              }
              rule='共同可见区有效 POI ≥ 95%（技术设计 §15.2）'
            />
            <Metric label='左相机有效' value={pctOf(s.status_counts.left)} />
            <Metric label='右相机有效' value={pctOf(s.status_counts.right)} />
            <Metric
              label='平均迭代'
              value={
                Number.isFinite(meanOf(s.iterations_mean))
                  ? meanOf(s.iterations_mean).toFixed(2)
                  : '—'
              }
            />
          </div>
          <TimeChart
            x={x}
            height={240}
            xLabel='时间 s'
            yLabel='有效比例 %'
            yRange={[0, 101]}
            y2Label='ZNSSD 中位数'
            syncKey={`dic-${name}`}
            series={[
              {
                label: '左有效',
                color: '#34d399',
                values: s.valid_left.map((v) => (v == null ? null : v * 100)),
              },
              {
                label: '右有效',
                color: '#a78bfa',
                values: s.valid_right.map((v) => (v == null ? null : v * 100)),
              },
              {
                label: 'ZNSSD',
                color: '#f59e0b',
                values: s.znssd_median,
                scale: 'y2',
                width: 1,
              },
            ]}
          />
          <StatusLegend counts={s.status_counts} />
        </CardContent>
      </Card>
      <StatusMapCard name={name} />
      <PointInspector name={name} />
    </div>
  )
}

function pctOf(c: Record<string, number>): string {
  const total = Object.values(c).reduce((a, b) => a + b, 0)
  return total ? `${(((c.OK ?? 0) / total) * 100).toFixed(2)}%` : '—'
}

function meanOf(v: (number | null)[]): number {
  const f = v.filter((x): x is number => x != null && Number.isFinite(x))
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : NaN
}

function StatusLegend({ counts }: { counts: DicSummary['status_counts'] }) {
  return (
    <div className='flex flex-wrap gap-2 text-xs'>
      {(['left', 'right'] as const).map((side) =>
        Object.entries(counts[side])
          .filter(([k]) => k !== 'OK')
          .map(([k, v]) => {
            const st = statusByName(k)
            return (
              <span
                key={side + k}
                className='flex items-center gap-1.5 rounded-md border px-2 py-0.5'
              >
                <span
                  className='size-2 rounded-full'
                  style={{ background: st?.color }}
                />
                {side === 'left' ? '左' : '右'} · {st?.label ?? k}
                <span className='font-mono text-muted-foreground'>
                  {v.toLocaleString()}
                </span>
              </span>
            )
          })
      )}
    </div>
  )
}

function StatusMapCard({ name }: { name: string }) {
  const [side, setSide] = useState<'left' | 'right' | 'recon'>('left')
  const { data } = useQuery({
    queryKey: [...qk.results(name), 'status-map', side],
    queryFn: () => api.statusMap(name, side),
    meta: { silent: true },
    retry: false,
  })
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = canvas.current
    if (!c || !data) return
    const tb = data.valid.length
    const nb = data.valid[0]?.length ?? 0
    c.width = tb
    c.height = nb
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(tb, nb)
    for (let t = 0; t < tb; t++)
      for (let n = 0; n < nb; n++) {
        const v = data.valid[t][n]
        const k = 4 * (n * tb + t)
        let rgb: [number, number, number]
        if (v >= 0.999) rgb = [22, 101, 52]
        else {
          const code = data.worst[t][n]
          const hex = STATUS_COLORS[code]?.color ?? '#ef4444'
          const base = [
            parseInt(hex.slice(1, 3), 16),
            parseInt(hex.slice(3, 5), 16),
            parseInt(hex.slice(5, 7), 16),
          ]
          const a = Math.min(1, (1 - v) * 4)
          rgb = [
            22 + (base[0] - 22) * a,
            101 + (base[1] - 101) * a,
            52 + (base[2] - 52) * a,
          ] as [number, number, number]
        }
        img.data[k] = rgb[0]
        img.data[k + 1] = rgb[1]
        img.data[k + 2] = rgb[2]
        img.data[k + 3] = 255
      }
    ctx.putImageData(img, 0, 0)
  }, [data])
  return (
    <Card>
      <CardHeader>
        <CardTitle className='text-base'>状态码时空图</CardTitle>
        <CardDescription>
          横轴为帧（时间），纵轴为测点（按网格顺序，自上而下即图像自上而下）。绿色为全部有效，颜色越深表示该格内失效比例越高，色相为最常见的失效原因。
        </CardDescription>
        <CardAction className='flex rounded-lg border p-0.5 text-sm'>
          {(
            [
              ['left', '左 DIC'],
              ['right', '右 DIC'],
              ['recon', '重建'],
            ] as const
          ).map(([k, l]) => (
            <button
              key={k}
              onClick={() => setSide(k)}
              className={cn(
                'rounded-md px-2.5 py-1',
                side === k ? 'bg-brand/15 text-brand' : 'text-muted-foreground'
              )}
            >
              {l}
            </button>
          ))}
        </CardAction>
      </CardHeader>
      <CardContent>
        {!data ? (
          <Skeleton className='h-48' />
        ) : (
          <div>
            <canvas
              ref={canvas}
              className='h-56 w-full rounded-md border'
              style={{ imageRendering: 'pixelated' }}
            />
            <div className='mt-1 flex justify-between font-mono text-xs text-muted-foreground'>
              <span>帧 0</span>
              <span>
                {data.n_points.toLocaleString()} 个测点 · 每格{' '}
                {Math.ceil(data.n_frames / data.valid.length)} 帧 ×{' '}
                {Math.ceil(data.n_points / (data.valid[0]?.length || 1))} 点
              </span>
              <span>帧 {data.n_frames - 1}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function PointInspector({ name }: { name: string }) {
  const ref = useQuery({
    queryKey: qk.reference(name),
    queryFn: () => api.reference(name),
    meta: { silent: true },
    retry: false,
  })
  const masks = useQuery({
    queryKey: qk.masks(name),
    queryFn: () => api.masks(name),
    meta: { silent: true },
    retry: false,
  })
  const [i, setI] = useState<number | null>(null)
  const pt = useQuery({
    queryKey: [...qk.results(name), 'dic-point', i],
    queryFn: () => api.dicPoint(name, i!),
    enabled: i != null,
    meta: { silent: true },
  })
  const x = useMemo(
    () => pt.data?.timestamps.map((t, k) => (t == null ? k : t)) ?? [],
    [pt.data]
  )
  return (
    <Card>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <MousePointerClick className='size-4 text-brand' /> 单点图像轨迹
        </CardTitle>
        <CardDescription>
          在左参考图上点选测点，查看它在左右图像中的位移
          u、v（px，相对参考帧）。
        </CardDescription>
      </CardHeader>
      <CardContent className='grid gap-4 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]'>
        {ref.data && masks.data ? (
          <PoiPicker
            name={name}
            r={ref.data}
            shape={masks.data.image_shape}
            selected={i}
            onSelect={setI}
          />
        ) : (
          <Skeleton className='aspect-[5/4]' />
        )}
        <div className='min-w-0 space-y-2'>
          {i == null && (
            <p className='text-sm text-muted-foreground'>选择一个测点</p>
          )}
          {pt.data && (
            <>
              <div className='text-sm'>
                测点 <span className='font-mono'>#{i}</span>
              </div>
              <TimeChart
                x={x}
                height={240}
                xLabel='时间 s'
                yLabel='位移 px'
                series={[
                  { label: '左 u', color: '#34d399', values: pt.data.left.u },
                  { label: '左 v', color: '#60a5fa', values: pt.data.left.v },
                  {
                    label: '右 u',
                    color: '#a78bfa',
                    values: pt.data.right.u,
                    dash: [4, 3],
                  },
                  {
                    label: '右 v',
                    color: '#f472b6',
                    values: pt.data.right.v,
                    dash: [4, 3],
                  },
                ]}
              />
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export function PoiPicker({
  name,
  r,
  shape,
  selected,
  onSelect,
  values,
}: {
  name: string
  r: ReferenceResult
  shape: [number, number]
  selected: number | null
  onSelect: (i: number) => void
  values?: (number | null)[]
}) {
  const [h, w] = shape
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, w, h)
    const s = Math.max(1.5, Math.max(w, h) / 350)
    const vmax = values
      ? Math.max(1e-12, ...values.map((v) => Math.abs(v ?? 0)))
      : 1
    for (let k = 0; k < r.n; k++) {
      const x = r.points_left[2 * k]
      const y = r.points_left[2 * k + 1]
      if (x == null || y == null) continue
      ctx.fillStyle = values
        ? viridisCss(Math.abs(values[k] ?? NaN) / vmax)
        : (STATUS_COLORS[r.status[k]]?.color ?? '#999')
      ctx.fillRect(x - s, y - s, 2 * s, 2 * s)
    }
    if (selected != null && r.points_left[2 * selected] != null) {
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = s / 1.2
      ctx.strokeRect(
        r.points_left[2 * selected]! - 4 * s,
        r.points_left[2 * selected + 1]! - 4 * s,
        8 * s,
        8 * s
      )
    }
  }, [r, w, h, selected, values])
  return (
    <div
      className='relative overflow-hidden rounded-xl border bg-black'
      style={{ aspectRatio: `${w} / ${h}` }}
    >
      <img
        src={api.frameUrl(name, 'left', r.frame_index)}
        alt=''
        className='absolute inset-0 h-full w-full opacity-60'
      />
      <canvas
        ref={canvas}
        width={w}
        height={h}
        className='absolute inset-0 h-full w-full cursor-crosshair'
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect()
          const x = ((e.clientX - rect.left) / rect.width) * w
          const y = ((e.clientY - rect.top) / rect.height) * h
          let best = -1
          let bd = Infinity
          for (let k = 0; k < r.n; k++) {
            const px = r.points_left[2 * k]
            const py = r.points_left[2 * k + 1]
            if (px == null || py == null) continue
            const d = (px - x) ** 2 + (py - y) ** 2
            if (d < bd) {
              bd = d
              best = k
            }
          }
          if (best >= 0) onSelect(best)
        }}
      />
    </div>
  )
}
