import { useQuery } from '@tanstack/react-query'
import { Clock, Link2 } from 'lucide-react'
import {
  api,
  qk,
  type FrameReport,
  type Sharpness,
  type SyncResult,
} from '@/lib/api'
import { pct } from '@/lib/format'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Metric, band } from './metric'

// ------------------------------------------------------------------ per-frame checks
export function FrameChecks({
  name,
  cam,
  frame,
}: {
  name: string
  cam: 'left' | 'right'
  frame: number
}) {
  const {
    data: r,
    isLoading,
    isError,
  } = useQuery({
    queryKey: qk.frameReport(name, cam, frame),
    queryFn: () => api.frameReport(name, cam, frame),
    placeholderData: (prev) => prev,
    meta: { silent: true },
  })
  return (
    <Card className='min-w-0 gap-3'>
      <CardHeader>
        <CardTitle className='text-sm'>
          {cam === 'left' ? '左相机' : '右相机'} · 第 {frame} 帧
        </CardTitle>
        {r && (
          <CardDescription className='truncate font-mono text-xs'>
            {r.name}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className='space-y-3'>
        {isLoading && <Skeleton className='h-40' />}
        {isError && <p className='text-sm text-bad'>无法读取该帧</p>}
        {r && (
          <>
            <Histogram r={r} />
            <div className='grid grid-cols-2 gap-2 sm:grid-cols-3'>
              <Metric
                label='饱和像素'
                value={pct(r.saturation_fraction, 2)}
                verdict={
                  r.saturation_fraction < 1e-3
                    ? 'ok'
                    : r.saturation_fraction < 1e-2
                      ? 'warn'
                      : 'bad'
                }
                rule={`达到满量程 ${r.full_scale} 的像素比例。< 0.1% 为好；饱和区的散斑信息丢失，会造成匹配偏差`}
              />
              <Metric
                label='量程利用'
                value={pct(r.p999 / r.full_scale, 0)}
                verdict={band(r.p999 / r.full_scale, [0.5, 0.95], [0.3, 0.99])}
                rule='99.9 分位灰度 / 满量程。50–95% 为好：太低信噪比差，太高容易饱和'
              />
              <Metric
                label='散斑尺寸'
                value={
                  Number.isFinite(r.speckle_px) ? r.speckle_px.toFixed(1) : '—'
                }
                unit='px'
                verdict={band(r.speckle_px, [3, 5], [2.2, 7])}
                rule='由自相关峰半高宽估计（±20%）。设计要求成像直径 3–5 px（技术设计 §3.1）'
              />
              <Metric
                label='平均梯度 MIG'
                value={((r.mig * 255) / r.full_scale).toFixed(1)}
                unit='GL₈/px'
                verdict={band(
                  (r.mig * 255) / r.full_scale,
                  [20, 1e9],
                  [10, 1e9]
                )}
                rule='平均灰度梯度，按 8 位量程折算。越大散斑对比越好；≥ 20 为好，< 10 时匹配噪声明显增大'
              />
              <Metric
                label='对比度'
                value={r.contrast.toFixed(2)}
                rule='灰度标准差 / 均值'
              />
              <Metric
                label='清晰度'
                value={r.sharpness.toFixed(3)}
                unit='px⁻²'
                rule='归一化梯度能量；运动模糊或失焦时下降，用于比较不同帧'
              />
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}

function Histogram({ r }: { r: FrameReport }) {
  const counts = r.histogram.counts
  const w = 320
  const h = 72
  const logs = counts.map((c) => Math.log10(1 + c))
  const max = Math.max(...logs, 1)
  const bw = w / counts.length
  return (
    <div>
      <svg
        viewBox={`0 0 ${w} ${h + 14}`}
        className='h-24 w-full'
        preserveAspectRatio='none'
        role='img'
        aria-label='灰度直方图'
      >
        {logs.map((v, i) => (
          <rect
            key={i}
            x={i * bw}
            y={h - (v / max) * h}
            width={Math.max(bw - 0.3, 0.5)}
            height={(v / max) * h}
            className='fill-brand/70'
          />
        ))}
        <line
          x1={w - 0.5}
          x2={w - 0.5}
          y1={0}
          y2={h}
          className='stroke-bad'
          strokeWidth={1}
          strokeDasharray='2 2'
        />
        <text x={0} y={h + 11} className='fill-muted-foreground text-[9px]'>
          0
        </text>
        <text
          x={w}
          y={h + 11}
          textAnchor='end'
          className='fill-muted-foreground text-[9px]'
        >
          {r.full_scale}
        </text>
        <text
          x={w / 2}
          y={h + 11}
          textAnchor='middle'
          className='fill-muted-foreground text-[9px]'
        >
          灰度（对数计数）
        </text>
      </svg>
    </div>
  )
}

// ------------------------------------------------------------------ sync
export function SyncCard({ name }: { name: string }) {
  const {
    data: s,
    isLoading,
    error,
  } = useQuery({
    queryKey: qk.sync(name),
    queryFn: () => api.sync(name),
    meta: { silent: true },
    retry: false,
  })
  return (
    <Card className='min-w-0'>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <Link2 className='size-4 text-brand' /> 左右同步与丢帧
        </CardTitle>
        <CardDescription>
          按时间戳把左右帧一一配对（互为最近邻）；有时间戳文件时才能发现丢帧和同步偏差
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className='h-32' />}
        {error && (
          <p className='text-sm text-bad'>
            {String(
              (error as { response?: { data?: { detail?: string } } }).response
                ?.data?.detail ?? error
            )}
          </p>
        )}
        {s && <SyncBody s={s} />}
      </CardContent>
    </Card>
  )
}

function SyncBody({ s }: { s: SyncResult }) {
  const limit = s.max_sync_delta_us
  const dmax = s.delta_us.max ?? 0
  const both = s.source.left === 'file' && s.source.right === 'file'
  return (
    <div className='space-y-3'>
      <div className='grid grid-cols-2 gap-2 md:grid-cols-4'>
        <Metric
          label='逐帧配对'
          value={s.index_aligned ? '一一对应' : `${s.n_pairs} 对`}
          verdict={s.index_aligned ? 'ok' : 'bad'}
          rule='第 k 帧左右是否为同一时刻。不对应时检查触发对齐（帧偏移）或丢帧'
        />
        <Metric
          label='同步偏差 max'
          value={both ? dmax.toFixed(2) : '—'}
          unit={both ? 'µs' : undefined}
          verdict={
            !both
              ? 'info'
              : limit == null
                ? 'info'
                : dmax <= limit
                  ? 'ok'
                  : 'bad'
          }
          rule={
            limit == null
              ? '配置中未设置 max_sync_delta_us，只显示数值'
              : `上限 ${limit} µs（cameras.max_sync_delta_us）`
          }
        />
        <Metric
          label='未配对帧'
          value={`${s.n_unpaired_left} / ${s.n_unpaired_right}`}
          verdict={s.n_unpaired_left + s.n_unpaired_right === 0 ? 'ok' : 'bad'}
          rule='左 / 右没有找到对应帧的数量'
        />
        <Metric
          label='时间戳来源'
          value={`${s.source.left === 'file' ? '文件' : '帧率'} / ${s.source.right === 'file' ? '文件' : '帧率'}`}
          verdict={both ? 'ok' : 'warn'}
          rule='只有导出了逐帧时间戳，才能真正检查同步和丢帧；按帧率生成的时间轴总是“对齐”的'
        />
      </div>
      <div className='grid gap-2 md:grid-cols-2'>
        {(['left', 'right'] as const).map((cam) => {
          const t = s[cam]
          return (
            <div key={cam} className='rounded-lg border p-3 text-sm'>
              <div className='mb-1 flex items-center gap-2 font-medium'>
                <Clock className='size-3.5 text-muted-foreground' />
                {cam === 'left' ? '左相机' : '右相机'}
                <span className='ms-auto font-mono text-xs text-muted-foreground'>
                  {t.n} 帧
                </span>
              </div>
              <div className='tabular grid grid-cols-3 gap-2 font-mono text-xs'>
                <span>{t.rate_hz ? `${t.rate_hz.toFixed(2)} fps` : '—'}</span>
                <span>
                  抖动{' '}
                  {t.jitter_us != null ? `${t.jitter_us.toFixed(2)} µs` : '—'}
                </span>
                <span className={cn(t.n_gaps ? 'text-bad' : 'text-ok')}>
                  间隙 {t.n_gaps}
                </span>
              </div>
              {t.gaps.length > 0 && (
                <div className='mt-1 text-xs text-bad'>
                  {t.gaps
                    .slice(0, 5)
                    .map((g) => `第 ${g.after_frame} 帧后缺 ${g.missing} 帧`)
                    .join('；')}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ sharpness
export function SharpnessChart({
  name,
  frame,
  referenceFrame,
  onFrame,
  onSetReference,
  saving,
}: {
  name: string
  frame: number
  referenceFrame: number
  onFrame: (k: number) => void
  onSetReference: (k: number) => void
  saving: boolean
}) {
  const { data, isLoading } = useQuery({
    queryKey: qk.sharpness(name),
    queryFn: () => api.sharpness(name),
    staleTime: 5 * 60_000,
    meta: { silent: true },
  })
  return (
    <Card className='min-w-0'>
      <CardHeader>
        <CardTitle className='text-base'>清晰度曲线 · 参考帧</CardTitle>
        <CardDescription>
          归一化梯度能量随帧变化（各自除以最大值）。参考帧应选运动模糊最小的时刻，例如激振前的静止帧。点击曲线跳到该帧。
        </CardDescription>
      </CardHeader>
      <CardContent className='space-y-3'>
        {isLoading && <Skeleton className='h-36' />}
        {data && (
          <div className='flex items-center gap-4 text-xs text-muted-foreground'>
            <span className='flex items-center gap-1.5'>
              <span className='h-0.5 w-4 rounded bg-chart-2' /> 左相机
            </span>
            <span className='flex items-center gap-1.5'>
              <span className='h-0.5 w-4 rounded bg-chart-4' /> 右相机
            </span>
            <span className='flex items-center gap-1.5'>
              <span className='h-0.5 w-4 border-t-2 border-dashed border-brand' />{' '}
              参考帧
            </span>
            <span className='flex items-center gap-1.5'>
              <span className='h-3 w-px bg-foreground/50' /> 当前帧
            </span>
          </div>
        )}
        {data && (
          <SharpPlot
            d={data}
            frame={frame}
            referenceFrame={referenceFrame}
            onFrame={onFrame}
          />
        )}
        {data && (
          <div className='flex flex-wrap items-center gap-2 text-sm'>
            <span className='text-muted-foreground'>
              当前参考帧{' '}
              <span className='font-mono text-foreground'>
                {referenceFrame}
              </span>
            </span>
            <Button
              size='sm'
              variant='outline'
              onClick={() => onFrame(bestFrame(data))}
            >
              跳到最清晰帧（{bestFrame(data)}）
            </Button>
            <Button
              size='sm'
              disabled={frame === referenceFrame || saving}
              onClick={() => onSetReference(frame)}
            >
              将第 {frame} 帧设为参考帧
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function bestFrame(d: Sharpness): number {
  let best = 0
  let bv = -Infinity
  d.frames.forEach((f, i) => {
    const v = Math.min(
      d.left[i] / Math.max(...d.left),
      d.right[i] / Math.max(...d.right)
    )
    if (v > bv) {
      bv = v
      best = f
    }
  })
  return best
}

function SharpPlot({
  d,
  frame,
  referenceFrame,
  onFrame,
}: {
  d: Sharpness
  frame: number
  referenceFrame: number
  onFrame: (k: number) => void
}) {
  const w = 800
  const h = 140
  const pad = { l: 34, r: 8, t: 8, b: 18 }
  const n = Math.max(d.n_frames - 1, 1)
  const norm = (v: number[]) => {
    const m = Math.max(...v)
    return v.map((x) => (m > 0 ? x / m : 0))
  }
  const L = norm(d.left)
  const R = norm(d.right)
  const lo = Math.min(...L, ...R, 0.9)
  const x = (f: number) => pad.l + (f / n) * (w - pad.l - pad.r)
  const y = (v: number) =>
    pad.t + (1 - (v - lo) / Math.max(1 - lo, 1e-6)) * (h - pad.t - pad.b)
  const path = (v: number[]) =>
    v
      .map(
        (val, i) =>
          `${i ? 'L' : 'M'}${x(d.frames[i]).toFixed(1)},${y(val).toFixed(1)}`
      )
      .join('')
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className='h-40 w-full cursor-crosshair'
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect()
        const px = ((e.clientX - r.left) / r.width) * w
        const f = Math.round(((px - pad.l) / (w - pad.l - pad.r)) * n)
        onFrame(Math.max(0, Math.min(d.n_frames - 1, f)))
      }}
      role='img'
      aria-label='清晰度曲线'
    >
      {[lo, (lo + 1) / 2, 1].map((t) => (
        <g key={t}>
          <line
            x1={pad.l}
            x2={w - pad.r}
            y1={y(t)}
            y2={y(t)}
            className='stroke-border'
          />
          <text
            x={pad.l - 4}
            y={y(t) + 3}
            textAnchor='end'
            className='fill-muted-foreground text-[10px]'
          >
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      <line
        x1={x(referenceFrame)}
        x2={x(referenceFrame)}
        y1={pad.t}
        y2={h - pad.b}
        className='stroke-brand'
        strokeDasharray='4 3'
        strokeWidth={1.5}
      />
      <line
        x1={x(frame)}
        x2={x(frame)}
        y1={pad.t}
        y2={h - pad.b}
        className='stroke-foreground/50'
      />
      <path
        d={path(L)}
        fill='none'
        className='stroke-chart-2'
        strokeWidth={1.6}
      />
      <path
        d={path(R)}
        fill='none'
        className='stroke-chart-4'
        strokeWidth={1.6}
      />
      <text x={pad.l} y={h - 4} className='fill-muted-foreground text-[10px]'>
        0
      </text>
      <text
        x={w - pad.r}
        y={h - 4}
        textAnchor='end'
        className='fill-muted-foreground text-[10px]'
      >
        {d.n_frames - 1}
      </text>
    </svg>
  )
}
