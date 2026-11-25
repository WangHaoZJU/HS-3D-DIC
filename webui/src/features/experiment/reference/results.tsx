import { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react'
import { api, type ReferenceResult } from '@/lib/api'
import {
  STATUS_COLORS,
  robustRange,
  statusByName,
  viridisCss,
} from '@/lib/colormap'
import { cn } from '@/lib/utils'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

const Shape3D = lazy(() => import('./shape-3d'))

export type ColorMode = 'status' | 'znssd' | 'reproj' | 'depth'

export function StereoResults({
  name,
  r,
  shape,
}: {
  name: string
  r: ReferenceResult
  shape: [number, number]
}) {
  const [mode, setMode] = useState<ColorMode>('status')
  return (
    <div className='space-y-4'>
      <div className='flex flex-wrap items-center gap-2 text-xs'>
        {Object.entries(r.status_counts)
          .sort((a, b) => b[1] - a[1])
          .map(([k, v]) => {
            const s = statusByName(k)
            return (
              <span
                key={k}
                className='flex items-center gap-1.5 rounded-md border px-2 py-1'
              >
                <span
                  className='size-2 rounded-full'
                  style={{ background: s?.color }}
                />
                {s?.label ?? k}
                <span className='font-mono text-muted-foreground'>
                  {v} · {((100 * v) / r.n).toFixed(1)}%
                </span>
              </span>
            )
          })}
        <span className='ms-auto text-muted-foreground'>
          参考帧 {r.frame_index} · {r.n} 个测点
          {r.depth_range &&
            ` · 深度 ${r.depth_range[0].toFixed(1)}–${r.depth_range[1].toFixed(1)} mm`}
        </span>
      </div>
      <Tabs defaultValue='overlay'>
        <div className='flex flex-wrap items-center gap-3'>
          <TabsList>
            <TabsTrigger value='overlay'>测点叠加</TabsTrigger>
            <TabsTrigger value='shape'>三维形貌</TabsTrigger>
          </TabsList>
          <div className='ms-auto flex rounded-lg border p-0.5 text-sm'>
            {(
              [
                ['status', '状态码'],
                ['znssd', 'ZNSSD'],
                ['reproj', '重投影误差'],
                ['depth', '深度'],
              ] as [ColorMode, string][]
            ).map(([k, label]) => (
              <button
                key={k}
                onClick={() => setMode(k)}
                className={cn(
                  'rounded-md px-2.5 py-1',
                  mode === k
                    ? 'bg-brand/15 text-brand'
                    : 'text-muted-foreground hover:text-foreground'
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <TabsContent value='overlay' className='mt-3'>
          <div className='grid gap-3 lg:grid-cols-2'>
            <PointOverlay
              name={name}
              cam='left'
              r={r}
              mode={mode}
              shape={shape}
            />
            <PointOverlay
              name={name}
              cam='right'
              r={r}
              mode={mode}
              shape={shape}
            />
          </div>
        </TabsContent>
        <TabsContent value='shape' className='mt-3'>
          <div className='h-[520px] overflow-hidden rounded-xl border'>
            <Suspense fallback={<Skeleton className='h-full' />}>
              <Shape3D r={r} mode={mode} />
            </Suspense>
          </div>
          <AxesNote r={r} />
        </TabsContent>
      </Tabs>
      <Legend r={r} mode={mode} />
    </div>
  )
}

export function valuesFor(
  r: ReferenceResult,
  mode: ColorMode
): (number | null)[] {
  if (mode === 'znssd') return r.znssd
  if (mode === 'reproj') return r.reprojection_error
  if (mode === 'depth')
    return Array.from({ length: r.n }, (_, i) => r.points_3d[3 * i + 2])
  return r.status
}

export function colorFor(r: ReferenceResult, mode: ColorMode) {
  if (mode === 'status')
    return (i: number) => STATUS_COLORS[r.status[i]]?.color ?? '#999'
  const v = valuesFor(r, mode)
  const [lo, hi] = robustRange(v)
  return (i: number) => {
    const x = v[i]
    return x == null ? '#555' : viridisCss((x - lo) / (hi - lo))
  }
}

function PointOverlay({
  name,
  cam,
  r,
  mode,
  shape,
}: {
  name: string
  cam: 'left' | 'right'
  r: ReferenceResult
  mode: ColorMode
  shape: [number, number]
}) {
  const [h, w] = shape
  const canvas = useRef<HTMLCanvasElement>(null)
  const pts = cam === 'left' ? r.points_left : r.points_right
  const color = useMemo(() => colorFor(r, mode), [r, mode])
  const [hover, setHover] = useState<number | null>(null)
  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, w, h)
    const s = Math.max(1.5, Math.max(w, h) / 400)
    for (let i = 0; i < r.n; i++) {
      const x = pts[2 * i]
      const y = pts[2 * i + 1]
      if (x == null || y == null) continue
      ctx.fillStyle = color(i)
      ctx.fillRect(x - s, y - s, 2 * s, 2 * s)
    }
    if (hover != null && pts[2 * hover] != null) {
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = s / 1.5
      ctx.strokeRect(
        pts[2 * hover]! - 3 * s,
        pts[2 * hover + 1]! - 3 * s,
        6 * s,
        6 * s
      )
    }
  }, [pts, color, r.n, w, h, hover])
  return (
    <div className='overflow-hidden rounded-xl border bg-black'>
      <div className='flex justify-between border-b border-white/10 px-3 py-1.5 text-xs text-white/80'>
        <span className='font-medium'>
          {cam === 'left' ? '左相机（参考）' : '右相机（立体匹配结果）'}
        </span>
        <span className='font-mono text-white/60'>
          {hover != null ? pointInfo(r, hover) : r.input?.[`reference_${cam}`]}
        </span>
      </div>
      <div className='relative' style={{ aspectRatio: `${w} / ${h}` }}>
        <img
          src={api.frameUrl(name, cam, r.frame_index)}
          alt=''
          className='absolute inset-0 h-full w-full opacity-70'
        />
        <canvas
          ref={canvas}
          width={w}
          height={h}
          className='absolute inset-0 h-full w-full'
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const x = ((e.clientX - rect.left) / rect.width) * w
            const y = ((e.clientY - rect.top) / rect.height) * h
            let best = -1
            let bd = Infinity
            for (let i = 0; i < r.n; i++) {
              const px = pts[2 * i]
              const py = pts[2 * i + 1]
              if (px == null || py == null) continue
              const d = (px - x) ** 2 + (py - y) ** 2
              if (d < bd) {
                bd = d
                best = i
              }
            }
            setHover(best >= 0 && bd < 400 ? best : null)
          }}
          onMouseLeave={() => setHover(null)}
        />
      </div>
    </div>
  )
}

function pointInfo(r: ReferenceResult, i: number): string {
  const st = STATUS_COLORS[r.status[i]]?.label ?? r.status[i]
  const z = r.znssd[i]
  const e = r.reprojection_error[i]
  return `#${i} · ${st} · ZNSSD ${z != null ? z.toFixed(4) : '—'} · 重投影 ${e != null ? e.toFixed(3) : '—'} px`
}

function Legend({ r, mode }: { r: ReferenceResult; mode: ColorMode }) {
  if (mode === 'status') return null
  const [lo, hi] = robustRange(valuesFor(r, mode))
  const unit = mode === 'depth' ? 'mm' : mode === 'reproj' ? 'px' : ''
  return (
    <div className='flex items-center gap-3 text-xs text-muted-foreground'>
      <span className='font-mono'>
        {lo.toPrecision(4)} {unit}
      </span>
      <span
        className='h-2 w-64 rounded-full'
        style={{
          background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => viridisCss(t)).join(',')})`,
        }}
      />
      <span className='font-mono'>
        {hi.toPrecision(4)} {unit}
      </span>
      <span>（2%–98% 分位）</span>
    </div>
  )
}

function AxesNote({ r }: { r: ReferenceResult }) {
  const a = r.mean_axes
  const f = (v?: number[]) =>
    v ? `(${v.map((x) => x.toFixed(2)).join(', ')})` : '—'
  return (
    <p className='mt-2 text-xs text-muted-foreground'>
      局部坐标系平均方向（左相机坐标）：弦向 c {f(a.c)} · 展向 s {f(a.s)} · 法向
      n {f(a.n)}。展向参考{' '}
      <span className='font-mono'>{f(r.span_direction)}</span>（配置
      recon.span_direction）；若 s 与叶片展向不符，在配置页修改后重跑立体匹配。
    </p>
  )
}
