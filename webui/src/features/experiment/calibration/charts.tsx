import type { CalibrationOverview } from '@/lib/api'
import { cn } from '@/lib/utils'

/** Per-view stereo RMS, left and right bars; click a view to inspect it. */
export function ViewRmsChart({
  pairs,
  views,
  excluded,
  selected,
  onSelect,
}: {
  pairs: CalibrationOverview['pairs']
  views: number
  excluded: Set<number>
  selected: number | null
  onSelect: (v: number) => void
}) {
  const w = 760
  const h = 170
  const pad = { l: 40, r: 8, t: 10, b: 22 }
  const byView = new Map(pairs.map((p) => [p.view, p]))
  const max =
    Math.max(0.05, ...pairs.map((p) => Math.max(p.rms_left, p.rms_right))) * 1.1
  const med = median(pairs.flatMap((p) => [p.rms_left, p.rms_right]))
  const slot = (w - pad.l - pad.r) / Math.max(views, 1)
  const y = (v: number) => pad.t + (1 - v / max) * (h - pad.t - pad.b)
  const bw = Math.max(2, Math.min(14, slot * 0.36))
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className='h-44 w-full'
      role='img'
      aria-label='逐视图 RMS'
    >
      {[0, max / 2, max].map((t) => (
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
            {t.toFixed(3)}
          </text>
        </g>
      ))}
      {med > 0 && (
        <line
          x1={pad.l}
          x2={w - pad.r}
          y1={y(med)}
          y2={y(med)}
          className='stroke-muted-foreground'
          strokeDasharray='3 3'
        />
      )}
      {Array.from({ length: views }, (_, v) => {
        const p = byView.get(v)
        const x0 = pad.l + v * slot + slot / 2
        const sel = v === selected
        return (
          <g key={v} className='cursor-pointer' onClick={() => onSelect(v)}>
            <rect
              x={x0 - slot / 2}
              y={pad.t}
              width={slot}
              height={h - pad.t - pad.b}
              className={cn('fill-transparent', sel && 'fill-brand/10')}
            />
            {p ? (
              <>
                <rect
                  x={x0 - bw}
                  y={y(p.rms_left)}
                  width={bw}
                  height={y(0) - y(p.rms_left)}
                  className='fill-chart-2'
                />
                <rect
                  x={x0}
                  y={y(p.rms_right)}
                  width={bw}
                  height={y(0) - y(p.rms_right)}
                  className='fill-chart-4'
                />
              </>
            ) : (
              <text
                x={x0}
                y={y(0) - 4}
                textAnchor='middle'
                className={cn(
                  'text-[10px]',
                  excluded.has(v) ? 'fill-warn' : 'fill-bad'
                )}
              >
                {excluded.has(v) ? '剔' : '×'}
              </text>
            )}
            {(views <= 40 || v % 5 === 0) && (
              <text
                x={x0}
                y={h - 8}
                textAnchor='middle'
                className={cn(
                  'text-[10px]',
                  sel ? 'fill-foreground' : 'fill-muted-foreground'
                )}
              >
                {v}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

/** All residual vectors of one camera as a scatter (px). */
export function ResidualScatter({
  points,
  label,
  tone,
}: {
  points: [number, number][]
  label: string
  tone: string
}) {
  const size = 200
  const lim =
    Math.max(
      0.05,
      ...points.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y)))
    ) * 1.1
  const s = (v: number) => size / 2 + (v / lim) * (size / 2 - 6)
  const rms = Math.sqrt(
    points.reduce((a, [x, y]) => a + x * x + y * y, 0) /
      Math.max(points.length, 1)
  )
  return (
    <div className='min-w-0'>
      <div className='mb-1 flex items-baseline justify-between text-xs'>
        <span className='font-medium'>{label}</span>
        <span className='font-mono text-muted-foreground'>
          RMS {rms.toFixed(4)} px
        </span>
      </div>
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className='aspect-square w-full max-w-56 rounded-lg border bg-muted/20'
        role='img'
        aria-label={`${label} 残差`}
      >
        <line
          x1={size / 2}
          x2={size / 2}
          y1={4}
          y2={size - 4}
          className='stroke-border'
        />
        <line
          y1={size / 2}
          y2={size / 2}
          x1={4}
          x2={size - 4}
          className='stroke-border'
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={(rms / lim) * (size / 2 - 6)}
          className='fill-none stroke-muted-foreground'
          strokeDasharray='3 2'
        />
        {points.map(([x, y], i) => (
          <circle
            key={i}
            cx={s(x)}
            cy={s(y)}
            r={1.3}
            className={tone}
            opacity={0.6}
          />
        ))}
        <text
          x={size - 6}
          y={size - 6}
          textAnchor='end'
          className='fill-muted-foreground text-[9px]'
        >
          ±{lim.toFixed(3)} px
        </text>
      </svg>
    </div>
  )
}

function median(v: number[]): number {
  if (!v.length) return 0
  const s = [...v].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]
}
