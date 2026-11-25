/** Thin React wrapper around uPlot for long time series (docs/UI_Design.md §3.2). */
import { useEffect, useMemo, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

export type ChartSeries = {
  label: string
  color: string
  values: (number | null)[]
  width?: number
  dash?: number[]
  scale?: string
  unit?: string
  points?: boolean
}

export type ChartMarker = { x: number; color: string; dash?: number[] }

function cssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim()
  return v || fallback
}

export function TimeChart({
  x,
  series,
  height = 220,
  xLabel,
  yLabel,
  y2Label,
  yRange,
  markers = [],
  syncKey,
  onPick,
  className,
}: {
  x: number[]
  series: ChartSeries[]
  height?: number
  xLabel?: string
  yLabel?: string
  y2Label?: string
  yRange?: [number, number]
  markers?: ChartMarker[]
  syncKey?: string
  onPick?: (x: number) => void
  className?: string
}) {
  const box = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  const markersRef = useRef(markers)
  const pickRef = useRef(onPick)
  useEffect(() => {
    markersRef.current = markers
    pickRef.current = onPick
  })
  const shape = useMemo(
    () =>
      JSON.stringify(
        series.map((s) => [
          s.label,
          s.color,
          s.scale,
          s.dash,
          s.width,
          s.points,
        ])
      ) +
      (yRange ?? '') +
      height +
      xLabel +
      yLabel +
      y2Label,
    [series, yRange, height, xLabel, yLabel, y2Label]
  )

  useEffect(() => {
    const el = box.current
    if (!el) return
    const axisColor = cssVar('--muted-foreground', '#8a94a6')
    const grid = cssVar('--border', 'rgba(128,128,128,0.2)')
    const hasY2 = series.some((s) => s.scale === 'y2')
    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height,
      legend: { show: true, live: true },
      cursor: {
        sync: syncKey ? { key: syncKey } : undefined,
        drag: { x: true, y: false },
      },
      scales: {
        x: { time: false },
        y: yRange ? { range: yRange } : {},
        ...(hasY2 ? { y2: {} } : {}),
      },
      axes: [
        {
          stroke: axisColor,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          label: xLabel,
          size: 36,
        },
        {
          stroke: axisColor,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          label: yLabel,
          size: 56,
        },
        ...(hasY2
          ? [
              {
                scale: 'y2',
                side: 1 as const,
                stroke: axisColor,
                grid: { show: false },
                label: y2Label,
                size: 56,
              },
            ]
          : []),
      ],
      series: [
        { label: xLabel ?? 'x' },
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: s.width ?? 1.4,
          dash: s.dash,
          scale: s.scale ?? 'y',
          points: { show: s.points ?? false, size: 4 },
          spanGaps: false,
          value: (_u: uPlot, v: number | null) =>
            v == null
              ? '—'
              : `${Number(v.toPrecision(5))}${s.unit ? ' ' + s.unit : ''}`,
        })),
      ],
      hooks: {
        draw: [
          (u) => {
            const ctx = u.ctx
            for (const m of markersRef.current) {
              const px = u.valToPos(m.x, 'x', true)
              if (!Number.isFinite(px)) continue
              ctx.save()
              ctx.strokeStyle = m.color
              ctx.lineWidth = 1.5 * devicePixelRatio
              ctx.setLineDash((m.dash ?? []).map((d) => d * devicePixelRatio))
              ctx.beginPath()
              ctx.moveTo(px, u.bbox.top)
              ctx.lineTo(px, u.bbox.top + u.bbox.height)
              ctx.stroke()
              ctx.restore()
            }
          },
        ],
      },
    }
    const u = new uPlot(
      opts,
      [x, ...series.map((s) => s.values)] as uPlot.AlignedData,
      el
    )
    plot.current = u
    const over = u.over
    const click = () => {
      const idx = u.cursor.idx
      if (idx != null && pickRef.current)
        pickRef.current(u.data[0][idx] as number)
    }
    over.addEventListener('click', click)
    const ro = new ResizeObserver(() =>
      u.setSize({ width: el.clientWidth, height })
    )
    ro.observe(el)
    return () => {
      ro.disconnect()
      over.removeEventListener('click', click)
      u.destroy()
      plot.current = null
    }
    // recreate only when the series layout changes; data updates go through setData below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shape, syncKey])

  useEffect(() => {
    plot.current?.setData([
      x,
      ...series.map((s) => s.values),
    ] as uPlot.AlignedData)
  }, [x, series])

  useEffect(() => {
    plot.current?.redraw(false)
  }, [markers])

  return <div ref={box} className={className ?? 'w-full'} />
}
