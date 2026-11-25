import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type Konva from 'konva'
import {
  Hand,
  Maximize,
  MinusSquare,
  PlusSquare,
  RotateCcw,
  Save,
  Square,
  Trash2,
  Anchor,
} from 'lucide-react'
import { Image as KImage, Layer, Line, Rect, Shape, Stage } from 'react-konva'
import { toast } from 'sonner'
import { api, qk, type MaskShape, type MasksInfo } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'

type Tool = 'pan' | 'roi' | 'poi-add' | 'poi-sub' | 'rbm'
type Kind = 'poi' | 'rbm'
type Pt = [number, number]
type View = { zoom: number; x: number; y: number }

const COLORS = {
  add: { stroke: '#34d399', fill: 'rgba(52,211,153,0.16)' },
  subtract: { stroke: '#f87171', fill: 'rgba(248,113,113,0.22)' },
  rbm: { stroke: '#60a5fa', fill: 'rgba(96,165,250,0.22)' },
  roi: '#facc15',
  poi: '#5fd0e6',
}

const TOOLS: { key: Tool; label: string; icon: typeof Hand; hint: string }[] = [
  { key: 'pan', label: '平移', icon: Hand, hint: '拖动平移，滚轮缩放' },
  {
    key: 'roi',
    label: 'ROI',
    icon: Square,
    hint: '拖出矩形测量区域（再次拖动替换）',
  },
  {
    key: 'poi-add',
    label: '加入区域',
    icon: PlusSquare,
    hint: '多边形：单击加点，双击 / Enter / 点回起点闭合，Esc 取消，Backspace 撤销上一点',
  },
  {
    key: 'poi-sub',
    label: '扣除区域',
    icon: MinusSquare,
    hint: '多边形：从测量区域中扣除（反光、遮挡、叶缘）',
  },
  {
    key: 'rbm',
    label: '刚体参考区',
    icon: Anchor,
    hint: '多边形：夹具上的静止散斑块，用于刚体运动补偿',
  },
]

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
  return v
}

function useHtmlImage(src: string): HTMLImageElement | null {
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  useEffect(() => {
    const i = new window.Image()
    i.onload = () => setImg(i)
    i.src = src
    return () => {
      i.onload = null
    }
  }, [src])
  return img
}

export function MaskEditor({ name, info }: { name: string; info: MasksInfo }) {
  const qc = useQueryClient()
  const [h, w] = info.image_shape
  const img = useHtmlImage(api.frameUrl(name, 'left', info.reference_frame))
  const [roi, setRoi] = useState<number[] | null>(info.roi)
  const [shapes, setShapes] = useState<Record<Kind, MaskShape[]>>({
    poi: info.poi.shapes,
    rbm: info.rbm.shapes,
  })
  const [subset, setSubset] = useState(info.subset_size)
  const [step, setStep] = useState(info.step)
  const [tool, setTool] = useState<Tool>('poi-add')
  const [draft, setDraft] = useState<Pt[]>([])
  const [roiDrag, setRoiDrag] = useState<Pt | null>(null)
  const [hover, setHover] = useState<Pt | null>(null)
  const [selected, setSelected] = useState<{ kind: Kind; i: number } | null>(
    null
  )
  const [showPois, setShowPois] = useState(true)
  const [view, setView] = useState<View>({ zoom: 1, x: 0, y: 0 })
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(900)
  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const fit = Math.min(width / w, (window.innerHeight * 0.72) / h)
  const height = h * fit
  const scale = fit * view.zoom

  // live POI preview of the unsaved state
  const previewReq = useDebounced(
    JSON.stringify({ roi, poi: shapes.poi, subset_size: subset, step }),
    350
  )
  const preview = useQuery({
    queryKey: ['poi-preview', name, previewReq],
    queryFn: () => api.poiPreview(name, JSON.parse(previewReq)),
    placeholderData: (p) => p,
    meta: { silent: true },
  })

  const initial = JSON.stringify({
    roi: info.roi,
    poi: info.poi.shapes,
    rbm: info.rbm.shapes,
    subset: info.subset_size,
    step: info.step,
  })
  const dirty =
    JSON.stringify({ roi, poi: shapes.poi, rbm: shapes.rbm, subset, step }) !==
    initial
  const save = useMutation({
    mutationFn: () =>
      api.saveMasks(name, {
        roi,
        poi: shapes.poi,
        rbm: shapes.rbm,
        subset_size: subset,
        step,
      }),
    onSuccess: () => {
      toast.success('区域与掩膜已保存', {
        description: '立体匹配及其后的阶段已过期，需要重新运行',
      })
      for (const k of [qk.masks(name), qk.experiment(name), qk.status(name)])
        qc.invalidateQueries({ queryKey: k })
    },
  })

  const toImage = (stage: Konva.Stage): Pt | null => {
    const p = stage.getPointerPosition()
    if (!p) return null
    return [(p.x - view.x) / scale, (p.y - view.y) / scale]
  }
  const finishPolygon = (pts: Pt[] = draft) => {
    // drop consecutive duplicates (a closing double-click adds the same point twice)
    const clean = pts.filter(
      (p, i) =>
        i === 0 ||
        Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) * scale >= 4
    )
    if (clean.length < 3) return // keep drawing; Esc cancels
    const kind: Kind = tool === 'rbm' ? 'rbm' : 'poi'
    const op = tool === 'poi-sub' ? 'subtract' : 'add'
    setShapes((s) => ({
      ...s,
      [kind]: [
        ...s[kind],
        { op, points: clean.map(([x, y]) => [round(x), round(y)]) as Pt[] },
      ],
    }))
    setDraft([])
  }

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA') return
      if (e.key === 'Escape') setDraft([])
      else if (e.key === 'Enter') finishPolygon()
      else if (e.key === 'Backspace' && draft.length)
        setDraft((d) => d.slice(0, -1))
      else if ((e.key === 'Delete' || e.key === 'Backspace') && selected) {
        setShapes((s) => ({
          ...s,
          [selected.kind]: s[selected.kind].filter((_, i) => i !== selected.i),
        }))
        setSelected(null)
      } else return
      e.preventDefault()
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  })

  const pts = preview.data?.points ?? []
  const sw = 1.5 / scale
  const polygonTool = tool === 'poi-add' || tool === 'poi-sub' || tool === 'rbm'
  const draftColor =
    tool === 'rbm'
      ? COLORS.rbm.stroke
      : tool === 'poi-sub'
        ? COLORS.subtract.stroke
        : COLORS.add.stroke

  return (
    <div className='space-y-3'>
      <div className='flex flex-wrap items-center gap-2'>
        <div className='flex rounded-lg border p-0.5'>
          {TOOLS.map((t) => (
            <Tooltip key={t.key}>
              <TooltipTrigger asChild>
                <button
                  onClick={() => {
                    setTool(t.key)
                    setDraft([])
                  }}
                  className={cn(
                    'flex items-center gap-1.5 rounded-md px-2.5 py-1 text-sm',
                    tool === t.key
                      ? 'bg-brand/15 text-brand'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                >
                  <t.icon className='size-4' /> {t.label}
                </button>
              </TooltipTrigger>
              <TooltipContent className='max-w-64'>{t.hint}</TooltipContent>
            </Tooltip>
          ))}
        </div>
        <Button
          size='sm'
          variant='ghost'
          onClick={() => setView({ zoom: 1, x: 0, y: 0 })}
        >
          <Maximize /> 适应
        </Button>
        <div className='ms-auto flex flex-wrap items-center gap-3'>
          <div className='flex items-center gap-1.5'>
            <Label className='text-xs text-muted-foreground'>子区</Label>
            <Input
              className='h-8 w-16 font-mono'
              value={subset}
              onChange={(e) => setSubset(Number(e.target.value) || 0)}
            />
          </div>
          <div className='flex items-center gap-1.5'>
            <Label className='text-xs text-muted-foreground'>步长</Label>
            <Input
              className='h-8 w-16 font-mono'
              value={step}
              onChange={(e) => setStep(Number(e.target.value) || 0)}
            />
          </div>
          <div className='flex items-center gap-1.5'>
            <Switch
              id='pois'
              checked={showPois}
              onCheckedChange={setShowPois}
            />
            <Label htmlFor='pois' className='text-sm'>
              测点
            </Label>
          </div>
          <span
            className={cn(
              'tabular font-mono text-sm',
              preview.data?.error ? 'text-bad' : 'text-brand'
            )}
          >
            {preview.data?.error
              ? preview.data.error
              : `${(preview.data?.n ?? 0).toLocaleString()} 个测点`}
          </span>
        </div>
      </div>

      <div className='grid gap-3 xl:grid-cols-[minmax(0,1fr)_260px]'>
        <div
          ref={box}
          className='flex justify-center overflow-hidden rounded-xl border bg-black'
        >
          <Stage
            width={width}
            height={height}
            draggable={tool === 'pan'}
            x={view.x}
            y={view.y}
            scaleX={scale}
            scaleY={scale}
            onDragEnd={(e) =>
              e.target === e.target.getStage() &&
              setView((v) => ({ ...v, x: e.target.x(), y: e.target.y() }))
            }
            onWheel={(e) => {
              e.evt.preventDefault()
              const stage = e.target.getStage()!
              const p = stage.getPointerPosition()!
              const zoom = Math.min(
                30,
                Math.max(1, view.zoom * Math.exp(-e.evt.deltaY * 0.0015))
              )
              const ns = fit * zoom
              const ix = (p.x - view.x) / scale
              const iy = (p.y - view.y) / scale
              setView({ zoom, x: p.x - ix * ns, y: p.y - iy * ns })
            }}
            onMouseDown={(e) => {
              if (tool !== 'roi') return
              const p = toImage(e.target.getStage()!)
              if (p) setRoiDrag(p)
            }}
            onMouseMove={(e) => {
              const p = toImage(e.target.getStage()!)
              setHover(p)
              if (roiDrag && p) setRoi(rectFrom(roiDrag, p, w, h))
            }}
            onMouseUp={() => setRoiDrag(null)}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              if (!polygonTool) return
              const p = toImage(e.target.getStage()!)
              if (!p) return
              if (draft.length >= 3) {
                const [x0, y0] = draft[0]
                if (Math.hypot(p[0] - x0, p[1] - y0) * scale < 8)
                  return finishPolygon()
              }
              setDraft((d) => [...d, p])
            }}
            onDblClick={() => {
              // only a double-click on one spot closes the polygon (two quick clicks elsewhere just add points)
              const n = draft.length
              if (!polygonTool || n < 2) return
              const [a, b] = [draft[n - 1], draft[n - 2]]
              if (Math.hypot(a[0] - b[0], a[1] - b[1]) * scale < 6)
                finishPolygon()
            }}
            style={{ cursor: tool === 'pan' ? 'grab' : 'crosshair' }}
          >
            <Layer listening={false}>
              {img && <KImage image={img} width={w} height={h} />}
            </Layer>
            <Layer>
              {(['poi', 'rbm'] as const).map((kind) =>
                shapes[kind].map((s, i) => {
                  const c = kind === 'rbm' ? COLORS.rbm : COLORS[s.op]
                  const sel = selected?.kind === kind && selected.i === i
                  return (
                    <Line
                      key={`${kind}-${i}`}
                      points={s.points.flat()}
                      closed
                      stroke={c.stroke}
                      strokeWidth={sel ? sw * 2.5 : sw}
                      fill={c.fill}
                      dash={kind === 'rbm' ? [6 / scale, 4 / scale] : undefined}
                      onClick={(e) => {
                        if (tool !== 'pan') return
                        e.cancelBubble = true
                        setSelected({ kind, i })
                      }}
                    />
                  )
                })
              )}
              {roi && (
                <Rect
                  x={roi[0]}
                  y={roi[1]}
                  width={roi[2]}
                  height={roi[3]}
                  stroke={COLORS.roi}
                  strokeWidth={sw}
                  dash={[8 / scale, 5 / scale]}
                  listening={false}
                />
              )}
            </Layer>
            <Layer listening={false}>
              {showPois && pts.length > 0 && (
                <Shape
                  sceneFunc={(ctx) => {
                    const r = Math.max(0.6, 1.6 / scale)
                    ctx.beginPath()
                    for (let k = 0; k < pts.length; k += 2)
                      ctx.rect(pts[k] - r, pts[k + 1] - r, 2 * r, 2 * r)
                    ctx.fillStyle = COLORS.poi
                    ctx.fill()
                  }}
                />
              )}
              {hover && showPois && subset > 0 && (
                <Rect
                  x={Math.round(hover[0]) - subset / 2}
                  y={Math.round(hover[1]) - subset / 2}
                  width={subset}
                  height={subset}
                  stroke='rgba(255,255,255,0.7)'
                  strokeWidth={sw}
                />
              )}
              {draft.length > 0 && (
                <Line
                  points={[...draft.flat(), ...(hover ?? [])]}
                  stroke={draftColor}
                  strokeWidth={sw * 1.4}
                  dash={[4 / scale, 3 / scale]}
                />
              )}
              {draft.map(([x, y], i) => (
                <Rect
                  key={i}
                  x={x - 3 / scale}
                  y={y - 3 / scale}
                  width={6 / scale}
                  height={6 / scale}
                  fill={i === 0 ? '#fff' : draftColor}
                />
              ))}
            </Layer>
          </Stage>
        </div>

        <div className='space-y-3 text-sm'>
          <ShapeList
            title='测量区域'
            shapes={shapes.poi}
            kind='poi'
            selected={selected}
            onSelect={setSelected}
            onDelete={(i) =>
              setShapes((s) => ({ ...s, poi: s.poi.filter((_, k) => k !== i) }))
            }
            empty='未绘制：整幅图（或 ROI）都布点'
          />
          <ShapeList
            title='刚体参考区'
            shapes={shapes.rbm}
            kind='rbm'
            selected={selected}
            onSelect={setSelected}
            onDelete={(i) =>
              setShapes((s) => ({ ...s, rbm: s.rbm.filter((_, k) => k !== i) }))
            }
            empty='未绘制：用全部有效点估计刚体运动'
          />
          <div className='rounded-lg border p-3'>
            <div className='mb-1 flex items-center justify-between font-medium'>
              ROI
              {roi && (
                <button
                  className='text-xs text-muted-foreground hover:text-bad'
                  onClick={() => setRoi(null)}
                >
                  清除
                </button>
              )}
            </div>
            <div className='font-mono text-xs text-muted-foreground'>
              {roi ? `x ${roi[0]}  y ${roi[1]}  ${roi[2]}×${roi[3]}` : '整幅图'}
            </div>
          </div>
          <p className='text-xs text-muted-foreground'>
            测点只布在整个子区都落在区域内的位置，叶缘会留下约半个子区宽的空带。
            {hover && `光标 (${hover[0].toFixed(0)}, ${hover[1].toFixed(0)})`}
          </p>
          <div className='flex gap-2'>
            <Button
              className='flex-1'
              disabled={!dirty || save.isPending || !!preview.data?.error}
              onClick={() => save.mutate()}
            >
              <Save /> 保存
            </Button>
            <Button
              variant='outline'
              size='icon'
              disabled={!dirty}
              title='放弃修改'
              onClick={() => {
                setRoi(info.roi)
                setShapes({ poi: info.poi.shapes, rbm: info.rbm.shapes })
                setSubset(info.subset_size)
                setStep(info.step)
              }}
            >
              <RotateCcw />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ShapeList({
  title,
  shapes,
  kind,
  selected,
  onSelect,
  onDelete,
  empty,
}: {
  title: string
  shapes: MaskShape[]
  kind: Kind
  selected: { kind: Kind; i: number } | null
  onSelect: (s: { kind: Kind; i: number }) => void
  onDelete: (i: number) => void
  empty: string
}) {
  return (
    <div className='rounded-lg border p-3'>
      <div className='mb-1 font-medium'>{title}</div>
      {shapes.length === 0 && (
        <div className='text-xs text-muted-foreground'>{empty}</div>
      )}
      {shapes.map((s, i) => (
        <div
          key={i}
          className={cn(
            'flex items-center gap-2 rounded px-1 py-0.5 text-xs',
            selected?.kind === kind && selected.i === i && 'bg-accent'
          )}
          onClick={() => onSelect({ kind, i })}
        >
          <span
            className='size-2.5 rounded-sm'
            style={{
              background:
                kind === 'rbm' ? COLORS.rbm.stroke : COLORS[s.op].stroke,
            }}
          />
          <span>
            {kind === 'rbm' ? '参考区' : s.op === 'add' ? '加入' : '扣除'}
          </span>
          <span className='text-muted-foreground'>{s.points.length} 点</span>
          <button
            className='ms-auto text-muted-foreground hover:text-bad'
            onClick={() => onDelete(i)}
            aria-label='删除'
          >
            <Trash2 className='size-3.5' />
          </button>
        </div>
      ))}
    </div>
  )
}

function round(v: number): number {
  return Math.round(v * 10) / 10
}

function rectFrom(a: Pt, b: Pt, w: number, h: number): number[] {
  const x0 = Math.max(0, Math.round(Math.min(a[0], b[0])))
  const y0 = Math.max(0, Math.round(Math.min(a[1], b[1])))
  const x1 = Math.min(w, Math.round(Math.max(a[0], b[0])))
  const y1 = Math.min(h, Math.round(Math.max(a[1], b[1])))
  return [x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)]
}
