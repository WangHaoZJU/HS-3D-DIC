import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ChevronFirst,
  ChevronLast,
  ChevronLeft,
  ChevronRight,
  Maximize,
  Pause,
  Play,
} from 'lucide-react'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'

type View = { scale: number; tx: number; ty: number }
const FIT: View = { scale: 1, tx: 0, ty: 0 }

export type WindowMode = 'auto' | 'full'

/** Left/right frames side by side with linked zoom/pan, frame stepping and playback. */
export function FramePairViewer({
  name,
  nFrames,
  frame,
  onFrame,
  fullScale,
  names,
  referenceFrame,
}: {
  name: string
  nFrames: number
  frame: number
  onFrame: (k: number) => void
  fullScale: { left?: number; right?: number }
  names?: { left?: string; right?: string }
  referenceFrame: number
}) {
  const [view, setView] = useState<View>(FIT)
  const [windowMode, setWindowMode] = useState<WindowMode>('auto')
  const [saturation, setSaturation] = useState(true)
  const [playing, setPlaying] = useState(false)
  const [fps, setFps] = useState(10)
  const loaded = useRef({ left: false, right: false })
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const step = useCallback(
    (d: number) => onFrame(Math.max(0, Math.min(nFrames - 1, frame + d))),
    [frame, nFrames, onFrame]
  )

  // playback: advance only after both images of the current frame have loaded
  const onLoaded = (cam: 'left' | 'right') => {
    loaded.current[cam] = true
    if (playing && loaded.current.left && loaded.current.right) {
      clearTimeout(timer.current)
      timer.current = setTimeout(() => {
        if (frame >= nFrames - 1) setPlaying(false)
        else onFrame(frame + 1)
      }, 1000 / fps)
    }
  }
  useEffect(() => {
    loaded.current = { left: false, right: false }
  }, [frame])
  useEffect(() => () => clearTimeout(timer.current), [])

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (
        t.tagName === 'INPUT' ||
        t.tagName === 'TEXTAREA' ||
        t.isContentEditable
      )
        return
      if (e.key === 'ArrowRight') step(e.shiftKey ? 10 : 1)
      else if (e.key === 'ArrowLeft') step(e.shiftKey ? -10 : -1)
      else if (e.key === ' ') {
        e.preventDefault()
        setPlaying((p) => !p)
      } else return
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [step])

  const opts = (cam: 'left' | 'right') => ({
    ...(windowMode === 'full' ? { lo: 0, hi: fullScale[cam] ?? 65535 } : {}),
    ...(saturation ? { overlay: 'saturation' as const } : {}),
  })

  return (
    <div className='space-y-3'>
      <div className='grid gap-3 lg:grid-cols-2'>
        {(['left', 'right'] as const).map((cam) => (
          <ZoomPane
            key={cam}
            src={api.frameUrl(name, cam, frame, opts(cam))}
            title={cam === 'left' ? '左相机' : '右相机'}
            caption={names?.[cam]}
            view={view}
            setView={setView}
            onLoad={() => onLoaded(cam)}
          />
        ))}
      </div>
      <div className='flex flex-wrap items-center gap-2'>
        <div className='flex items-center'>
          <Button
            size='icon'
            variant='ghost'
            className='size-8'
            onClick={() => onFrame(0)}
            aria-label='第一帧'
          >
            <ChevronFirst />
          </Button>
          <Button
            size='icon'
            variant='ghost'
            className='size-8'
            onClick={() => step(-1)}
            aria-label='上一帧'
          >
            <ChevronLeft />
          </Button>
          <Button
            size='icon'
            variant='outline'
            className='size-8'
            onClick={() => setPlaying((p) => !p)}
            aria-label='播放'
          >
            {playing ? <Pause /> : <Play />}
          </Button>
          <Button
            size='icon'
            variant='ghost'
            className='size-8'
            onClick={() => step(1)}
            aria-label='下一帧'
          >
            <ChevronRight />
          </Button>
          <Button
            size='icon'
            variant='ghost'
            className='size-8'
            onClick={() => onFrame(nFrames - 1)}
            aria-label='最后一帧'
          >
            <ChevronLast />
          </Button>
        </div>
        <div className='relative min-w-48 flex-1'>
          <input
            type='range'
            min={0}
            max={Math.max(0, nFrames - 1)}
            value={frame}
            onChange={(e) => onFrame(Number(e.target.value))}
            className='w-full accent-[var(--brand)]'
            aria-label='帧'
          />
          {nFrames > 1 && (
            <span
              className='pointer-events-none absolute -top-1 h-2 w-0.5 -translate-x-1/2 rounded bg-brand'
              style={{ left: `${(referenceFrame / (nFrames - 1)) * 100}%` }}
              title='参考帧'
            />
          )}
        </div>
        <span className='tabular w-28 text-end font-mono text-sm'>
          {frame} / {Math.max(0, nFrames - 1)}
        </span>
        <Select value={String(fps)} onValueChange={(v) => setFps(Number(v))}>
          <SelectTrigger size='sm' className='w-24'>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {[2, 5, 10, 25].map((f) => (
              <SelectItem key={f} value={String(f)}>
                {f} 帧/s
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select
          value={windowMode}
          onValueChange={(v) => setWindowMode(v as WindowMode)}
        >
          <SelectTrigger size='sm' className='w-32'>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value='auto'>自动对比度</SelectItem>
            <SelectItem value='full'>全量程</SelectItem>
          </SelectContent>
        </Select>
        <div className='flex items-center gap-2'>
          <Switch
            id='sat'
            checked={saturation}
            onCheckedChange={setSaturation}
          />
          <Label htmlFor='sat' className='text-sm'>
            饱和标红
          </Label>
        </div>
        <Button size='sm' variant='ghost' onClick={() => setView(FIT)}>
          <Maximize /> 适应窗口
        </Button>
      </div>
      <p className='text-xs text-muted-foreground'>
        滚轮缩放、拖动平移，左右图联动；← / → 逐帧，Shift 跳 10 帧，空格播放。
      </p>
    </div>
  )
}

function ZoomPane({
  src,
  title,
  caption,
  view,
  setView,
  onLoad,
}: {
  src: string
  title: string
  caption?: string
  view: View
  setView: (v: View | ((v: View) => View)) => void
  onLoad: () => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(
    null
  )
  // keep the previous frame on screen until the next one has decoded (no flicker):
  // a hidden <img> loads `src`, the visible one shows the last decoded frame
  const [shown, setShown] = useState(src)
  const loading = shown !== src

  useEffect(() => {
    const el = box.current
    if (!el) return
    const wheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = el.getBoundingClientRect()
      const cx = e.clientX - r.left
      const cy = e.clientY - r.top
      setView((v) => {
        const scale = Math.min(
          40,
          Math.max(1, v.scale * Math.exp(-e.deltaY * 0.0015))
        )
        const k = scale / v.scale
        return { scale, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k }
      })
    }
    el.addEventListener('wheel', wheel, { passive: false })
    return () => el.removeEventListener('wheel', wheel)
  }, [setView])

  return (
    <div className='overflow-hidden rounded-xl border bg-black'>
      <div className='flex items-center justify-between border-b border-white/10 px-3 py-1.5 text-xs text-white/80'>
        <span className='font-medium'>{title}</span>
        <span className='truncate font-mono text-white/60'>
          {loading && <span className='me-2 text-white/40'>加载中…</span>}
          {caption}
        </span>
      </div>
      <div
        ref={box}
        className={cn(
          'relative aspect-[5/4] cursor-grab touch-none select-none active:cursor-grabbing'
        )}
        onPointerDown={(e) => {
          ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
          drag.current = {
            x: e.clientX,
            y: e.clientY,
            tx: view.tx,
            ty: view.ty,
          }
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (!d) return
          setView((v) => ({
            ...v,
            tx: d.tx + e.clientX - d.x,
            ty: d.ty + e.clientY - d.y,
          }))
        }}
        onPointerUp={() => (drag.current = null)}
        onDoubleClick={() => setView(FIT)}
      >
        {loading && (
          <img
            src={src}
            alt=''
            className='hidden'
            onLoad={() => {
              setShown(src)
              onLoad()
            }}
          />
        )}
        <img
          src={shown}
          alt={title}
          draggable={false}
          onLoad={shown === src ? onLoad : undefined}
          className='absolute inset-0 h-full w-full origin-top-left object-contain'
          style={{
            transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`,
            imageRendering: view.scale > 2.5 ? 'pixelated' : 'auto',
          }}
        />
        {view.scale > 1.01 && (
          <span className='absolute end-2 bottom-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-white/80'>
            ×{view.scale.toFixed(1)}
          </span>
        )}
      </div>
    </div>
  )
}
