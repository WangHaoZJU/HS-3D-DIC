import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type CalibrationViewDetail } from '@/lib/api'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'

const GAINS = [20, 50, 100, 200, 500]

/** One calibration view, both cameras: detected corners and residual vectors (magnified). */
export function ViewDetail({
  name,
  view,
  shape,
}: {
  name: string
  view: number
  shape: [number, number]
}) {
  const [gain, setGain] = useState(100)
  const { data, isLoading, isError } = useQuery({
    queryKey: ['calibration', name, 'view', view],
    queryFn: () => api.calibrationView(name, view),
    meta: { silent: true },
    retry: false,
  })
  return (
    <div className='space-y-3'>
      <div className='flex flex-wrap items-center gap-3 text-sm'>
        <span className='font-medium'>视图 #{view}</span>
        {data?.pair && (
          <span className='font-mono text-xs text-muted-foreground'>
            RMS 左 {data.pair.rms_left.toFixed(4)} / 右{' '}
            {data.pair.rms_right.toFixed(4)} px · {data.pair.ids.length}{' '}
            个共同角点
          </span>
        )}
        {data && !data.pair && (
          <span className='text-xs text-warn'>该视图未参与双目标定</span>
        )}
        <div className='ms-auto flex items-center gap-2'>
          <Label className='text-xs text-muted-foreground'>残差放大</Label>
          {GAINS.map((g) => (
            <button
              key={g}
              onClick={() => setGain(g)}
              className={`rounded border px-1.5 py-0.5 font-mono text-xs ${g === gain ? 'border-brand bg-brand/15 text-brand' : 'hover:bg-accent'}`}
            >
              ×{g}
            </button>
          ))}
        </div>
      </div>
      {isLoading && <Skeleton className='aspect-[5/2]' />}
      {isError && (
        <p className='text-sm text-muted-foreground'>没有该视图的检测结果。</p>
      )}
      {data && (
        <div className='grid gap-3 md:grid-cols-2'>
          {(['left', 'right'] as const).map((cam) => (
            <Overlay
              key={cam}
              name={name}
              cam={cam}
              data={data}
              shape={shape}
              gain={gain}
            />
          ))}
        </div>
      )}
      <p className='text-xs text-muted-foreground'>
        ○ 检测到的角点（亚像素精化后）；线段为重投影残差（重投影点 −
        检测点），按所选倍数放大。
      </p>
    </div>
  )
}

function Overlay({
  name,
  cam,
  data,
  shape,
  gain,
}: {
  name: string
  cam: 'left' | 'right'
  data: CalibrationViewDetail
  shape: [number, number]
  gain: number
}) {
  const [h, w] = shape
  const det = data[cam]
  const pair = data.pair?.[cam]
  const r = Math.max(w, h) / 110
  return (
    <div className='overflow-hidden rounded-xl border bg-black'>
      <div className='flex justify-between border-b border-white/10 px-3 py-1.5 text-xs text-white/80'>
        <span className='font-medium'>
          {cam === 'left' ? '左相机' : '右相机'}
        </span>
        <span className='font-mono text-white/60'>
          {data.names[cam === 'left' ? 0 : 1]} ·{' '}
          {det ? `${det.ids.length} 角点` : '未检测到'}
        </span>
      </div>
      <div className='relative' style={{ aspectRatio: `${w} / ${h}` }}>
        <img
          src={api.calibrationImageUrl(name, data.index, cam)}
          alt=''
          className='absolute inset-0 h-full w-full'
        />
        <svg
          viewBox={`-0.5 -0.5 ${w} ${h}`}
          className='absolute inset-0 h-full w-full'
        >
          {det?.points.map(([x, y], i) => (
            <circle
              key={i}
              cx={x}
              cy={y}
              r={r}
              fill='none'
              stroke='#34d399'
              strokeWidth={r / 3}
            />
          ))}
          {pair?.points.map(([x, y], i) => {
            const [dx, dy] = pair.residuals[i]
            return (
              <line
                key={i}
                x1={x}
                y1={y}
                x2={x + dx * gain}
                y2={y + dy * gain}
                stroke='#f472b6'
                strokeWidth={r / 2.5}
                strokeLinecap='round'
              />
            )
          })}
        </svg>
      </div>
    </div>
  )
}
