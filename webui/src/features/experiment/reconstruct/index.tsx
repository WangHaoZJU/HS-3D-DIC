import { Suspense, lazy, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Anchor, Move3d, Pause, Play, Waves } from 'lucide-react'
import { toast } from 'sonner'
import { api, qk } from '@/lib/api'
import { divergingCss, viridisCss } from '@/lib/colormap'
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
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { TimeChart } from '@/components/time-chart'
import { Metric } from '../import/metric'

const FieldMesh = lazy(() => import('@/components/field-mesh'))

type Comp = 'c' | 's' | 'n' | 'mag'
const COMP_LABEL: Record<Comp, string> = {
  c: '弦向 c',
  s: '展向 s',
  n: '法向 n',
  mag: '模长',
}

export function ReconstructPage({ name }: { name: string }) {
  const qc = useQueryClient()
  const status = useQuery({
    queryKey: qk.status(name),
    queryFn: () => api.status(name),
  })
  const busy = !!status.data?.busy
  const info = useQuery({
    queryKey: [...qk.results(name), 'disp-info'],
    queryFn: () => api.displacementInfo(name),
    enabled: !busy,
    retry: false,
    meta: { silent: true },
  })
  const submit = useMutation({
    mutationFn: () =>
      api.submit(name, 'pipeline', { stages: ['reconstruct', 'rbm'] }),
    onSuccess: () => {
      toast.success('已提交重建 + 刚体补偿')
      qc.invalidateQueries({ queryKey: qk.jobs })
    },
  })
  const st = status.data?.stages
  return (
    <div className='space-y-6'>
      <Card>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <Move3d className='size-4 text-brand' /> 三维重建与刚体补偿
          </CardTitle>
          <CardDescription>
            逐帧非线性三角化得到三维位移（相对参考帧），转到叶片局部坐标 (c, s,
            n)；再用刚体参考区估计并扣除支架/夹具的整体运动（技术设计 §9）。
          </CardDescription>
          <CardAction>
            <Button
              size='sm'
              disabled={busy || submit.isPending}
              onClick={() => submit.mutate()}
            >
              <Play /> 运行重建 + 补偿
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className='grid grid-cols-2 gap-2 sm:grid-cols-4'>
          <Metric
            label='重建'
            value={stateLabel(st?.reconstruct.state)}
            verdict={st?.reconstruct.state === 'done' ? 'ok' : 'warn'}
          />
          <Metric
            label='刚体补偿'
            value={stateLabel(st?.rbm.state)}
            verdict={st?.rbm.state === 'done' ? 'ok' : 'warn'}
          />
          <Metric
            label='帧数 × 测点'
            value={
              info.data
                ? `${info.data.n_frames} × ${info.data.n_points.toLocaleString()}`
                : '—'
            }
          />
          <Metric
            label='有效比例（抽样）'
            value={
              info.data?.valid_fraction != null
                ? `${(info.data.valid_fraction * 100).toFixed(1)}%`
                : '—'
            }
            verdict={
              info.data?.valid_fraction != null
                ? info.data.valid_fraction >= 0.95
                  ? 'ok'
                  : 'warn'
                : 'info'
            }
          />
        </CardContent>
      </Card>
      {busy && (
        <p className='text-sm text-muted-foreground'>
          该实验有任务正在写结果文件，完成后显示结果。
        </p>
      )}
      {!busy && info.isError && (
        <p className='py-8 text-center text-sm text-muted-foreground'>
          还没有重建结果，先完成时序 DIC。
        </p>
      )}
      {!busy && info.data && (
        <>
          <FieldViewer name={name} />
          {info.data.rbm && <RbmCard name={name} />}
        </>
      )}
    </div>
  )
}

function stateLabel(s?: string): string {
  return (
    (
      {
        done: '最新',
        stale: '过期',
        missing: '未运行',
        partial: '中断',
        running: '运行中',
        unknown: '未知',
      } as Record<string, string>
    )[s ?? 'unknown'] ?? '—'
  )
}

function FieldViewer({ name }: { name: string }) {
  const info = useQuery({
    queryKey: [...qk.results(name), 'disp-info'],
    queryFn: () => api.displacementInfo(name),
  })
  const ref = useQuery({
    queryKey: qk.reference(name),
    queryFn: () => api.reference(name),
    meta: { silent: true },
  })
  const kinds = info.data?.kinds ?? []
  const [kind, setKind] = useState<'local' | 'rbm_local'>('local')
  const [comp, setComp] = useState<Comp>('n')
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [gain, setGain] = useState(0)
  const [sel, setSel] = useState<number | null>(null)
  const T = info.data?.n_frames ?? 1
  const frame = useQuery({
    queryKey: [...qk.results(name), 'disp-frame', kind, t],
    queryFn: () => api.displacementFrame(name, t, kind),
    placeholderData: (p) => p,
    meta: { silent: true },
  })
  const warpKind = kind === 'rbm_local' ? 'rbm' : 'raw'
  const warp = useQuery({
    queryKey: [...qk.results(name), 'disp-frame', warpKind, t],
    queryFn: () => api.displacementFrame(name, t, warpKind),
    enabled: gain > 0,
    placeholderData: (p) => p,
    meta: { silent: true },
  })
  useEffect(() => {
    if (!playing) return
    if (frame.isFetching) return
    const id = setTimeout(() => setT((x) => (x + 1) % T), 80)
    return () => clearTimeout(id)
  }, [playing, frame.isFetching, frame.data, T])

  const values = useMemo(() => {
    const d = frame.data?.d
    if (!d) return null
    const ci = { c: 0, s: 1, n: 2 }[comp as 'c']
    return d.map((v) =>
      v[0] == null
        ? null
        : comp === 'mag'
          ? Math.hypot(v[0]!, v[1]!, v[2]!)
          : v[ci]
    )
  }, [frame.data, comp])
  const range = info.data?.ranges[kind]
    ? comp === 'mag'
      ? info.data.ranges[kind].mag
      : info.data.ranges[kind].abs[{ c: 0, s: 1, n: 2 }[comp as 'c']]
    : 1
  const signed = comp !== 'mag'
  const time = info.data?.timestamps[t]

  return (
    <div className='grid gap-4 2xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]'>
      <Card className='min-w-0'>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <Waves className='size-4 text-brand' /> 全场位移
          </CardTitle>
          <CardDescription>
            按所选分量着色（色标固定为全部帧抽样的 99%
            分位，便于逐帧比较）；“放大”把位移按倍数叠加到形貌上。点击网格选中测点。
          </CardDescription>
        </CardHeader>
        <CardContent className='space-y-3'>
          <div className='flex flex-wrap items-center gap-2 text-sm'>
            <Seg
              value={kind}
              onChange={(v) => setKind(v as 'local')}
              options={[
                ['local', '原始'],
                ...(kinds.includes('rbm_local')
                  ? [['rbm_local', '刚体补偿后'] as [string, string]]
                  : []),
              ]}
            />
            <Seg
              value={comp}
              onChange={(v) => setComp(v as Comp)}
              options={(Object.keys(COMP_LABEL) as Comp[]).map((k) => [
                k,
                COMP_LABEL[k],
              ])}
            />
            <div className='ms-auto flex items-center gap-2'>
              <Label className='text-xs text-muted-foreground'>放大</Label>
              <select
                className='h-8 rounded-md border bg-background px-2 font-mono text-xs'
                value={gain}
                onChange={(e) => setGain(Number(e.target.value))}
              >
                {[0, 10, 50, 100, 500, 1000].map((g) => (
                  <option key={g} value={g}>
                    {g ? `×${g}` : '关'}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className='h-[460px] overflow-hidden rounded-xl border'>
            {ref.data && (
              <Suspense fallback={<Skeleton className='h-full' />}>
                <FieldMesh
                  points3d={ref.data.points_3d}
                  connectivity={ref.data.connectivity}
                  values={values}
                  range={range}
                  signed={signed}
                  warp={gain > 0 ? warp.data?.d : null}
                  gain={gain}
                  selected={sel}
                  onPick={setSel}
                />
              </Suspense>
            )}
          </div>
          <div className='flex items-center gap-3'>
            <Button
              size='icon'
              variant='outline'
              className='size-8'
              onClick={() => setPlaying((p) => !p)}
              aria-label='播放'
            >
              {playing ? <Pause /> : <Play />}
            </Button>
            <input
              type='range'
              min={0}
              max={T - 1}
              value={t}
              onChange={(e) => setT(Number(e.target.value))}
              className='flex-1 accent-[var(--brand)]'
              aria-label='帧'
            />
            <span className='tabular w-44 text-end font-mono text-xs'>
              帧 {t} · {time != null ? `${(time * 1e3).toFixed(2)} ms` : ''}
            </span>
          </div>
          <ColorBar signed={signed} range={range} />
        </CardContent>
      </Card>
      <PointSeriesCard name={name} i={sel} />
    </div>
  )
}

function Seg({
  value,
  onChange,
  options,
}: {
  value: string
  onChange: (v: string) => void
  options: [string, string][]
}) {
  return (
    <div className='flex rounded-lg border p-0.5'>
      {options.map(([k, l]) => (
        <button
          key={k}
          onClick={() => onChange(k)}
          className={cn(
            'rounded-md px-2.5 py-1',
            value === k
              ? 'bg-brand/15 text-brand'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          {l}
        </button>
      ))}
    </div>
  )
}

function ColorBar({ signed, range }: { signed: boolean; range: number }) {
  const um = range * 1e3
  const stops = signed
    ? [-1, -0.5, 0, 0.5, 1].map(divergingCss)
    : [0, 0.25, 0.5, 0.75, 1].map(viridisCss)
  return (
    <div className='flex items-center gap-3 font-mono text-xs text-muted-foreground'>
      <span>{signed ? `−${um.toPrecision(3)}` : '0'} µm</span>
      <span
        className='h-2 w-64 rounded-full'
        style={{ background: `linear-gradient(90deg, ${stops.join(',')})` }}
      />
      <span>{um.toPrecision(3)} µm</span>
    </div>
  )
}

function PointSeriesCard({ name, i }: { name: string; i: number | null }) {
  const q = useQuery({
    queryKey: [...qk.results(name), 'disp-point', i],
    queryFn: () => api.displacementPoint(name, i!),
    enabled: i != null,
    meta: { silent: true },
  })
  const um = (v: (number | null)[]) =>
    v.map((x) => (x == null ? null : x * 1e3))
  return (
    <Card className='min-w-0'>
      <CardHeader>
        <CardTitle className='text-base'>单点三分量时间历程</CardTitle>
        <CardDescription>
          局部坐标 (c, s, n)，单位 µm；虚线为刚体补偿后。
        </CardDescription>
      </CardHeader>
      <CardContent className='space-y-3'>
        {i == null && (
          <p className='text-sm text-muted-foreground'>
            在左侧网格上点击一个测点
          </p>
        )}
        {q.data?.local && (
          <>
            <div className='text-sm'>
              测点 <span className='font-mono'>#{i}</span>
            </div>
            {(['n', 'c', 's'] as const).map((k) => (
              <TimeChart
                key={k}
                x={q.data!.local!.t.map((t, j) => t ?? j)}
                height={150}
                xLabel='时间 s'
                yLabel={`${k} µm`}
                syncKey={`pt-${name}`}
                series={[
                  {
                    label: `${k} 原始`,
                    color:
                      k === 'n' ? '#60a5fa' : k === 'c' ? '#f87171' : '#34d399',
                    values: um(q.data!.local![k]),
                  },
                  ...(q.data!.rbm_local &&
                  q.data!.rbm_local.t.length === q.data!.local!.t.length
                    ? [
                        {
                          label: `${k} 补偿后`,
                          color: '#e5e7eb',
                          values: um(q.data!.rbm_local[k]),
                          dash: [4, 3],
                          width: 1,
                        },
                      ]
                    : []),
                ]}
              />
            ))}
          </>
        )}
      </CardContent>
    </Card>
  )
}

function RbmCard({ name }: { name: string }) {
  const { data } = useQuery({
    queryKey: [...qk.results(name), 'rbm'],
    queryFn: () => api.rbm(name),
    meta: { silent: true },
  })
  if (!data) return <Skeleton className='h-64' />
  const x = data.t.map((t, j) => t ?? j)
  const um = (v: (number | null)[]) =>
    v.map((a) => (a == null ? null : a * 1e3))
  const mdeg = (v: (number | null)[]) =>
    v.map((a) => (a == null ? null : a * 1e3))
  return (
    <Card>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <Anchor className='size-4 text-brand' /> 刚体运动
        </CardTitle>
        <CardDescription>
          {data.reference_points != null
            ? `由刚体参考区内 ${data.reference_points} 个测点估计（加权 Kabsch）。`
            : '未配置刚体参考区：用全部有效测点估计，会把叶片自身的整体运动也当作刚体运动扣除。建议在参考帧页绘制夹具上的静止散斑块。'}
        </CardDescription>
      </CardHeader>
      <CardContent className='grid gap-4 xl:grid-cols-3'>
        <TimeChart
          x={x}
          height={180}
          xLabel='时间 s'
          yLabel='平移 µm'
          syncKey={`rbm-${name}`}
          series={[
            { label: 'tx', color: '#f87171', values: um(data.tx) },
            { label: 'ty', color: '#34d399', values: um(data.ty) },
            { label: 'tz', color: '#60a5fa', values: um(data.tz) },
          ]}
        />
        <TimeChart
          x={x}
          height={180}
          xLabel='时间 s'
          yLabel='转角 m°'
          syncKey={`rbm-${name}`}
          series={[
            { label: 'rx', color: '#f87171', values: mdeg(data.rx) },
            { label: 'ry', color: '#34d399', values: mdeg(data.ry) },
            { label: 'rz', color: '#60a5fa', values: mdeg(data.rz) },
          ]}
        />
        <TimeChart
          x={x}
          height={180}
          xLabel='时间 s'
          yLabel='拟合残差 µm'
          syncKey={`rbm-${name}`}
          series={[
            { label: 'RMS', color: '#f59e0b', values: um(data.residual_rms) },
          ]}
        />
      </CardContent>
    </Card>
  )
}
