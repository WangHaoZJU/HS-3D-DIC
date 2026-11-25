import { Suspense, lazy, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Gauge, Play, ShieldCheck } from 'lucide-react'
import { toast } from 'sonner'
import { api, qk } from '@/lib/api'
import { viridisCss } from '@/lib/colormap'
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
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { TimeChart } from '@/components/time-chart'
import { Metric } from '../import/metric'

const FieldMesh = lazy(() => import('@/components/field-mesh'))
const COMP = ['c', 's', 'n']
const COLORS = ['#f87171', '#34d399', '#60a5fa']

export function QualityPage({ name }: { name: string }) {
  const qc = useQueryClient()
  const status = useQuery({
    queryKey: qk.status(name),
    queryFn: () => api.status(name),
  })
  const busy = !!status.data?.busy
  const q = useQuery({
    queryKey: [...qk.results(name), 'quality'],
    queryFn: () => api.quality(name),
    enabled: !busy,
    retry: false,
    meta: { silent: true },
  })
  const ref = useQuery({
    queryKey: qk.reference(name),
    queryFn: () => api.reference(name),
    meta: { silent: true },
    retry: false,
  })
  const [useRbm, setUseRbm] = useState(false)
  const [sigma, setSigma] = useState(0.01)
  const [mc, setMc] = useState(0)
  const submit = useMutation({
    mutationFn: ({
      stage,
      params,
    }: {
      stage: string
      params: Record<string, unknown>
    }) => api.submit(name, stage, params),
    onSuccess: () => {
      toast.success('已提交')
      qc.invalidateQueries({ queryKey: qk.jobs })
    },
  })
  const reconstructed = status.data?.stages.reconstruct.state === 'done'
  const stereo = status.data?.stages.stereo.state === 'done'
  return (
    <div className='space-y-6'>
      <Card>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <ShieldCheck className='size-4 text-brand' /> 静态噪声底（E1）
          </CardTitle>
          <CardDescription>
            叶片静止、与正式试验相同的帧率、曝光和照明下采集一段序列，走完整个流程后在这里计算逐点、逐方向的噪声标准差和噪声谱。这是工程上的不确定度依据（技术设计
            §10.2）；振动分析页可引用本实验作为显著性门限。
          </CardDescription>
          <CardAction className='flex items-center gap-3'>
            <label className='flex items-center gap-2 text-sm'>
              <Switch checked={useRbm} onCheckedChange={setUseRbm} /> 刚体补偿后
            </label>
            <Button
              size='sm'
              disabled={busy || !reconstructed || submit.isPending}
              onClick={() =>
                submit.mutate({
                  stage: 'noise_floor',
                  params: { use_rbm: useRbm },
                })
              }
            >
              <Play /> 计算噪声底
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent>
          {q.data?.noise_floor ? (
            <NoiseFloor nf={q.data.noise_floor} r={ref.data} />
          ) : (
            <p className='text-sm text-muted-foreground'>
              {reconstructed
                ? '尚未计算。只对静态采集有意义。'
                : '需要先完成三维重建。'}
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <Gauge className='size-4 text-brand' /> 三角化不确定度
          </CardTitle>
          <CardDescription>
            把二维匹配噪声
            σp（px）经三角化一阶传播到三维（可选蒙特卡洛校验）。只含像素观测噪声，不含标定、同步和运动模糊等系统误差；σp
            宜取 E1 实测值。
          </CardDescription>
          <CardAction className='flex items-end gap-2'>
            <div className='space-y-1'>
              <Label className='text-xs text-muted-foreground'>σp (px)</Label>
              <Input
                className='h-8 w-20 font-mono'
                value={sigma}
                onChange={(e) => setSigma(Number(e.target.value))}
              />
            </div>
            <div className='space-y-1'>
              <Label className='text-xs text-muted-foreground'>蒙特卡洛</Label>
              <Input
                className='h-8 w-20 font-mono'
                value={mc}
                onChange={(e) => setMc(Number(e.target.value))}
              />
            </div>
            <Button
              size='sm'
              disabled={busy || !stereo || submit.isPending || !(sigma > 0)}
              onClick={() =>
                submit.mutate({
                  stage: 'uncertainty',
                  params: { pixel_sigma: sigma, monte_carlo: mc },
                })
              }
            >
              <Play /> 计算
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent>
          {q.data?.uncertainty.length ? (
            <Uncertainty u={q.data.uncertainty} r={ref.data} />
          ) : (
            <p className='text-sm text-muted-foreground'>尚未计算。</p>
          )}
        </CardContent>
      </Card>
      {q.data?.valid_fraction != null && (
        <div className='grid grid-cols-2 gap-2 sm:grid-cols-4'>
          <Metric
            label='有效测点（时间平均）'
            value={`${(q.data.valid_fraction * 100).toFixed(2)}%`}
            verdict={q.data.valid_fraction >= 0.95 ? 'ok' : 'warn'}
            rule='共同可见区有效 POI ≥ 95%'
          />
          <Metric label='帧数' value={q.data.n_frames ?? '—'} />
        </div>
      )}
    </div>
  )
}

function NoiseFloor({
  nf,
  r,
}: {
  nf: NonNullable<Awaited<ReturnType<typeof api.quality>>['noise_floor']>
  r?: Awaited<ReturnType<typeof api.reference>>
}) {
  const [comp, setComp] = useState(2)
  const values = useMemo(() => nf.std.map((s) => s?.[comp] ?? null), [nf, comp])
  const max = Math.max(1e-12, ...values.map((v) => v ?? 0))
  const x = nf.spectrum.frequencies.map((f) => f ?? 0).slice(1)
  return (
    <div className='space-y-4'>
      <div className='grid grid-cols-3 gap-2 lg:grid-cols-6'>
        {COMP.map((c, i) => (
          <Metric
            key={c}
            label={`${c} 中位数 σ`}
            value={(nf.median_std_mm[i] * 1e3).toFixed(3)}
            unit='µm'
          />
        ))}
        {COMP.map((c, i) => (
          <Metric
            key={c + 'p'}
            label={`${c} 95% 分位 σ`}
            value={(nf.p95_std_mm[i] * 1e3).toFixed(3)}
            unit='µm'
          />
        ))}
      </div>
      <div className='grid gap-4 xl:grid-cols-2'>
        <div className='space-y-2'>
          <div className='flex items-center gap-2 text-sm'>
            逐点标准差
            <div className='ms-auto flex rounded-lg border p-0.5'>
              {COMP.map((c, i) => (
                <button
                  key={c}
                  onClick={() => setComp(i)}
                  className={cn(
                    'rounded-md px-2.5 py-0.5',
                    comp === i
                      ? 'bg-brand/15 text-brand'
                      : 'text-muted-foreground'
                  )}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
          <div className='h-80 overflow-hidden rounded-xl border'>
            {r ? (
              <Suspense fallback={<Skeleton className='h-full' />}>
                <FieldMesh
                  points3d={r.points_3d}
                  connectivity={r.connectivity}
                  values={values}
                  range={max}
                  signed={false}
                />
              </Suspense>
            ) : (
              <Skeleton className='h-full' />
            )}
          </div>
          <div className='flex items-center gap-2 font-mono text-xs text-muted-foreground'>
            <span>0</span>
            <span
              className='h-2 w-48 rounded-full'
              style={{
                background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map(viridisCss).join(',')})`,
              }}
            />
            <span>{(max * 1e3).toPrecision(3)} µm</span>
          </div>
        </div>
        <div className='space-y-2'>
          <div className='text-sm'>
            噪声谱（全场平均）· 检查照明频闪、环境振动等谱线
          </div>
          <TimeChart
            x={x}
            height={320}
            logY
            xLabel='频率 Hz'
            yLabel='幅值 µm'
            series={nf.spectrum.amplitude.map((a, c) => ({
              label: COMP[c],
              color: COLORS[c],
              values: a
                .slice(1)
                .map((y) => (y == null || y <= 0 ? null : y * 1e3)),
            }))}
          />
        </div>
      </div>
    </div>
  )
}

function Uncertainty({
  u,
  r,
}: {
  u: Awaited<ReturnType<typeof api.quality>>['uncertainty']
  r?: Awaited<ReturnType<typeof api.reference>>
}) {
  const item = u[0]
  const values = useMemo(() => item.std.map((s) => s?.[2] ?? null), [item])
  const max = Math.max(1e-12, ...values.map((v) => v ?? 0))
  return (
    <div className='grid gap-4 xl:grid-cols-2'>
      <div className='space-y-2'>
        {u.map((it) => (
          <div key={it.key} className='grid grid-cols-3 gap-2'>
            {['x', 'y', 'z'].map((c, i) => (
              <Metric
                key={c}
                label={`${it.key} · σ${c}（σp=${it.pixel_sigma}）`}
                value={(it.median_std_mm[i] * 1e3).toFixed(3)}
                unit='µm'
                rule={
                  it.monte_carlo_median_std_mm
                    ? `蒙特卡洛 ${it.monte_carlo} 次：${(it.monte_carlo_median_std_mm[i] * 1e3).toFixed(3)} µm`
                    : undefined
                }
              />
            ))}
          </div>
        ))}
        <p className='text-xs text-muted-foreground'>
          z 为左相机光轴方向（深度），通常比面内大 1/tan(α/2) 倍左右（技术设计
          §3.4）。
        </p>
      </div>
      <div className='space-y-2'>
        <div className='text-sm'>深度方向 σz（参考形貌上）</div>
        <div className='h-72 overflow-hidden rounded-xl border'>
          {r ? (
            <Suspense fallback={<Skeleton className='h-full' />}>
              <FieldMesh
                points3d={r.points_3d}
                connectivity={r.connectivity}
                values={values}
                range={max}
                signed={false}
              />
            </Suspense>
          ) : (
            <Skeleton className='h-full' />
          )}
        </div>
      </div>
    </div>
  )
}
