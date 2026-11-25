import { Suspense, lazy, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Activity, Download, Play, Radar, Sigma, Waves } from 'lucide-react'
import { toast } from 'sonner'
import {
  api,
  qk,
  type AnalyzeOptions,
  type Harmonic,
  type ReferenceResult,
  type Vibration,
} from '@/lib/api'
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
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { PathPicker } from '@/components/path-picker'
import { TimeChart } from '@/components/time-chart'
import { Metric } from '../import/metric'

const FieldMesh = lazy(() => import('@/components/field-mesh'))
const COMP = ['c 弦向', 's 展向', 'n 法向']
const COLORS = ['#f87171', '#34d399', '#60a5fa']

const DEFAULTS: AnalyzeOptions = {
  direction: 2,
  use_rbm: false,
  min_frequency: 0,
  max_frequency: null,
  max_peaks: 10,
  nperseg: null,
  frequencies: [],
  fit_peaks: 3,
  refine: true,
  n_harmonics: 1,
  fdd: true,
  n_modes: 3,
  force_file: null,
}

export function VibrationPage({ name }: { name: string }) {
  const qc = useQueryClient()
  const status = useQuery({
    queryKey: qk.status(name),
    queryFn: () => api.status(name),
  })
  const experiments = useQuery({
    queryKey: qk.experiments,
    queryFn: api.experiments,
  })
  const [noiseFrom, setNoiseFrom] = useState<string | null>(() => {
    try {
      return localStorage.getItem(`noise-from:${name}`)
    } catch {
      return null
    }
  })
  const busy = !!status.data?.busy
  const vib = useQuery({
    queryKey: [...qk.results(name), 'vibration', noiseFrom],
    queryFn: () => api.vibration(name, noiseFrom),
    enabled: !busy,
    retry: false,
    meta: { silent: true },
  })
  const submit = useMutation({
    mutationFn: (options: AnalyzeOptions) =>
      api.submit(name, 'analyze', { options }),
    onSuccess: () => {
      toast.success('已提交振动分析')
      qc.invalidateQueries({ queryKey: qk.jobs })
    },
  })
  const saved: AnalyzeOptions = {
    ...DEFAULTS,
    ...(vib.data?.options ?? {}),
    frequencies: vib.data?.options?.frequencies ?? [],
  }
  const setNoise = (v: string | null) => {
    setNoiseFrom(v)
    try {
      if (v) localStorage.setItem(`noise-from:${name}`, v)
      else localStorage.removeItem(`noise-from:${name}`)
    } catch {
      /* private mode */
    }
  }
  const reconstructed = status.data?.stages.reconstruct.state === 'done'
  return (
    <div className='space-y-6'>
      <OptionsCard
        key={JSON.stringify(vib.data?.options ?? null)}
        initial={saved}
        onRun={(o) => submit.mutate(o)}
        disabled={busy || submit.isPending || !reconstructed}
        experiments={(experiments.data ?? []).map((e) => e.name)}
        noiseFrom={noiseFrom}
        setNoiseFrom={setNoise}
        hint={!reconstructed ? '需要先完成三维重建' : undefined}
      />
      {busy && (
        <p className='text-sm text-muted-foreground'>
          该实验有任务正在写结果文件，完成后显示结果。
        </p>
      )}
      {!busy && vib.isError && (
        <p className='py-8 text-center text-sm text-muted-foreground'>
          还没有振动分析结果。
        </p>
      )}
      {!busy && vib.isLoading && <Skeleton className='h-80' />}
      {!busy && vib.data && (
        <>
          <SpectrumCard
            v={vib.data}
            name={name}
            onFit={(f) =>
              submit.mutate({ ...saved, frequencies: [f], fit_peaks: 0 })
            }
          />
          <HarmonicCard name={name} v={vib.data} noiseFrom={noiseFrom} />
          <div className='grid gap-4 xl:grid-cols-2'>
            {vib.data.fdd && <FddCard name={name} v={vib.data} />}
            <MacCard name={name} />
          </div>
          {vib.data.frf && <FrfCard v={vib.data} />}
        </>
      )}
    </div>
  )
}

function OptionsCard({
  initial,
  onRun,
  disabled,
  experiments,
  noiseFrom,
  setNoiseFrom,
  hint,
}: {
  initial: AnalyzeOptions
  onRun: (o: AnalyzeOptions) => void
  disabled: boolean
  experiments: string[]
  noiseFrom: string | null
  setNoiseFrom: (v: string | null) => void
  hint?: string
}) {
  const [opts, setOpts] = useState<AnalyzeOptions>(initial)
  const [freqText, setFreqText] = useState(initial.frequencies.join(', '))
  const num = (
    k: keyof AnalyzeOptions,
    label: string,
    nullable = false,
    unit?: string
  ) => (
    <div className='space-y-1'>
      <Label className='text-xs text-muted-foreground'>
        {label}
        {unit && ` (${unit})`}
      </Label>
      <Input
        className='h-8 font-mono'
        value={opts[k] == null ? '' : String(opts[k])}
        placeholder={nullable ? '自动' : undefined}
        onChange={(e) => {
          const v = e.target.value.trim()
          setOpts((o) => ({
            ...o,
            [k]: v === '' && nullable ? null : Number(v),
          }))
        }}
      />
    </div>
  )
  return (
    <Card>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <Waves className='size-4 text-brand' /> 振动分析
        </CardTitle>
        <CardDescription>
          宽带激励（T4）：全场平均 Welch 谱找峰、半功率带宽估阻尼、FDD
          看振型；定频驻留（T5）：在激振频率上做谐波拟合得到全场振幅、相位与
          ODS。拟合频率会按全场谐波能量最大化精化（消除频率误差导致的振幅偏小）。
        </CardDescription>
        <CardAction>
          <Button
            size='sm'
            disabled={disabled}
            title={hint}
            onClick={() => {
              const fr = freqText
                .split(/[,\s，]+/)
                .map(Number)
                .filter((x) => Number.isFinite(x) && x > 0)
              onRun({ ...opts, frequencies: fr })
            }}
          >
            <Play /> 运行分析
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className='space-y-4'>
        <div className='grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8'>
          <div className='space-y-1'>
            <Label className='text-xs text-muted-foreground'>分析分量</Label>
            <Select
              value={String(opts.direction)}
              onValueChange={(v) =>
                setOpts((o) => ({ ...o, direction: Number(v) as 0 }))
              }
            >
              <SelectTrigger size='sm' className='w-full'>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {COMP.map((c, i) => (
                  <SelectItem key={i} value={String(i)}>
                    {c}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {num('min_frequency', '最低频率', false, 'Hz')}
          {num('max_frequency', '最高频率', true, 'Hz')}
          {num('max_peaks', '峰数')}
          {num('nperseg', 'Welch 段长', true)}
          {num('fit_peaks', '拟合前 N 个峰')}
          {num('n_modes', 'FDD 奇异值数')}
          {num('n_harmonics', '谐波阶数')}
        </div>
        <div className='grid gap-3 lg:grid-cols-3'>
          <div className='space-y-1'>
            <Label className='text-xs text-muted-foreground'>
              拟合频率（定频激振，逗号分隔；留空则拟合最强的前 N 个峰）
            </Label>
            <Input
              className='h-8 font-mono'
              value={freqText}
              placeholder='例如 1000'
              onChange={(e) => setFreqText(e.target.value)}
            />
          </div>
          <div className='space-y-1'>
            <Label className='text-xs text-muted-foreground'>
              力信号文件（可选，计算 H1 与相干函数）
            </Label>
            <PathPicker
              mode='file'
              value={opts.force_file}
              onChange={(v) => setOpts((o) => ({ ...o, force_file: v }))}
            />
          </div>
          <div className='space-y-1'>
            <Label className='text-xs text-muted-foreground'>
              噪声底参考实验（静态采集，用于显著性门限）
            </Label>
            <Select
              value={noiseFrom ?? '__none__'}
              onValueChange={(v) => setNoiseFrom(v === '__none__' ? null : v)}
            >
              <SelectTrigger size='sm' className='w-full'>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value='__none__'>不使用</SelectItem>
                {experiments.map((e) => (
                  <SelectItem key={e} value={e}>
                    {e}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className='flex flex-wrap gap-6 text-sm'>
          {(
            [
              ['use_rbm', '用刚体补偿后的位移'],
              ['refine', '精化拟合频率'],
              ['fdd', 'FDD'],
            ] as const
          ).map(([k, l]) => (
            <label key={k} className='flex items-center gap-2'>
              <Switch
                checked={opts[k] as boolean}
                onCheckedChange={(v) => setOpts((o) => ({ ...o, [k]: v }))}
              />
              {l}
            </label>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function SpectrumCard({
  v,
  name,
  onFit,
}: {
  v: Vibration
  name: string
  onFit: (f: number) => void
}) {
  const x = useMemo(
    () => v.spectrum.frequencies.map((f) => f ?? 0).slice(1),
    [v]
  )
  const um = (a: (number | null)[]) =>
    a.slice(1).map((y) => (y == null || y <= 0 ? null : y * 1e3))
  const d = v.options.direction ?? 2
  const noise = v.noise
  const noiseOnGrid = useMemo(() => {
    if (!noise) return null
    const nf = noise.frequencies.map((f) => f ?? 0)
    const na = noise.amplitude[d]
    return x.map((f) => {
      let k = 0
      while (k < nf.length - 1 && nf[k + 1] < f) k++
      const y = na[k]
      return y == null || y <= 0 ? null : y * 1e3
    })
  }, [noise, x, d])
  const df = v.sample_rate_hz / Math.min(v.options.nperseg ?? 1024, v.n_frames)
  return (
    <Card>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <Activity className='size-4 text-brand' /> 全场平均谱
        </CardTitle>
        <CardDescription>
          各测点 Welch
          功率谱的全场平均（对功率求平均，扭转等反相区域不会互相抵消）。采样率{' '}
          {v.sample_rate_hz.toFixed(1)} Hz，时长 {v.duration_s.toFixed(3)}{' '}
          s，谱分辨率约 {df.toFixed(2)} Hz。
        </CardDescription>
        <CardAction>
          <Button size='sm' variant='outline' asChild>
            <a href={api.exportUrl2(name, 'peaks.csv')} download>
              <Download /> 峰值 CSV
            </a>
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className='space-y-4'>
        <TimeChart
          x={x}
          height={280}
          logY
          xLabel='频率 Hz'
          yLabel='幅值 µm'
          markers={v.peaks.map((p) => ({
            x: p.frequency_hz,
            color: 'rgba(148,163,184,0.6)',
            dash: [4, 4],
          }))}
          series={[
            ...v.spectrum.amplitude.map((a, c) => ({
              label: COMP[c],
              color: COLORS[c],
              values: um(a),
              width: c === d ? 1.8 : 1,
            })),
            ...(noiseOnGrid
              ? [
                  {
                    label: `噪声底（${noise!.experiment}）`,
                    color: '#94a3b8',
                    values: noiseOnGrid,
                    dash: [5, 4],
                    width: 1,
                  },
                ]
              : []),
          ]}
        />
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>#</TableHead>
              <TableHead className='text-end'>频率 Hz</TableHead>
              <TableHead className='text-end'>幅值 µm</TableHead>
              <TableHead className='text-end'>半功率带宽 Hz</TableHead>
              <TableHead className='text-end'>阻尼比</TableHead>
              <TableHead className='w-28' />
            </TableRow>
          </TableHeader>
          <TableBody>
            {v.peaks.map((p, i) => (
              <TableRow key={i}>
                <TableCell className='font-mono'>{i + 1}</TableCell>
                <TableCell className='text-end font-mono'>
                  {p.frequency_hz.toFixed(3)}
                </TableCell>
                <TableCell className='text-end font-mono'>
                  {(p.amplitude_mm * 1e3).toFixed(3)}
                </TableCell>
                <TableCell className='text-end font-mono'>
                  {p.damping != null && Number.isFinite(p.right_hz - p.left_hz)
                    ? (p.right_hz - p.left_hz).toFixed(3)
                    : '—'}
                </TableCell>
                <TableCell className='text-end font-mono'>
                  {p.damping == null ? '—' : p.damping.toFixed(4)}
                  {p.damping_limited && (
                    <span
                      className='ms-1 text-xs text-warn'
                      title='半功率带宽不到 4 个频率分辨单元，只能作为上限；延长采集时长或用自由衰减法'
                    >
                      ≤
                    </span>
                  )}
                </TableCell>
                <TableCell className='text-end'>
                  <Button
                    size='sm'
                    variant='ghost'
                    onClick={() => onFit(p.frequency_hz)}
                  >
                    谐波拟合
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {v.damping_resolution_hz != null && (
          <p className='text-xs text-muted-foreground'>
            阻尼用全长单段谱（分辨率 {v.damping_resolution_hz.toFixed(3)}{' '}
            Hz）估计；标“≤”的峰带宽受分辨率限制，真实阻尼更小。
          </p>
        )}
      </CardContent>
    </Card>
  )
}

type HView = 'amp' | 'phase' | 'ods'

function HarmonicCard({
  name,
  v,
  noiseFrom,
}: {
  name: string
  v: Vibration
  noiseFrom: string | null
}) {
  const [key, setKey] = useState<string | null>(v.harmonic[0]?.key ?? null)
  const [comp, setComp] = useState<number>(v.options.direction ?? 2)
  const [view, setView] = useState<HView>('amp')
  const [gain, setGain] = useState(200)
  const [phi, setPhi] = useState(0)
  const [playing, setPlaying] = useState(true)
  const ref = useQuery({
    queryKey: qk.reference(name),
    queryFn: () => api.reference(name),
    meta: { silent: true },
  })
  const hq = useQuery({
    queryKey: [...qk.results(name), 'harmonic', key, noiseFrom],
    queryFn: () => api.harmonic(name, key!, noiseFrom),
    enabled: !!key,
    meta: { silent: true },
  })
  useEffect(() => {
    if (view !== 'ods' || !playing) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      setPhi(
        (p) => (p + ((now - last) / 1000) * 2 * Math.PI * 0.6) % (2 * Math.PI)
      )
      last = now
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [view, playing])
  if (!v.harmonic.length) return null
  const sel = v.harmonic.find((x) => x.key === key)
  return (
    <Card>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <Sigma className='size-4 text-brand' /> 谐波拟合 · 振幅、相位与 ODS
        </CardTitle>
        <CardDescription>
          按实际时间戳做最小二乘谐波拟合（支持欠采样），响应约定 u =
          Re(R·e^(iωt))。
          {noiseFrom
            ? `灰色测点：振幅未超过 5σ 噪声门限（${noiseFrom}）。`
            : '可在上方选择静态噪声底实验，用 5σ 门限标出不显著的测点。'}
        </CardDescription>
        {key && (
          <CardAction>
            <Button size='sm' variant='outline' asChild>
              <a href={api.exportUrl2(name, `harmonic/${key}.csv`)} download>
                <Download /> CSV
              </a>
            </Button>
          </CardAction>
        )}
      </CardHeader>
      <CardContent className='space-y-3'>
        <div className='flex flex-wrap items-center gap-2'>
          {v.harmonic.map((x) => (
            <button
              key={x.key}
              onClick={() => setKey(x.key)}
              className={cn(
                'rounded-md border px-2.5 py-1 font-mono text-sm',
                x.key === key
                  ? 'border-brand bg-brand/15 text-brand'
                  : 'hover:bg-accent'
              )}
              title={
                x.requested_hz != null
                  ? `请求 ${x.requested_hz.toFixed(4)} Hz，精化后 ${x.frequency_hz.toFixed(4)} Hz`
                  : undefined
              }
            >
              {x.frequency_hz.toFixed(3)} Hz
            </button>
          ))}
          <div className='ms-auto flex flex-wrap gap-2'>
            <Seg
              value={String(comp)}
              onChange={(c) => setComp(Number(c))}
              options={COMP.map((c, i) => [String(i), c])}
            />
            <Seg
              value={view}
              onChange={(x) => setView(x as HView)}
              options={[
                ['amp', '振幅'],
                ['phase', '相位'],
                ['ods', 'ODS 动画'],
              ]}
            />
          </div>
        </div>
        {sel && (
          <div className='grid grid-cols-3 gap-2 md:grid-cols-6'>
            <Metric
              label='拟合频率'
              value={sel.frequency_hz.toFixed(4)}
              unit='Hz'
            />
            {sel.max_amplitude_mm.map((a, c) => (
              <Metric
                key={c}
                label={`最大振幅 ${COMP[c]}`}
                value={a == null ? '—' : (a * 1e3).toFixed(3)}
                unit='µm'
              />
            ))}
            <Metric label='位移' value={sel.use_rbm ? '刚体补偿后' : '原始'} />
          </div>
        )}
        <div className='grid gap-3 xl:grid-cols-[minmax(0,1fr)_260px]'>
          <div className='h-[460px] overflow-hidden rounded-xl border'>
            {ref.data && hq.data ? (
              <HarmonicMesh
                r={ref.data}
                h={hq.data}
                comp={comp}
                view={view}
                gain={gain}
                phi={phi}
              />
            ) : (
              <Skeleton className='h-full' />
            )}
          </div>
          <div className='space-y-3 text-sm'>
            {view === 'ods' && (
              <>
                <div className='flex items-center gap-2'>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() => setPlaying((p) => !p)}
                  >
                    {playing ? '暂停' : '播放'}
                  </Button>
                  <span className='font-mono text-xs text-muted-foreground'>
                    相位 {((phi * 180) / Math.PI).toFixed(0)}°
                  </span>
                </div>
                <div className='space-y-1'>
                  <Label className='text-xs text-muted-foreground'>
                    位移放大 ×{gain}
                  </Label>
                  <input
                    type='range'
                    min={0}
                    max={3000}
                    step={50}
                    value={gain}
                    onChange={(e) => setGain(Number(e.target.value))}
                    className='w-full accent-[var(--brand)]'
                  />
                </div>
                <p className='text-xs text-muted-foreground'>
                  形貌按三个分量的 Re(R·e^(iφ))
                  变形并放大，颜色为所选分量的瞬时值；定频激励且模态间隔良好时
                  ODS 近似该阶振型。
                </p>
              </>
            )}
            {view !== 'ods' && hq.data && (
              <Legend h={hq.data} comp={comp} view={view} />
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function HarmonicMesh({
  r,
  h,
  comp,
  view,
  gain,
  phi,
}: {
  r: ReferenceResult
  h: Harmonic
  comp: number
  view: HView
  gain: number
  phi: number
}) {
  const n = r.n
  const sig = h.significant
  const cos = Math.cos(phi)
  const sin = Math.sin(phi)
  const values = useMemo(() => {
    const out: (number | null)[] = new Array(n)
    for (let i = 0; i < n; i++) {
      if (sig && !sig[i]?.[comp]) {
        out[i] = null
        continue
      }
      if (view === 'amp') out[i] = h.amplitude[i]?.[comp] ?? null
      else if (view === 'phase') out[i] = h.phase[i]?.[comp] ?? null
      else {
        const re = h.re[i]?.[comp]
        const im = h.im[i]?.[comp]
        out[i] = re == null || im == null ? null : re * cos - im * sin
      }
    }
    return out
  }, [h, comp, view, cos, sin, n, sig])
  const warp = useMemo(() => {
    if (view !== 'ods') return null
    const out: (number | null)[][] = new Array(n)
    for (let i = 0; i < n; i++) {
      const re = h.re[i]
      const im = h.im[i]
      if (
        !re ||
        !im ||
        re.some((x) => x == null) ||
        im.some((x) => x == null)
      ) {
        out[i] = [null, null, null]
        continue
      }
      const loc = [0, 1, 2].map((k) => re[k]! * cos - im[k]! * sin) // (c, s, n) at this phase
      const F = r.local_frames.slice(9 * i, 9 * i + 9) as number[] // rows c, s, n in the left-camera frame
      out[i] = [0, 1, 2].map(
        (a) => loc[0] * F[a] + loc[1] * F[3 + a] + loc[2] * F[6 + a]
      )
    }
    return out
  }, [h, view, cos, sin, n, r.local_frames])
  const maxAmp = useMemo(
    () => Math.max(1e-12, ...h.amplitude.map((a) => a?.[comp] ?? 0)),
    [h, comp]
  )
  return (
    <Suspense fallback={<Skeleton className='h-full' />}>
      <FieldMesh
        points3d={r.points_3d}
        connectivity={r.connectivity}
        values={values}
        range={maxAmp}
        signed={view === 'ods'}
        cyclicColors={view === 'phase'}
        warp={warp}
        gain={view === 'ods' ? gain : 0}
      />
    </Suspense>
  )
}

function Legend({ h, comp, view }: { h: Harmonic; comp: number; view: HView }) {
  const max = Math.max(...h.amplitude.map((a) => a?.[comp] ?? 0))
  if (view === 'phase')
    return (
      <div className='space-y-2'>
        <div
          className='h-3 w-full rounded-full'
          style={{
            background:
              'conic-gradient(from 90deg, hsl(0 65% 55%), hsl(90 65% 55%), hsl(180 65% 55%), hsl(270 65% 55%), hsl(360 65% 55%))',
          }}
        />
        <p className='text-xs text-muted-foreground'>
          相位 −π…π，循环色标；同色即同相，反色（相差
          π）即反相，节线两侧相位翻转。
        </p>
      </div>
    )
  return (
    <div className='space-y-2 font-mono text-xs text-muted-foreground'>
      <div
        className='h-2 w-full rounded-full'
        style={{
          background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map(viridisCss).join(',')})`,
        }}
      />
      <div className='flex justify-between'>
        <span>0</span>
        <span>{(max * 1e3).toPrecision(3)} µm</span>
      </div>
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
    <div className='flex rounded-lg border p-0.5 text-sm'>
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

function FddCard({ name, v }: { name: string; v: Vibration }) {
  const fdd = v.fdd!
  const [p, setP] = useState(0)
  const ref = useQuery({
    queryKey: qk.reference(name),
    queryFn: () => api.reference(name),
    meta: { silent: true },
  })
  const { data: modeData } = useQuery({
    queryKey: [...qk.results(name), 'fdd', p],
    queryFn: () => api.fddMode(name, p),
    meta: { silent: true },
  })
  const x = fdd.frequencies.map((f) => f ?? 0).slice(1)
  const values = useMemo(
    () =>
      modeData?.re.map((re, i) =>
        re == null
          ? null
          : Math.hypot(re, modeData!.im[i] ?? 0) * Math.sign(re || 1)
      ) ?? null,
    [modeData]
  )
  return (
    <Card className='min-w-0'>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <Radar className='size-4 text-brand' /> 频域分解 FDD
        </CardTitle>
        <CardDescription>
          第一奇异值谱的峰对应模态，其奇异向量即振型（只保存各峰处，单分量，归一化）。
        </CardDescription>
      </CardHeader>
      <CardContent className='space-y-3'>
        <TimeChart
          x={x}
          height={200}
          logY
          xLabel='频率 Hz'
          yLabel='奇异值'
          markers={fdd.peak_frequencies
            .filter((f): f is number => f != null)
            .map((f, i) => ({
              x: f,
              color: i === p ? '#5fd0e6' : 'rgba(148,163,184,0.5)',
              dash: [4, 4],
            }))}
          series={fdd.singular_values.map((sv, k) => ({
            label: `σ${k + 1}`,
            color: ['#5fd0e6', '#a78bfa', '#f59e0b', '#94a3b8'][k % 4],
            values: sv.slice(1).map((y) => (y == null || y <= 0 ? null : y)),
            width: k ? 1 : 1.6,
          }))}
        />
        <div className='flex flex-wrap gap-1'>
          {fdd.peak_frequencies.map((f, i) => (
            <button
              key={i}
              onClick={() => setP(i)}
              className={cn(
                'rounded-md border px-2 py-0.5 font-mono text-xs',
                i === p
                  ? 'border-brand bg-brand/15 text-brand'
                  : 'hover:bg-accent'
              )}
            >
              {f?.toFixed(2)} Hz
            </button>
          ))}
        </div>
        <div className='h-72 overflow-hidden rounded-xl border'>
          {ref.data && values ? (
            <Suspense fallback={<Skeleton className='h-full' />}>
              <FieldMesh
                points3d={ref.data.points_3d}
                connectivity={ref.data.connectivity}
                values={values}
                range={1}
                signed
              />
            </Suspense>
          ) : (
            <Skeleton className='h-full' />
          )}
        </div>
        <div className='flex items-center gap-2 font-mono text-xs text-muted-foreground'>
          <span>−1</span>
          <span
            className='h-2 w-48 rounded-full'
            style={{
              background: `linear-gradient(90deg, ${[-1, -0.5, 0, 0.5, 1].map(divergingCss).join(',')})`,
            }}
          />
          <span>+1（模长 × 实部符号）</span>
        </div>
      </CardContent>
    </Card>
  )
}

function MacCard({ name }: { name: string }) {
  const { data, isError } = useQuery({
    queryKey: [...qk.results(name), 'mac'],
    queryFn: () => api.mac(name),
    meta: { silent: true },
    retry: false,
  })
  return (
    <Card className='min-w-0'>
      <CardHeader>
        <CardTitle className='text-base'>MAC 矩阵</CardTitle>
        <CardDescription>
          谐波 ODS 与 FDD 振型两两之间的模态置信准则：对角为
          1，不同阶振型之间应接近 0；与有限元比对时 ≥ 0.9 视为一致（技术设计
          §11.6）。
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isError && (
          <p className='text-sm text-muted-foreground'>没有可比较的振型。</p>
        )}
        {data && (
          <div className='overflow-auto'>
            <table className='text-xs'>
              <tbody>
                <tr>
                  <td />
                  {data.labels.map((l) => (
                    <td
                      key={l}
                      className='px-1 pb-1 font-mono text-muted-foreground [writing-mode:vertical-rl]'
                    >
                      {l}
                    </td>
                  ))}
                </tr>
                {data.mac.map((row, i) => (
                  <tr key={i}>
                    <td className='pe-2 font-mono whitespace-nowrap text-muted-foreground'>
                      {data.labels[i]}
                    </td>
                    {row.map((m, j) => (
                      <td
                        key={j}
                        className='size-11 border border-background text-center font-mono'
                        style={{
                          background: viridisCss(m ?? 0),
                          color: (m ?? 0) > 0.6 ? '#111' : '#eee',
                        }}
                      >
                        {m == null ? '—' : m.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function FrfCard({ v }: { v: Vibration }) {
  const frf = v.frf!
  const x = frf.frequencies.map((f) => f ?? 0).slice(1)
  return (
    <Card>
      <CardHeader>
        <CardTitle className='text-base'>频响函数 H1 与相干函数</CardTitle>
        <CardDescription>
          H1 = G_uF / G_FF（全场平均幅值），相干函数接近 1 的频段结果可信。
        </CardDescription>
      </CardHeader>
      <CardContent className='grid gap-4 xl:grid-cols-2'>
        <TimeChart
          x={x}
          height={220}
          logY
          xLabel='频率 Hz'
          yLabel='|H1| mm/N'
          series={[
            {
              label: '|H1|',
              color: '#5fd0e6',
              values: frf.H1_mean_abs.slice(1).map((y) =>
                y == null || y <= 0 ? null : y
              ),
            },
          ]}
        />
        <TimeChart
          x={x}
          height={220}
          xLabel='频率 Hz'
          yLabel='相干'
          yRange={[0, 1.02]}
          series={[
            {
              label: 'γ²',
              color: '#34d399',
              values: frf.coherence_mean.slice(1),
            },
          ]}
        />
      </CardContent>
    </Card>
  )
}
