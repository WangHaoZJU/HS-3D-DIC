/**
 * Acquisition planner (docs/UI_Design.md §7.12): pure front-end formulas from
 * StereoDIC_Technical_Design.md §3.2 / §3.4 / §11.3 and 振动实验台设定.md §5.
 */
import { useState } from 'react'
import { Calculator, Camera, Clock, Radio, Target } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Page } from '@/components/layout/page'
import { Metric, type Verdict } from '@/features/experiment/import/metric'

type P = {
  width: number
  height: number
  bits: number
  memoryGB: number
  fps: number
  f: number
  A: number // mm
  s: number // mm/px
  blur: number // px
  alpha: number // deg
  sigmaP: number // px
  nFrames: number
}

const DEFAULTS: P = {
  width: 1280,
  height: 1024,
  bits: 12,
  memoryGB: 32,
  fps: 5000,
  f: 1000,
  A: 0.05,
  s: 0.1,
  blur: 0.1,
  alpha: 30,
  sigmaP: 0.01,
  nFrames: 2000,
}

const PRESETS: { name: string; p: Partial<P>; note: string }[] = [
  {
    name: 'T4 宽带识频',
    p: { fps: 3000, f: 1000, A: 0.05, nFrames: 6000 },
    note: '覆盖到 1000 Hz，fs ≥ 2.5 f',
  },
  {
    name: 'T5 定频 1000 Hz',
    p: { fps: 5000, f: 1000, A: 0.05, nFrames: 2000 },
    note: '直接看清波形',
  },
  {
    name: 'T5 欠采样',
    p: { fps: 930, f: 1000, A: 0.02, nFrames: 2000 },
    note: '相机帧率受限时错开整数倍',
  },
]

function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—'
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2)
  return Number(v.toPrecision(digits)).toString()
}

export function PlannerPage() {
  const [p, setP] = useState<P>(DEFAULTS)
  const set = (k: keyof P) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setP((o) => ({ ...o, [k]: Number(e.target.value) }))

  // recording
  const bytesPerPx = p.bits > 8 ? 2 : 1
  const frameBytes = p.width * p.height * bytesPerPx
  const maxFrames = Math.floor((p.memoryGB * 1024 ** 3) / frameBytes)
  const maxSeconds = maxFrames / p.fps
  const recSeconds = p.nFrames / p.fps
  const df = 1 / recSeconds

  // aliasing (§11.3)
  const nyq = p.fps / 2
  const k = Math.round(p.f / p.fps)
  const alias = Math.abs(p.f - k * p.fps)
  const aliasRel = alias / p.fps
  let sampling: { label: string; verdict: Verdict; note: string }
  if (p.fps >= 2.5 * p.f)
    sampling = {
      label: '直接采样',
      verdict: 'ok',
      note: `fs ≥ 2.5 f，宽带识频与看波形都可以`,
    }
  else if (p.fps > 2 * p.f)
    sampling = {
      label: '接近奈奎斯特',
      verdict: 'warn',
      note: 'fs 在 2f–2.5f 之间：峰值可见，但窗函数泄漏和抗混叠余量不足',
    }
  else if (aliasRel < 0.02 || aliasRel > 0.48)
    sampling = {
      label: '欠采样 · 危险',
      verdict: 'bad',
      note: `混叠到 ${fmt(alias)} Hz，接近 0 或 fs/2，振幅相位无法稳定恢复；调整帧率`,
    }
  else
    sampling = {
      label: '欠采样 · 可用',
      verdict: 'warn',
      note: `混叠到 ${fmt(alias)} Hz；只适用于激振频率已知的定频驻留（T5），不能做宽带识频`,
    }

  // exposure (§3.2)
  const tExp = (p.blur * p.s) / (2 * Math.PI * p.f * p.A) // s
  const tFrame = 1 / p.fps

  // noise (§3.4, §11.3)
  const a = (p.alpha * Math.PI) / 180
  const sLat = (p.s * p.sigmaP) / (Math.SQRT2 * Math.cos(a / 2)) // mm
  const sDepth = (p.s * p.sigmaP) / (Math.SQRT2 * Math.sin(a / 2))
  const sFit = Math.sqrt(2 / p.nFrames) * sDepth
  const minAmp = 10 * sFit

  const field = (k: keyof P, label: string, unit?: string) => (
    <div className='space-y-1'>
      <Label htmlFor={`pl-${k}`} className='text-xs text-muted-foreground'>
        {label}
        {unit && <span className='ms-1'>({unit})</span>}
      </Label>
      <Input
        id={`pl-${k}`}
        inputMode='decimal'
        className='h-8 font-mono'
        value={String(p[k])}
        onChange={set(k)}
      />
    </div>
  )

  return (
    <Page
      title='规划计算器'
      description='采集前估算录制时长、频率分辨率、混叠、曝光上限和噪声底（公式见技术设计 §3.2、§3.4、§11.3）'
      actions={
        <div className='flex flex-wrap gap-2'>
          {PRESETS.map((pr) => (
            <Button
              key={pr.name}
              size='sm'
              variant='outline'
              title={pr.note}
              onClick={() => setP((o) => ({ ...o, ...pr.p }))}
            >
              {pr.name}
            </Button>
          ))}
        </div>
      }
    >
      <div className='grid gap-4 xl:grid-cols-2'>
        <Section
          icon={Camera}
          title='相机与录制'
          desc='高速相机先存机身内存：单次可录时长受内存限制'
        >
          <div className='grid grid-cols-3 gap-3'>
            {field('width', '宽', 'px')}
            {field('height', '高', 'px')}
            {field('bits', '位深', 'bit')}
            {field('memoryGB', '机身内存', 'GB')}
            {field('fps', '帧率', 'fps')}
            {field('nFrames', '计划帧数')}
          </div>
          <div className='grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-2 2xl:grid-cols-4'>
            <Metric
              label='最多可录'
              value={maxFrames.toLocaleString()}
              unit='帧'
            />
            <Metric label='最长时长' value={fmt(maxSeconds)} unit='s' />
            <Metric
              label='计划时长'
              value={fmt(recSeconds)}
              unit='s'
              verdict={p.nFrames <= maxFrames ? 'ok' : 'bad'}
              rule={
                p.nFrames <= maxFrames
                  ? '在机身内存范围内'
                  : '超过机身内存，减少帧数、分辨率或位深'
              }
            />
            <Metric
              label='频率分辨率 Δf'
              value={fmt(df)}
              unit='Hz'
              rule='Δf = 1 / T，只取决于采集时长，与帧率无关'
            />
          </div>
          <p className='text-xs text-muted-foreground'>
            每帧 {fmt(frameBytes / 1024 ** 2)} MB（{bytesPerPx}{' '}
            字节/像素）。机身内存按实际型号填写；千眼狼 S1315
            公开资料：1280×1024 下最高约 15000 fps。
          </p>
        </Section>

        <Section
          icon={Radio}
          title='采样与混叠'
          desc='目标频率 f 与帧率 fs 的关系'
        >
          <div className='grid grid-cols-3 gap-3'>
            {field('f', '目标频率 f', 'Hz')}
            {field('fps', '帧率 fs', 'fps')}
          </div>
          <div className='grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-2 2xl:grid-cols-4'>
            <Metric
              label='判定'
              value={sampling.label}
              verdict={sampling.verdict}
              rule={sampling.note}
            />
            <Metric label='奈奎斯特' value={fmt(nyq)} unit='Hz' />
            <Metric
              label='fs / f'
              value={fmt(p.fps / p.f)}
              verdict={p.fps >= 2.5 * p.f ? 'ok' : 'warn'}
              rule='宽带识频要求 ≥ 2.5；看清波形取 5–10'
            />
            <Metric
              label='混叠后频率'
              value={p.fps > 2 * p.f ? '—' : fmt(alias)}
              unit={p.fps > 2 * p.f ? undefined : 'Hz'}
            />
          </div>
          <p
            className={cn(
              'text-xs',
              sampling.verdict === 'bad' ? 'text-bad' : 'text-muted-foreground'
            )}
          >
            {sampling.note}
          </p>
        </Section>

        <Section
          icon={Clock}
          title='曝光时间上限'
          desc='t ≲ δ·s / (2π f A)：振动最快时运动模糊不超过 δ 像素'
        >
          <div className='grid grid-cols-4 gap-3'>
            {field('A', '振幅 A', 'mm')}
            {field('s', '物方比例 s', 'mm/px')}
            {field('blur', '允许模糊 δ', 'px')}
            {field('f', '频率 f', 'Hz')}
          </div>
          <div className='grid grid-cols-2 gap-2 sm:grid-cols-3'>
            <Metric
              label='曝光上限'
              value={fmt(tExp * 1e6)}
              unit='µs'
              verdict={tExp < 5e-6 ? 'warn' : 'ok'}
              rule='曝光短于约 5 µs 时照明通常成为瓶颈'
            />
            <Metric label='帧间隔' value={fmt(tFrame * 1e6)} unit='µs' />
            <Metric
              label='最大速度'
              value={fmt(2 * Math.PI * p.f * p.A)}
              unit='mm/s'
            />
          </div>
        </Section>

        <Section
          icon={Target}
          title='噪声底'
          desc='对称会聚双目、各相机二维噪声 σp，谐波拟合 N 帧后的振幅噪声'
        >
          <div className='grid grid-cols-4 gap-3'>
            {field('alpha', '夹角 α', '°')}
            {field('sigmaP', '匹配噪声 σp', 'px')}
            {field('s', '物方比例 s', 'mm/px')}
            {field('nFrames', '帧数 N')}
          </div>
          <div className='grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-2 2xl:grid-cols-4'>
            <Metric label='面内 σ（单帧）' value={fmt(sLat * 1e3)} unit='µm' />
            <Metric
              label='离面 σ（单帧）'
              value={fmt(sDepth * 1e3)}
              unit='µm'
            />
            <Metric
              label='拟合后 σA'
              value={fmt(sFit * 1e3)}
              unit='µm'
              rule='σA ≈ √(2/N) σn，白噪声假设；固定参考帧的相关误差不随 N 下降，以 E1 实测为准'
            />
            <Metric
              label='可测最小振幅'
              value={fmt(minAmp * 1e3)}
              unit='µm'
              verdict={p.A >= minAmp ? 'ok' : 'bad'}
              rule={`按 10 倍噪声底（技术设计 §15.2）；当前振幅 ${fmt(p.A * 1e3)} µm`}
            />
          </div>
        </Section>
      </div>
      <p className='mt-4 flex items-center gap-2 text-xs text-muted-foreground'>
        <Calculator className='size-3.5' /> 只做量级估算；三维精度以 E1
        静态噪声、E2 位移台实验为准。
      </p>
    </Page>
  )
}

function Section({
  icon: Icon,
  title,
  desc,
  children,
}: {
  icon: typeof Camera
  title: string
  desc: string
  children: React.ReactNode
}) {
  return (
    <Card className='min-w-0'>
      <CardHeader>
        <CardTitle className='flex items-center gap-2 text-base'>
          <Icon className='size-4 text-brand' /> {title}
        </CardTitle>
        <CardDescription>{desc}</CardDescription>
      </CardHeader>
      <CardContent className='space-y-4'>{children}</CardContent>
    </Card>
  )
}
