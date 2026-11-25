import { Suspense, lazy, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Box,
  Camera,
  Grid3x3,
  Play,
  RefreshCcw,
  ScanSearch,
  Save,
} from 'lucide-react'
import { toast } from 'sonner'
import {
  api,
  qk,
  type CalibrationOverview,
  type CalibrationReportCamera,
  type ExperimentConfig,
} from '@/lib/api'
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
import { Checkbox } from '@/components/ui/checkbox'
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Metric, band } from '../import/metric'
import { ResidualScatter, ViewRmsChart } from './charts'
import { ViewDetail } from './view-detail'

const Poses3D = lazy(() => import('./poses-3d'))

export function CalibrationPage({ name }: { name: string }) {
  const qc = useQueryClient()
  const ov = useQuery({
    queryKey: qk.calibration(name),
    queryFn: () => api.calibration(name),
    meta: { silent: true },
    retry: false,
  })
  const detail = useQuery({
    queryKey: qk.experiment(name),
    queryFn: () => api.experiment(name),
  })
  const submit = useMutation({
    mutationFn: ({ stage, force }: { stage: string; force?: boolean }) =>
      api.submit(name, stage, {}, force),
    onSuccess: (j) => {
      toast.success(j.stage === 'calibrate' ? '已提交标定' : '已提交角点检测')
      qc.invalidateQueries({ queryKey: qk.jobs })
    },
  })
  if (ov.isError)
    return (
      <Card>
        <CardContent className='py-10 text-center text-sm text-muted-foreground'>
          该实验没有标定设置（calibration）或标定文件路径。可以在“配置”页启用“ChArUco
          标定”。
        </CardContent>
      </Card>
    )
  if (!ov.data || !detail.data) return <Skeleton className='h-96' />
  return (
    <CalibrationBody
      key={`${detail.dataUpdatedAt}:${ov.data.exclude_views.join(',')}`}
      name={name}
      ov={ov.data}
      config={detail.data.config}
      onRun={(stage, force) => submit.mutate({ stage, force })}
      running={submit.isPending}
    />
  )
}

function CalibrationBody({
  name,
  ov,
  config,
  onRun,
  running,
}: {
  name: string
  ov: CalibrationOverview
  config: ExperimentConfig
  onRun: (stage: string, force?: boolean) => void
  running: boolean
}) {
  const qc = useQueryClient()
  const views = ov.detections?.views ?? []
  const [excluded, setExcluded] = useState<Set<number>>(
    () => new Set(ov.exclude_views)
  )
  const [selected, setSelected] = useState<number | null>(
    ov.pairs[0]?.view ?? (views.length ? 0 : null)
  )
  const excludeDirty = useMemo(
    () =>
      JSON.stringify([...excluded].sort((a, b) => a - b)) !==
      JSON.stringify([...ov.exclude_views].sort((a, b) => a - b)),
    [excluded, ov.exclude_views]
  )
  const saveConfig = useMutation({
    mutationFn: (patch: (c: ExperimentConfig) => ExperimentConfig) =>
      api.saveConfig(name, { config: patch(structuredClone(config)) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.experiment(name) })
      qc.invalidateQueries({ queryKey: qk.calibration(name) })
      qc.invalidateQueries({ queryKey: qk.status(name) })
    },
  })
  const applyExclusions = () =>
    saveConfig.mutate(
      (c) => {
        ;(c.calibration as Record<string, unknown>).exclude_views = [
          ...excluded,
        ].sort((a, b) => a - b)
        return c
      },
      { onSuccess: () => onRun('calibrate') }
    )

  const rep = ov.report
  const pairByView = new Map(ov.pairs.map((p) => [p.view, p]))
  const shape = ov.detections?.image_shapes?.[0] ??
    ov.images?.shape ?? [1024, 1280]
  const stale = ov.status.state === 'stale'

  return (
    <div className='space-y-6'>
      {/* status + actions */}
      <div className='flex flex-wrap items-center gap-3 rounded-xl border bg-card px-4 py-3 text-sm'>
        <StatusDot state={ov.busy ? 'running' : ov.status.state} />
        <span>
          {ov.busy
            ? '标定任务运行中…'
            : ov.status.state === 'done'
              ? '标定结果是最新的'
              : ov.status.state === 'stale'
                ? '标定设置已改变，结果过期'
                : '尚未标定'}
        </span>
        <span className='text-muted-foreground'>
          · 标定图像{' '}
          {ov.images
            ? `${ov.images.left} / ${ov.images.right} 张`
            : ov.images_error}
          {ov.detections &&
            ` · 角点检测${ov.detections.up_to_date ? '已缓存' : '需要重新检测'}`}
        </span>
        <div className='ms-auto flex flex-wrap gap-2'>
          <Button
            variant='outline'
            size='sm'
            disabled={ov.busy || running}
            onClick={() => onRun('calibrate_detect', true)}
          >
            <ScanSearch /> 重新检测角点
          </Button>
          {excludeDirty ? (
            <Button
              size='sm'
              disabled={ov.busy || running || saveConfig.isPending}
              onClick={applyExclusions}
            >
              <RefreshCcw /> 应用剔除（{excluded.size}）并重新标定
            </Button>
          ) : (
            <Button
              size='sm'
              disabled={ov.busy || running}
              onClick={() => onRun('calibrate')}
            >
              <Play /> {rep ? '重新标定' : '标定'}
            </Button>
          )}
        </div>
      </div>

      {/* results */}
      {rep && (
        <Card className={cn(stale && 'border-warn/40')}>
          <CardHeader>
            <CardTitle className='flex items-center gap-2 text-base'>
              <Camera className='size-4 text-brand' /> 标定结果
            </CardTitle>
            <CardDescription>
              重投影 RMS 只用于诊断，三维精度以 E2
              位移台实验为准；窄视场下主点、畸变等内参逐项比较意义不大（技术设计
              §3.5）。
            </CardDescription>
          </CardHeader>
          <CardContent className='space-y-5'>
            <div className='grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6'>
              <Metric
                label='立体 RMS'
                value={rep.stereo_rms_px.toFixed(4)}
                unit='px'
                verdict={band(rep.stereo_rms_px, [0, 0.1], [0, 0.3])}
                rule='< 0.1 px 为好（仅诊断）'
              />
              <Metric
                label='基线'
                value={rep.baseline_mm.toFixed(3)}
                unit='mm'
              />
              <Metric
                label='光轴夹角'
                value={rep.optical_axis_angle_deg.toFixed(2)}
                unit='°'
                verdict={band(rep.optical_axis_angle_deg, [25, 35], [15, 45])}
                rule='建议 25°–35°：夹角越大离面灵敏度越高，但遮挡和透视变形更大'
              />
              <Metric
                label='参与视图'
                value={`${rep.pairs.length} / ${views.length || rep.pairs.length}`}
              />
              <Metric label='角点数' value={rep.n_corners} />
              <Metric
                label='最大残差 左/右'
                value={`${rep.residual_left_max_px?.toFixed(3)} / ${rep.residual_right_max_px?.toFixed(3)}`}
                unit='px'
              />
            </div>
            <div className='grid gap-4 lg:grid-cols-2'>
              <Intrinsics label='左相机' cam={rep.left} />
              <Intrinsics label='右相机' cam={rep.right} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* views */}
      {ov.detections ? (
        <div className='grid gap-4 2xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]'>
          <Card className='min-w-0 gap-2 p-0'>
            <CardHeader className='px-4 pt-4'>
              <CardTitle className='text-base'>视图</CardTitle>
              <CardDescription>
                勾选要剔除的视图（两台相机同时剔除），然后重新标定。
              </CardDescription>
            </CardHeader>
            <div className='max-h-[520px] overflow-auto'>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className='w-10'>剔除</TableHead>
                    <TableHead>#</TableHead>
                    <TableHead className='text-end'>左</TableHead>
                    <TableHead className='text-end'>右</TableHead>
                    <TableHead className='text-end'>共同</TableHead>
                    <TableHead className='text-end'>RMS 左/右</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {views.map((v) => {
                    const p = pairByView.get(v.index)
                    const ex = excluded.has(v.index)
                    return (
                      <TableRow
                        key={v.index}
                        className={cn(
                          'cursor-pointer',
                          selected === v.index && 'bg-brand/10',
                          ex && 'opacity-50'
                        )}
                        onClick={() => setSelected(v.index)}
                      >
                        <TableCell onClick={(e) => e.stopPropagation()}>
                          <Checkbox
                            checked={ex}
                            onCheckedChange={(c) =>
                              setExcluded((s) => {
                                const n = new Set(s)
                                if (c) n.add(v.index)
                                else n.delete(v.index)
                                return n
                              })
                            }
                            aria-label={`剔除视图 ${v.index}`}
                          />
                        </TableCell>
                        <TableCell className='font-mono'>
                          {v.index}
                          <span className='ms-2 text-xs text-muted-foreground'>
                            {v.name_left}
                          </span>
                        </TableCell>
                        <TableCell
                          className={cn(
                            'text-end font-mono',
                            !v.n_left && 'text-bad'
                          )}
                        >
                          {v.n_left || '—'}
                        </TableCell>
                        <TableCell
                          className={cn(
                            'text-end font-mono',
                            !v.n_right && 'text-bad'
                          )}
                        >
                          {v.n_right || '—'}
                        </TableCell>
                        <TableCell className='text-end font-mono'>
                          {v.n_common || '—'}
                        </TableCell>
                        <TableCell className='text-end font-mono text-xs'>
                          {p ? (
                            `${p.rms_left.toFixed(3)} / ${p.rms_right.toFixed(3)}`
                          ) : (
                            <span className='text-muted-foreground'>
                              未参与
                            </span>
                          )}
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
          </Card>
          <Card className='min-w-0'>
            <CardContent className='pt-6'>
              {selected != null ? (
                <ViewDetail
                  name={name}
                  view={selected}
                  shape={shape as [number, number]}
                />
              ) : (
                <p className='text-sm text-muted-foreground'>选择一个视图</p>
              )}
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card className='border-dashed'>
          <CardContent className='flex flex-col items-center gap-3 py-10 text-sm text-muted-foreground'>
            <Grid3x3 className='size-8' />
            还没有角点检测结果。先核对下方的标定板参数，然后检测角点（或直接标定）。
            <Button
              size='sm'
              onClick={() => onRun('calibrate_detect')}
              disabled={ov.busy || running}
            >
              <ScanSearch /> 检测角点
            </Button>
          </CardContent>
        </Card>
      )}

      {rep && ov.pairs.length > 0 && (
        <div className='grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]'>
          <Card className='min-w-0'>
            <CardHeader>
              <CardTitle className='text-base'>逐视图 RMS</CardTitle>
              <CardDescription>
                <span className='me-3 inline-flex items-center gap-1'>
                  <span className='size-2 rounded-sm bg-chart-2' /> 左
                </span>
                <span className='me-3 inline-flex items-center gap-1'>
                  <span className='size-2 rounded-sm bg-chart-4' /> 右
                </span>
                虚线为中位数；明显偏高的视图可考虑剔除。点击柱子查看该视图。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ViewRmsChart
                pairs={ov.pairs}
                views={
                  views.length || Math.max(...ov.pairs.map((p) => p.view)) + 1
                }
                excluded={excluded}
                selected={selected}
                onSelect={setSelected}
              />
              {ov.residuals && (
                <div className='mt-4 grid grid-cols-2 gap-4'>
                  <ResidualScatter
                    points={ov.residuals.left}
                    label='左相机残差'
                    tone='fill-chart-2'
                  />
                  <ResidualScatter
                    points={ov.residuals.right}
                    label='右相机残差'
                    tone='fill-chart-4'
                  />
                </div>
              )}
            </CardContent>
          </Card>
          <Card className='min-w-0'>
            <CardHeader>
              <CardTitle className='flex items-center gap-2 text-base'>
                <Box className='size-4 text-brand' /> 相机与标定板位姿
              </CardTitle>
              <CardDescription>
                左相机坐标系，单位 mm；标定板颜色由绿到红表示 RMS
                由低到高。拖动旋转，滚轮缩放，点击标定板选中视图。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <PosesPanel
                name={name}
                selected={selected}
                onSelect={setSelected}
              />
            </CardContent>
          </Card>
        </div>
      )}

      <BoardCard
        name={name}
        config={config}
        onSave={(patch) =>
          saveConfig.mutate(patch, {
            onSuccess: () => toast.success('标定设置已保存'),
          })
        }
        saving={saveConfig.isPending}
      />
    </div>
  )
}

function PosesPanel({
  name,
  selected,
  onSelect,
}: {
  name: string
  selected: number | null
  onSelect: (v: number) => void
}) {
  const { data } = useQuery({
    queryKey: ['calibration', name, 'poses'],
    queryFn: () => api.calibrationPoses(name),
    meta: { silent: true },
    retry: false,
  })
  if (!data) return <Skeleton className='h-80' />
  return (
    <div className='h-96 overflow-hidden rounded-xl border'>
      <Suspense fallback={<Skeleton className='h-full' />}>
        <Poses3D poses={data} selected={selected} onSelect={onSelect} />
      </Suspense>
    </div>
  )
}

function Intrinsics({
  label,
  cam,
}: {
  label: string
  cam: CalibrationReportCamera
}) {
  const K = cam.K
  const std = cam.std_fx_fy_cx_cy
  const rows: [string, number, number | undefined][] = [
    ['fx', K[0][0], std?.[0]],
    ['fy', K[1][1], std?.[1]],
    ['cx', K[0][2], std?.[2]],
    ['cy', K[1][2], std?.[3]],
  ]
  const distNames = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
  return (
    <div className='rounded-lg border p-3'>
      <div className='mb-2 flex items-baseline justify-between text-sm'>
        <span className='font-medium'>{label}</span>
        <span className='font-mono text-xs text-muted-foreground'>
          单目 RMS {cam.mono_rms_px.toFixed(4)} px · {cam.views_used}/
          {cam.views_total} 视图 · {cam.image_size.join('×')}
        </span>
      </div>
      <div className='tabular grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs'>
        {rows.map(([k, v, s]) => (
          <div key={k} className='flex justify-between'>
            <span className='text-muted-foreground'>{k}</span>
            <span>
              {v.toFixed(2)}
              {s != null && (
                <span className='text-muted-foreground'> ± {s.toFixed(2)}</span>
              )}
            </span>
          </div>
        ))}
        {cam.dist.map((d, i) => (
          <div key={i} className='flex justify-between'>
            <span className='text-muted-foreground'>
              {distNames[i] ?? `d${i}`}
            </span>
            <span>{d.toExponential(3)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function BoardCard({
  name,
  config,
  onSave,
  saving,
}: {
  name: string
  config: ExperimentConfig
  onSave: (patch: (c: ExperimentConfig) => ExperimentConfig) => void
  saving: boolean
}) {
  const cal = config.calibration as Record<string, unknown> | null
  const initial = (cal?.board ?? {}) as Record<string, unknown>
  const [board, setBoard] = useState<Record<string, unknown>>(initial)
  const [distortion, setDistortion] = useState(
    String(cal?.distortion ?? 'k1k2p1p2')
  )
  const dirty =
    JSON.stringify(board) !== JSON.stringify(initial) ||
    distortion !== cal?.distortion
  if (!cal) return null
  const num = (k: string, label: string, unit?: string) => (
    <div className='space-y-1'>
      <Label className='text-xs text-muted-foreground'>
        {label}
        {unit && ` (${unit})`}
      </Label>
      <Input
        className='h-8 font-mono'
        value={String(board[k] ?? '')}
        onChange={(e) =>
          setBoard((b) => ({ ...b, [k]: Number(e.target.value) }))
        }
      />
    </div>
  )
  const sq = Number(board.square_mm)
  const mk = Number(board.marker_mm)
  return (
    <Card>
      <CardHeader>
        <CardTitle className='text-base'>标定板与模型</CardTitle>
        <CardDescription>
          与打印的标定板逐项核对；下图按当前参数实时渲染。改动后会使角点检测缓存失效。
        </CardDescription>
        <CardAction>
          <Button
            size='sm'
            disabled={!dirty || saving}
            onClick={() =>
              onSave((c) => ({
                ...c,
                calibration: {
                  ...(c.calibration as object),
                  board,
                  distortion,
                },
              }))
            }
          >
            <Save /> 保存
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className='grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]'>
        <div className='space-y-4'>
          <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
            {num('squares_x', '横向方格')}
            {num('squares_y', '纵向方格')}
            {num('square_mm', '方格', 'mm')}
            {num('marker_mm', '码', 'mm')}
          </div>
          <div className='grid grid-cols-2 gap-3'>
            <div className='space-y-1'>
              <Label className='text-xs text-muted-foreground'>
                ArUco 字典
              </Label>
              <Input
                className='h-8 font-mono'
                value={String(board.dictionary ?? '')}
                onChange={(e) =>
                  setBoard((b) => ({ ...b, dictionary: e.target.value }))
                }
              />
            </div>
            <div className='space-y-1'>
              <Label className='text-xs text-muted-foreground'>畸变模型</Label>
              <Select value={distortion} onValueChange={setDistortion}>
                <SelectTrigger size='sm' className='w-full font-mono'>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value='k1k2p1p2'>k1k2p1p2（默认）</SelectItem>
                  <SelectItem value='k1k2p1p2k3'>k1k2p1p2k3</SelectItem>
                  <SelectItem value='rational'>rational</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className='grid grid-cols-3 gap-2'>
            <Metric
              label='板面尺寸'
              value={`${(Number(board.squares_x) * sq).toFixed(1)} × ${(Number(board.squares_y) * sq).toFixed(1)}`}
              unit='mm'
            />
            <Metric
              label='内角点'
              value={
                (Number(board.squares_x) - 1) * (Number(board.squares_y) - 1)
              }
            />
            <Metric
              label='码 / 方格'
              value={(mk / sq).toFixed(2)}
              verdict={mk > 0 && mk < sq ? 'ok' : 'bad'}
              rule='码必须小于方格；白边越宽，亚像素精化窗口越大（窗口取白边的 80%）'
            />
          </div>
          <p className='text-xs text-muted-foreground'>
            采集 20–40
            个不同深度和倾角的位姿，覆盖整个测量体积；标定与正式采集的分辨率、ROI、对焦、光圈必须一致（技术设计
            §3.5）。
          </p>
        </div>
        <div className='flex items-center justify-center rounded-xl border bg-white p-3'>
          <img
            src={api.boardUrl(name, board)}
            alt='标定板预览'
            className='max-h-72 w-auto'
          />
        </div>
      </CardContent>
    </Card>
  )
}

function StatusDot({ state }: { state: string }) {
  const c =
    state === 'done'
      ? 'bg-ok'
      : state === 'stale'
        ? 'bg-warn'
        : state === 'running'
          ? 'bg-brand animate-pulse'
          : 'bg-idle/60'
  return <span className={cn('size-2.5 rounded-full', c)} />
}
