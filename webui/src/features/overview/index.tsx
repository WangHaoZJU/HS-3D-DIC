import { useQueries, useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import {
  ArrowRight,
  Cpu,
  FlaskConical,
  HardDrive,
  ListTodo,
  Plus,
} from 'lucide-react'
import { useCurrentExperiment } from '@/stores/current-experiment'
import { api, qk, type ExperimentSummary } from '@/lib/api'
import { ago, duration, elapsed, pct } from '@/lib/format'
import { STAGE_LABEL } from '@/lib/stages'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { JobStateBadge } from '@/components/job-state-badge'
import { Page } from '@/components/layout/page'
import { ProgressBar } from '@/components/progress-bar'
import { StageStepper } from '@/components/stage-stepper'

export function Overview() {
  const experiments = useQuery({
    queryKey: qk.experiments,
    queryFn: api.experiments,
  })
  const jobs = useQuery({
    queryKey: [...qk.jobs, 'recent'],
    queryFn: () => api.jobs({ limit: 8 }),
  })
  const system = useQuery({
    queryKey: qk.system,
    queryFn: () => api.system(),
    staleTime: 30_000,
  })

  return (
    <Page
      title='概览'
      description='实验、计算资源与最近任务'
      actions={
        <Button asChild>
          <Link to='/experiments' search={{ create: true }}>
            <Plus /> 新建实验
          </Link>
        </Button>
      }
    >
      <div className='grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-3'>
        <Card className='min-w-0 xl:col-span-2'>
          <CardHeader>
            <div>
              <CardTitle className='flex items-center gap-2 text-base'>
                <FlaskConical className='size-4 text-brand' /> 实验
              </CardTitle>
              <CardDescription>工作区中的实验及其流程进度</CardDescription>
            </div>
            <CardAction>
              <Button asChild variant='ghost' size='sm'>
                <Link to='/experiments'>
                  全部 <ArrowRight />
                </Link>
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className='space-y-1'>
            {experiments.isLoading && <Skeleton className='h-24' />}
            {experiments.data?.length === 0 && <EmptyExperiments />}
            {experiments.data?.map((e) => (
              <ExperimentRow key={e.name} e={e} />
            ))}
          </CardContent>
        </Card>

        <div className='grid min-w-0 grid-cols-[minmax(0,1fr)] gap-4'>
          <Card>
            <CardHeader>
              <CardTitle className='flex items-center gap-2 text-base'>
                <Cpu className='size-4 text-brand' /> 计算资源
              </CardTitle>
            </CardHeader>
            <CardContent className='space-y-3 text-sm'>
              {system.isLoading && <Skeleton className='h-20' />}
              {system.data && (
                <>
                  <KV
                    k='默认后端'
                    v={
                      <span className='font-mono'>
                        {system.data.backends.auto ?? '—'}
                      </span>
                    }
                  />
                  <KV
                    k='可用后端'
                    v={
                      <span className='font-mono'>
                        {(system.data.backends.available ?? []).join(' · ') ||
                          '—'}
                      </span>
                    }
                  />
                  {system.data.gpus.map((g) => (
                    <div key={g.index} className='space-y-1'>
                      <KV k={`GPU ${g.index}`} v={g.name} />
                      <ProgressBar
                        value={g.memory_used_mb / g.memory_total_mb}
                        tone='info'
                      />
                      <div className='tabular text-xs text-muted-foreground'>
                        显存 {(g.memory_used_mb / 1024).toFixed(1)} /{' '}
                        {(g.memory_total_mb / 1024).toFixed(0)} GB · 利用率{' '}
                        {g.utilization_pct}%
                      </div>
                    </div>
                  ))}
                  {system.data.gpus.length === 0 && (
                    <KV
                      k='GPU'
                      v={
                        <span className='text-muted-foreground'>
                          未检测到（CPU 后端）
                        </span>
                      }
                    />
                  )}
                </>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className='flex items-center gap-2 text-base'>
                <HardDrive className='size-4 text-brand' /> 存储
              </CardTitle>
            </CardHeader>
            <CardContent className='space-y-3 text-sm'>
              {system.data &&
                Object.entries(system.data.disks).map(([k, d]) =>
                  d ? (
                    <div key={k} className='space-y-1'>
                      <KV
                        k={
                          {
                            data: '数据根目录',
                            results: '结果根目录',
                            workspace: '工作区',
                          }[k] ?? k
                        }
                        v={
                          <span className='tabular'>
                            剩余 {d.free_gb.toFixed(0)} GB
                          </span>
                        }
                      />
                      <ProgressBar
                        value={d.used_gb / d.total_gb}
                        tone={d.free_gb < 50 ? 'warn' : 'ok'}
                      />
                      <div
                        className='truncate font-mono text-xs text-muted-foreground'
                        title={d.path}
                      >
                        {d.path}
                      </div>
                    </div>
                  ) : null
                )}
            </CardContent>
          </Card>
        </div>

        <Card className='min-w-0 xl:col-span-3'>
          <CardHeader>
            <CardTitle className='flex items-center gap-2 text-base'>
              <ListTodo className='size-4 text-brand' /> 最近任务
            </CardTitle>
            <CardAction>
              <Button asChild variant='ghost' size='sm'>
                <Link to='/jobs'>
                  任务队列 <ArrowRight />
                </Link>
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent>
            {jobs.data?.length === 0 && (
              <p className='text-sm text-muted-foreground'>还没有任务。</p>
            )}
            <div className='divide-y'>
              {jobs.data?.map((j) => (
                <Link
                  key={j.id}
                  to='/jobs'
                  search={{ job: j.id }}
                  className='flex items-center gap-3 py-2 text-sm hover:bg-accent/40'
                >
                  <JobStateBadge
                    state={j.state}
                    className='w-16 justify-center'
                  />
                  <span className='w-40 truncate font-medium'>
                    {j.experiment || '—'}
                  </span>
                  <span className='w-24 text-muted-foreground'>
                    {STAGE_LABEL[j.stage] ?? j.stage}
                  </span>
                  <span className='hidden flex-1 md:block'>
                    {j.state === 'running' && (
                      <ProgressBar value={j.fraction} className='max-w-xs' />
                    )}
                    {j.state === 'failed' && (
                      <span className='truncate text-xs text-bad'>
                        {j.error}
                      </span>
                    )}
                  </span>
                  <span className='tabular ms-auto text-xs text-muted-foreground'>
                    {j.state === 'running'
                      ? `已用 ${duration(elapsed(j.started_utc))}`
                      : ago(j.created_utc)}
                  </span>
                </Link>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </Page>
  )
}

function KV({ k, v }: { k: React.ReactNode; v: React.ReactNode }) {
  return (
    <div className='flex items-center justify-between gap-3'>
      <span className='text-muted-foreground'>{k}</span>
      <span className='truncate text-end'>{v}</span>
    </div>
  )
}

function EmptyExperiments() {
  return (
    <div className='flex flex-col items-center gap-2 rounded-lg border border-dashed py-10 text-center'>
      <FlaskConical className='size-8 text-muted-foreground' />
      <p className='text-sm text-muted-foreground'>
        工作区还没有实验。可以新建一个，或运行
      </p>
      <code className='rounded bg-muted px-2 py-1 font-mono text-xs'>
        uv run python scripts/make_demo_dataset.py
      </code>
      <p className='text-sm text-muted-foreground'>生成一套合成演示数据。</p>
    </div>
  )
}

function ExperimentRow({ e }: { e: ExperimentSummary }) {
  const setName = useCurrentExperiment((s) => s.setName)
  const [status] = useQueries({
    queries: [
      {
        queryKey: qk.status(e.name),
        queryFn: () => api.status(e.name),
        enabled: e.valid,
      },
    ],
  })
  const fractions = status.data
    ? Object.values(status.data.stages).filter((s) => s.state === 'done')
        .length / 7
    : null
  return (
    <Link
      to='/experiments/$name/$section'
      params={{ name: e.name, section: 'config' }}
      onClick={() => setName(e.name)}
      className='flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg px-3 py-2.5 transition-colors hover:bg-accent/50'
    >
      <div className='min-w-0 flex-1'>
        <div className='flex items-center gap-2'>
          <span className='truncate font-medium'>{e.name}</span>
          {!e.valid && (
            <span className='rounded border border-bad/40 px-1 text-xs text-bad'>
              配置无效
            </span>
          )}
        </div>
        <div className='truncate text-xs text-muted-foreground'>
          {e.description || e.error || '—'}
        </div>
      </div>
      <StageStepper name={e.name} status={status.data} compact />
      <span className='tabular w-24 text-end text-xs text-muted-foreground'>
        {fractions != null ? `${pct(fractions)} · ` : ''}
        {ago(e.modified_utc)}
      </span>
    </Link>
  )
}
