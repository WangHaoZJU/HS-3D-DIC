import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { FlaskRound, RotateCcw, ScrollText, Skull, Square } from 'lucide-react'
import { toast } from 'sonner'
import { useLive } from '@/stores/live'
import { api, qk, type Job } from '@/lib/api'
import { clock, duration, elapsed } from '@/lib/format'
import { STAGE_LABEL } from '@/lib/stages'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { JobStateBadge } from '@/components/job-state-badge'
import { Page } from '@/components/layout/page'
import { ProgressBar } from '@/components/progress-bar'

export function JobsPage() {
  const search = useSearch({ from: '/_app/jobs' })
  const navigate = useNavigate({ from: '/jobs' })
  const { data: jobs = [], isLoading } = useQuery({
    queryKey: [...qk.jobs, 'all'],
    queryFn: () => api.jobs({ limit: 200 }),
    refetchInterval: 10_000,
  })
  const live = useLive((s) => s.jobs)
  const rows = useMemo(() => jobs.map((j) => live[j.id] ?? j), [jobs, live])
  const qc = useQueryClient()
  const noop = useMutation({
    mutationFn: () => api.submit('', 'noop', { seconds: 12, steps: 120 }),
    onSuccess: () => {
      toast.success('测试任务已提交')
      qc.invalidateQueries({ queryKey: qk.jobs })
    },
  })
  const selected = rows.find((j) => j.id === search.job)

  return (
    <Page
      title='任务队列'
      description='一次只运行一个任务（单 GPU），其余排队；取消在帧边界安全生效，之后可续算'
      actions={
        <Button
          variant='outline'
          onClick={() => noop.mutate()}
          disabled={noop.isPending}
        >
          <FlaskRound /> 运行测试任务
        </Button>
      }
    >
      <Card className='p-0'>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className='w-24'>状态</TableHead>
              <TableHead>实验</TableHead>
              <TableHead>阶段</TableHead>
              <TableHead className='w-[30%]'>进度</TableHead>
              <TableHead>提交</TableHead>
              <TableHead>耗时</TableHead>
              <TableHead className='w-32 text-end'>操作</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading && (
              <TableRow>
                <TableCell colSpan={7} className='text-muted-foreground'>
                  加载中…
                </TableCell>
              </TableRow>
            )}
            {!isLoading && rows.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={7}
                  className='py-10 text-center text-muted-foreground'
                >
                  还没有任务。可以点右上角“运行测试任务”看看进度推送。
                </TableCell>
              </TableRow>
            )}
            {rows.map((j) => (
              <JobRow
                key={j.id}
                job={j}
                onOpen={() => navigate({ search: { job: j.id } })}
                active={j.id === search.job}
              />
            ))}
          </TableBody>
        </Table>
      </Card>
      <Sheet
        open={!!selected}
        onOpenChange={(o) => !o && navigate({ search: {} })}
      >
        <SheetContent className='flex w-full flex-col gap-0 sm:max-w-2xl'>
          {selected && <JobDetail key={selected.id} job={selected} />}
        </SheetContent>
      </Sheet>
    </Page>
  )
}

function JobRow({
  job,
  onOpen,
  active,
}: {
  job: Job
  onOpen: () => void
  active: boolean
}) {
  const p = useLive((s) => s.progress[job.id])
  const isActive = job.state === 'running' || job.state === 'cancelling'
  const done = isActive && p ? p.done : job.done
  const total = isActive && p ? p.total : job.total
  const phase = isActive && p ? p.phase : job.phase
  return (
    <TableRow
      className={cn('cursor-pointer', active && 'bg-accent/50')}
      onClick={onOpen}
    >
      <TableCell>
        <JobStateBadge state={job.state} />
      </TableCell>
      <TableCell className='max-w-48 truncate font-medium'>
        {job.experiment || '—'}
      </TableCell>
      <TableCell>
        {STAGE_LABEL[job.stage] ?? job.stage}
        {job.force && <span className='ms-1 text-xs text-warn'>强制</span>}
      </TableCell>
      <TableCell>
        {job.state === 'failed' ? (
          <span className='line-clamp-1 text-xs text-bad' title={job.error}>
            {job.error}
          </span>
        ) : (
          <div className='flex items-center gap-2'>
            <ProgressBar
              value={
                total
                  ? done / total
                  : isActive
                    ? null
                    : job.state === 'succeeded'
                      ? 1
                      : 0
              }
              tone={
                job.state === 'succeeded'
                  ? 'ok'
                  : job.state === 'cancelled'
                    ? 'warn'
                    : 'brand'
              }
              className='flex-1'
            />
            <span className='tabular w-28 truncate text-xs text-muted-foreground'>
              {total ? `${done}/${total}` : ''} {phase}
            </span>
          </div>
        )}
      </TableCell>
      <TableCell className='tabular text-xs text-muted-foreground'>
        {clock(job.created_utc)}
      </TableCell>
      <TableCell className='tabular text-xs'>
        {duration(elapsed(job.started_utc, job.finished_utc))}
      </TableCell>
      <TableCell className='text-end' onClick={(e) => e.stopPropagation()}>
        <JobActions job={job} />
      </TableCell>
    </TableRow>
  )
}

function JobActions({ job }: { job: Job }) {
  const qc = useQueryClient()
  const refresh = () => qc.invalidateQueries({ queryKey: qk.jobs })
  const cancel = useMutation({
    mutationFn: () => api.cancel(job.id),
    onSuccess: refresh,
  })
  const retry = useMutation({
    mutationFn: () => api.retry(job.id),
    onSuccess: () => {
      toast.success(
        job.state === 'cancelled' ? '已重新提交（从断点续算）' : '已重新提交'
      )
      refresh()
    },
  })
  const kill = useMutation({
    mutationFn: () => api.kill(job.id),
    onSuccess: refresh,
  })
  const [confirmKill, setConfirmKill] = useState(false)
  const active =
    job.state === 'queued' ||
    job.state === 'running' ||
    job.state === 'cancelling'
  return (
    <div className='flex justify-end gap-1'>
      {active && job.state !== 'cancelling' && (
        <IconAction
          label='取消'
          onClick={() => cancel.mutate()}
          icon={Square}
        />
      )}
      {job.state === 'cancelling' && (
        <IconAction
          label='强制终止'
          onClick={() => setConfirmKill(true)}
          icon={Skull}
          className='text-bad'
        />
      )}
      {!active && (
        <IconAction
          label={job.state === 'cancelled' ? '续算' : '重试'}
          onClick={() => retry.mutate()}
          icon={RotateCcw}
        />
      )}
      <ConfirmDialog
        open={confirmKill}
        onOpenChange={setConfirmKill}
        title='强制终止任务？'
        desc='立即结束 worker 进程。如果它正在写 HDF5 结果文件，文件可能损坏，需要重跑该阶段。协作取消通常会在当前帧结束后生效。'
        confirmText='强制终止'
        destructive
        handleConfirm={() => {
          kill.mutate()
          setConfirmKill(false)
        }}
      />
    </div>
  )
}

function IconAction({
  label,
  onClick,
  icon: Icon,
  className,
}: {
  label: string
  onClick: () => void
  icon: typeof Square
  className?: string
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          size='icon'
          variant='ghost'
          className={cn('size-8', className)}
          onClick={onClick}
          aria-label={label}
        >
          <Icon className='size-4' />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function JobDetail({ job }: { job: Job }) {
  const [text, setText] = useState('')
  const offset = useRef(0)
  const box = useRef<HTMLPreElement>(null)
  const follow = useRef(true)
  const active =
    job.state === 'queued' ||
    job.state === 'running' ||
    job.state === 'cancelling'

  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const r = await api.log(job.id, offset.current)
        if (stop) return
        if (r.text) {
          offset.current = r.offset
          setText((t) => (t + r.text).slice(-400_000))
        }
      } catch {
        /* ignore */
      }
    }
    tick()
    const id = active ? setInterval(tick, 1000) : undefined
    return () => {
      stop = true
      clearInterval(id)
    }
  }, [job.id, active])

  useEffect(() => {
    if (follow.current && box.current)
      box.current.scrollTop = box.current.scrollHeight
  }, [text])

  return (
    <>
      <SheetHeader className='border-b'>
        <SheetTitle className='flex items-center gap-2'>
          <ScrollText className='size-4' />
          {STAGE_LABEL[job.stage] ?? job.stage}
          <JobStateBadge state={job.state} />
        </SheetTitle>
        <SheetDescription className='font-mono text-xs'>
          {job.id} · {job.experiment || '无实验'} · 提交于{' '}
          {clock(job.created_utc)}
        </SheetDescription>
        {job.error && (
          <p className='rounded-md border border-bad/30 bg-bad/10 p-2 text-sm text-bad'>
            {job.error}
          </p>
        )}
        {Object.keys(job.params).length > 0 && (
          <pre className='rounded-md bg-muted p-2 font-mono text-xs'>
            {JSON.stringify(job.params, null, 2)}
          </pre>
        )}
      </SheetHeader>
      <pre
        ref={box}
        onScroll={(e) => {
          const el = e.currentTarget
          follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
        }}
        className='flex-1 overflow-auto p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground'
      >
        {text.split('\n').map((line, i) => (
          <div
            key={i}
            className={cn(
              / ERROR /.test(line) && 'text-bad',
              / WARNING /.test(line) && 'text-warn'
            )}
          >
            {line}
          </div>
        ))}
        {!text && (active ? '等待日志…' : '没有日志。')}
      </pre>
    </>
  )
}
