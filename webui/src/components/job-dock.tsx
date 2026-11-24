import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { ChevronUp, LoaderCircle, Square } from 'lucide-react'
import { toast } from 'sonner'
import { useLive } from '@/stores/live'
import { api, errorMessage, qk, type Job } from '@/lib/api'
import { duration, elapsed } from '@/lib/format'
import { STAGE_LABEL } from '@/lib/stages'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ProgressBar } from '@/components/progress-bar'

/** Bottom bar with the running job (docs/UI_Design.md §7.1). */
export function JobDock() {
  const [open, setOpen] = useState(false)
  const { data: active = [] } = useQuery({
    queryKey: [...qk.jobs, 'active'],
    queryFn: () => api.jobs({ active: true }),
    refetchInterval: 15_000,
  })
  const liveJobs = useLive((s) => s.jobs)
  const running = active
    .map((j) => liveJobs[j.id] ?? j)
    .find((j) => j.state === 'running' || j.state === 'cancelling')
  const queued = active.filter((j) => j.state === 'queued').length
  if (!running && !queued) return null
  return (
    <div className='sticky bottom-0 z-40 mt-auto border-t bg-background/85 backdrop-blur-lg'>
      {running ? (
        <RunningJob
          job={running}
          queued={queued}
          open={open}
          setOpen={setOpen}
        />
      ) : (
        <div className='flex h-11 items-center gap-2 px-4 text-sm text-muted-foreground'>
          <LoaderCircle className='size-4 animate-spin' /> {queued} 个任务排队中
        </div>
      )}
    </div>
  )
}

function RunningJob({
  job,
  queued,
  open,
  setOpen,
}: {
  job: Job
  queued: number
  open: boolean
  setOpen: (v: boolean) => void
}) {
  const p = useLive((s) => s.progress[job.id])
  const logs = useLive((s) => s.logs[job.id])
  const cancel = useMutation({
    mutationFn: () => api.cancel(job.id),
    onError: (e) => toast.error(errorMessage(e)),
  })
  const done = p?.done ?? job.done
  const total = p?.total ?? job.total
  const fraction = total ? done / total : null
  const stage = p?.stage || job.stage
  const phase = p?.phase || job.phase
  return (
    <div>
      {open && (
        <pre className='max-h-56 overflow-auto border-b px-4 py-2 font-mono text-xs leading-relaxed text-muted-foreground'>
          {(logs ?? []).slice(-60).map((l, i) => (
            <div
              key={i}
              className={cn(
                l.level === 'ERROR' && 'text-bad',
                l.level === 'WARNING' && 'text-warn'
              )}
            >
              {l.text}
            </div>
          ))}
          {!logs?.length && '等待日志…'}
        </pre>
      )}
      <div className='flex h-12 items-center gap-3 px-4 text-sm'>
        <LoaderCircle className='size-4 shrink-0 animate-spin text-brand' />
        <Link
          to='/jobs'
          className='flex min-w-0 items-baseline gap-2 hover:underline'
          title={job.id}
        >
          <span className='truncate font-medium'>{job.experiment || '—'}</span>
          <span className='text-muted-foreground'>·</span>
          <span className='shrink-0'>{STAGE_LABEL[stage] ?? stage}</span>
          {phase && (
            <span className='hidden truncate text-muted-foreground sm:inline'>
              {phase}
            </span>
          )}
        </Link>
        <ProgressBar value={fraction} className='mx-2 hidden w-56 md:block' />
        <span className='tabular hidden font-mono text-xs text-muted-foreground lg:inline'>
          {total ? `${done}/${total}` : ''}
          {p?.rate ? ` · ${p.rate.toFixed(1)}/s` : ''}
          {p?.eta_s != null ? ` · 剩余 ${duration(p.eta_s)}` : ''}
          {` · 已用 ${duration(elapsed(job.started_utc))}`}
        </span>
        {queued > 0 && (
          <span className='hidden text-xs text-muted-foreground xl:inline'>
            +{queued} 排队
          </span>
        )}
        <div className='ms-auto flex items-center gap-1'>
          <Button
            size='sm'
            variant='ghost'
            disabled={job.state === 'cancelling' || cancel.isPending}
            onClick={() => cancel.mutate()}
          >
            <Square className='size-3.5' />
            {job.state === 'cancelling' ? '取消中…' : '取消'}
          </Button>
          <Button
            size='icon'
            variant='ghost'
            className='size-8'
            onClick={() => setOpen(!open)}
            aria-label='日志'
          >
            <ChevronUp
              className={cn(
                'size-4 transition-transform',
                open && 'rotate-180'
              )}
            />
          </Button>
        </div>
      </div>
    </div>
  )
}
