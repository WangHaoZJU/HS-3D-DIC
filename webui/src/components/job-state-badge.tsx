import type { JobState } from '@/lib/api'
import { JOB_STATE } from '@/lib/stages'
import { cn } from '@/lib/utils'

export function JobStateBadge({
  state,
  className,
}: {
  state: JobState
  className?: string
}) {
  const meta = JOB_STATE[state]
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-xs font-medium',
        meta.className,
        className
      )}
    >
      {(state === 'running' || state === 'cancelling') && (
        <span className='size-1.5 animate-pulse rounded-full bg-current' />
      )}
      {meta.label}
    </span>
  )
}
