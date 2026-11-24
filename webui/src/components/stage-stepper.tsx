import { Link } from '@tanstack/react-router'
import {
  Check,
  CircleDashed,
  LoaderCircle,
  PauseCircle,
  TriangleAlert,
  HelpCircle,
} from 'lucide-react'
import type { ExperimentStatus, StageState } from '@/lib/api'
import { PIPELINE, STAGE_STATE } from '@/lib/stages'
import { cn } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'

const ICON: Record<StageState, typeof Check> = {
  done: Check,
  stale: TriangleAlert,
  partial: PauseCircle,
  missing: CircleDashed,
  unknown: HelpCircle,
  running: LoaderCircle,
}

/** Pipeline status strip (docs/UI_Design.md §4.2). */
export function StageStepper({
  name,
  status,
  compact = false,
}: {
  name: string
  status: ExperimentStatus | undefined
  compact?: boolean
}) {
  return (
    <ol
      className={cn(
        'flex items-center',
        compact ? 'gap-1' : 'no-scrollbar gap-0.5 overflow-x-auto'
      )}
    >
      {PIPELINE.map((step, i) => {
        const rec = status?.stages[step.key]
        const state: StageState = rec?.state ?? 'unknown'
        const meta = STAGE_STATE[state]
        const Icon = ICON[state]
        const tip = (
          <div className='space-y-0.5 text-xs'>
            <div className='font-medium'>
              {step.label} · {meta.label}
              {state === 'partial' && rec?.fraction != null
                ? ` ${(rec.fraction * 100).toFixed(0)}%`
                : ''}
            </div>
            {rec?.detail && (
              <div className='max-w-72 opacity-80'>{rec.detail}</div>
            )}
            {rec?.done_utc && (
              <div className='opacity-60'>
                {new Date(rec.done_utc).toLocaleString('zh-CN', {
                  hour12: false,
                })}
              </div>
            )}
          </div>
        )
        if (compact) {
          return (
            <li key={step.key}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className={cn('block h-1.5 w-6 rounded-full', meta.dot)}
                  />
                </TooltipTrigger>
                <TooltipContent>{tip}</TooltipContent>
              </Tooltip>
            </li>
          )
        }
        return (
          <li key={step.key} className='flex items-center'>
            {i > 0 && (
              <span className='mx-0.5 h-px w-3 bg-border' aria-hidden />
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  to='/experiments/$name/$section'
                  params={{ name, section: step.section }}
                  className={cn(
                    'flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs whitespace-nowrap transition-colors hover:bg-accent',
                    state === 'running' && 'border-brand/40 bg-brand/10',
                    state === 'stale' && 'border-warn/30'
                  )}
                >
                  <Icon
                    className={cn(
                      'size-3.5',
                      meta.text,
                      state === 'running' && 'animate-spin'
                    )}
                  />
                  <span>{step.label}</span>
                </Link>
              </TooltipTrigger>
              <TooltipContent>{tip}</TooltipContent>
            </Tooltip>
          </li>
        )
      })}
    </ol>
  )
}
