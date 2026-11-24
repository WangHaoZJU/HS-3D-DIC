import { CheckCircle2, CircleAlert, CircleX, Info } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'

export type Verdict = 'ok' | 'warn' | 'bad' | 'info'

const ICON = { ok: CheckCircle2, warn: CircleAlert, bad: CircleX, info: Info }
const TONE = {
  ok: 'text-ok',
  warn: 'text-warn',
  bad: 'text-bad',
  info: 'text-muted-foreground',
}

/** Small labelled value with a pass / warn / fail mark and the rule in a tooltip. */
export function Metric({
  label,
  value,
  unit,
  verdict = 'info',
  rule,
  className,
}: {
  label: string
  value: React.ReactNode
  unit?: string
  verdict?: Verdict
  rule?: string
  className?: string
}) {
  const Icon = ICON[verdict]
  const body = (
    <div
      className={cn(
        'min-w-0 rounded-lg border bg-muted/20 px-3 py-2',
        className
      )}
    >
      <div className='flex items-center gap-1.5 text-xs text-muted-foreground'>
        <Icon className={cn('size-3.5 shrink-0', TONE[verdict])} />
        <span className='truncate'>{label}</span>
      </div>
      <div className='tabular mt-0.5 truncate font-mono text-sm'>
        {value}
        {unit && (
          <span className='ms-1 text-xs text-muted-foreground'>{unit}</span>
        )}
      </div>
    </div>
  )
  if (!rule) return body
  return (
    <Tooltip>
      <TooltipTrigger asChild>{body}</TooltipTrigger>
      <TooltipContent className='max-w-72'>{rule}</TooltipContent>
    </Tooltip>
  )
}

export function band(
  v: number | null | undefined,
  ok: [number, number],
  warn: [number, number]
): Verdict {
  if (v == null || !Number.isFinite(v)) return 'info'
  if (v >= ok[0] && v <= ok[1]) return 'ok'
  if (v >= warn[0] && v <= warn[1]) return 'warn'
  return 'bad'
}
