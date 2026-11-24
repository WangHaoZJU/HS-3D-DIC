import { cn } from '@/lib/utils'

/** Thin progress bar; indeterminate (animated) when value is null. */
export function ProgressBar({
  value,
  className,
  tone = 'brand',
}: {
  value: number | null | undefined
  className?: string
  tone?: 'brand' | 'ok' | 'warn' | 'bad' | 'info'
}) {
  const color = {
    brand: 'bg-brand',
    ok: 'bg-ok',
    warn: 'bg-warn',
    bad: 'bg-bad',
    info: 'bg-info',
  }[tone]
  return (
    <div
      className={cn(
        'relative h-1.5 overflow-hidden rounded-full bg-muted',
        className
      )}
    >
      {value == null ? (
        <div
          className={cn(
            'absolute inset-y-0 w-1/3 animate-[indeterminate_1.4s_ease-in-out_infinite] rounded-full',
            color
          )}
        />
      ) : (
        <div
          className={cn(
            'h-full rounded-full transition-[width] duration-300',
            color
          )}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
      )}
    </div>
  )
}
