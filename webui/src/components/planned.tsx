import type { LucideIcon } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

/** Placeholder for a page scheduled in a later milestone (docs/UI_Design.md §11). */
export function Planned({
  icon: Icon,
  milestone,
  items,
  note,
}: {
  icon: LucideIcon
  milestone: string
  items: string[]
  note?: string
}) {
  return (
    <Card className='overflow-hidden border-dashed'>
      <CardContent className='grid gap-6 p-6 md:grid-cols-[auto_1fr]'>
        <div className='flex size-14 items-center justify-center rounded-xl bg-brand/10 text-brand'>
          <Icon className='size-7' />
        </div>
        <div>
          <div className='flex items-center gap-2'>
            <span className='rounded-md border border-brand/30 bg-brand/10 px-1.5 py-0.5 font-mono text-xs text-brand'>
              {milestone}
            </span>
            <span className='text-sm text-muted-foreground'>计划中的页面</span>
          </div>
          <ul className='mt-3 space-y-1.5 text-sm'>
            {items.map((t) => (
              <li key={t} className='flex gap-2'>
                <span className='mt-2 size-1 shrink-0 rounded-full bg-muted-foreground/60' />
                <span>{t}</span>
              </li>
            ))}
          </ul>
          {note && <p className='mt-4 text-sm text-muted-foreground'>{note}</p>}
        </div>
      </CardContent>
    </Card>
  )
}
