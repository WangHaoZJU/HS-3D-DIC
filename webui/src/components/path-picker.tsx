import { useState } from 'react'
import { FolderSearch } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { FileBrowserDialog } from '@/components/file-browser'

/** Text input with a server-side file/directory browser. */
export function PathPicker({
  value,
  onChange,
  mode,
  root = 'data',
  placeholder,
  className,
  id,
}: {
  value: string | null | undefined
  onChange: (v: string | null) => void
  mode: 'dir' | 'file'
  root?: string
  placeholder?: string
  className?: string
  id?: string
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className={cn('flex gap-2', className)}>
      <Input
        id={id}
        value={value ?? ''}
        placeholder={
          placeholder ?? (mode === 'dir' ? '目录路径' : '文件路径（可选）')
        }
        className='font-mono text-xs'
        onChange={(e) =>
          onChange(e.target.value === '' ? null : e.target.value)
        }
      />
      <Button
        type='button'
        variant='outline'
        size='icon'
        onClick={() => setOpen(true)}
        aria-label='浏览'
      >
        <FolderSearch />
      </Button>
      <FileBrowserDialog
        open={open}
        onOpenChange={setOpen}
        mode={mode}
        initialRoot={root}
        onPick={(p) => onChange(p)}
      />
    </div>
  )
}
