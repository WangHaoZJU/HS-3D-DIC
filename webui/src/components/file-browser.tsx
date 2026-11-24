import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ChevronRight,
  File,
  FileImage,
  Folder,
  FolderOpen,
  CornerLeftUp,
} from 'lucide-react'
import { api, qk } from '@/lib/api'
import { bytes } from '@/lib/format'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

const ROOT_LABEL: Record<string, string> = {
  data: '数据',
  results: '结果',
  workspace: '工作区',
}
const IMAGE = new Set(['.tif', '.tiff', '.png', '.bmp'])

/** Server-side file picker limited to the configured roots (UI_Design §6.1 /fs). */
export function FileBrowserDialog({
  open,
  onOpenChange,
  mode,
  initialRoot = 'data',
  title,
  onPick,
}: {
  open: boolean
  onOpenChange: (o: boolean) => void
  mode: 'dir' | 'file'
  initialRoot?: string
  title?: string
  onPick: (absPath: string) => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className='sm:max-w-2xl'>
        {/* content unmounts when closed, so the browser state starts fresh on every open */}
        <BrowserBody
          mode={mode}
          initialRoot={initialRoot}
          title={title}
          onPick={onPick}
          onClose={() => onOpenChange(false)}
        />
      </DialogContent>
    </Dialog>
  )
}

function BrowserBody({
  mode,
  initialRoot,
  title,
  onPick,
  onClose,
}: {
  mode: 'dir' | 'file'
  initialRoot: string
  title?: string
  onPick: (absPath: string) => void
  onClose: () => void
}) {
  const onOpenChange = (o: boolean) => !o && onClose()
  const [root, setRoot] = useState(initialRoot)
  const [path, setPath] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const system = useQuery({
    queryKey: qk.system,
    queryFn: () => api.system(),
    staleTime: 60_000,
  })
  const listing = useQuery({
    queryKey: qk.fs(root, path),
    queryFn: () => api.fs(root, path),
    meta: { silent: true },
  })
  const roots = Object.keys(system.data?.roots ?? { data: '', workspace: '' })
  const crumbs = path ? path.split('/') : []
  const join = (name: string) => (path ? `${path}/${name}` : name)
  const abs = (name?: string) =>
    listing.data
      ? name
        ? `${listing.data.abs_path}/${name}`
        : listing.data.abs_path
      : ''
  const pickValue = mode === 'dir' ? abs() : selected ? abs(selected) : ''

  return (
    <>
      <DialogHeader>
        <DialogTitle>
          {title ?? (mode === 'dir' ? '选择目录' : '选择文件')}
        </DialogTitle>
        <DialogDescription>只能浏览服务器上配置的根目录。</DialogDescription>
      </DialogHeader>
      <Tabs
        value={root}
        onValueChange={(r) => {
          setRoot(r)
          setPath('')
          setSelected(null)
        }}
      >
        <TabsList>
          {roots.map((r) => (
            <TabsTrigger key={r} value={r}>
              {ROOT_LABEL[r] ?? r}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      <div className='flex flex-wrap items-center gap-1 text-sm'>
        <button
          className='rounded px-1 text-muted-foreground hover:text-foreground'
          onClick={() => setPath('')}
        >
          {ROOT_LABEL[root] ?? root}
        </button>
        {crumbs.map((c, i) => (
          <span key={i} className='flex items-center gap-1'>
            <ChevronRight className='size-3 text-muted-foreground' />
            <button
              className='rounded px-1 hover:underline'
              onClick={() => setPath(crumbs.slice(0, i + 1).join('/'))}
            >
              {c}
            </button>
          </span>
        ))}
      </div>
      <ScrollArea className='h-80 rounded-md border'>
        {listing.isError && (
          <div className='p-4 text-sm text-bad'>
            无法读取该目录（根目录不存在或无权限）。
          </div>
        )}
        {listing.data?.parent != null && (
          <Entry
            icon={CornerLeftUp}
            name='..'
            onClick={() => setPath(listing.data!.parent ?? '')}
            muted
          />
        )}
        {listing.data?.entries.map((e) =>
          e.kind === 'dir' ? (
            <Entry
              key={e.name}
              icon={Folder}
              name={e.name}
              onClick={() => setPath(join(e.name))}
            />
          ) : (
            <Entry
              key={e.name}
              icon={IMAGE.has(e.ext) ? FileImage : File}
              name={e.name}
              meta={bytes(e.size)}
              muted={mode === 'dir'}
              active={selected === e.name}
              onClick={() => mode === 'file' && setSelected(e.name)}
              onDoubleClick={() => {
                if (mode === 'file') {
                  onPick(abs(e.name))
                  onOpenChange(false)
                }
              }}
            />
          )
        )}
        {listing.data && listing.data.entries.length === 0 && (
          <div className='p-4 text-sm text-muted-foreground'>空目录</div>
        )}
      </ScrollArea>
      <div className='flex items-center gap-2 text-xs text-muted-foreground'>
        <FolderOpen className='size-3.5' />
        <span className='truncate font-mono'>{pickValue || '—'}</span>
        {listing.data && listing.data.n_images > 0 && (
          <span className='ms-auto shrink-0 rounded bg-brand/10 px-1.5 py-0.5 text-brand'>
            {listing.data.n_images} 张图像
          </span>
        )}
      </div>
      <DialogFooter>
        <Button variant='outline' onClick={() => onOpenChange(false)}>
          取消
        </Button>
        <Button
          disabled={!pickValue}
          onClick={() => {
            onPick(pickValue)
            onOpenChange(false)
          }}
        >
          {mode === 'dir' ? '选择此目录' : '选择文件'}
        </Button>
      </DialogFooter>
    </>
  )
}

function Entry({
  icon: Icon,
  name,
  meta,
  onClick,
  onDoubleClick,
  muted,
  active,
}: {
  icon: typeof Folder
  name: string
  meta?: string
  onClick?: () => void
  onDoubleClick?: () => void
  muted?: boolean
  active?: boolean
}) {
  return (
    <button
      type='button'
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      className={cn(
        'flex w-full items-center gap-2 px-3 py-1.5 text-start text-sm hover:bg-accent',
        muted && 'text-muted-foreground',
        active && 'bg-brand/15 text-foreground'
      )}
    >
      <Icon className='size-4 shrink-0' />
      <span className='truncate'>{name}</span>
      {meta && (
        <span className='tabular ms-auto shrink-0 text-xs text-muted-foreground'>
          {meta}
        </span>
      )}
    </button>
  )
}
