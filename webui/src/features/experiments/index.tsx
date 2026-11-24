import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useSearch } from '@tanstack/react-router'
import { Copy, FlaskConical, MoreHorizontal, Plus, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { useCurrentExperiment } from '@/stores/current-experiment'
import { api, errorMessage, qk, type ExperimentSummary } from '@/lib/api'
import { ago } from '@/lib/format'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { Page } from '@/components/layout/page'
import { StageStepper } from '@/components/stage-stepper'

const TEMPLATE = '__template__'

export function ExperimentsPage() {
  const search = useSearch({ from: '/_app/experiments/' })
  const navigate = useNavigate({ from: '/experiments/' })
  const { data: experiments = [], isLoading } = useQuery({
    queryKey: qk.experiments,
    queryFn: api.experiments,
  })
  const [copyFrom, setCopyFrom] = useState<string | null>(null)
  const open = !!search.create
  const setOpen = (o: boolean) =>
    navigate({ search: o ? { create: true } : { create: undefined } })

  return (
    <Page
      title='实验列表'
      description='每个实验是工作区里的一个目录：experiment.yaml、掩膜、标定与结果文件'
      actions={
        <Button
          onClick={() => {
            setCopyFrom(null)
            setOpen(true)
          }}
        >
          <Plus /> 新建实验
        </Button>
      }
    >
      <Card className='divide-y p-0'>
        {isLoading && (
          <div className='p-6 text-sm text-muted-foreground'>加载中…</div>
        )}
        {!isLoading && experiments.length === 0 && (
          <div className='flex flex-col items-center gap-2 py-14 text-center text-sm text-muted-foreground'>
            <FlaskConical className='size-8' />
            还没有实验
          </div>
        )}
        {experiments.map((e) => (
          <Row
            key={e.name}
            e={e}
            onCopy={() => {
              setCopyFrom(e.name)
              setOpen(true)
            }}
          />
        ))}
      </Card>
      <CreateDialog
        open={open}
        onOpenChange={setOpen}
        experiments={experiments}
        copyFrom={copyFrom}
      />
    </Page>
  )
}

function Row({ e, onCopy }: { e: ExperimentSummary; onCopy: () => void }) {
  const qc = useQueryClient()
  const { name: current, setName } = useCurrentExperiment()
  const { data: status } = useQuery({
    queryKey: qk.status(e.name),
    queryFn: () => api.status(e.name),
    enabled: e.valid,
  })
  const [confirm, setConfirm] = useState(false)
  const del = useMutation({
    mutationFn: () => api.deleteExperiment(e.name),
    onSuccess: (r) => {
      toast.success(`已移到回收目录：${r.moved_to}`)
      if (current === e.name) setName(null)
      qc.invalidateQueries({ queryKey: qk.experiments })
    },
  })
  return (
    <div className='flex flex-wrap items-center gap-x-6 gap-y-2 px-5 py-4'>
      <Link
        to='/experiments/$name/$section'
        params={{ name: e.name, section: 'config' }}
        onClick={() => setName(e.name)}
        className='min-w-0 flex-1'
      >
        <div className='flex items-center gap-2'>
          <span className='font-medium hover:underline'>{e.name}</span>
          {e.name === current && (
            <span className='rounded bg-brand/15 px-1.5 text-xs text-brand'>
              当前
            </span>
          )}
          {!e.valid && (
            <span className='rounded border border-bad/40 px-1 text-xs text-bad'>
              配置无效
            </span>
          )}
        </div>
        <p className='mt-0.5 line-clamp-1 text-sm text-muted-foreground'>
          {e.description || e.error || '无描述'}
        </p>
      </Link>
      {e.valid && <StageStepper name={e.name} status={status} />}
      <span className='w-24 text-end text-xs text-muted-foreground'>
        {ago(e.modified_utc)}
      </span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            size='icon'
            variant='ghost'
            className='size-8'
            aria-label='更多'
          >
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align='end'>
          <DropdownMenuItem onClick={onCopy}>
            <Copy /> 复制为新实验
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <a href={api.exportUrl(e.name)} download>
              导出 YAML
            </a>
          </DropdownMenuItem>
          <DropdownMenuItem
            className='text-bad focus:text-bad'
            onClick={() => setConfirm(true)}
          >
            <Trash2 /> 删除
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`删除实验 ${e.name}？`}
        desc='整个实验目录（配置、掩膜、结果文件）会移到工作区的 .trash 目录，可以手动恢复。原始图像不受影响。'
        confirmText='移到回收目录'
        destructive
        handleConfirm={() => {
          del.mutate()
          setConfirm(false)
        }}
      />
    </div>
  )
}

function CreateDialog({
  open,
  onOpenChange,
  experiments,
  copyFrom,
}: {
  open: boolean
  onOpenChange: (o: boolean) => void
  experiments: ExperimentSummary[]
  copyFrom: string | null
}) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const setCurrent = useCurrentExperiment((s) => s.setName)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [source, setSource] = useState<string>(copyFrom ?? TEMPLATE)
  const [error, setError] = useState<string | null>(null)
  const create = useMutation({
    mutationFn: () =>
      api.createExperiment({
        name,
        description,
        copy_from: source === TEMPLATE ? null : source,
      }),
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: qk.experiments })
      setCurrent(d.name)
      onOpenChange(false)
      navigate({
        to: '/experiments/$name/$section',
        params: { name: d.name, section: 'config' },
      })
      toast.success(`已创建 ${d.name}`)
    },
    onError: (e) => setError(errorMessage(e)),
  })
  const valid = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(name)
  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (o) setSource(copyFrom ?? TEMPLATE)
        setError(null)
        onOpenChange(o)
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>新建实验</DialogTitle>
          <DialogDescription>
            在工作区创建{' '}
            <code className='font-mono'>&lt;name&gt;/experiment.yaml</code>
            。复制已有实验时，标定文件共用，结果文件独立。
          </DialogDescription>
        </DialogHeader>
        <form
          className='space-y-4'
          onSubmit={(ev) => {
            ev.preventDefault()
            if (valid) create.mutate()
          }}
        >
          <div className='space-y-1.5'>
            <Label htmlFor='exp-name'>名称</Label>
            <Input
              id='exp-name'
              autoFocus
              placeholder='blade_T5_1000Hz'
              value={name}
              onChange={(e) => setName(e.target.value)}
              className='font-mono'
            />
            <p className='text-xs text-muted-foreground'>
              字母、数字、_ - .，不超过 64 个字符；同时作为目录名。
            </p>
          </div>
          <div className='space-y-1.5'>
            <Label>来源</Label>
            <Select value={source} onValueChange={setSource}>
              <SelectTrigger className='w-full'>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={TEMPLATE}>
                  模板（configs/experiments/example.yaml）
                </SelectItem>
                {experiments.map((e) => (
                  <SelectItem key={e.name} value={e.name}>
                    复制 {e.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className='space-y-1.5'>
            <Label htmlFor='exp-desc'>描述</Label>
            <Textarea
              id='exp-desc'
              rows={2}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>
          {error && <p className='text-sm text-bad'>{error}</p>}
          <DialogFooter>
            <Button
              type='button'
              variant='outline'
              onClick={() => onOpenChange(false)}
            >
              取消
            </Button>
            <Button type='submit' disabled={!valid || create.isPending}>
              创建
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
