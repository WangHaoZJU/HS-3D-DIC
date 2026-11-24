import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Outlet, useParams } from '@tanstack/react-router'
import { ChevronDown, Play } from 'lucide-react'
import { toast } from 'sonner'
import { useCurrentExperiment } from '@/stores/current-experiment'
import { api, qk } from '@/lib/api'
import { RUNNABLE, SECTIONS } from '@/lib/stages'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Label } from '@/components/ui/label'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { Page } from '@/components/layout/page'
import { StageStepper } from '@/components/stage-stepper'

export function ExperimentLayout() {
  const { name, section } = useParams({ strict: false }) as {
    name: string
    section?: string
  }
  const setName = useCurrentExperiment((s) => s.setName)
  useEffect(() => setName(name), [name, setName])
  const status = useQuery({
    queryKey: qk.status(name),
    queryFn: () => api.status(name),
    refetchInterval: (q) => (q.state.data?.busy ? 5_000 : false),
  })
  const meta = SECTIONS.find((s) => s.key === section)
  return (
    <Page
      title={
        <span className='flex items-baseline gap-3'>
          {meta?.label ?? '实验'}
          <span className='text-base font-normal text-muted-foreground'>
            {name}
          </span>
        </span>
      }
      description={meta?.description}
      headerExtra={<StageStepper name={name} status={status.data} />}
      actions={<RunMenu name={name} />}
      fluid
    >
      <Outlet />
    </Page>
  )
}

function RunMenu({ name }: { name: string }) {
  const qc = useQueryClient()
  const [force, setForce] = useState(false)
  const [confirmStage, setConfirmStage] = useState<string | null>(null)
  const { data: status } = useQuery({
    queryKey: qk.status(name),
    queryFn: () => api.status(name),
  })
  const submit = useMutation({
    mutationFn: (stage: string) => api.submit(name, stage, {}, force),
    onSuccess: (job) => {
      toast.success(`已提交：${job.stage}`, { description: job.id })
      qc.invalidateQueries({ queryKey: qk.jobs })
    },
  })
  const downstreamDone =
    status &&
    ['temporal', 'reconstruct', 'rbm', 'analyze'].some((s) =>
      ['done', 'stale', 'partial'].includes(
        status.stages[s as 'temporal'].state
      )
    )
  const run = (stage: string) => {
    if (
      (stage === 'stereo' || (stage === 'pipeline' && force)) &&
      downstreamDone
    )
      setConfirmStage(stage)
    else submit.mutate(stage)
  }
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button>
            <Play /> 运行 <ChevronDown className='opacity-70' />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align='end' className='w-72'>
          <DropdownMenuLabel className='text-xs text-muted-foreground'>
            提交到任务队列
          </DropdownMenuLabel>
          {RUNNABLE.map((r) => (
            <DropdownMenuItem
              key={r.key}
              onClick={() => run(r.key)}
              className='flex-col items-start gap-0'
            >
              <span>{r.label}</span>
              {r.hint && (
                <span className='text-xs text-muted-foreground'>{r.hint}</span>
              )}
            </DropdownMenuItem>
          ))}
          <DropdownMenuSeparator />
          <div
            className='flex items-center gap-2 px-2 py-1.5'
            onClick={(e) => e.stopPropagation()}
          >
            <Checkbox
              id='force'
              checked={force}
              onCheckedChange={(v) => setForce(!!v)}
            />
            <Label htmlFor='force' className='text-xs'>
              强制重算（忽略已完成标记）
            </Label>
          </div>
        </DropdownMenuContent>
      </DropdownMenu>
      <ConfirmDialog
        open={confirmStage != null}
        onOpenChange={(o) => !o && setConfirmStage(null)}
        title='重跑参考帧立体匹配？'
        desc='立体阶段会重建整个结果文件，已有的时序 DIC、重建、刚体补偿和分析结果都会被清空，需要重新计算。'
        confirmText='清空并重跑'
        destructive
        handleConfirm={() => {
          if (confirmStage) submit.mutate(confirmStage)
          setConfirmStage(null)
        }}
      />
    </>
  )
}
