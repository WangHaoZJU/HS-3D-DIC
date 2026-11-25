import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { Crosshair, Mountain, Play } from 'lucide-react'
import { toast } from 'sonner'
import { api, qk } from '@/lib/api'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { MaskEditor } from './mask-editor'
import { StereoResults } from './results'

export function ReferencePage({ name }: { name: string }) {
  const qc = useQueryClient()
  const masks = useQuery({
    queryKey: qk.masks(name),
    queryFn: () => api.masks(name),
    meta: { silent: true },
    retry: false,
  })
  const ref = useQuery({
    queryKey: qk.reference(name),
    queryFn: () => api.reference(name),
    meta: { silent: true },
    retry: false,
  })
  const status = useQuery({
    queryKey: qk.status(name),
    queryFn: () => api.status(name),
  })
  const run = useMutation({
    mutationFn: () => api.submit(name, 'stereo'),
    onSuccess: () => {
      toast.success('已提交立体匹配')
      qc.invalidateQueries({ queryKey: qk.jobs })
    },
  })
  const st = status.data?.stages.stereo.state
  const downstream =
    status.data &&
    ['temporal', 'reconstruct', 'rbm'].some(
      (s) => status.data.stages[s as 'temporal'].state !== 'missing'
    )
  return (
    <div className='space-y-6'>
      <Card>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <Crosshair className='size-4 text-brand' /> 测量区域与测点网格
          </CardTitle>
          <CardDescription>
            在左相机参考帧（第 {masks.data?.reference_frame ?? '…'} 帧，可在
            <Link
              to='/experiments/$name/$section'
              params={{ name, section: 'import' }}
              className='mx-1 text-brand hover:underline'
            >
              数据导入
            </Link>
            页按清晰度选择）上绘制
            ROI、测量区域和刚体参考区；修改子区和步长时测点网格实时更新。
          </CardDescription>
        </CardHeader>
        <CardContent>
          {masks.isLoading && <Skeleton className='h-96' />}
          {masks.isError && (
            <p className='text-sm text-bad'>
              无法打开左相机序列，请先在数据导入页设置图像目录。
            </p>
          )}
          {masks.data && (
            <MaskEditor
              key={JSON.stringify(masks.data)}
              name={name}
              info={masks.data}
            />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <Mountain className='size-4 text-brand' /> 立体匹配与参考形貌
          </CardTitle>
          <CardDescription>
            左参考帧 L0 ↔ 右参考帧 R0：SIFT 种子 + 可靠性引导传播 + 二阶
            IC-GN，再三角化得到参考曲面和局部坐标系（技术设计 §7.5–§9.2）。
          </CardDescription>
          <CardAction>
            <Button
              size='sm'
              disabled={run.isPending || status.data?.busy}
              onClick={() => {
                if (
                  downstream &&
                  !confirm(
                    '重跑立体匹配会清空时序 DIC、重建与分析结果，继续吗？'
                  )
                )
                  return
                run.mutate()
              }}
            >
              <Play /> {st === 'done' ? '重新匹配' : '运行立体匹配'}
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent>
          {st === 'stale' && (
            <p className='mb-3 text-sm text-warn'>
              设置已改变，下面是上一次的结果；重新运行立体匹配后更新。
            </p>
          )}
          {ref.isLoading && <Skeleton className='h-80' />}
          {ref.isError && (
            <p className='py-8 text-center text-sm text-muted-foreground'>
              {status.data?.busy ? '立体匹配运行中…' : '还没有立体匹配结果。'}
            </p>
          )}
          {ref.data && masks.data && (
            <StereoResults
              name={name}
              r={ref.data}
              shape={masks.data.image_shape}
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
}
