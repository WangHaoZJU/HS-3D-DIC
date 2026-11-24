import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import { api, qk } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Page } from '@/components/layout/page'
import { ProgressBar } from '@/components/progress-bar'

export function SystemPage() {
  const qc = useQueryClient()
  const { data, isLoading } = useQuery({
    queryKey: qk.system,
    queryFn: () => api.system(),
  })
  const refresh = useMutation({
    mutationFn: () => api.system(true),
    onSuccess: (d) => qc.setQueryData(qk.system, d),
  })
  return (
    <Page
      title='系统'
      description='运行环境、计算后端、GPU 与存储（等价于 scripts/check_env.py）'
      actions={
        <Button
          variant='outline'
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending}
        >
          <RefreshCw className={refresh.isPending ? 'animate-spin' : ''} />{' '}
          重新检测后端
        </Button>
      }
    >
      {isLoading && <Skeleton className='h-64' />}
      {data && (
        <div className='grid gap-4 lg:grid-cols-2'>
          <Card>
            <CardHeader>
              <CardTitle className='text-base'>运行环境</CardTitle>
            </CardHeader>
            <CardContent>
              <Dl
                rows={[
                  ['主机', data.environment.hostname],
                  ['平台', data.environment.platform],
                  ['Python', data.environment.python],
                  ['stereodic', data.environment.stereodic_version],
                  [
                    'git commit',
                    <span className='font-mono'>
                      {data.git.commit ?? '—'}
                      {data.git.dirty ? '（有未提交修改）' : ''}
                    </span>,
                  ],
                  [
                    '前端托管',
                    data.static_ui
                      ? '已构建（webui/dist）'
                      : '开发模式（Vite）',
                  ],
                ]}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className='text-base'>计算后端</CardTitle>
            </CardHeader>
            <CardContent>
              {data.backends.error ? (
                <p className='text-sm text-bad'>{data.backends.error}</p>
              ) : (
                <Dl
                  rows={[
                    [
                      'auto 选择',
                      <span className='font-mono'>{data.backends.auto}</span>,
                    ],
                    [
                      '可用',
                      <span className='font-mono'>
                        {data.backends.available?.join(', ')}
                      </span>,
                    ],
                    ['Numba', <Json v={data.backends.numba} />],
                    ['CUDA', <Json v={data.backends.cuda} />],
                  ]}
                />
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className='text-base'>GPU</CardTitle>
            </CardHeader>
            <CardContent className='space-y-4'>
              {data.gpus.length === 0 && (
                <p className='text-sm text-muted-foreground'>
                  没有检测到 NVIDIA GPU（nvidia-smi 不可用）。
                </p>
              )}
              {data.gpus.map((g) => (
                <div key={g.index} className='space-y-1.5 text-sm'>
                  <div className='flex justify-between'>
                    <span className='font-medium'>
                      #{g.index} {g.name}
                    </span>
                    <span className='tabular text-muted-foreground'>
                      {g.temperature_c} °C · {g.utilization_pct}%
                    </span>
                  </div>
                  <ProgressBar
                    value={g.memory_used_mb / g.memory_total_mb}
                    tone='info'
                  />
                  <div className='tabular text-xs text-muted-foreground'>
                    显存 {(g.memory_used_mb / 1024).toFixed(1)} /{' '}
                    {(g.memory_total_mb / 1024).toFixed(1)} GB
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className='text-base'>目录与磁盘</CardTitle>
            </CardHeader>
            <CardContent className='space-y-4'>
              {Object.entries(data.roots).map(([k, path]) => {
                const d = data.disks[k]
                return (
                  <div key={k} className='space-y-1.5 text-sm'>
                    <div className='flex justify-between gap-3'>
                      <span className='font-medium'>
                        {{
                          data: '数据根目录 data_root',
                          results: '结果根目录 results_root',
                          workspace: '实验工作区',
                        }[k] ?? k}
                      </span>
                      {d && (
                        <span className='tabular text-muted-foreground'>
                          剩余 {d.free_gb.toFixed(0)} / {d.total_gb.toFixed(0)}{' '}
                          GB
                        </span>
                      )}
                    </div>
                    {d && (
                      <ProgressBar
                        value={d.used_gb / d.total_gb}
                        tone={d.free_gb < 50 ? 'warn' : 'ok'}
                      />
                    )}
                    <div className='truncate font-mono text-xs text-muted-foreground'>
                      {path}
                    </div>
                  </div>
                )
              })}
            </CardContent>
          </Card>
        </div>
      )}
    </Page>
  )
}

function Dl({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className='grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm'>
      {rows.map(([k, v]) => (
        <div key={k} className='contents'>
          <dt className='text-muted-foreground'>{k}</dt>
          <dd className='min-w-0 break-words'>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

function Json({ v }: { v: unknown }) {
  if (!v || typeof v !== 'object') return <span>—</span>
  return (
    <span className='font-mono text-xs'>
      {Object.entries(v as Record<string, unknown>)
        .map(([k, x]) => `${k}=${String(x)}`)
        .join('  ')}
    </span>
  )
}
