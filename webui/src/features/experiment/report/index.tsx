import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Download, ExternalLink, FileText } from 'lucide-react'
import { api, qk } from '@/lib/api'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'

export function ReportPage({ name }: { name: string }) {
  const status = useQuery({
    queryKey: qk.status(name),
    queryFn: () => api.status(name),
  })
  const vib = useQuery({
    queryKey: [...qk.results(name), 'vibration', null],
    queryFn: () => api.vibration(name),
    enabled: !status.data?.busy,
    retry: false,
    meta: { silent: true },
  })
  const [point, setPoint] = useState('0')
  const reconstructed = status.data?.stages.reconstruct.state === 'done'
  const stereo = status.data?.stages.stereo.state === 'done'
  const items: { label: string; desc: string; href: string; ok: boolean }[] = [
    {
      label: '实验配置 YAML',
      desc: '可提交到仓库存档',
      href: api.exportUrl(name),
      ok: true,
    },
    {
      label: '参考形貌 PLY',
      desc: '三维点与三角网格，左相机坐标 mm',
      href: api.exportUrl2(name, 'shape.ply'),
      ok: !!stereo,
    },
    {
      label: '峰值表 CSV',
      desc: '频率、幅值、阻尼比',
      href: api.exportUrl2(name, 'peaks.csv'),
      ok: !!vib.data,
    },
    ...(vib.data?.harmonic ?? []).map((h) => ({
      label: `谐波拟合 ${h.frequency_hz.toFixed(3)} Hz CSV`,
      desc: '逐点坐标、三分量振幅与相位',
      href: api.exportUrl2(name, `harmonic/${h.key}.csv`),
      ok: true,
    })),
  ]
  return (
    <div className='space-y-6'>
      <Card>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-base'>
            <FileText className='size-4 text-brand' /> 测量报告
          </CardTitle>
          <CardDescription>
            自包含的 HTML 报告：实验信息与 git
            commit、流程状态、标定、有效率、频谱与峰值表、各拟合频率的振幅分布、噪声底。在浏览器中打开后可直接打印为
            PDF。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button asChild disabled={!stereo}>
            <a href={api.reportUrl(name)} target='_blank' rel='noreferrer'>
              <ExternalLink /> 打开报告
            </a>
          </Button>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className='text-base'>数据导出</CardTitle>
          <CardDescription>
            完整数据在实验目录的 results.h5（HDF5，schema
            v1）中；这里导出常用的小文件。
          </CardDescription>
        </CardHeader>
        <CardContent className='divide-y'>
          {items.map((it) => (
            <div key={it.label} className='flex items-center gap-3 py-2.5'>
              <div className='min-w-0 flex-1'>
                <div className='text-sm font-medium'>{it.label}</div>
                <div className='text-xs text-muted-foreground'>{it.desc}</div>
              </div>
              <Button size='sm' variant='outline' asChild disabled={!it.ok}>
                <a
                  href={it.ok ? it.href : undefined}
                  download
                  aria-disabled={!it.ok}
                  className={it.ok ? '' : 'pointer-events-none opacity-50'}
                >
                  <Download /> 下载
                </a>
              </Button>
            </div>
          ))}
          <div className='flex items-center gap-3 py-2.5'>
            <div className='min-w-0 flex-1'>
              <div className='text-sm font-medium'>单点三分量时间序列 CSV</div>
              <div className='text-xs text-muted-foreground'>
                时间、局部位移 c/s/n（原始与刚体补偿后），mm
              </div>
            </div>
            <Input
              className='h-8 w-24 font-mono'
              value={point}
              onChange={(e) => setPoint(e.target.value.replace(/\D/g, ''))}
              placeholder='测点号'
            />
            <Button
              size='sm'
              variant='outline'
              asChild
              disabled={!reconstructed}
            >
              <a
                href={api.exportUrl2(name, `point/${point || 0}.csv`)}
                download
                className={
                  reconstructed ? '' : 'pointer-events-none opacity-50'
                }
              >
                <Download /> 下载
              </a>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
