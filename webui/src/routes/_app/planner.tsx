import { createFileRoute } from '@tanstack/react-router'
import { Calculator } from 'lucide-react'
import { Page } from '@/components/layout/page'
import { Planned } from '@/components/planned'

export const Route = createFileRoute('/_app/planner')({
  component: () => (
    <Page
      title='规划计算器'
      description='帧率、曝光、混叠与录制时长估算（纯前端，不依赖实验数据）'
    >
      <Planned
        icon={Calculator}
        milestone='U1'
        items={[
          '帧率、分辨率、机身内存 → 可录制时长、频率分辨率 Δf = 1/T',
          '目标频率与帧率 → 是否满足 fs ≥ 2.5f、混叠后的频率、是否接近 0 或 fs/2',
          '振幅、物方比例、允许模糊 → 曝光时间上限 t ≲ δs / (2πfA)',
          '夹角、像素噪声、帧数 → 单帧面内/离面噪声与谐波拟合后的噪声底',
        ]}
      />
    </Page>
  ),
})
