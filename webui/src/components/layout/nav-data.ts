import {
  Calculator,
  LayoutDashboard,
  ListTodo,
  Server,
  FlaskConical,
} from 'lucide-react'
import { useCurrentExperiment } from '@/stores/current-experiment'
import { SECTIONS } from '@/lib/stages'
import { type NavGroup } from './types'

export function useNavGroups(activeJobs = 0): NavGroup[] {
  const name = useCurrentExperiment((s) => s.name)
  const groups: NavGroup[] = [
    {
      title: '总览',
      items: [
        { title: '概览', url: '/', icon: LayoutDashboard },
        { title: '实验列表', url: '/experiments', icon: FlaskConical },
        {
          title: '任务队列',
          url: '/jobs',
          icon: ListTodo,
          badge: activeJobs ? String(activeJobs) : undefined,
        },
      ],
    },
  ]
  if (name) {
    groups.push({
      title: `实验 · ${name}`,
      items: SECTIONS.map((s) => ({
        title: s.label,
        url: `/experiments/${encodeURIComponent(name)}/${s.key}`,
        icon: s.icon,
        badge: s.milestone,
      })),
    })
  }
  groups.push({
    title: '工具',
    items: [
      { title: '规划计算器', url: '/planner', icon: Calculator },
      { title: '系统', url: '/system', icon: Server },
    ],
  })
  return groups
}
