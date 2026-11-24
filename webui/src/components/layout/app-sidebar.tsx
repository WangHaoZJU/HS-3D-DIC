import { useQuery } from '@tanstack/react-query'
import { GitCommitHorizontal } from 'lucide-react'
import { useLive } from '@/stores/live'
import { api, qk } from '@/lib/api'
import { cn } from '@/lib/utils'
import { useLayout } from '@/context/layout-provider'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
} from '@/components/ui/sidebar'
import { ExperimentSwitcher } from './experiment-switcher'
import { useNavGroups } from './nav-data'
import { NavGroup } from './nav-group'

export function AppSidebar() {
  const { collapsible, variant } = useLayout()
  const { data: active = [] } = useQuery({
    queryKey: [...qk.jobs, 'active'],
    queryFn: () => api.jobs({ active: true }),
    refetchInterval: 15_000,
  })
  const groups = useNavGroups(active.length)
  return (
    <Sidebar collapsible={collapsible} variant={variant}>
      <SidebarHeader>
        <ExperimentSwitcher />
      </SidebarHeader>
      <SidebarContent>
        {groups.map((props) => (
          <NavGroup key={props.title} {...props} />
        ))}
      </SidebarContent>
      <SidebarFooter>
        <ConnectionStatus />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}

function ConnectionStatus() {
  const connected = useLive((s) => s.connected)
  const { data } = useQuery({
    queryKey: qk.system,
    queryFn: () => api.system(),
    staleTime: 60_000,
  })
  const commit = data?.git.commit?.slice(0, 7)
  return (
    <div className='flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground group-data-[collapsible=icon]:hidden'>
      <span
        className={cn(
          'size-2 rounded-full',
          connected ? 'bg-ok' : 'animate-pulse bg-bad'
        )}
        aria-hidden
      />
      <span>{connected ? '已连接' : '连接中…'}</span>
      {commit && (
        <span
          className='ms-auto flex items-center gap-1 font-mono'
          title={data?.git.dirty ? '工作区有未提交修改' : ''}
        >
          <GitCommitHorizontal className='size-3.5' />
          {commit}
          {data?.git.dirty ? '*' : ''}
        </span>
      )}
    </div>
  )
}
