import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { ChevronsUpDown, FlaskConical, Plus } from 'lucide-react'
import { Logo } from '@/assets/logo'
import { useCurrentExperiment } from '@/stores/current-experiment'
import { api, qk } from '@/lib/api'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from '@/components/ui/sidebar'

export function ExperimentSwitcher() {
  const { isMobile } = useSidebar()
  const navigate = useNavigate()
  const { name, setName } = useCurrentExperiment()
  const { data: experiments = [] } = useQuery({
    queryKey: qk.experiments,
    queryFn: api.experiments,
  })

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton
              size='lg'
              className='data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground'
            >
              <div className='flex aspect-square size-8 items-center justify-center rounded-lg bg-brand text-brand-foreground'>
                <Logo className='size-5' />
              </div>
              <div className='grid flex-1 text-start text-sm leading-tight'>
                <span className='truncate font-semibold'>StereoDIC</span>
                <span className='truncate text-xs text-muted-foreground'>
                  {name ?? '未选择实验'}
                </span>
              </div>
              <ChevronsUpDown className='ms-auto' />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            className='w-(--radix-dropdown-menu-trigger-width) min-w-56 rounded-lg'
            align='start'
            side={isMobile ? 'bottom' : 'right'}
            sideOffset={4}
          >
            <DropdownMenuLabel className='text-xs text-muted-foreground'>
              实验
            </DropdownMenuLabel>
            {experiments.map((e) => (
              <DropdownMenuItem
                key={e.name}
                onClick={() => {
                  setName(e.name)
                  navigate({
                    to: '/experiments/$name/$section',
                    params: { name: e.name, section: 'config' },
                  })
                }}
                className='gap-2 p-2'
              >
                <FlaskConical className='size-4 shrink-0 text-muted-foreground' />
                <span className='truncate'>{e.name}</span>
                {e.name === name && (
                  <span className='ms-auto size-1.5 rounded-full bg-brand' />
                )}
              </DropdownMenuItem>
            ))}
            {experiments.length === 0 && (
              <div className='px-2 py-1.5 text-sm text-muted-foreground'>
                还没有实验
              </div>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className='gap-2 p-2'
              onClick={() =>
                navigate({ to: '/experiments', search: { create: true } })
              }
            >
              <div className='flex size-6 items-center justify-center rounded-md border bg-background'>
                <Plus className='size-4' />
              </div>
              <div className='font-medium text-muted-foreground'>新建实验</div>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
