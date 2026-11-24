import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { ArrowRight, FlaskConical, Laptop, Moon, Sun } from 'lucide-react'
import { useCurrentExperiment } from '@/stores/current-experiment'
import { api, qk } from '@/lib/api'
import { useSearch } from '@/context/search-provider'
import { useTheme } from '@/context/theme-provider'
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command'
import { useNavGroups } from './layout/nav-data'
import { ScrollArea } from './ui/scroll-area'

export function CommandMenu() {
  const navigate = useNavigate()
  const { setTheme } = useTheme()
  const { open, setOpen } = useSearch()
  const groups = useNavGroups()
  const setExperiment = useCurrentExperiment((s) => s.setName)
  const { data: experiments = [] } = useQuery({
    queryKey: qk.experiments,
    queryFn: api.experiments,
    enabled: open,
  })

  const run = React.useCallback(
    (command: () => unknown) => {
      setOpen(false)
      command()
    },
    [setOpen]
  )

  return (
    <CommandDialog modal open={open} onOpenChange={setOpen}>
      <CommandInput placeholder='输入页面、实验或命令…' />
      <CommandList>
        <ScrollArea type='hover' className='h-80 pe-1'>
          <CommandEmpty>没有匹配项</CommandEmpty>
          {groups.map((group) => (
            <CommandGroup key={group.title} heading={group.title}>
              {group.items.map((item, i) =>
                item.url ? (
                  <CommandItem
                    key={`${item.url}-${i}`}
                    value={`${group.title} ${item.title}`}
                    onSelect={() =>
                      run(() => navigate({ to: item.url as string }))
                    }
                  >
                    <div className='flex size-4 items-center justify-center'>
                      <ArrowRight className='size-2 text-muted-foreground/80' />
                    </div>
                    {item.title}
                  </CommandItem>
                ) : null
              )}
            </CommandGroup>
          ))}
          {experiments.length > 0 && (
            <CommandGroup heading='切换实验'>
              {experiments.map((e) => (
                <CommandItem
                  key={e.name}
                  value={`实验 ${e.name} ${e.description}`}
                  onSelect={() =>
                    run(() => {
                      setExperiment(e.name)
                      navigate({
                        to: '/experiments/$name/$section',
                        params: { name: e.name, section: 'config' },
                      })
                    })
                  }
                >
                  <FlaskConical className='text-muted-foreground' />
                  {e.name}
                </CommandItem>
              ))}
            </CommandGroup>
          )}
          <CommandSeparator />
          <CommandGroup heading='主题'>
            <CommandItem onSelect={() => run(() => setTheme('light'))}>
              <Sun /> <span>浅色</span>
            </CommandItem>
            <CommandItem onSelect={() => run(() => setTheme('dark'))}>
              <Moon className='scale-90' />
              <span>深色</span>
            </CommandItem>
            <CommandItem onSelect={() => run(() => setTheme('system'))}>
              <Laptop />
              <span>跟随系统</span>
            </CommandItem>
          </CommandGroup>
        </ScrollArea>
      </CommandList>
    </CommandDialog>
  )
}
