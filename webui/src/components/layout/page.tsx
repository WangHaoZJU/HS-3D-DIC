import { type ReactNode } from 'react'
import { ConfigDrawer } from '@/components/config-drawer'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'
import { Header } from './header'
import { Main } from './main'

/** Standard page: sticky header with search/theme, then content. */
export function Page({
  title,
  description,
  actions,
  children,
  fluid,
  headerExtra,
}: {
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
  fluid?: boolean
  headerExtra?: ReactNode
}) {
  return (
    <>
      <Header fixed>
        <div className='min-w-0 flex-1'>{headerExtra}</div>
        <div className='ms-auto flex items-center gap-2'>
          <Search placeholder='搜索页面与命令' />
          <ThemeSwitch />
          <ConfigDrawer />
        </div>
      </Header>
      <Main fluid={fluid} className='pb-24'>
        <div className='mb-5 flex flex-wrap items-end justify-between gap-3'>
          <div className='min-w-0'>
            <h1 className='text-2xl font-semibold tracking-tight'>{title}</h1>
            {description && (
              <p className='mt-1 text-sm text-muted-foreground'>
                {description}
              </p>
            )}
          </div>
          {actions && (
            <div className='flex flex-wrap items-center gap-2'>{actions}</div>
          )}
        </div>
        {children}
      </Main>
    </>
  )
}
