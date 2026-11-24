import { type SVGProps } from 'react'
import { cn } from '@/lib/utils'

/** Two converging views over a vibrating surface. */
export function Logo({ className, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox='0 0 24 24'
      xmlns='http://www.w3.org/2000/svg'
      height='24'
      width='24'
      fill='none'
      stroke='currentColor'
      strokeWidth='1.8'
      strokeLinecap='round'
      strokeLinejoin='round'
      className={cn('size-6', className)}
      {...props}
    >
      <title>StereoDIC</title>
      <circle cx='5' cy='5' r='2.2' />
      <circle cx='19' cy='5' r='2.2' />
      <path d='M6.4 6.8 11 13.2M17.6 6.8 13 13.2' opacity='0.7' />
      <path d='M2.5 18c2.4-3 4.3-3 6.3 0s3.9 3 6.4 0 3.9-3 6.3 0' />
    </svg>
  )
}
