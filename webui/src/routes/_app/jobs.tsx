import { createFileRoute } from '@tanstack/react-router'
import { JobsPage } from '@/features/jobs'

export const Route = createFileRoute('/_app/jobs')({
  component: JobsPage,
  validateSearch: (s: Record<string, unknown>): { job?: string } => ({
    job: typeof s.job === 'string' ? s.job : undefined,
  }),
})
