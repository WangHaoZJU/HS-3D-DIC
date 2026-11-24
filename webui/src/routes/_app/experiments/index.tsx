import { createFileRoute } from '@tanstack/react-router'
import { ExperimentsPage } from '@/features/experiments'

export const Route = createFileRoute('/_app/experiments/')({
  component: ExperimentsPage,
  validateSearch: (s: Record<string, unknown>): { create?: boolean } => ({
    create: s.create === true || s.create === 'true' ? true : undefined,
  }),
})
