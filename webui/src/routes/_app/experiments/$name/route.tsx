import { createFileRoute } from '@tanstack/react-router'
import { ExperimentLayout } from '@/features/experiment/layout'

export const Route = createFileRoute('/_app/experiments/$name')({
  component: ExperimentLayout,
})
