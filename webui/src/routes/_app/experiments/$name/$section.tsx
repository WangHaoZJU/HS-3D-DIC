import { createFileRoute } from '@tanstack/react-router'
import { ExperimentSection } from '@/features/experiment/section'

export const Route = createFileRoute('/_app/experiments/$name/$section')({
  component: ExperimentSection,
})
