import { createFileRoute, redirect } from '@tanstack/react-router'

export const Route = createFileRoute('/_app/experiments/$name/')({
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/experiments/$name/$section',
      params: { name: params.name, section: 'config' },
    })
  },
})
