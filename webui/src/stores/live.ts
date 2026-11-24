/** Live job events from the server WebSocket (/api/ws). */
import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { create } from 'zustand'
import { qk, type Job } from '@/lib/api'

export type LiveProgress = {
  stage: string
  phase: string
  done: number
  total: number
  message: string
  rate: number | null
  eta_s: number | null
  at: number
}

export type LogLine = { level: string; text: string }

type LiveState = {
  connected: boolean
  jobs: Record<string, Job>
  progress: Record<string, LiveProgress>
  logs: Record<string, LogLine[]>
  summaries: Record<string, Record<string, unknown>[]>
  setConnected: (c: boolean) => void
  apply: (msg: Record<string, unknown>) => void
}

const MAX_LOG = 500
const MAX_SUMMARIES = 20000

export const useLive = create<LiveState>((set) => ({
  connected: false,
  jobs: {},
  progress: {},
  logs: {},
  summaries: {},
  setConnected: (connected) => set({ connected }),
  apply: (msg) =>
    set((s) => {
      const id = msg.job as string
      switch (msg.type) {
        case 'job': {
          const job = msg.job as unknown as Job
          return { jobs: { ...s.jobs, [job.id]: job } }
        }
        case 'progress': {
          const p: LiveProgress = {
            stage: String(msg.stage ?? ''),
            phase: String(msg.phase ?? ''),
            done: Number(msg.done ?? 0),
            total: Number(msg.total ?? 0),
            message: String(msg.message ?? ''),
            rate: (msg.rate as number | null) ?? null,
            eta_s: (msg.eta_s as number | null) ?? null,
            at: Date.now(),
          }
          return { progress: { ...s.progress, [id]: p } }
        }
        case 'log': {
          const lines = [
            ...(s.logs[id] ?? []),
            { level: String(msg.level), text: String(msg.text) },
          ]
          return { logs: { ...s.logs, [id]: lines.slice(-MAX_LOG) } }
        }
        case 'summary': {
          const { type: _t, job: _j, ...rest } = msg
          const list = [...(s.summaries[id] ?? []), rest]
          return {
            summaries: { ...s.summaries, [id]: list.slice(-MAX_SUMMARIES) },
          }
        }
        default:
          return {}
      }
    }),
}))

/** Connect once (in the app shell); reconnects with backoff and refreshes queries on job changes. */
export function useLiveEvents() {
  const queryClient = useQueryClient()
  useEffect(() => {
    let ws: WebSocket | null = null
    let closed = false
    let retry = 0
    let timer: ReturnType<typeof setTimeout> | undefined

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/api/ws`)
      ws.onopen = () => {
        retry = 0
        useLive.getState().setConnected(true)
        queryClient.invalidateQueries({ queryKey: qk.jobs })
      }
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'ping' || msg.type === 'hello') return
        useLive.getState().apply(msg)
        if (msg.type === 'job') {
          const job = msg.job as Job
          queryClient.invalidateQueries({ queryKey: qk.jobs })
          if (job.experiment) {
            queryClient.invalidateQueries({
              queryKey: qk.status(job.experiment),
            })
          }
        }
      }
      ws.onclose = () => {
        useLive.getState().setConnected(false)
        if (closed) return
        retry = Math.min(retry + 1, 6)
        timer = setTimeout(connect, 500 * 2 ** retry)
      }
      ws.onerror = () => ws?.close()
    }
    connect()
    return () => {
      closed = true
      clearTimeout(timer)
      ws?.close()
    }
  }, [queryClient])
}
