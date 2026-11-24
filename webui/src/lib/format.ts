import { formatDistanceToNowStrict } from 'date-fns'
import { zhCN } from 'date-fns/locale'

export function ago(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    return formatDistanceToNowStrict(new Date(iso), {
      addSuffix: true,
      locale: zhCN,
    })
  } catch {
    return iso
  }
}

export function clock(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString('zh-CN', { hour12: false })
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (h)
    return `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
  return `${m}:${String(r).padStart(2, '0')}`
}

export function elapsed(
  start: string | null,
  end?: string | null
): number | null {
  if (!start) return null
  const t1 = end ? new Date(end).getTime() : Date.now()
  return (t1 - new Date(start).getTime()) / 1000
}

export function bytes(n: number | null | undefined): string {
  if (n == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = n
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 && i ? 1 : 0)} ${u[i]}`
}

export function pct(f: number | null | undefined, digits = 0): string {
  return f == null ? '—' : `${(f * 100).toFixed(digits)}%`
}
