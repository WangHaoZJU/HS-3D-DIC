/** Scientific colour maps (docs/UI_Design.md §8): viridis for magnitudes, status colours. */

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
]

/** t in [0, 1] -> [r, g, b] in 0..255 (NaN -> grey). */
export function viridis(t: number): [number, number, number] {
  if (!Number.isFinite(t)) return [110, 110, 110]
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1)
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x))
  const f = x - i
  const a = VIRIDIS[i]
  const b = VIRIDIS[i + 1]
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ]
}

export function viridisCss(t: number): string {
  const [r, g, b] = viridis(t)
  return `rgb(${r | 0},${g | 0},${b | 0})`
}

/** Status code colours (UI_Design §8), index = code. */
export const STATUS_COLORS: Record<
  number,
  { name: string; label: string; color: string }
> = {
  0: { name: 'NOT_COMPUTED', label: '未计算', color: '#64748b' },
  1: { name: 'OK', label: '有效', color: '#34d399' },
  2: { name: 'OUT_OF_BOUNDS', label: '越界', color: '#94a3b8' },
  3: { name: 'LOW_TEXTURE', label: '纹理不足', color: '#b45309' },
  4: { name: 'ILL_CONDITIONED', label: '病态', color: '#a855f7' },
  5: { name: 'NOT_CONVERGED', label: '未收敛', color: '#f97316' },
  6: { name: 'HIGH_RESIDUAL', label: '残差过大', color: '#ef4444' },
  7: { name: 'GEOMETRY_INCONSISTENT', label: '几何不一致', color: '#ec4899' },
  8: { name: 'TEMPORAL_JUMP', label: '时间跳变', color: '#eab308' },
  9: { name: 'MASKED', label: '掩膜', color: '#334155' },
}

export function statusByName(name: string) {
  return Object.values(STATUS_COLORS).find((s) => s.name === name)
}

/** Robust range (percentiles) of finite values. */
export function robustRange(
  values: (number | null)[],
  lo = 0.02,
  hi = 0.98
): [number, number] {
  const v = values
    .filter((x): x is number => x != null && Number.isFinite(x))
    .sort((a, b) => a - b)
  if (!v.length) return [0, 1]
  const a = v[Math.floor(lo * (v.length - 1))]
  const b = v[Math.ceil(hi * (v.length - 1))]
  return a === b ? [a, a + 1e-9] : [a, b]
}

/** Diverging blue–white–red for signed values; t in [-1, 1]. */
export function diverging(t: number): [number, number, number] {
  if (!Number.isFinite(t)) return [110, 110, 110]
  const x = Math.min(1, Math.max(-1, t))
  const lo: [number, number, number] = [33, 102, 172]
  const mid: [number, number, number] = [242, 242, 242]
  const hi: [number, number, number] = [178, 24, 43]
  const [a, b, f] = x < 0 ? [mid, lo, -x] : [mid, hi, x]
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ]
}

export function divergingCss(t: number): string {
  const [r, g, b] = diverging(t)
  return `rgb(${r | 0},${g | 0},${b | 0})`
}

/** Cyclic colour map for phase (rad), hue wheel. */
export function cyclic(phase: number): [number, number, number] {
  if (!Number.isFinite(phase)) return [110, 110, 110]
  const h = (((phase / (2 * Math.PI)) % 1) + 1) % 1
  const s = 0.65
  const l = 0.55
  const k = (n: number) => (n + h * 12) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) =>
    l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [f(0) * 255, f(8) * 255, f(4) * 255]
}
