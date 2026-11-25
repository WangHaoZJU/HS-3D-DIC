import { useMemo, useState } from 'react'
import { Line, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import * as THREE from 'three'
import type { ReferenceResult } from '@/lib/api'
import { STATUS_COLORS, robustRange, viridis } from '@/lib/colormap'
import { type ColorMode, valuesFor } from './results'

/** Reference surface mesh (left-camera frame, mm) with per-vertex colour. */
export default function Shape3D({
  r,
  mode,
}: {
  r: ReferenceResult
  mode: ColorMode
}) {
  const [wire, setWire] = useState(false)
  const { geom, center, size } = useMemo(() => {
    const n = r.n
    const pos = new Float32Array(3 * n)
    const valid = new Uint8Array(n)
    const acc = new THREE.Vector3()
    let count = 0
    for (let i = 0; i < n; i++) {
      const x = r.points_3d[3 * i]
      const y = r.points_3d[3 * i + 1]
      const z = r.points_3d[3 * i + 2]
      if (x == null || y == null || z == null) continue
      // OpenCV (x right, y down, z forward) -> three.js (x right, y up, z toward viewer)
      pos[3 * i] = x
      pos[3 * i + 1] = -y
      pos[3 * i + 2] = -z
      valid[i] = 1
      acc.add(new THREE.Vector3(x, -y, -z))
      count++
    }
    const idx: number[] = []
    for (let t = 0; t < r.connectivity.length; t += 3) {
      const a = r.connectivity[t]
      const b = r.connectivity[t + 1]
      const c = r.connectivity[t + 2]
      if (valid[a] && valid[b] && valid[c]) idx.push(a, b, c)
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setIndex(idx)
    g.computeVertexNormals()
    g.computeBoundingBox()
    const colors = new Float32Array(3 * n)
    const v = valuesFor(r, mode)
    const [lo, hi] = robustRange(v)
    const col = new THREE.Color()
    for (let i = 0; i < n; i++) {
      if (mode === 'status')
        col.set(STATUS_COLORS[r.status[i]]?.color ?? '#888')
      else {
        const [rr, gg, bb] = viridis(
          v[i] == null ? NaN : ((v[i] as number) - lo) / (hi - lo)
        )
        col.setRGB(rr / 255, gg / 255, bb / 255, THREE.SRGBColorSpace)
      }
      colors.set([col.r, col.g, col.b], 3 * i)
    }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    const center = count ? acc.multiplyScalar(1 / count) : new THREE.Vector3()
    const size = g.boundingBox
      ? g.boundingBox.getSize(new THREE.Vector3()).length()
      : 50
    return { geom: g, center, size: Math.max(size, 1) }
  }, [r, mode])

  const axes = (['c', 's', 'n'] as const)
    .map((k, j) => {
      const a = r.mean_axes[k]
      if (!a) return null
      const tip = center
        .clone()
        .add(new THREE.Vector3(a[0], -a[1], -a[2]).multiplyScalar(size * 0.2))
      return {
        k,
        pts: [center, tip],
        color: ['#f87171', '#34d399', '#60a5fa'][j],
      }
    })
    .filter(Boolean) as { k: string; pts: THREE.Vector3[]; color: string }[]

  const camPos = center
    .clone()
    .add(new THREE.Vector3(size * 0.15, size * 0.35, size * 1.1))
  return (
    <div className='relative h-full w-full'>
      <Canvas
        camera={{
          position: camPos.toArray(),
          fov: 40,
          near: size / 1000,
          far: size * 50,
        }}
        dpr={[1, 2]}
      >
        <color attach='background' args={['#0b0f14']} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[size, size, size]} intensity={0.9} />
        <OrbitControls target={center} makeDefault />
        <mesh geometry={geom}>
          <meshStandardMaterial
            vertexColors
            side={THREE.DoubleSide}
            roughness={0.85}
            metalness={0}
            wireframe={wire}
          />
        </mesh>
        {axes.map((a) => (
          <Line key={a.k} points={a.pts} color={a.color} lineWidth={2.5} />
        ))}
      </Canvas>
      <div className='absolute top-2 left-2 flex items-center gap-3 rounded-md bg-black/60 px-2 py-1 text-xs text-white/80'>
        <span className='text-[#f87171]'>c 弦向</span>
        <span className='text-[#34d399]'>s 展向</span>
        <span className='text-[#60a5fa]'>n 法向</span>
        <button
          className='rounded border border-white/20 px-1.5'
          onClick={() => setWire((w) => !w)}
        >
          {wire ? '实体' : '网格线'}
        </button>
      </div>
    </div>
  )
}
