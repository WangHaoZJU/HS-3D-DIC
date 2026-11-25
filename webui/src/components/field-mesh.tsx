/** Displacement field on the reference mesh (three.js): per-vertex colour, optional warp, picking. */
import { useMemo } from 'react'
import { OrbitControls } from '@react-three/drei'
import { Canvas, type ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { diverging, viridis } from '@/lib/colormap'

export type FieldMeshProps = {
  points3d: (number | null)[] // (N*3) reference positions, left-camera frame (mm)
  connectivity: number[] // (M*3)
  values: (number | null)[] | null // per vertex
  range: number // colour scale: [-range, range] (signed) or [0, range]
  signed: boolean
  warp?: (number | null)[][] | null // (N, 3) camera-frame displacement (mm) to add, times gain
  gain?: number
  selected?: number | null
  onPick?: (i: number) => void
}

const cv = (x: number, y: number, z: number) => new THREE.Vector3(x, -y, -z)

export default function FieldMesh(props: FieldMeshProps) {
  const { points3d, connectivity } = props
  const base = useMemo(() => {
    const n = points3d.length / 3
    const valid = new Uint8Array(n)
    const acc = new THREE.Vector3()
    let count = 0
    for (let i = 0; i < n; i++) {
      if (
        points3d[3 * i] != null &&
        points3d[3 * i + 1] != null &&
        points3d[3 * i + 2] != null
      ) {
        valid[i] = 1
        acc.add(
          cv(points3d[3 * i]!, points3d[3 * i + 1]!, points3d[3 * i + 2]!)
        )
        count++
      }
    }
    const idx: number[] = []
    for (let t = 0; t < connectivity.length; t += 3) {
      const [a, b, c] = [
        connectivity[t],
        connectivity[t + 1],
        connectivity[t + 2],
      ]
      if (valid[a] && valid[b] && valid[c]) idx.push(a, b, c)
    }
    const center = count ? acc.multiplyScalar(1 / count) : new THREE.Vector3()
    let ext = 1
    for (let i = 0; i < n; i++)
      if (valid[i])
        ext = Math.max(
          ext,
          cv(
            points3d[3 * i]!,
            points3d[3 * i + 1]!,
            points3d[3 * i + 2]!
          ).distanceTo(center)
        )
    return { n, valid, idx, center, size: 2 * ext }
  }, [points3d, connectivity])

  const geom = useMemo(() => {
    const { n, valid, idx } = base
    const pos = new Float32Array(3 * n)
    const col = new Float32Array(3 * n)
    const g = props.gain ?? 0
    const c = new THREE.Color()
    for (let i = 0; i < n; i++) {
      if (!valid[i]) continue
      let [x, y, z] = [
        points3d[3 * i]!,
        points3d[3 * i + 1]!,
        points3d[3 * i + 2]!,
      ]
      const d = props.warp?.[i]
      if (g && d && d[0] != null && d[1] != null && d[2] != null) {
        x += g * d[0]
        y += g * d[1]
        z += g * d[2]
      }
      pos.set([x, -y, -z], 3 * i)
      const v = props.values?.[i]
      const [r, gg, b] =
        v == null
          ? [90, 90, 90]
          : props.signed
            ? diverging(v / (props.range || 1))
            : viridis(Math.abs(v) / (props.range || 1))
      c.setRGB(r / 255, gg / 255, b / 255, THREE.SRGBColorSpace)
      col.set([c.r, c.g, c.b], 3 * i)
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
    geo.setIndex(idx)
    geo.computeVertexNormals()
    return geo
  }, [
    base,
    points3d,
    props.values,
    props.range,
    props.signed,
    props.warp,
    props.gain,
  ])

  const pick = (e: ThreeEvent<MouseEvent>) => {
    if (!props.onPick || e.face == null) return
    e.stopPropagation()
    const p = e.point
    const cand = [e.face.a, e.face.b, e.face.c]
    const pos = geom.getAttribute('position')
    let best = cand[0]
    let bd = Infinity
    for (const k of cand) {
      const d = new THREE.Vector3(
        pos.getX(k),
        pos.getY(k),
        pos.getZ(k)
      ).distanceTo(p)
      if (d < bd) {
        bd = d
        best = k
      }
    }
    props.onPick(best)
  }

  const sel = props.selected != null && base.valid[props.selected]
  const selPos = sel
    ? cv(
        points3d[3 * props.selected!]!,
        points3d[3 * props.selected! + 1]!,
        points3d[3 * props.selected! + 2]!
      )
    : null
  const camPos = base.center
    .clone()
    .add(new THREE.Vector3(base.size * 0.15, base.size * 0.35, base.size * 1.2))
  return (
    <Canvas
      camera={{
        position: camPos.toArray(),
        fov: 40,
        near: base.size / 1000,
        far: base.size * 50,
      }}
      dpr={[1, 2]}
    >
      <color attach='background' args={['#0b0f14']} />
      <ambientLight intensity={0.6} />
      <directionalLight
        position={[base.size, base.size, base.size]}
        intensity={0.8}
      />
      <OrbitControls target={base.center} makeDefault />
      <mesh geometry={geom} onClick={pick}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          roughness={0.9}
          metalness={0}
        />
      </mesh>
      {selPos && (
        <mesh position={selPos}>
          <sphereGeometry args={[base.size * 0.012, 16, 16]} />
          <meshBasicMaterial color='#ffffff' />
        </mesh>
      )}
    </Canvas>
  )
}
