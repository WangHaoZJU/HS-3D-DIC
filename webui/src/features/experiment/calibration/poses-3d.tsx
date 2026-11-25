import { useMemo } from 'react'
import { Grid, Line, OrbitControls, Text } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import * as THREE from 'three'
import type { CalibrationPoses } from '@/lib/api'

/**
 * Cameras (frustums) and board poses in the left-camera frame.
 * OpenCV camera axes (x right, y down, z forward) are mapped to three.js (x right, y up, z toward viewer)
 * by (x, y, z) -> (x, -y, -z); units mm.
 */
export default function Poses3D({
  poses,
  selected,
  onSelect,
}: {
  poses: CalibrationPoses
  selected: number | null
  onSelect: (v: number) => void
}) {
  const cv = (p: number[]) => new THREE.Vector3(p[0], -p[1], -p[2])
  const { target, position, dist } = useMemo(() => {
    const pts = [
      ...poses.boards.flatMap((b) => b.corners),
      ...poses.cameras.map((c) => c.center),
    ]
    const v = pts.map(cv)
    const box = new THREE.Box3().setFromPoints(
      v.length ? v : [new THREE.Vector3(0, 0, -250), new THREE.Vector3()]
    )
    const t = box.getCenter(new THREE.Vector3())
    const span = Math.max(50, box.getSize(new THREE.Vector3()).length())
    const dir = new THREE.Vector3(0.85, 0.5, 0.55).normalize()
    return {
      target: t,
      position: t.clone().add(dir.multiplyScalar(span * 1.25)),
      dist: span,
    }
  }, [poses])
  const maxRms = Math.max(1e-6, ...poses.boards.map((b) => b.rms))
  return (
    <Canvas
      camera={{
        position: position.toArray(),
        fov: 45,
        near: 1,
        far: dist * 20,
      }}
      dpr={[1, 2]}
    >
      <color attach='background' args={['#0b0f14']} />
      <ambientLight intensity={0.8} />
      <OrbitControls target={target} makeDefault />
      <Grid
        position={[target.x, target.y - dist * 0.4, target.z]}
        args={[dist * 3, dist * 3]}
        cellSize={10}
        sectionSize={50}
        cellColor='#1f2933'
        sectionColor='#334155'
        fadeDistance={dist * 4}
        infiniteGrid
      />
      {poses.cameras.map((c) => (
        <Frustum key={c.name} cam={c} depth={dist * 0.12} />
      ))}
      {poses.boards.map((b) => {
        const pts = [...b.corners, b.corners[0]].map(cv)
        const t = b.rms / maxRms
        const color = new THREE.Color().setHSL(0.33 - 0.33 * t, 0.75, 0.55)
        const sel = b.view === selected
        return (
          <group
            key={b.view}
            onClick={(e) => (e.stopPropagation(), onSelect(b.view))}
          >
            <Quad
              pts={b.corners.map(cv)}
              color={color}
              opacity={sel ? 0.45 : 0.14}
            />
            <Line
              points={pts}
              color={sel ? '#5fd0e6' : color}
              lineWidth={sel ? 2.5 : 1.2}
            />
            <Text
              position={cv(b.corners[0]).add(new THREE.Vector3(0, 3, 0))}
              fontSize={dist * 0.02}
              color='#94a3b8'
            >
              {String(b.view)}
            </Text>
          </group>
        )
      })}
    </Canvas>
  )
}

function Quad({
  pts,
  color,
  opacity,
}: {
  pts: THREE.Vector3[]
  color: THREE.Color
  opacity: number
}) {
  const geom = useMemo(() => {
    const [a, b, c, d] = pts
    return new THREE.BufferGeometry().setFromPoints([a, b, c, a, c, d])
  }, [pts])
  return (
    <mesh geometry={geom}>
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}

function Frustum({
  cam,
  depth,
}: {
  cam: CalibrationPoses['cameras'][number]
  depth: number
}) {
  const [w, h] = cam.image_size
  const K = cam.K
  const R = cam.R
  const c = cam.center
  // corners of the image plane at `depth` in camera coordinates -> left frame: X = R^T x + C
  const toWorld = (x: number[]) => [
    R[0][0] * x[0] + R[1][0] * x[1] + R[2][0] * x[2] + c[0],
    R[0][1] * x[0] + R[1][1] * x[1] + R[2][1] * x[2] + c[1],
    R[0][2] * x[0] + R[1][2] * x[1] + R[2][2] * x[2] + c[2],
  ]
  const ray = (u: number, v: number) => [
    ((u - K[0][2]) / K[0][0]) * depth,
    ((v - K[1][2]) / K[1][1]) * depth,
    depth,
  ]
  const corners = [ray(0, 0), ray(w, 0), ray(w, h), ray(0, h)].map(toWorld)
  const cv = (p: number[]) => new THREE.Vector3(p[0], -p[1], -p[2])
  const o = cv(c)
  const color = cam.name === 'left' ? '#38bdf8' : '#f59e0b'
  return (
    <group>
      {corners.map((p, i) => (
        <Line key={i} points={[o, cv(p)]} color={color} lineWidth={1} />
      ))}
      <Line
        points={[...corners, corners[0]].map(cv)}
        color={color}
        lineWidth={1.5}
      />
      <Text
        position={o.clone().add(new THREE.Vector3(0, depth * 0.25, 0))}
        fontSize={depth * 0.18}
        color={color}
      >
        {cam.name === 'left' ? 'L' : 'R'}
      </Text>
    </group>
  )
}
