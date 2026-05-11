import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { CameraInfo, MetaMessage } from "./types";
import type { DecodedFrame } from "./socket";
import { projectPoints } from "./projection";
import { drawPoseOverlay, kpToPoints } from "./poseOverlay";
import type { Point2D } from "./poseOverlay";

type Props = {
  meta: MetaMessage | null;
  frame: DecodedFrame | null;
  prevFrame: DecodedFrame | null;
  receiveTime: number;
  showGroundTruth: boolean;
  cameras: CameraInfo[];
  enabledCams: Set<number> | null; // null = all
  showRaw2D: boolean;
  showReproj3D: boolean;
  showReprojGT: boolean;
  onExpandCam: (camId: number) => void;
};

// Fallback COCO edges if meta hasn't arrived yet
const COCO17_EDGES: [number, number][] = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4],
];

const H36M17_EDGES: [number, number][] = [
  [0, 16], [16, 1], [1, 15], [15, 14],
  [1, 2], [2, 3], [3, 4],
  [1, 5], [5, 6], [6, 7],
  [14, 8], [8, 9], [9, 10],
  [14, 11], [11, 12], [12, 13],
];

export default function Scene3D({ meta, cameras, frame, prevFrame, receiveTime, showGroundTruth, enabledCams, showRaw2D, showReproj3D, showReprojGT, onExpandCam }: Props) {
  const cocoEdges = (meta?.skeletons.coco17.edges as [number, number][]) ?? COCO17_EDGES;
  const h36mEdges = (meta?.skeletons.h36m17.edges as [number, number][]) ?? H36M17_EDGES;

  return (
    <Canvas
      camera={{ position: [3, 3, 3], fov: 50, near: 0.01, far: 100 }}
      style={{ background: "#05080d" }}
      gl={{ antialias: true }}
    >
      <ambientLight intensity={0.4} />
      <pointLight position={[5, 5, 5]} intensity={0.6} />
      <Grid
        cellColor="#0a3a3a"
        sectionColor="#00ffd5"
        sectionThickness={1}
        cellSize={0.5}
        sectionSize={1}
        fadeDistance={20}
        fadeStrength={1}
        infiniteGrid
        position={[0, 0, 0]}
      />

      {/* Cameras + their video planes */}
      {cameras.map((cam) => (
        <CameraView
          key={cam.cam_id}
          cam={cam}
          frame={frame}
          meta={meta}
          dimmed={enabledCams !== null && !enabledCams.has(cam.cam_id)}
          showRaw2D={showRaw2D}
          showReproj3D={showReproj3D}
          showReprojGT={showReprojGT}
          showGroundTruth={showGroundTruth}
          cocoEdges={cocoEdges}
          h36mEdges={h36mEdges}
          onExpandCam={onExpandCam}
        />
      ))}

      {/* Predicted skeleton — COCO-17 wiring, cyan — with interpolation */}
      {frame && (
        <Skeleton
          points3d={frame.msg.points_3d}
          valid={frame.msg.valid_3d}
          prevPoints3d={prevFrame?.msg.points_3d ?? null}
          prevValid={prevFrame?.msg.valid_3d ?? null}
          receiveTime={receiveTime}
          fps={frame.msg.fps}
          color="#00ffd5"
          edges={cocoEdges}
        />
      )}

      {/* Ground truth skeleton — H36M-17 wiring, orange — no interpolation needed */}
      {frame && showGroundTruth && frame.msg.gt_3d && (
        <Skeleton
          points3d={frame.msg.gt_3d}
          valid={frame.msg.gt_3d.map((p) => p !== null && p[0] !== null)}
          prevPoints3d={null}
          prevValid={null}
          receiveTime={receiveTime}
          fps={frame.msg.fps}
          color="#ff8a3d"
          edges={h36mEdges}
        />
      )}

      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
    </Canvas>
  );
}

// MPI uses millimeters; we scale down to display in meters for nicer Three.js camera distances.
const WORLD_SCALE = 0.001;

function CameraView({ cam, frame, meta, dimmed, showRaw2D, showReproj3D, showReprojGT, showGroundTruth, cocoEdges, h36mEdges, onExpandCam }: {
  cam: CameraInfo;
  frame: DecodedFrame | null;
  meta: MetaMessage | null;
  dimmed: boolean;
  showRaw2D: boolean;
  showReproj3D: boolean;
  showReprojGT: boolean;
  showGroundTruth: boolean;
  cocoEdges: [number, number][];
  h36mEdges: [number, number][];
  onExpandCam: (camId: number) => void;
}) {
  const textureRef = useRef<THREE.Texture | null>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  // Build R^T (camera to world rotation) and camera world position once
  const { worldPos, worldQuat } = useMemo(() => {
    const R = new THREE.Matrix3().set(
      cam.R[0][0], cam.R[0][1], cam.R[0][2],
      cam.R[1][0], cam.R[1][1], cam.R[1][2],
      cam.R[2][0], cam.R[2][1], cam.R[2][2],
    );
    // World position C = -R^T * t
    const t = new THREE.Vector3(cam.t[0], cam.t[1], cam.t[2]);
    const Rt_inv = R.clone().transpose();
    const C = t.clone().applyMatrix3(Rt_inv).multiplyScalar(-WORLD_SCALE);

    // Build a 4x4 matrix where the rotation part is R^T (camera -> world)
    const m = new THREE.Matrix4();
    m.set(
      cam.R[0][0], cam.R[1][0], cam.R[2][0], 0,
      cam.R[0][1], cam.R[1][1], cam.R[2][1], 0,
      cam.R[0][2], cam.R[1][2], cam.R[2][2], 0,
      0, 0, 0, 1,
    );
    // OpenCV camera convention: +Z forward, +X right, +Y down. Three.js uses +Z back.
    // Apply 180-deg flip around X to convert.
    const flip = new THREE.Matrix4().makeRotationX(Math.PI);
    m.multiply(flip);
    const q = new THREE.Quaternion().setFromRotationMatrix(m);

    return { worldPos: C, worldQuat: q };
  }, [cam]);

  // Update texture from incoming JPEG blob + overlays for this camera
  useEffect(() => {
    if (!frame) return;
    const blob = frame.jpegByCam.get(cam.cam_id);
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      // Get or create offscreen canvas for compositing
      if (!offscreenRef.current) {
        offscreenRef.current = document.createElement("canvas");
      }
      const osc = offscreenRef.current;
      osc.width = img.naturalWidth;
      osc.height = img.naturalHeight;
      const ctx = osc.getContext("2d")!;

      // Draw base video frame
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);

      // Draw overlays (same logic as FeedGrid)
      const calib = meta?.cameras[String(cam.cam_id)];
      const origW = calib?.image_size?.[0] ?? osc.width;
      const origH = calib?.image_size?.[1] ?? osc.height;
      const scaleX = osc.width / origW;
      const scaleY = osc.height / origH;
      const scalePoints = (pts: Point2D[]): Point2D[] =>
        pts.map((p) => (p ? [p[0] * scaleX, p[1] * scaleY] : null));

      const pose = frame.msg.poses_2d[String(cam.cam_id)];

      // Raw 2D detection — cyan
      if (showRaw2D && pose?.detected) {
        const raw: Point2D[] = kpToPoints(pose.kp, pose.conf);
        drawPoseOverlay(ctx, scalePoints(raw), pose.conf, cocoEdges, "rgba(0,255,213,0.85)");
      }

      // Reprojected 3D prediction — magenta
      if (showReproj3D && calib && frame.msg.points_3d) {
        const reproj = projectPoints(calib, frame.msg.points_3d);
        const allConf = new Array(reproj.length).fill(1);
        const valid = frame.msg.valid_3d;
        const maskedReproj: Point2D[] = reproj.map((p, i) => (valid[i] ? p : null));
        drawPoseOverlay(ctx, scalePoints(maskedReproj), allConf, cocoEdges, "rgba(255,0,255,0.85)", undefined, 0);
      }

      // Reprojected GT — yellow
      if (showReprojGT && showGroundTruth && calib && frame.msg.gt_3d) {
        const reproj = projectPoints(calib, frame.msg.gt_3d);
        const allConf = new Array(reproj.length).fill(1);
        const maskedReproj: Point2D[] = reproj.map((p, i) => {
          const g = frame.msg.gt_3d![i];
          return g && g[0] != null ? p : null;
        });
        drawPoseOverlay(ctx, scalePoints(maskedReproj), allConf, h36mEdges, "rgba(255,213,74,0.85)", undefined, 0);
      }

      // Create texture from composited canvas
      const tex = new THREE.CanvasTexture(osc);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      // Dispose previous
      if (textureRef.current) textureRef.current.dispose();
      textureRef.current = tex;
      const mesh = meshRef.current;
      if (mesh && mesh.material instanceof THREE.MeshBasicMaterial) {
        mesh.material.map = tex;
        mesh.material.needsUpdate = true;
      }
    };
    img.src = url;
  }, [frame, cam.cam_id, meta, showRaw2D, showReproj3D, showReprojGT, showGroundTruth, cocoEdges, h36mEdges]);

  // Frustum geometry: a camera-frustum-shaped wireframe + a textured plane at depth=1m
  const planeWidth = 0.6;
  const aspect = cam.width / cam.height;
  const planeHeight = planeWidth / aspect;
  const planeDist = 0.6;

  // Lines for frustum edges
  const frustumLines = useMemo(() => {
    const half_w = planeWidth / 2;
    const half_h = planeHeight / 2;
    const z = -planeDist; // forward in camera space (after flip)
    const corners = [
      new THREE.Vector3(-half_w, -half_h, z),
      new THREE.Vector3(half_w, -half_h, z),
      new THREE.Vector3(half_w, half_h, z),
      new THREE.Vector3(-half_w, half_h, z),
    ];
    const origin = new THREE.Vector3(0, 0, 0);
    const positions: number[] = [];
    for (const c of corners) {
      positions.push(origin.x, origin.y, origin.z, c.x, c.y, c.z);
    }
    for (let i = 0; i < 4; i++) {
      const a = corners[i];
      const b = corners[(i + 1) % 4];
      positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return geom;
  }, [planeWidth, planeHeight, planeDist]);

  return (
    <group position={worldPos} quaternion={worldQuat}>
      <lineSegments geometry={frustumLines}>
        <lineBasicMaterial color={dimmed ? "#444444" : "#00ffd5"} transparent opacity={dimmed ? 0.3 : 0.7} />
      </lineSegments>
      {/* Video plane (textured) — slightly transparent, two-sided */}
      <mesh 
        ref={meshRef} 
        position={[0, 0, -planeDist]} 
        onClick={(e) => { e.stopPropagation(); onExpandCam(cam.cam_id); }}
        onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = 'pointer'; }}
        onPointerOut={(e) => { e.stopPropagation(); document.body.style.cursor = 'auto'; }}
      >
        <planeGeometry args={[planeWidth, planeHeight]} />
        <meshBasicMaterial
          color="#ffffff"
          side={THREE.DoubleSide}
          transparent
          opacity={dimmed ? 0.25 : 0.92}
        />
      </mesh>
      {/* Tiny sphere at the camera origin */}
      <mesh>
        <sphereGeometry args={[0.025, 12, 12]} />
        <meshBasicMaterial color={dimmed ? "#444444" : "#00ffd5"} />
      </mesh>
    </group>
  );
}

function Skeleton({
  points3d,
  valid,
  prevPoints3d,
  prevValid,
  receiveTime,
  fps,
  color,
  edges,
}: {
  points3d: (number | null)[][];
  valid: boolean[];
  prevPoints3d: (number | null)[][] | null;
  prevValid: boolean[] | null;
  receiveTime: number;
  fps: number;
  color: string;
  edges: [number, number][];
}) {
  const linesRef = useRef<THREE.LineSegments>(null);
  const jointsRef = useRef<THREE.InstancedMesh>(null);
  const jointObj = useMemo(() => new THREE.Object3D(), []);

  // Store targets in refs so useFrame always sees the latest values
  const currRef = useRef(points3d);
  const prevRef = useRef(prevPoints3d);
  const currValidRef = useRef(valid);
  const prevValidRef = useRef(prevValid);
  const receiveTimeRef = useRef(receiveTime);
  const fpsRef = useRef(fps);

  // Sync refs on each render
  currRef.current = points3d;
  prevRef.current = prevPoints3d;
  currValidRef.current = valid;
  prevValidRef.current = prevValid;
  receiveTimeRef.current = receiveTime;
  fpsRef.current = fps;

  useFrame(() => {
    const curr = currRef.current;
    const prev = prevRef.current;
    const currValid = currValidRef.current;
    const prevValidArr = prevValidRef.current;
    const rt = receiveTimeRef.current;
    const backendFps = Math.max(fpsRef.current, 1);

    // Compute lerp factor: clamp t in [0, 1] based on time since last WS frame
    const elapsed = performance.now() - rt;
    const frameDuration = 1000 / backendFps;
    const t = Math.min(elapsed / frameDuration, 1);

    // Interpolate a single joint position
    const lerp = (j: number): [number, number, number] | null => {
      const cp = curr[j];
      if (!cp || cp[0] === null) return null;
      const cx = cp[0] as number;
      const cy = cp[1] as number;
      const cz = cp[2] as number;

      if (!prev || !prevValidArr || !prevValidArr[j] || !prev[j] || prev[j][0] === null) {
        return [cx, cy, cz];
      }
      const pp = prev[j];
      const px = pp[0] as number;
      const py = pp[1] as number;
      const pz = pp[2] as number;
      return [
        px + (cx - px) * t,
        py + (cy - py) * t,
        pz + (cz - pz) * t,
      ];
    };

    // Update line geometry
    if (linesRef.current) {
      const positions: number[] = [];
      for (const [a, b] of edges) {
        if (!currValid[a] || !currValid[b]) continue;
        const pa = lerp(a);
        const pb = lerp(b);
        if (!pa || !pb) continue;
        positions.push(
          pa[0] * WORLD_SCALE, pa[1] * WORLD_SCALE, pa[2] * WORLD_SCALE,
          pb[0] * WORLD_SCALE, pb[1] * WORLD_SCALE, pb[2] * WORLD_SCALE,
        );
      }
      const geom = linesRef.current.geometry as THREE.BufferGeometry;
      geom.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions, 3),
      );
      geom.computeBoundingSphere();
    }

    // Update joint instances
    if (jointsRef.current) {
      let i = 0;
      for (let j = 0; j < curr.length; j++) {
        if (!currValid[j]) continue;
        const p = lerp(j);
        if (!p) continue;
        jointObj.position.set(p[0] * WORLD_SCALE, p[1] * WORLD_SCALE, p[2] * WORLD_SCALE);
        jointObj.updateMatrix();
        jointsRef.current.setMatrixAt(i++, jointObj.matrix);
      }
      jointsRef.current.count = i;
      jointsRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <group>
      <lineSegments ref={linesRef}>
        <bufferGeometry />
        <lineBasicMaterial color={color} linewidth={2} />
      </lineSegments>
      <instancedMesh ref={jointsRef} args={[undefined, undefined, points3d.length]}>
        <sphereGeometry args={[0.03, 12, 12]} />
        <meshBasicMaterial color={color} />
      </instancedMesh>
    </group>
  );
}
