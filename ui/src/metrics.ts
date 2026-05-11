/**
 * Frontend-computed temporal-consistency metrics.
 * All pure functions — no side effects.
 */

import { useRef, useEffect, useState } from "react";
import type { MetaMessage } from "./types";
import type { DecodedFrame } from "./socket";
import { projectPoints } from "./projection";

// ---- Types ------------------------------------------------------------------

export type FrameSnapshot = {
  pts3d: (number | null)[][];   // COCO-17 predicted
  valid3d: boolean[];
  gt3d: (number | null)[][] | null; // H36M-17 GT or null
  poses2d: Record<string, { kp: number[][]; conf: number[]; detected: boolean }>;
  fps: number;
  stageTiming?: Record<string, number>; // ms per stage (Phase 4)
};

export type BoneStats = {
  bone: string;       // e.g. "L_shoulder–L_elbow"
  meanLen: number;
  cv: number;         // coefficient of variation = std/mean
  history: number[];  // rolling window of lengths
};

export type JointJitter = {
  joint: string;
  jitter: number;     // position std-dev over window
  validPct: number;   // % frames this joint was valid
  history: number[];
};

export type CameraReproj = {
  camId: string;
  meanErr: number;    // mean pixel reprojection error
  worstJoint: string;
  history: number[];
};

export type OverallMetrics = {
  mpjpe: number | null;   // mm, null if no GT
  detectionRate: number;  // 0-1
  fps: number;
  fpsHistory: number[];
  mpjpeHistory: number[];
  detRateHistory: number[];
  stageTiming?: Record<string, number>;
};

// ---- Bone stats -------------------------------------------------------------

const COCO17_JOINT_NAMES = [
  "nose", "L_eye", "R_eye", "L_ear", "R_ear",
  "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
  "L_wrist", "R_wrist", "L_hip", "R_hip",
  "L_knee", "R_knee", "L_ankle", "R_ankle",
];

const COCO17_SKELETON: [number, number][] = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

function boneLabel(a: number, b: number): string {
  return `${COCO17_JOINT_NAMES[a]}–${COCO17_JOINT_NAMES[b]}`;
}

function dist3d(a: (number | null)[], b: (number | null)[]): number | null {
  if (!a || !b || a[0] == null || b[0] == null) return null;
  const dx = (a[0] as number) - (b[0] as number);
  const dy = (a[1] as number) - (b[1] as number);
  const dz = (a[2] as number) - (b[2] as number);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function mean(arr: number[]): number {
  return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;
}

function std(arr: number[], m?: number): number {
  if (arr.length < 2) return 0;
  const mu = m ?? mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - mu) ** 2, 0) / arr.length);
}

// ---- Ring buffer hook -------------------------------------------------------

const WINDOW = 300; // ~10s @ 30fps

// ---- Main metrics hook ------------------------------------------------------

export type MetricsState = {
  bones: BoneStats[];
  joints: JointJitter[];
  cameras: CameraReproj[];
  overall: OverallMetrics;
};

export function useMetrics(
  frame: DecodedFrame | null,
  meta: MetaMessage | null,
): MetricsState {
  // Per-bone history: bone index -> number[]
  const boneHistRef = useRef<Map<number, number[]>>(new Map());
  // Per-joint positions history for jitter: joint index -> [x,y,z][]
  const jointHistRef = useRef<Map<number, number[][]>>(new Map());
  // Per-joint valid count
  const jointValidRef = useRef<Map<number, number>>(new Map());
  const frameCountRef = useRef(0);
  // Per-camera reproj error history
  const camErrHistRef = useRef<Map<string, number[]>>(new Map());
  // Overall metrics history
  const fpsHistRef = useRef<number[]>([]);
  const mpjpeHistRef = useRef<number[]>([]);
  const detHistRef = useRef<number[]>([]);

  const [metrics, setMetrics] = useState<MetricsState>({
    bones: [],
    joints: [],
    cameras: [],
    overall: { mpjpe: null, detectionRate: 0, fps: 0, fpsHistory: [], mpjpeHistory: [], detRateHistory: [] },
  });

  const throttleRef = useRef(0);

  useEffect(() => {
    if (!frame || !meta) return;

    const msg = frame.msg;
    frameCountRef.current++;

    // ---- Bone lengths -------------------------------------------------------
    const usedEdges = COCO17_SKELETON;

    usedEdges.forEach(([a, b], i) => {
      const d = dist3d(msg.points_3d[a], msg.points_3d[b]);
      if (d == null || !msg.valid_3d[a] || !msg.valid_3d[b]) return;
      const hist = boneHistRef.current.get(i) ?? [];
      hist.push(d);
      if (hist.length > WINDOW) hist.shift();
      boneHistRef.current.set(i, hist);
    });

    // ---- Joint jitter -------------------------------------------------------
    msg.points_3d.forEach((p, i) => {
      if (!msg.valid_3d[i] || !p || p[0] == null) return;
      const hist = jointHistRef.current.get(i) ?? [];
      hist.push([p[0] as number, p[1] as number, p[2] as number]);
      if (hist.length > WINDOW) hist.shift();
      jointHistRef.current.set(i, hist);
      jointValidRef.current.set(i, (jointValidRef.current.get(i) ?? 0) + 1);
    });

    // ---- Camera reprojection error ------------------------------------------
    Object.keys(meta.cameras).forEach((cidStr) => {
      const calib = meta.cameras[cidStr];
      const pose = msg.poses_2d[cidStr];
      if (!pose?.detected) return;
      const reproj = projectPoints(calib, msg.points_3d);
      let errSum = 0; let errCount = 0;
      reproj.forEach((rp, i) => {
        if (!rp || !msg.valid_3d[i] || pose.conf[i] < 0.3) return;
        const dx = rp[0] - pose.kp[i][0];
        const dy = rp[1] - pose.kp[i][1];
        errSum += Math.sqrt(dx * dx + dy * dy);
        errCount++;
      });
      if (errCount === 0) return;
      const meanErr = errSum / errCount;
      const hist = camErrHistRef.current.get(cidStr) ?? [];
      hist.push(meanErr);
      if (hist.length > WINDOW) hist.shift();
      camErrHistRef.current.set(cidStr, hist);
    });

    // ---- Overall ------------------------------------------------------------
    // Detection rate
    const validCount = msg.valid_3d.filter(Boolean).length;
    const detRate = validCount / (msg.valid_3d.length || 1);
    detHistRef.current.push(detRate);
    if (detHistRef.current.length > WINDOW) detHistRef.current.shift();

    fpsHistRef.current.push(msg.fps);
    if (fpsHistRef.current.length > WINDOW) fpsHistRef.current.shift();

    // MPJPE vs GT
    let mpjpe: number | null = null;
    if (msg.gt_3d && meta.h36m_to_coco_subset?.length) {
      let errSum = 0; let count = 0;
      for (const [h36mIdx, cocoIdx] of meta.h36m_to_coco_subset) {
        const gt = msg.gt_3d[h36mIdx];
        const pred = msg.points_3d[cocoIdx];
        if (!gt || gt[0] == null || !pred || pred[0] == null || !msg.valid_3d[cocoIdx]) continue;
        const dx = (gt[0] as number) - (pred[0] as number);
        const dy = (gt[1] as number) - (pred[1] as number);
        const dz = (gt[2] as number) - (pred[2] as number);
        errSum += Math.sqrt(dx * dx + dy * dy + dz * dz);
        count++;
      }
      if (count > 0) mpjpe = errSum / count;
    }
    if (mpjpe != null) {
      mpjpeHistRef.current.push(mpjpe);
      if (mpjpeHistRef.current.length > WINDOW) mpjpeHistRef.current.shift();
    }

    // ---- Throttle render: ~5Hz ----------------------------------------------
    const now = Date.now();
    if (now - throttleRef.current < 200) return;
    throttleRef.current = now;

    const totalFrames = frameCountRef.current;

    const bones: BoneStats[] = COCO17_SKELETON.map(([a, b], i) => {
      const hist = boneHistRef.current.get(i) ?? [];
      const m = mean(hist);
      const s = std(hist, m);
      return { bone: boneLabel(a, b), meanLen: m, cv: m > 0 ? s / m : 0, history: [...hist] };
    });

    const joints: JointJitter[] = COCO17_JOINT_NAMES.map((name, i) => {
      const hist = jointHistRef.current.get(i) ?? [];
      const validCnt = jointValidRef.current.get(i) ?? 0;
      // Jitter = std of per-axis positions concatenated magnitude
      let jitter = 0;
      if (hist.length > 1) {
        const mx = mean(hist.map((p) => p[0]));
        const my = mean(hist.map((p) => p[1]));
        const mz = mean(hist.map((p) => p[2]));
        const stds = [
          std(hist.map((p) => p[0]), mx),
          std(hist.map((p) => p[1]), my),
          std(hist.map((p) => p[2]), mz),
        ];
        jitter = Math.sqrt(stds[0] ** 2 + stds[1] ** 2 + stds[2] ** 2);
      }
      return {
        joint: name,
        jitter,
        validPct: totalFrames > 0 ? validCnt / totalFrames : 0,
        history: hist.map((p) => Math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)),
      };
    });

    const cameras: CameraReproj[] = Array.from(camErrHistRef.current.entries()).map(([cidStr, hist]) => {
      const m = mean(hist);
      // Worst joint: find joint with highest err in last frame
      const calib = meta.cameras[cidStr];
      let worstJoint = "—";
      if (calib && frame.msg.points_3d) {
        const reproj = projectPoints(calib, frame.msg.points_3d);
        const pose = frame.msg.poses_2d[cidStr];
        let maxErr = -1;
        reproj.forEach((rp, i) => {
          if (!rp || !frame.msg.valid_3d[i] || !pose?.detected || pose.conf[i] < 0.3) return;
          const dx = rp[0] - pose.kp[i][0];
          const dy = rp[1] - pose.kp[i][1];
          const e = Math.sqrt(dx * dx + dy * dy);
          if (e > maxErr) { maxErr = e; worstJoint = COCO17_JOINT_NAMES[i]; }
        });
      }
      return { camId: cidStr, meanErr: m, worstJoint, history: [...hist] };
    });

    setMetrics({
      bones,
      joints,
      cameras,
      overall: {
        mpjpe: mpjpe ?? (mpjpeHistRef.current.length ? mpjpeHistRef.current[mpjpeHistRef.current.length - 1] : null),
        detectionRate: detRate,
        fps: msg.fps,
        fpsHistory: [...fpsHistRef.current],
        mpjpeHistory: [...mpjpeHistRef.current],
        detRateHistory: [...detHistRef.current],
      },
    });
  }, [frame, meta]);

  return metrics;
}
