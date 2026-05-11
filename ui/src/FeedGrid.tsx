import { useEffect, useMemo, useRef } from "react";
import type { DecodedFrame } from "./socket";
import type { MetaMessage } from "./types";
import { projectPoints } from "./projection";
import { drawPoseOverlay, kpToPoints } from "./poseOverlay";
import type { Point2D } from "./poseOverlay";

const COCO17_LINKS: [number, number][] = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4],
];

const H36M17_LINKS: [number, number][] = [
  [0, 16], [16, 1], [1, 15], [15, 14],
  [1, 2], [2, 3], [3, 4],
  [1, 5], [5, 6], [6, 7],
  [14, 8], [8, 9], [9, 10],
  [14, 11], [11, 12], [12, 13],
];

type Props = {
  frame: DecodedFrame | null;
  meta: MetaMessage | null;
  selectedCam: number | null;
  onSelect: (camId: number | null) => void;
  showRaw2D: boolean;
  showReproj3D: boolean;
  showReprojGT: boolean;
  showGroundTruth: boolean;
  enabledCams: Set<number> | null; // null = all enabled
  onToggleCam: (camId: number) => void;
  onExpand: (camId: number) => void;
};

export default function FeedGrid({
  frame, meta, selectedCam, onSelect,
  showRaw2D, showReproj3D, showReprojGT, showGroundTruth,
  enabledCams, onToggleCam, onExpand,
}: Props) {
  const camIds = frame?.msg.cam_ids ?? [];
  return (
    <div className="panel p-2 h-full overflow-auto">
      <div className="label mb-2">Cameras ({camIds.length})</div>
      <div className="grid grid-cols-2 gap-2">
        {camIds.map((cid) => {
          const enabled = enabledCams === null || enabledCams.has(cid);
          return (
            <FeedCell
              key={cid}
              camId={cid}
              frame={frame}
              meta={meta}
              selected={selectedCam === cid}
              onClick={() => onSelect(selectedCam === cid ? null : cid)}
              showRaw2D={showRaw2D}
              showReproj3D={showReproj3D}
              showReprojGT={showReprojGT}
              showGroundTruth={showGroundTruth}
              enabled={enabled}
              onToggleEnabled={() => onToggleCam(cid)}
              onExpand={() => onExpand(cid)}
            />
          );
        })}
      </div>
    </div>
  );
}

function FeedCell({
  camId, frame, meta, selected, onClick,
  showRaw2D, showReproj3D, showReprojGT, showGroundTruth,
  enabled, onToggleEnabled, onExpand,
}: {
  camId: number;
  frame: DecodedFrame | null;
  meta: MetaMessage | null;
  selected: boolean;
  onClick: () => void;
  showRaw2D: boolean;
  showReproj3D: boolean;
  showReprojGT: boolean;
  showGroundTruth: boolean;
  enabled: boolean;
  onToggleEnabled: () => void;
  onExpand: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const blob = frame?.jpegByCam.get(camId);
  const pose = frame?.msg.poses_2d[String(camId)];
  const cocoLinks = useMemo(
    () => (meta?.skeletons.coco17.edges as [number, number][]) ?? COCO17_LINKS,
    [meta],
  );
  const h36mLinks = useMemo(
    () => (meta?.skeletons.h36m17.edges as [number, number][]) ?? H36M17_LINKS,
    [meta],
  );

  useEffect(() => {
    if (!blob) return;
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      if (cancelled) { URL.revokeObjectURL(url); return; }

      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);

      const calib = meta?.cameras[String(camId)];

      // Keypoints & reprojected points are in original image coordinates,
      // but the canvas is at JPEG resolution (resized). Compute scale factors.
      const origW = calib?.image_size?.[0] ?? canvas.width;
      const origH = calib?.image_size?.[1] ?? canvas.height;
      const scaleX = canvas.width / origW;
      const scaleY = canvas.height / origH;

      const scalePoints = (pts: Point2D[]): Point2D[] =>
        pts.map((p) => (p ? [p[0] * scaleX, p[1] * scaleY] : null));

      // Raw 2D detection — cyan
      if (showRaw2D && pose?.detected) {
        const raw: Point2D[] = kpToPoints(pose.kp, pose.conf);
        drawPoseOverlay(ctx, scalePoints(raw), pose.conf, cocoLinks, "rgba(0,255,213,0.85)");
      }

      // Reprojected 3D prediction — magenta
      if (showReproj3D && calib && frame?.msg.points_3d) {
        const reproj = projectPoints(calib, frame.msg.points_3d);
        const allConf = new Array(reproj.length).fill(1);
        const valid = frame.msg.valid_3d;
        const maskedReproj: Point2D[] = reproj.map((p, i) => (valid[i] ? p : null));
        drawPoseOverlay(ctx, scalePoints(maskedReproj), allConf, cocoLinks, "rgba(255,0,255,0.85)", undefined, 0);
      }

      // Reprojected GT — yellow
      if (showReprojGT && showGroundTruth && calib && frame?.msg.gt_3d) {
        const reproj = projectPoints(calib, frame.msg.gt_3d);
        const allConf = new Array(reproj.length).fill(1);
        const maskedReproj: Point2D[] = reproj.map((p, i) => {
          const g = frame.msg.gt_3d![i];
          return g && g[0] != null ? p : null;
        });
        drawPoseOverlay(ctx, scalePoints(maskedReproj), allConf, h36mLinks, "rgba(255,213,74,0.85)", undefined, 0);
      }
    };
    img.src = url;
    return () => { cancelled = true; };
  }, [blob, pose, meta, frame, camId, cocoLinks, h36mLinks, showRaw2D, showReproj3D, showReprojGT, showGroundTruth]);

  return (
    <div
      className={`relative aspect-video bg-bg-0 border rounded overflow-hidden transition-all cursor-pointer ${
        selected ? "border-teal-500 shadow-glow" : "border-edge hover:border-teal-700"
      } ${!enabled ? "opacity-40" : ""}`}
      onClick={onClick}
    >
      <canvas ref={canvasRef} className="w-full h-full object-contain" />
      <div className="absolute top-1 left-1 flex items-center gap-1 bg-bg-0/80 px-1.5 py-0.5 rounded">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => { e.stopPropagation(); onToggleEnabled(); }}
          onClick={(e) => e.stopPropagation()}
          className="accent-teal-500 w-3 h-3"
        />
        <span className="text-teal-400 font-mono text-[10px]">CAM {camId}</span>
      </div>
      {/* Expand / maximize button */}
      <button
        className="absolute top-1 right-1 flex items-center justify-center w-5 h-5 rounded bg-bg-0/80 border border-edge text-ink-2 hover:text-teal-400 hover:border-teal-700 transition-colors"
        title="Expand feed"
        onClick={(e) => { e.stopPropagation(); onExpand(); }}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M1 4V1h3M8 1h3v3M11 8v3H8M4 11H1V8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {pose && !pose.detected && (
        <div className="absolute bottom-1 left-1 text-[10px] text-ink-2 font-mono bg-bg-0/80 px-1 rounded">
          no person
        </div>
      )}
    </div>
  );
}
