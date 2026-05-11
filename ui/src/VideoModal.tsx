import { useEffect, useRef, useMemo, useCallback, useState } from "react";
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

const COCO17_NAMES = [
  "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
  "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
  "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
  "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
];

const H36M17_NAMES = [
  "Hip", "R Hip", "R Knee", "R Ankle",
  "L Hip (unused)", "L Hip", "L Knee", "L Ankle",
  "Spine", "Neck", "Head Top",
  "L Shoulder (unused)", "L Shoulder", "L Elbow", "L Wrist",
  "R Shoulder", "Thorax",
];

/** Map a 0–1 confidence to a heatmap color: red → yellow → green */
function confToColor(c: number): string {
  const clamped = Math.max(0, Math.min(1, c));
  if (clamped < 0.5) {
    // red (0) → yellow (0.5)
    const t = clamped / 0.5;
    const r = 255;
    const g = Math.round(200 * t);
    return `rgb(${r},${g},30)`;
  } else {
    // yellow (0.5) → green (1)
    const t = (clamped - 0.5) / 0.5;
    const r = Math.round(255 * (1 - t));
    const g = Math.round(200 + 55 * t);
    return `rgb(${r},${g},30)`;
  }
}

type Props = {
  camId: number;
  frame: DecodedFrame | null;
  meta: MetaMessage | null;
  showRaw2D: boolean;
  showReproj3D: boolean;
  showReprojGT: boolean;
  showGroundTruth: boolean;
  onToggleRaw2D?: (val: boolean) => void;
  onToggleReproj3D?: (val: boolean) => void;
  onToggleReprojGT?: (val: boolean) => void;
  onToggleGT?: (val: boolean) => void;
  onClose: () => void;
};

export default function VideoModal({
  camId, frame, meta,
  showRaw2D, showReproj3D, showReprojGT, showGroundTruth,
  onToggleRaw2D, onToggleReproj3D, onToggleReprojGT, onToggleGT,
  onClose,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);
  const [cocoOpen, setCocoOpen] = useState(true);
  const [h36mOpen, setH36mOpen] = useState(true);

  const cocoLinks = useMemo(
    () => (meta?.skeletons.coco17.edges as [number, number][]) ?? COCO17_LINKS,
    [meta],
  );
  const h36mLinks = useMemo(
    () => (meta?.skeletons.h36m17.edges as [number, number][]) ?? H36M17_LINKS,
    [meta],
  );

  const cocoNames = useMemo(
    () => (meta?.skeletons.coco17.joints) ?? COCO17_NAMES,
    [meta],
  );
  const h36mNames = useMemo(
    () => (meta?.skeletons.h36m17.joints) ?? H36M17_NAMES,
    [meta],
  );

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  // Click-outside-to-close
  const handleBackdropClick = useCallback((e: React.MouseEvent) => {
    if (e.target === backdropRef.current) onClose();
  }, [onClose]);

  // Draw the video + overlays at full resolution
  useEffect(() => {
    if (!frame) return;
    const blob = frame.jpegByCam.get(camId);
    if (!blob) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let cancelled = false;
    const img = new Image();
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      if (cancelled) { URL.revokeObjectURL(url); return; }

      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);

      const calib = meta?.cameras[String(camId)];
      const origW = calib?.image_size?.[0] ?? canvas.width;
      const origH = calib?.image_size?.[1] ?? canvas.height;
      const scaleX = canvas.width / origW;
      const scaleY = canvas.height / origH;
      const scalePoints = (pts: Point2D[]): Point2D[] =>
        pts.map((p) => (p ? [p[0] * scaleX, p[1] * scaleY] : null));

      const pose = frame.msg.poses_2d[String(camId)];

      // Raw 2D — cyan
      if (showRaw2D && pose?.detected) {
        const raw: Point2D[] = kpToPoints(pose.kp, pose.conf);
        drawPoseOverlay(ctx, scalePoints(raw), pose.conf, cocoLinks, "rgba(0,255,213,0.85)");
      }

      // Reprojected 3D — magenta
      if (showReproj3D && calib && frame.msg.points_3d) {
        const reproj = projectPoints(calib, frame.msg.points_3d);
        const allConf = new Array(reproj.length).fill(1);
        const valid = frame.msg.valid_3d;
        const maskedReproj: Point2D[] = reproj.map((p, i) => (valid[i] ? p : null));
        drawPoseOverlay(ctx, scalePoints(maskedReproj), allConf, cocoLinks, "rgba(255,0,255,0.85)", undefined, 0);
      }

      // Reprojected GT — yellow
      if (showReprojGT && showGroundTruth && calib && frame.msg.gt_3d) {
        const reproj = projectPoints(calib, frame.msg.gt_3d);
        const allConf = new Array(reproj.length).fill(1);
        const maskedReproj: Point2D[] = reproj.map((p, i) => {
          const g = frame.msg.gt_3d![i];
          return g && g[0] != null ? p : null;
        });
        drawPoseOverlay(ctx, scalePoints(maskedReproj), allConf, h36mLinks, "rgba(255,213,74,0.85)", undefined, 0);
      }

      // Draw confidence heatmap dots on each joint
      if (showRaw2D && pose?.detected) {
        const raw: Point2D[] = kpToPoints(pose.kp, pose.conf, 0);
        const scaled = scalePoints(raw);
        const r = Math.max(4, canvas.width / 200);
        ctx.save();
        for (let i = 0; i < scaled.length; i++) {
          const p = scaled[i];
          if (!p) continue;
          const c = pose.conf[i] ?? 0;
          const col = confToColor(c);
          // Glow halo
          ctx.shadowColor = col;
          ctx.shadowBlur = r * 1.5;
          ctx.fillStyle = col;
          ctx.strokeStyle = "rgba(0,0,0,0.6)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
        ctx.restore();
      }
    };
    img.src = url;
    return () => { cancelled = true; };
  }, [frame, camId, meta, cocoLinks, h36mLinks, showRaw2D, showReproj3D, showReprojGT, showGroundTruth]);

  return (
    <div
      ref={backdropRef}
      className="video-modal-backdrop"
      onClick={handleBackdropClick}
    >
      <div className="video-modal-content">
        {/* Header */}
        <div className="video-modal-header">
          <div className="video-modal-title">
            <div className="video-modal-dot" />
            <span>CAM {camId}</span>
            <span className="video-modal-subtitle">Expanded View</span>
          </div>
          <button className="video-modal-close" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M15 5L5 15M5 5l10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Body: canvas + legend */}
        <div className="video-modal-body">
          {/* Canvas */}
          <div className="video-modal-canvas-wrap">
            <canvas ref={canvasRef} className="video-modal-canvas" />
            {frame && !frame.msg.poses_2d[String(camId)]?.detected && (
              <div className="video-modal-no-person">No person detected</div>
            )}
          </div>

          {/* Legend sidebar */}
          <div className="video-modal-legend">
            {/* Overlay color legend */}
            <div className="video-modal-legend-section">
              <div className="video-modal-legend-title">Overlay Colors</div>
              <div className="video-modal-legend-items">
                <LegendEntry color="rgba(0,255,213,0.85)" label="Raw 2D Detection" active={showRaw2D} onClick={() => onToggleRaw2D?.(!showRaw2D)} />
                <LegendEntry color="rgba(255,0,255,0.85)" label="Reprojected 3D" active={showReproj3D} onClick={() => onToggleReproj3D?.(!showReproj3D)} />
                <LegendEntry color="rgba(255,213,74,0.85)" label="Reprojected GT" active={showReprojGT && showGroundTruth} onClick={() => {
                  if (!showGroundTruth) onToggleGT?.(true);
                  onToggleReprojGT?.(!showReprojGT);
                }} />
                <LegendEntry color="#ff8a3d" label="Ground Truth 3D" active={showGroundTruth} onClick={() => onToggleGT?.(!showGroundTruth)} />
              </div>
            </div>

            {/* Confidence heatmap legend */}
            <div className="video-modal-legend-section">
              <div className="video-modal-legend-title">
                Confidence Heatmap
              </div>
              <div className="video-modal-heatmap-bar-wrap">
                <div className="video-modal-heatmap-bar" />
                <div className="video-modal-heatmap-labels">
                  <span>0.0</span>
                  <span>0.5</span>
                  <span>1.0</span>
                </div>
                <div className="video-modal-heatmap-labels">
                  <span style={{ color: confToColor(0) }}>Low</span>
                  <span style={{ color: confToColor(0.5) }}>Med</span>
                  <span style={{ color: confToColor(1) }}>High</span>
                </div>
              </div>
            </div>

            {/* Per-joint confidences */}
            <div className="video-modal-legend-section">
              <div 
                className="video-modal-legend-title"
                onClick={() => setCocoOpen(!cocoOpen)}
                style={{ cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center" }}
              >
                <span>
                  Joint Confidences
                  <span className="video-modal-legend-badge">{cocoNames.length}</span>
                </span>
                <span>{cocoOpen ? "▼" : "▶"}</span>
              </div>
              {cocoOpen && (
                <div className="video-modal-legend-kp-list">
                  {cocoNames.map((name, i) => {
                    const pose = frame?.msg.poses_2d[String(camId)];
                    const c = pose?.conf[i] ?? 0;
                    return (
                      <div key={i} className="video-modal-legend-kp">
                        <span
                          className="video-modal-legend-kp-dot"
                          style={{ background: confToColor(c) }}
                        />
                        <span className="video-modal-legend-kp-idx">{i}</span>
                        <span className="video-modal-legend-kp-name">{name}</span>
                        <span className="video-modal-legend-conf" style={{ color: confToColor(c) }}>
                          {c.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* H36M joints — shown only with GT */}
            {showGroundTruth && (
              <div className="video-modal-legend-section">
                <div 
                  className="video-modal-legend-title"
                  onClick={() => setH36mOpen(!h36mOpen)}
                  style={{ cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center" }}
                >
                  <span>
                    H36M-17 Joints
                    <span className="video-modal-legend-badge">{h36mNames.length}</span>
                  </span>
                  <span>{h36mOpen ? "▼" : "▶"}</span>
                </div>
                {h36mOpen && (
                  <div className="video-modal-legend-kp-list">
                    {h36mNames.map((name, i) => (
                      <div key={i} className="video-modal-legend-kp">
                        <span
                          className="video-modal-legend-kp-dot"
                          style={{ background: "#ff8a3d" }}
                        />
                        <span className="video-modal-legend-kp-idx">{i}</span>
                        <span className="video-modal-legend-kp-name">{name}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function LegendEntry({ color, label, active, onClick }: { color: string; label: string; active: boolean; onClick?: () => void }) {
  return (
    <div 
      className={`video-modal-legend-entry ${!active ? "video-modal-legend-entry-inactive" : ""}`}
      onClick={onClick}
      style={{ cursor: onClick ? "pointer" : "default" }}
    >
      <span className="video-modal-legend-swatch" style={{ background: color }} />
      <span>{label}</span>
      {!active && <span className="video-modal-legend-off">OFF</span>}
    </div>
  );
}
