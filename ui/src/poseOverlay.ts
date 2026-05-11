/**
 * Generic canvas-2D pose overlay helper.
 * Accepts raw 2D keypoints OR reprojected pixel coords from projection.ts.
 * Null / NaN points are silently skipped.
 */

export type Point2D = [number, number] | null;

/**
 * Draw a skeleton overlay onto a 2D canvas context.
 *
 * @param ctx       Canvas 2D context to draw on
 * @param points    Array of [px, py] or null per joint
 * @param conf      Per-joint confidence (0–1). Pass all-1s if not available.
 * @param links     Skeleton edge list [[a, b], ...]
 * @param color     CSS colour string
 * @param lineWidth Line width in canvas pixels (default auto-scaled)
 * @param confThreshold  Min confidence to draw a joint/bone (default 0.3)
 */
export function drawPoseOverlay(
  ctx: CanvasRenderingContext2D,
  points: Point2D[],
  conf: number[],
  links: [number, number][],
  color: string,
  lineWidth?: number,
  confThreshold = 0.3,
): void {
  const lw = lineWidth ?? Math.max(2, ctx.canvas.width / 400);
  const r = Math.max(3, ctx.canvas.width / 300);

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;

  // Bones
  for (const [a, b] of links) {
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) continue;
    if ((conf[a] ?? 1) < confThreshold || (conf[b] ?? 1) < confThreshold) continue;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  }

  // Joints
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (!p) continue;
    if ((conf[i] ?? 1) < confThreshold) continue;
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

/** Convert raw kp array [J][2] to Point2D[], skipping zeros (undetected). */
export function kpToPoints(kp: number[][], conf: number[], confThreshold = 0.3): Point2D[] {
  return kp.map((xy, i) => {
    if ((conf[i] ?? 0) < confThreshold) return null;
    return [xy[0], xy[1]];
  });
}
