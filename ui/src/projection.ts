/**
 * Pure JS mirror of livepose/core.py:project_points().
 * Projects world-space 3D points into pixel coordinates for a single camera.
 */

import type { CameraCalib } from "./types";

/**
 * Project an array of 3D world points into 2D pixel coordinates.
 *
 * @param calib  Camera calibration (K 3x3, R 3x3, t [3])
 * @param pts3d  Array of [x, y, z] world points (nulls/NaN skipped)
 * @returns      Array of [px, py] pixel coords, or null where projection
 *               fails (point behind camera, or input invalid).
 */
export function projectPoints(
  calib: CameraCalib,
  pts3d: (number | null)[][],
): ([number, number] | null)[] {
  const { K, R, t } = calib;

  return pts3d.map((p) => {
    if (!p || p[0] == null || p[1] == null || p[2] == null) return null;
    const [X, Y, Z] = p as [number, number, number];
    if (!isFinite(X) || !isFinite(Y) || !isFinite(Z)) return null;

    // X_cam = R @ X_world + t
    const xc = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0];
    const yc = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1];
    const zc = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2];

    if (zc <= 0) return null; // behind camera

    // px = K @ [xc, yc, zc]^T / zc
    const px = (K[0][0] * xc + K[0][1] * yc + K[0][2] * zc) / zc;
    const py = (K[1][0] * xc + K[1][1] * yc + K[1][2] * zc) / zc;

    if (!isFinite(px) || !isFinite(py)) return null;
    return [px, py];
  });
}
