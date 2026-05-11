"""Core types: Camera, Skeleton, calibration parsing, triangulation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Camera:
    """A calibrated camera. Translation in millimeters (MPI convention)."""

    cam_id: int
    K: np.ndarray  # 3x3 intrinsic
    R: np.ndarray  # 3x3 rotation (world -> camera)
    t: np.ndarray  # 3,   translation (world -> camera)
    width: int
    height: int

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix: x = P @ [X, Y, Z, 1]^T."""
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt

    @property
    def position(self) -> np.ndarray:
        """Camera center in world coordinates: C = -R^T @ t."""
        return -self.R.T @ self.t


def parse_mpi_calibration(path: str | Path) -> dict[int, Camera]:
    """Parse the MPI-INF-3DHP `camera.calibration` file.

    Format is a custom Skeletool text format. Each `name N` block contains
    `intrinsic` (16 floats, 4x4 row-major; we use top-left 3x3) and
    `extrinsic` (16 floats, 4x4 row-major; we use top-left 3x3 R and 3x1 t).
    """
    text = Path(path).read_text()
    cams: dict[int, Camera] = {}

    blocks = re.split(r"^name\s+", text, flags=re.MULTILINE)[1:]
    for block in blocks:
        lines = block.splitlines()
        cam_id = int(lines[0].strip())
        kv: dict[str, str] = {}
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                kv[parts[0]] = parts[1]

        size_parts = kv.get("size", "0 0").split()
        width, height = int(size_parts[0]), int(size_parts[1])

        intrinsic = np.array([float(x) for x in kv["intrinsic"].split()]).reshape(4, 4)
        extrinsic = np.array([float(x) for x in kv["extrinsic"].split()]).reshape(4, 4)

        K = intrinsic[:3, :3].copy()
        R = extrinsic[:3, :3].copy()
        t = extrinsic[:3, 3].copy()

        cams[cam_id] = Camera(cam_id=cam_id, K=K, R=R, t=t, width=width, height=height)

    return cams


def project_points(cam: Camera, points_3d: np.ndarray) -> np.ndarray:
    """Project 3D points (N,3) to 2D pixels (N,2) using camera P."""
    N = points_3d.shape[0]
    homog = np.hstack([points_3d, np.ones((N, 1))])
    proj = (cam.P @ homog.T).T  # (N, 3)
    return proj[:, :2] / proj[:, 2:3]


def compute_fundamental_matrix(cam_a: Camera, cam_b: Camera) -> np.ndarray:
    """Compute fundamental matrix F such that x_b^T @ F @ x_a = 0.

    Derived analytically from calibrated camera parameters (no RANSAC needed).
    """
    R_rel = cam_b.R @ cam_a.R.T
    t_rel = cam_b.t - cam_b.R @ cam_a.R.T @ cam_a.t
    tx = np.array([
        [0,         -t_rel[2],  t_rel[1]],
        [t_rel[2],   0,        -t_rel[0]],
        [-t_rel[1],  t_rel[0],  0],
    ])
    E = tx @ R_rel
    F = np.linalg.inv(cam_b.K).T @ E @ np.linalg.inv(cam_a.K)
    denom = F[2, 2]
    return F / denom if abs(denom) > 1e-12 else F


def epipolar_distance(F: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray) -> float:
    """Symmetric point-to-epipolar-line distance in pixels.

    Returns the average of the distance from pt_b to its epipolar line in
    image B and the distance from pt_a to its epipolar line in image A.
    """
    pa = np.array([pt_a[0], pt_a[1], 1.0])
    pb = np.array([pt_b[0], pt_b[1], 1.0])

    # Epipolar line in B for point in A: l_b = F @ pa
    l_b = F @ pa
    d_b = abs(float(pb @ l_b)) / (np.sqrt(l_b[0] ** 2 + l_b[1] ** 2) + 1e-12)

    # Epipolar line in A for point in B: l_a = F.T @ pb
    l_a = F.T @ pb
    d_a = abs(float(pa @ l_a)) / (np.sqrt(l_a[0] ** 2 + l_a[1] ** 2) + 1e-12)

    return float(0.5 * (d_a + d_b))


def check_epipolar_consistency(
    cameras: list[Camera],
    keypoints: list[np.ndarray],  # one (J, 2) per camera
    threshold_px: float = 15.0,
) -> list[int]:
    """Return indices of cameras consistent with the majority.

    For each camera pair, compute median epipolar distance across joints.
    A camera whose median distance to the majority is above threshold_px is
    flagged as an outlier. Returns list of local indices (into `cameras`)
    of cameras to keep.

    With ≤2 cameras, always returns [0, 1] (no way to vote out either).
    """
    n = len(cameras)
    if n <= 2:
        return list(range(n))

    # Pre-compute fundamental matrices for all pairs
    F_cache: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n):
        for j in range(i + 1, n):
            F_cache[(i, j)] = compute_fundamental_matrix(cameras[i], cameras[j])

    J = keypoints[0].shape[0]

    # For each camera, accumulate median distances to all other cameras
    cam_scores = np.zeros(n)
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            F = F_cache.get((min(i, j), max(i, j)))
            if F is None:
                continue
            # Use F such that x_j^T @ F @ x_i = 0 when i < j
            F_ij = F if i < j else F.T
            joint_dists = []
            for k in range(J):
                d = epipolar_distance(F_ij, keypoints[i][k], keypoints[j][k])
                joint_dists.append(d)
            dists.append(float(np.median(joint_dists)))
        cam_scores[i] = float(np.median(dists)) if dists else 0.0

    # Cameras within threshold of the group median are consistent
    group_median = float(np.median(cam_scores))
    consistent = [i for i in range(n) if cam_scores[i] <= group_median + threshold_px]

    # Always keep at least 2 cameras
    if len(consistent) < 2:
        consistent = list(np.argsort(cam_scores)[:2])

    return consistent


def _solve_dlt(rows: list[np.ndarray]) -> np.ndarray | None:
    """Solve DLT system via SVD. Returns 3D point or None if degenerate."""
    A = np.stack(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-9:
        return None
    return X[:3] / X[3]


def triangulate_dlt(
    cameras: list[Camera],
    points_2d: list[np.ndarray],
    confidences: list[np.ndarray] | None = None,
    min_views: int = 2,
    conf_threshold: float = 0.3,
    refine: bool = True,
    outlier_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a set of joint observations across multiple cameras.

    Uses confidence-weighted DLT with optional iterative outlier rejection
    based on per-camera reprojection error standard deviation.

    Args:
        cameras: list of K cameras with valid projection matrices
        points_2d: list of K arrays, each (J, 2) — pixel coords per joint per cam
        confidences: list of K arrays, each (J,) — per-joint confidence per cam
        min_views: minimum cameras per joint to attempt triangulation
        conf_threshold: drop observations below this confidence
        refine: if True, do a second pass rejecting outlier cameras
        outlier_sigma: reject cameras with reproj error > mean + sigma*std

    Returns:
        points_3d: (J, 3) triangulated joints (NaN where untriangulable)
        valid: (J,) bool mask of joints successfully triangulated
        reproj_errors: (J,) mean reprojection error per joint (px); 0.0 where invalid
    """
    K = len(cameras)
    if K < 2:
        raise ValueError("Need at least 2 cameras to triangulate.")
    J = points_2d[0].shape[0]

    if confidences is None:
        confidences = [np.ones(J, dtype=np.float32) for _ in range(K)]

    points_3d = np.full((J, 3), np.nan, dtype=np.float64)
    valid = np.zeros(J, dtype=bool)
    reproj_errors = np.zeros(J, dtype=np.float64)

    for j in range(J):
        # Gather contributing cameras (passing confidence threshold)
        contrib_cams: list[int] = []
        rows: list[np.ndarray] = []
        for k in range(K):
            if confidences[k][j] < conf_threshold:
                continue
            contrib_cams.append(k)
            x, y = points_2d[k][j]
            P = cameras[k].P
            w = float(confidences[k][j])  # confidence weight
            rows.append(w * (x * P[2] - P[0]))
            rows.append(w * (y * P[2] - P[1]))

        if len(rows) < 2 * min_views:
            continue

        pt = _solve_dlt(rows)
        if pt is None:
            continue

        # Track which cameras contribute to the final point (may be reduced by refinement)
        final_cams = contrib_cams

        # Refinement pass: reject outlier cameras by reprojection error
        if refine and len(contrib_cams) > min_views:
            pt_h = np.append(pt, 1.0)
            errors = []
            for k in contrib_cams:
                proj = cameras[k].P @ pt_h
                if abs(proj[2]) < 1e-9:
                    errors.append(1e6)
                    continue
                proj_px = proj[:2] / proj[2]
                err = float(np.linalg.norm(proj_px - points_2d[k][j]))
                errors.append(err)

            err_arr = np.array(errors)
            mu = float(np.mean(err_arr))
            sigma = float(np.std(err_arr))

            if sigma > 0:
                threshold = mu + outlier_sigma * sigma
                kept = [k for k, e in zip(contrib_cams, errors) if e <= threshold]

                if len(kept) >= min_views and len(kept) < len(contrib_cams):
                    # Re-triangulate with kept cameras only
                    rows2 = []
                    for k in kept:
                        x, y = points_2d[k][j]
                        P = cameras[k].P
                        w = float(confidences[k][j])
                        rows2.append(w * (x * P[2] - P[0]))
                        rows2.append(w * (y * P[2] - P[1]))
                    pt2 = _solve_dlt(rows2)
                    if pt2 is not None:
                        pt = pt2
                        final_cams = kept

        points_3d[j] = pt
        valid[j] = True

        # Compute mean reprojection error for this joint using final cameras
        pt_h = np.append(pt, 1.0)
        cam_errors: list[float] = []
        for k in final_cams:
            proj = cameras[k].P @ pt_h
            if abs(proj[2]) < 1e-9:
                continue
            proj_px = proj[:2] / proj[2]
            cam_errors.append(float(np.linalg.norm(proj_px - points_2d[k][j])))
        reproj_errors[j] = float(np.mean(cam_errors)) if cam_errors else 0.0

    return points_3d, valid, reproj_errors


# COCO-17 skeleton (matches YOLO-Pose output)
COCO17_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO17_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
    (0, 1), (0, 2), (1, 3), (2, 4),              # face
]

# Human3.6M-17 joint ordering returned by the MPI-INF-3DHP ground-truth loader.
# Indices: 0=head_top, 1=neck, 2=R_shoulder, 3=R_elbow, 4=R_wrist,
#          5=L_shoulder, 6=L_elbow, 7=L_wrist, 8=R_hip, 9=R_knee, 10=R_ankle,
#          11=L_hip, 12=L_knee, 13=L_ankle, 14=pelvis, 15=spine, 16=head
H36M17_JOINTS = [
    "head_top", "neck", "R_shoulder", "R_elbow", "R_wrist",
    "L_shoulder", "L_elbow", "L_wrist", "R_hip", "R_knee", "R_ankle",
    "L_hip", "L_knee", "L_ankle", "pelvis", "spine", "head",
]

H36M17_SKELETON = [
    (0, 16),   # head_top - head
    (16, 1),   # head - neck
    (1, 15),   # neck - spine
    (15, 14),  # spine - pelvis
    (1, 2),    # neck - R_shoulder
    (2, 3),    # R_shoulder - R_elbow
    (3, 4),    # R_elbow - R_wrist
    (1, 5),    # neck - L_shoulder
    (5, 6),    # L_shoulder - L_elbow
    (6, 7),    # L_elbow - L_wrist
    (14, 8),   # pelvis - R_hip
    (8, 9),    # R_hip - R_knee
    (9, 10),   # R_knee - R_ankle
    (14, 11),  # pelvis - L_hip
    (11, 12),  # L_hip - L_knee
    (12, 13),  # L_knee - L_ankle
]

# Shared joints between COCO-17 (predicted) and H36M-17 (GT).
# Each tuple: (h36m_idx, coco_idx) for anatomically equivalent joints.
# Used for MPJPE computation in the frontend metrics panel.
H36M_TO_COCO_SUBSET: list[tuple[int, int]] = [
    (2, 6),   # R_shoulder
    (5, 5),   # L_shoulder
    (3, 8),   # R_elbow
    (6, 7),   # L_elbow
    (4, 10),  # R_wrist
    (7, 9),   # L_wrist
    (8, 12),  # R_hip
    (11, 11), # L_hip
    (9, 14),  # R_knee
    (12, 13), # L_knee
    (10, 16), # R_ankle
    (13, 15), # L_ankle
]
