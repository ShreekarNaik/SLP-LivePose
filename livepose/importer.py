"""Auto-detection and loading of multi-camera datasets.

Currently supports MPI-INF-3DHP. Layout:
    {root}/S{n}/Seq{m}/imageSequence/video_{cam}.avi
    {root}/S{n}/Seq{m}/camera.calibration
    {root}/S{n}/Seq{m}/annot.mat (optional, for ground truth overlay)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import scipy.io

from livepose.core import Camera, parse_mpi_calibration


# ---- Discovery types --------------------------------------------------------

@dataclass
class MpiSequence:
    """One subject/sequence within an MPI-INF-3DHP root."""

    root: Path                # path to e.g. .../mpi_inf_3dhp
    subject: str              # "S1"
    name: str                 # "Seq1"
    seq_path: Path            # .../S1/Seq1
    video_map: dict[int, Path]
    calibration_path: Path
    annot_path: Optional[Path]

    @property
    def display_name(self) -> str:
        return f"{self.subject} / {self.name}"

    @property
    def has_ground_truth(self) -> bool:
        return self.annot_path is not None


@dataclass
class DiscoveredDataset:
    kind: str
    root: Path
    sequences: list[MpiSequence] = field(default_factory=list)


# ---- Session (loaded view of a sequence) ------------------------------------

@dataclass
class Session:
    name: str
    cameras: dict[int, Camera]
    video_map: dict[int, Path]
    fps: float
    total_frames: int
    ground_truth_loader: Optional[Callable[[int], Optional[np.ndarray]]] = None
    sample_thumbnails: dict[int, bytes] = field(default_factory=dict)  # cam_id -> jpeg


# ---- Discovery --------------------------------------------------------------

_VIDEO_RE = re.compile(r"video_(\d+)\.(?:avi|mp4|mov|mkv)$", re.IGNORECASE)


def scan_for_datasets(root: str | Path) -> list[DiscoveredDataset]:
    """Walk a folder and detect supported dataset layouts."""
    root = Path(root)
    if not root.exists():
        return []

    found: list[DiscoveredDataset] = []
    mpi = _scan_mpi(root)
    if mpi.sequences:
        found.append(mpi)
    return found


def _scan_mpi(root: Path) -> DiscoveredDataset:
    sequences: list[MpiSequence] = []

    # Search for any S*/Seq*/camera.calibration under root
    for cal in sorted(root.glob("S*/Seq*/camera.calibration")):
        seq_path = cal.parent
        subject = seq_path.parent.name
        name = seq_path.name
        img_dir = seq_path / "imageSequence"
        if not img_dir.exists():
            continue

        video_map: dict[int, Path] = {}
        for v in sorted(img_dir.iterdir()):
            m = _VIDEO_RE.match(v.name)
            if m:
                video_map[int(m.group(1))] = v
        if not video_map:
            continue

        annot = seq_path / "annot.mat"
        sequences.append(
            MpiSequence(
                root=root,
                subject=subject,
                name=name,
                seq_path=seq_path,
                video_map=video_map,
                calibration_path=cal,
                annot_path=annot if annot.exists() else None,
            )
        )

    return DiscoveredDataset(kind="mpi_inf_3dhp", root=root, sequences=sequences)


# ---- Loading ----------------------------------------------------------------

def load_mpi_session(
    seq: MpiSequence,
    include_ground_truth: bool = True,
    thumbnail_width: int = 320,
) -> Session:
    """Materialize a MpiSequence into a runnable Session."""
    all_cams = parse_mpi_calibration(seq.calibration_path)
    cameras = {cid: cam for cid, cam in all_cams.items() if cid in seq.video_map}

    # Probe FPS / frame count from the first video
    first_video = next(iter(seq.video_map.values()))
    cap = cv2.VideoCapture(str(first_video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    gt_loader = None
    if include_ground_truth and seq.has_ground_truth:
        gt_loader = _make_mpi_gt_loader(seq.annot_path, list(cameras.keys()))

    thumbnails = _generate_thumbnails(seq.video_map, thumbnail_width)

    return Session(
        name=seq.display_name,
        cameras=cameras,
        video_map=dict(seq.video_map),
        fps=fps,
        total_frames=total_frames,
        ground_truth_loader=gt_loader,
        sample_thumbnails=thumbnails,
    )


def _generate_thumbnails(video_map: dict[int, Path], target_width: int) -> dict[int, bytes]:
    """Grab one frame per camera, encode as JPEG. Used for import preview."""
    out: dict[int, bytes] = {}
    for cam_id, path in video_map.items():
        cap = cv2.VideoCapture(str(path))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = img.shape[:2]
        if w > target_width:
            scale = target_width / w
            img = cv2.resize(img, (target_width, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            out[cam_id] = buf.tobytes()
    return out


def _make_mpi_gt_loader(
    annot_path: Path,
    cam_ids: list[int],
) -> Callable[[int], Optional[np.ndarray]]:
    """Return a callable: frame_idx -> (J,3) world-coord 3D ground truth.

    MPI annotations are stored per-camera in camera coordinates. We pick the
    first available camera and convert to world coordinates using its extrinsic.
    """
    mat = scipy.io.loadmat(str(annot_path))
    annot3 = mat["annot3"]  # (14, 1), each (n_frames, 84)

    # Find an MPI camera_id we have video for; we'll use cam 0 if present
    use_cam_id = cam_ids[0] if cam_ids else 0

    # We also need this camera's extrinsic to invert into world coords. The
    # annotation cam-coord 3D for camera k is X_cam_k. We convert to world via:
    #     X_world = R_k^T @ (X_cam_k - t_k)
    # We'll close over the calibration parsed at load time.
    cal_path = annot_path.parent / "camera.calibration"
    cams = parse_mpi_calibration(cal_path)
    if use_cam_id not in cams:
        # Fallback to any cam present in annotations
        use_cam_id = 0

    cam = cams[use_cam_id]
    R, t = cam.R, cam.t

    # Joint subset for COCO-17-ish skeleton: MPI 'relevant' joints map roughly
    # to COCO. We use the 17-joint Human3.6m-compatible subset (1-indexed in
    # mpii_get_joint_set):
    #   [head_top, neck, R_shoulder, R_elbow, R_wrist, L_shoulder, L_elbow,
    #    L_wrist, R_hip, R_knee, R_ankle, L_hip, L_knee, L_ankle, pelvis,
    #    spine, head]
    # MPI 0-indexed indices for this subset (from mpii_get_joint_set 'relevant'):
    h36m_to_mpi = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

    annot_for_cam = annot3[use_cam_id, 0]  # (n_frames, 84)
    n_frames = annot_for_cam.shape[0]

    def loader(frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0 or frame_idx >= n_frames:
            return None
        row = annot_for_cam[frame_idx].reshape(-1, 3)  # (28, 3) in cam coords
        subset_cam = row[h36m_to_mpi]                   # (17, 3)
        # Convert to world: X_world = R^T @ (X_cam - t)
        return (R.T @ (subset_cam.T - t.reshape(3, 1))).T  # (17, 3)

    return loader
