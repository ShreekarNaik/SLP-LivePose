"""Charuco-based multi-camera calibration.

Workflow:
    1. CalibrationCollector opens N cameras, accumulates board observations per
       camera (intrinsic samples) and pairs of simultaneous-detection samples
       (extrinsic samples).
    2. After enough samples, `compute_calibration` calls cv2.aruco's intrinsic
       calibration per camera, then derives relative poses from any frame where
       camera-k saw the board at the same time as the reference camera.
    3. Result is saved as JSON, loadable later as `Camera` objects compatible
       with the rest of the pipeline.

Coordinate conventions: world frame is the reference camera's frame after
calibration (camera 0 is at origin with identity rotation). Translations in
millimeters to match MPI dataset units.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from livepose.core import Camera


# ---- Board spec -------------------------------------------------------------

@dataclass(frozen=True)
class CharucoBoardSpec:
    squares_x: int = 5
    squares_y: int = 3
    square_length_mm: float = 80.0
    marker_length_mm: float = 60.0
    dictionary_id: int = cv2.aruco.DICT_4X4_50

    @property
    def square_length_m(self) -> float:
        return self.square_length_mm / 1000.0

    @property
    def marker_length_m(self) -> float:
        return self.marker_length_mm / 1000.0


DEFAULT_BOARD = CharucoBoardSpec()


def make_board(spec: CharucoBoardSpec) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(spec.dictionary_id)
    # Use meters for board sizing — keeps OpenCV happy and translations are scaled to mm later.
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        aruco_dict,
    )
    return board, aruco_dict


# ---- Detection --------------------------------------------------------------

def detect_charuco(
    image_bgr: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    aruco_dict: cv2.aruco.Dictionary,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect charuco corners. Returns (corners (N,1,2), ids (N,1)) or (None, None)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
        return None, None
    return charuco_corners, charuco_ids


def estimate_board_pose(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    K: np.ndarray,
    dist: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Estimate the board's pose in camera frame. Returns (R, t) in meters or None."""
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < 4:
        return None
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not success:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


# ---- Per-camera intrinsic calibration --------------------------------------

@dataclass
class IntrinsicResult:
    K: np.ndarray
    dist: np.ndarray
    rms_reproj_error: float
    n_views: int


def calibrate_intrinsics(
    samples: list[tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int],
) -> IntrinsicResult:
    """samples: list of (corners, ids). image_size: (width, height)."""
    valid = [(c, i) for c, i in samples if c is not None and i is not None and len(i) >= 4]
    if len(valid) < 5:
        raise ValueError(f"Need ≥5 valid charuco views, got {len(valid)}")
    corners = [c for c, _ in valid]
    ids = [i for _, i in valid]
    rms, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
        corners, ids, board, image_size, None, None
    )
    return IntrinsicResult(K=K, dist=dist.reshape(-1), rms_reproj_error=float(rms), n_views=len(valid))


# ---- Relative pose averaging ------------------------------------------------

def average_relative_pose(
    ref_poses: list[tuple[np.ndarray, np.ndarray]],
    other_poses: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Given N synchronized (R,t) board pose pairs, average the relative transform.

    For each pair: T_other = T_rel @ T_ref, so T_rel = T_other @ T_ref^-1.
    Translations are averaged arithmetically. Rotations are averaged via
    quaternions (hemisphere-aligned to the first sample).
    """
    if not ref_poses or len(ref_poses) != len(other_poses):
        raise ValueError("Need equal-length non-empty pose lists")

    rels: list[tuple[np.ndarray, np.ndarray]] = []
    for (R_ref, t_ref), (R_other, t_other) in zip(ref_poses, other_poses):
        R_rel = R_other @ R_ref.T
        t_rel = t_other - R_rel @ t_ref
        rels.append((R_rel, t_rel))

    quats = np.stack([ScipyRotation.from_matrix(r).as_quat() for r, _ in rels])
    # Align hemispheres to the first quaternion to avoid cancellation
    flips = np.sign(quats @ quats[0]) ; flips[flips == 0] = 1.0
    quats *= flips[:, None]
    avg_q = quats.mean(axis=0)
    avg_q /= np.linalg.norm(avg_q)
    R_avg = ScipyRotation.from_quat(avg_q).as_matrix()
    t_avg = np.mean(np.stack([t for _, t in rels]), axis=0)
    return R_avg, t_avg


# ---- Live calibration collection -------------------------------------------

@dataclass
class CamCollectorState:
    intrinsic_samples: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    last_corners: Optional[np.ndarray] = None
    last_ids: Optional[np.ndarray] = None
    image_size: Optional[tuple[int, int]] = None


@dataclass
class PairSample:
    """One frame where two cameras saw the board simultaneously."""
    cam_a: int
    cam_b: int
    corners_a: np.ndarray
    ids_a: np.ndarray
    corners_b: np.ndarray
    ids_b: np.ndarray


@dataclass
class CalibrationProgress:
    per_cam_samples: dict[int, int]            # cam_id -> intrinsic sample count
    per_pair_shared: dict[tuple[int, int], int] # (ref, other) -> shared frames count
    last_corners_per_cam: dict[int, int]       # cam_id -> # corners detected on latest frame
    image_sizes: dict[int, tuple[int, int]]    # cam_id -> (w, h)
    ready: bool                                # whether compute is allowed


class CalibrationCollector:
    """Owns N cameras, runs detection in a thread, exposes progress + last detection."""

    MIN_INTRINSIC_VIEWS = 15
    MIN_SHARED_VIEWS = 8
    MIN_SECONDS_BETWEEN_SAMPLES = 0.4  # avoid trivially similar consecutive frames

    def __init__(self, source, board_spec: CharucoBoardSpec = DEFAULT_BOARD) -> None:
        from livepose.sources import CameraSource  # local import to avoid cycle
        if not isinstance(source, CameraSource):
            raise TypeError("CalibrationCollector requires a CameraSource")
        self.source = source
        self.spec = board_spec
        self.board, self.aruco_dict = make_board(board_spec)
        self.cam_states: dict[int, CamCollectorState] = {
            cid: CamCollectorState() for cid in source.cam_ids
        }
        self.pair_samples: list[PairSample] = []
        self.last_jpeg_per_cam: dict[int, bytes] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_sample_time: dict[int, float] = {cid: 0.0 for cid in source.cam_ids}

    def start(self) -> None:
        if self._running:
            return
        self.source.open()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.source.close()

    def _run(self) -> None:
        while self._running:
            mf = self.source.read()
            if mf is None:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            per_frame_detections: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            for cam_id, frame in mf.frames.items():
                corners, ids = detect_charuco(frame.image, self.board, self.aruco_dict)
                state = self.cam_states[cam_id]
                state.image_size = (frame.image.shape[1], frame.image.shape[0])

                # Save annotated thumbnail for live UI preview
                annotated = frame.image.copy()
                if corners is not None and ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(annotated, corners, ids, (0, 255, 213))
                # Downscale + JPEG encode
                h, w = annotated.shape[:2]
                if w > 480:
                    s = 480 / w
                    annotated = cv2.resize(annotated, (480, int(h * s)))
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with self._lock:
                        self.last_jpeg_per_cam[cam_id] = buf.tobytes()

                if corners is None:
                    continue

                state.last_corners = corners
                state.last_ids = ids

                if (now - self._last_sample_time[cam_id]) >= self.MIN_SECONDS_BETWEEN_SAMPLES:
                    state.intrinsic_samples.append((corners, ids))
                    self._last_sample_time[cam_id] = now
                per_frame_detections[cam_id] = (corners, ids)

            # If multiple cameras saw the board this frame, save shared-view samples
            seen_ids = sorted(per_frame_detections.keys())
            if len(seen_ids) >= 2:
                ref = seen_ids[0]
                for other in seen_ids[1:]:
                    ca, ia = per_frame_detections[ref]
                    cb, ib = per_frame_detections[other]
                    with self._lock:
                        self.pair_samples.append(
                            PairSample(cam_a=ref, cam_b=other,
                                       corners_a=ca, ids_a=ia,
                                       corners_b=cb, ids_b=ib)
                        )

    def progress(self) -> CalibrationProgress:
        with self._lock:
            per_cam = {cid: len(s.intrinsic_samples) for cid, s in self.cam_states.items()}
            pair_counts: dict[tuple[int, int], int] = {}
            for p in self.pair_samples:
                key = (p.cam_a, p.cam_b)
                pair_counts[key] = pair_counts.get(key, 0) + 1
            last_corners = {
                cid: (0 if s.last_corners is None else len(s.last_corners))
                for cid, s in self.cam_states.items()
            }
            sizes = {cid: s.image_size or (0, 0) for cid, s in self.cam_states.items()}

            ready = (
                all(n >= self.MIN_INTRINSIC_VIEWS for n in per_cam.values())
                and (
                    len(self.cam_states) == 1
                    or all(pair_counts.get(k, 0) >= self.MIN_SHARED_VIEWS
                           for k in [(min(self.cam_states), c)
                                     for c in self.cam_states if c != min(self.cam_states)])
                )
            )
            return CalibrationProgress(
                per_cam_samples=per_cam,
                per_pair_shared=pair_counts,
                last_corners_per_cam=last_corners,
                image_sizes=sizes,
                ready=ready,
            )

    # ---- Computation ----

    def compute(self) -> dict[int, Camera]:
        """Run intrinsic + extrinsic calibration. Returns Camera objects keyed by cam_id."""
        with self._lock:
            states = {cid: s for cid, s in self.cam_states.items()}
            pair_samples = list(self.pair_samples)

        # 1. Intrinsics per camera
        intrinsics: dict[int, IntrinsicResult] = {}
        for cam_id, state in states.items():
            if state.image_size is None:
                raise RuntimeError(f"No image size recorded for cam {cam_id}")
            intrinsics[cam_id] = calibrate_intrinsics(
                state.intrinsic_samples, self.board, state.image_size
            )

        # 2. For each non-reference camera, compute relative pose to reference
        #    using shared-view board pose pairs.
        ref = min(states.keys())
        cameras: dict[int, Camera] = {}
        ref_intrinsic = intrinsics[ref]
        cameras[ref] = _intrinsic_to_camera(
            cam_id=ref, intr=ref_intrinsic, R=np.eye(3), t_meters=np.zeros(3),
            image_size=states[ref].image_size,
        )

        for cam_id in sorted(states.keys()):
            if cam_id == ref:
                continue
            other_intrinsic = intrinsics[cam_id]
            ref_poses: list[tuple[np.ndarray, np.ndarray]] = []
            other_poses: list[tuple[np.ndarray, np.ndarray]] = []
            for ps in pair_samples:
                if ps.cam_a != ref or ps.cam_b != cam_id:
                    continue
                pose_ref = estimate_board_pose(
                    ps.corners_a, ps.ids_a, self.board,
                    ref_intrinsic.K, ref_intrinsic.dist,
                )
                pose_other = estimate_board_pose(
                    ps.corners_b, ps.ids_b, self.board,
                    other_intrinsic.K, other_intrinsic.dist,
                )
                if pose_ref is None or pose_other is None:
                    continue
                ref_poses.append(pose_ref)
                other_poses.append(pose_other)

            if len(ref_poses) < 3:
                raise RuntimeError(
                    f"Cam {cam_id} has only {len(ref_poses)} valid shared-view pose pairs"
                )

            R_rel, t_rel_m = average_relative_pose(ref_poses, other_poses)
            cameras[cam_id] = _intrinsic_to_camera(
                cam_id=cam_id, intr=other_intrinsic, R=R_rel, t_meters=t_rel_m,
                image_size=states[cam_id].image_size,
            )

        return cameras


def _intrinsic_to_camera(
    cam_id: int,
    intr: IntrinsicResult,
    R: np.ndarray,
    t_meters: np.ndarray,
    image_size: tuple[int, int],
) -> Camera:
    """Wrap into our Camera type. We convert translation from meters to mm to match MPI scale."""
    return Camera(
        cam_id=cam_id,
        K=intr.K.copy(),
        R=R.copy(),
        t=(t_meters * 1000.0).copy(),
        width=image_size[0],
        height=image_size[1],
    )


# ---- Save / Load ------------------------------------------------------------

def save_calibration(
    path: str | Path,
    cameras: dict[int, dict | Camera],
    board_spec: CharucoBoardSpec = DEFAULT_BOARD,
) -> None:
    """Persist calibration to JSON. Accepts Camera objects or plain dicts."""
    out: dict = {
        "board": {
            "squares_x": board_spec.squares_x,
            "squares_y": board_spec.squares_y,
            "square_length_mm": board_spec.square_length_mm,
            "marker_length_mm": board_spec.marker_length_mm,
            "dictionary_id": board_spec.dictionary_id,
        },
        "cameras": {},
    }
    for cid, cam in cameras.items():
        if isinstance(cam, Camera):
            cam_dict = {
                "K": cam.K.tolist(),
                "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
                "R": cam.R.tolist(),
                "t": cam.t.tolist(),
                "width": cam.width,
                "height": cam.height,
            }
        else:
            cam_dict = cam
        out["cameras"][str(cid)] = cam_dict

    Path(path).write_text(json.dumps(out, indent=2))


def load_calibration(path: str | Path) -> tuple[dict[int, dict], CharucoBoardSpec]:
    raw = json.loads(Path(path).read_text())
    cams = {int(k): v for k, v in raw["cameras"].items()}
    b = raw["board"]
    spec = CharucoBoardSpec(
        squares_x=b["squares_x"],
        squares_y=b["squares_y"],
        square_length_mm=b["square_length_mm"],
        marker_length_mm=b["marker_length_mm"],
        dictionary_id=b["dictionary_id"],
    )
    return cams, spec


def calibration_dict_to_camera(cam_id: int, cam_dict: dict) -> Camera:
    return Camera(
        cam_id=cam_id,
        K=np.asarray(cam_dict["K"], dtype=float),
        R=np.asarray(cam_dict["R"], dtype=float),
        t=np.asarray(cam_dict["t"], dtype=float),
        width=int(cam_dict["width"]),
        height=int(cam_dict["height"]),
    )

def render_board_image(spec: CharucoBoardSpec, pixels_per_mm: float = 4.0) -> np.ndarray:
    """Generate a grayscale image of the board, sized for printing.
    Defaults to 4 px/mm; an A4 sheet @ 300dpi is ~3.94 px/mm. Print at 100% scale."""
    board, _ = make_board(spec)
    w_mm = spec.squares_x * spec.square_length_mm
    h_mm = spec.squares_y * spec.square_length_mm
    img = board.generateImage(
        (int(w_mm * pixels_per_mm), int(h_mm * pixels_per_mm)),
        marginSize=int(10 * pixels_per_mm),
        borderBits=1,
    )
    return img
