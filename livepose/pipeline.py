"""Async pipeline: pulls multi-frames, runs tracker per camera, triangulates,
broadcasts to the WebSocket bus.

Pipeline flow (per tick):
  1. Read frames from all cameras
  2. Pose backend inference → list[list[Pose2D]] per camera
  3. _select_detections():
       - Score by reprojection proximity (last-frame 3D projected to 2D)
       - Fallback: 2D centroid proximity to last frame
       - Fallback: highest confidence (cold start)
       - Max-distance gate: reject if no detection within threshold
  4. Epipolar consistency check → drop outlier cameras
  5. Triangulate (DLT) with remaining cameras
  6. Anthropometric bounds check → reject frame if implausible
  7. One-Euro smoothing
  8. Progressive bone model: observe smoothed skeleton
  9. XPBD bone-length constraint solving
  10. Broadcast
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from livepose.core import Camera, check_epipolar_consistency, project_points, triangulate_dlt
from livepose.filters import BoneModel, PoseFilter, check_anthropometric_bounds, xpbd_solve
from livepose.sources import FrameSource, MultiFrame, VideoSource
from livepose.tracker import Pose2D, PoseTracker


@dataclass
class ProcessingConfig:
    """Mutable runtime settings; read by the pipeline thread each tick."""
    smoothing_enabled: bool = True
    min_cutoff: float = 3.0      # One-Euro min cutoff (Hz) — higher = less lag at rest
    beta: float = 0.05           # One-Euro beta — higher = faster response during motion
    imgsz: int = 320             # Backend inference resolution (px); lower = faster
    conf_threshold: float = 0.3  # Keypoint confidence threshold for triangulation
    backend: str = "yolo"        # Active backend name
    backend_variant: str = "nano"  # Variant within the backend
    # Detection selection params
    detection_max_distance_px: float = 150.0  # max 2D distance for detection matching
    epipolar_threshold_px: float = 15.0       # epipolar consistency threshold
    # Bone IK params
    bone_ik_enabled: bool = True


@dataclass
class PipelineEvent:
    """One time-step of pipeline output."""

    sequence: int
    timestamp: float
    fps: float
    frame_index: int
    total_frames: int
    cam_ids: list[int]
    poses_2d: dict[int, Pose2D]   # per camera (selected detection)
    points_3d: np.ndarray         # (J, 3), NaN where invalid
    valid_3d: np.ndarray          # (J,) bool
    gt_3d: Optional[np.ndarray] = None  # (J, 3) or None
    jpeg_per_cam: dict[int, bytes] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


_EMA_ALPHA = 0.1  # smoothing factor for EMA timing

# Confidence threshold for computing detection centroid
_CENTROID_CONF_MIN = 0.3


def _detection_centroid(pose: Pose2D, conf_min: float = _CENTROID_CONF_MIN) -> np.ndarray | None:
    """Return (2,) centroid of high-confidence keypoints, or None if no such points."""
    mask = pose.confidences >= conf_min
    if not mask.any():
        return None
    return pose.keypoints[mask].mean(axis=0)


class Pipeline:
    """Owns the source, a single shared tracker, filter, and the broadcast bus."""

    def __init__(
        self,
        source: FrameSource,
        cameras: dict[int, Camera],
        ground_truth_loader: Optional[callable] = None,
        jpeg_quality: int = 70,
        max_jpeg_width: int = 640,
    ) -> None:
        self.source = source
        self.cameras = cameras
        self.ground_truth_loader = ground_truth_loader
        self.jpeg_quality = jpeg_quality
        self.max_jpeg_width = max_jpeg_width

        # Processing config — mutated from the API thread (GIL-safe for simple attrs)
        self.config = ProcessingConfig()

        # Single shared tracker — batched inference across all cameras
        self._tracker = PoseTracker(model_size=self.config.backend_variant)

        # One-Euro pose smoother
        self._pose_filter = PoseFilter(
            min_cutoff=self.config.min_cutoff,
            beta=self.config.beta,
        )

        # Bone model (progressive confidence-weighted; never freezes)
        self._bone_model = BoneModel()

        self._filter_reset_pending = False  # set True after seek to discard stale state

        # Per-camera tracking state for detection selection
        self._last_centroids: dict[int, np.ndarray | None] = {}  # cam_id -> (2,) px centroid
        self._last_3d: np.ndarray | None = None          # (17, 3) last valid triangulated output
        self._last_3d_valid: np.ndarray | None = None    # (17,) bool
        self._last_3d_frame: int = 0                     # frame index of last _last_3d update

        self._executor = ThreadPoolExecutor(max_workers=max(2, len(source.cam_ids)))
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: list[asyncio.Queue[PipelineEvent]] = []
        self._running = False
        self._fps_smoothed = 0.0
        # EMA per-stage timings (ms)
        self._timings_ema: dict[str, float] = {}
        # Camera selection: None = use all, otherwise only these for triangulation
        self.enabled_cam_ids: Optional[set[int]] = None

        # Frame-result cache: frame_index -> (poses_2d, pts_3d, valid_3d, reproj_errors)
        # Keyed by (imgsz, conf_threshold, backend, backend_variant) so config
        # changes automatically invalidate it.
        self._result_cache: dict[int, tuple[dict, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._cache_key: tuple = (
            self.config.imgsz,
            self.config.conf_threshold,
            self.config.backend,
            self.config.backend_variant,
        )

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running:
            return
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                import warnings
                warnings.warn("Pipeline thread did not stop within timeout; captures left open")
            else:
                self.source.close()
            self._thread = None

    def subscribe(self) -> asyncio.Queue[PipelineEvent]:
        q: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=4)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[PipelineEvent]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def reset_filter(self) -> None:
        """Schedule a filter state reset (call after seek)."""
        self._filter_reset_pending = True

    def _reset_all_filters(self) -> None:
        self._pose_filter.reset()
        self._bone_model.reset()
        self._last_centroids.clear()
        self._last_3d = None
        self._last_3d_valid = None
        self._last_3d_frame = 0

    def swap_backend(self, name: str, variant: str) -> None:
        """Hot-swap the pose backend. Resets filter state and cache."""
        self._tracker.swap_backend(name, variant)
        self._reset_all_filters()
        self._result_cache.clear()

    def _invalidate_cache_if_needed(self, cfg: "ProcessingConfig") -> None:
        """Clear the result cache if any cache-busting config field changed."""
        new_key = (cfg.imgsz, cfg.conf_threshold, cfg.backend, cfg.backend_variant)
        if new_key != self._cache_key:
            self._result_cache.clear()
            self._cache_key = new_key

    def _encode_jpeg(self, cam_id: int, image: np.ndarray) -> tuple[int, bytes]:
        h, w = image.shape[:2]
        if w > self.max_jpeg_width:
            scale = self.max_jpeg_width / w
            image = cv2.resize(image, (self.max_jpeg_width, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return cam_id, (buf.tobytes() if ok else b"")

    def _ema(self, key: str, val_ms: float) -> float:
        prev = self._timings_ema.get(key, val_ms)
        updated = prev * (1 - _EMA_ALPHA) + val_ms * _EMA_ALPHA
        self._timings_ema[key] = updated
        return updated

    def _select_detections(
        self,
        cam_ids: list[int],
        all_detections: dict[int, list[Pose2D]],
        cfg: ProcessingConfig,
    ) -> dict[int, Pose2D]:
        """Pick one Pose2D per camera using temporal consistency.

        Priority order:
          1. Reprojection of last-frame 3D → score by 2D distance (most accurate)
          2. Last-frame 2D centroid per camera → score by centroid distance
          3. Cold start → pick highest mean confidence

        If no detection is within cfg.detection_max_distance_px, returns
        Pose2D(detected=False) — treat as occluded rather than jump to wrong person.
        """
        max_dist = cfg.detection_max_distance_px
        selected: dict[int, Pose2D] = {}
        _no_detect = Pose2D(
            keypoints=np.zeros((17, 2), dtype=np.float32),
            confidences=np.zeros(17, dtype=np.float32),
            detected=False,
        )

        for cam_id in cam_ids:
            detections = all_detections.get(cam_id, [])
            if not detections:
                selected[cam_id] = _no_detect
                self._last_centroids[cam_id] = None
                continue

            cam = self.cameras.get(cam_id)

            # --- Compute prior centroid for scoring ---
            prior_centroid: np.ndarray | None = None

            if self._last_3d is not None and self._last_3d_valid is not None and cam is not None:
                # Use reprojected 3D as prior
                valid_mask = self._last_3d_valid
                if valid_mask.any():
                    proj = project_points(cam, self._last_3d)  # (17, 2)
                    proj_valid = proj[valid_mask]
                    prior_centroid = proj_valid.mean(axis=0)

            if prior_centroid is None:
                prior_centroid = self._last_centroids.get(cam_id)

            # --- Score detections ---
            if prior_centroid is not None:
                best_pose = None
                best_score = float("inf")
                for pose in detections:
                    c = _detection_centroid(pose)
                    if c is None:
                        continue
                    dist = float(np.linalg.norm(c - prior_centroid))
                    # Tiebreak: small confidence bonus
                    score = dist - 0.1 * float(pose.confidences.mean()) * max_dist
                    if score < best_score:
                        best_score = score
                        best_pose = pose

                if best_pose is None or best_score > max_dist:
                    # No detection within threshold — treat as occluded
                    selected[cam_id] = _no_detect
                    # Keep last centroid (don't update — person is occluded)
                    continue
                selected[cam_id] = best_pose
            else:
                # Cold start: pick highest mean confidence
                selected[cam_id] = detections[0]  # already sorted by conf desc

            # Update last centroid for this camera
            c = _detection_centroid(selected[cam_id])
            self._last_centroids[cam_id] = c

        return selected

    def _run(self) -> None:
        import traceback
        self.source.open()
        last_t = time.perf_counter()

        try:
            while self._running:
                try:
                    t0 = time.perf_counter()

                    # --- Read frames -----------------------------------------
                    mf = self.source.read()
                    t_read = time.perf_counter()

                    if mf is None:
                        time.sleep(0.05)
                        continue

                    # --- Apply pending filter reset (after seek) -------------
                    if self._filter_reset_pending:
                        self._reset_all_filters()
                        self._filter_reset_pending = False

                    # --- Snapshot config (read once per tick) ----------------
                    cfg = self.config
                    self._invalidate_cache_if_needed(cfg)

                    # --- Determine frame index for caching ------------------
                    frame_idx = next(iter(mf.frames.values())).frame_index

                    # --- Staleness: clear _last_3d if prior is too old ------
                    # 90 frames ≈ 3 s at 30 fps — long enough to survive brief
                    # occlusion without losing the tracked person.
                    if (self._last_3d is not None
                            and frame_idx - self._last_3d_frame > 90):
                        self._last_3d = None
                        self._last_3d_valid = None

                    cached = self._result_cache.get(frame_idx)

                    if cached is not None:
                        # Cache hit — skip inference + triangulation entirely
                        poses_2d, pts_3d, valid_3d, reproj_errors = cached
                        t_infer = time.perf_counter()
                        t_tri = t_infer
                    else:
                        # --- Batched pose inference --------------------------
                        cam_ids_ordered = list(mf.frames.keys())
                        images = [mf.frames[cid].image for cid in cam_ids_ordered]
                        all_detections_list = self._tracker.process_batch(
                            images, imgsz=cfg.imgsz
                        )
                        all_detections: dict[int, list[Pose2D]] = dict(
                            zip(cam_ids_ordered, all_detections_list)
                        )
                        t_infer = time.perf_counter()

                        # --- Select one detection per camera ----------------
                        enabled = self.enabled_cam_ids
                        cam_ids_with_pose = [
                            cid for cid in mf.cam_ids
                            if cid in self.cameras
                            and (enabled is None or cid in enabled)
                        ]
                        poses_2d = self._select_detections(
                            cam_ids_with_pose, all_detections, cfg
                        )

                        # Cold-start fallback: only when there is NO 3D prior
                        # (initial startup or after staleness timeout cleared it).
                        # Do NOT trigger just because cameras missed one frame —
                        # that would pick random people per camera and corrupt tracking.
                        if (self._last_3d is None
                                and not any(poses_2d[cid].detected
                                            for cid in cam_ids_with_pose)):
                            saved_cents = dict(self._last_centroids)
                            self._last_centroids.clear()
                            poses_2d = self._select_detections(
                                cam_ids_with_pose, all_detections, cfg
                            )
                            if not any(poses_2d[cid].detected for cid in cam_ids_with_pose):
                                self._last_centroids = saved_cents

                        # --- Epipolar consistency filter --------------------
                        cams_list      = [self.cameras[cid] for cid in cam_ids_with_pose]
                        kpts_list      = [poses_2d[cid].keypoints for cid in cam_ids_with_pose]
                        confs_list     = [poses_2d[cid].confidences for cid in cam_ids_with_pose]
                        detected_flags = [poses_2d[cid].detected for cid in cam_ids_with_pose]

                        detected_indices = [i for i, d in enumerate(detected_flags) if d]
                        if len(detected_indices) >= 2:
                            consistent_local = check_epipolar_consistency(
                                [cams_list[i] for i in detected_indices],
                                [kpts_list[i] for i in detected_indices],
                                threshold_px=cfg.epipolar_threshold_px,
                            )
                            consistent_global = [detected_indices[i] for i in consistent_local]
                            tri_cams  = [cams_list[i] for i in consistent_global]
                            tri_kpts  = [kpts_list[i] for i in consistent_global]
                            tri_confs = [confs_list[i] for i in consistent_global]
                        else:
                            tri_cams  = [cams_list[i] for i in detected_indices]
                            tri_kpts  = [kpts_list[i] for i in detected_indices]
                            tri_confs = [confs_list[i] for i in detected_indices]

                        # --- Triangulate ------------------------------------
                        if len(tri_cams) >= 2:
                            pts_3d, valid_3d, reproj_errors = triangulate_dlt(
                                tri_cams, tri_kpts, tri_confs,
                                min_views=2, conf_threshold=cfg.conf_threshold,
                            )
                        else:
                            pts_3d        = np.full((17, 3), np.nan)
                            valid_3d      = np.zeros(17, dtype=bool)
                            reproj_errors = np.zeros(17, dtype=np.float64)
                        t_tri = time.perf_counter()

                        # --- Anthropometric bounds check --------------------
                        if valid_3d.any():
                            score = check_anthropometric_bounds(pts_3d, valid_3d)
                            if score < 0.5 and self._last_3d is not None:
                                pts_3d   = self._last_3d.copy()
                                valid_3d = self._last_3d_valid.copy()

                        # Store in cache (video sources only)
                        if isinstance(self.source, VideoSource):
                            self._result_cache[frame_idx] = (
                                poses_2d,
                                pts_3d.copy(),
                                valid_3d.copy(),
                                reproj_errors.copy(),
                            )

                    # --- Temporal smoothing (One-Euro) -----------------------
                    if cfg.smoothing_enabled and valid_3d.any():
                        self._pose_filter.update_params(cfg.min_cutoff, cfg.beta)
                        pts_3d, valid_3d = self._pose_filter.filter(
                            time.time(), pts_3d, valid_3d
                        )

                    # --- Bone-model update + XPBD IK -------------------------
                    if valid_3d.any():
                        self._bone_model.observe(pts_3d, valid_3d, reproj_errors)
                        if cfg.bone_ik_enabled:
                            pts_3d = xpbd_solve(pts_3d, valid_3d, self._bone_model)

                    # --- Update last-frame 3D for next tick ------------------
                    if valid_3d.any():
                        self._last_3d       = pts_3d.copy()
                        self._last_3d_valid = valid_3d.copy()
                        self._last_3d_frame = frame_idx

                    t_smooth = time.perf_counter()

                    # --- JPEG encode in parallel -----------------------------
                    jpeg_futures = [
                        self._executor.submit(self._encode_jpeg, cid, fr.image)
                        for cid, fr in mf.frames.items()
                    ]
                    jpeg_per_cam: dict[int, bytes] = {}
                    for fut in jpeg_futures:
                        cid, b = fut.result()
                        if b:
                            jpeg_per_cam[cid] = b
                    t_encode = time.perf_counter()

                    # --- Ground-truth lookup ---------------------------------
                    gt_3d = None
                    if self.ground_truth_loader is not None:
                        first_frame_idx = next(iter(mf.frames.values())).frame_index
                        gt_3d = self.ground_truth_loader(first_frame_idx)

                    # --- FPS (EMA) -------------------------------------------
                    now = time.perf_counter()
                    dt = now - last_t
                    last_t = now
                    inst_fps = 1.0 / dt if dt > 0 else 0.0
                    self._fps_smoothed = (
                        inst_fps if self._fps_smoothed == 0
                        else 0.9 * self._fps_smoothed + 0.1 * inst_fps
                    )

                    # --- Per-stage EMA timings (ms) --------------------------
                    stage_timings_ms = {
                        "read":        self._ema("read",        (t_read - t0) * 1000),
                        "infer":       self._ema("infer",       (t_infer - t_read) * 1000),
                        "triangulate": self._ema("triangulate", (t_tri - t_infer) * 1000),
                        "smooth":      self._ema("smooth",      (t_smooth - t_tri) * 1000),
                        "encode":      self._ema("encode",      (t_encode - t_smooth) * 1000),
                        "total":       self._ema("total",       (now - t0) * 1000),
                    }

                    total_frames = (
                        self.source.total_frames
                        if isinstance(self.source, VideoSource) else 0
                    )
                    current_idx = (
                        self.source.current_frame
                        if isinstance(self.source, VideoSource) else mf.sequence
                    )

                    evt = PipelineEvent(
                        sequence=mf.sequence,
                        timestamp=time.time(),
                        fps=self._fps_smoothed,
                        frame_index=current_idx,
                        total_frames=total_frames,
                        cam_ids=mf.cam_ids,
                        poses_2d=poses_2d,
                        points_3d=pts_3d,
                        valid_3d=valid_3d,
                        gt_3d=gt_3d,
                        jpeg_per_cam=jpeg_per_cam,
                        stage_timings_ms=stage_timings_ms,
                    )

                    # --- Broadcast (drop oldest if backpressured) ------------
                    for q in list(self._subscribers):
                        if self._loop is None:
                            break
                        if q.full():
                            try:
                                self._loop.call_soon_threadsafe(q.get_nowait)
                            except Exception:
                                pass
                        try:
                            self._loop.call_soon_threadsafe(q.put_nowait, evt)
                        except RuntimeError:
                            pass  # event loop closed

                except Exception:
                    traceback.print_exc()
                    time.sleep(0.1)
                    continue

        except Exception:
            traceback.print_exc()
