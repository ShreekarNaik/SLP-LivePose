"""Pose tracker: pluggable backend protocol + YOLO and MediaPipe implementations.

All backends output COCO-17 keypoints so the rest of the pipeline doesn't
care which backend is active.  Backends are hot-swappable via PoseTracker.swap_backend().

Backends:
  - YoloBackend   (default) — batched inference, fast on CPU with imgsz=320
  - MediaPipeBackend        — BlazePose, single-image loop, no batch
  - RTMPoseBackend          — rtmlib/ONNX, lazy import (not installed by default)

Each backend returns list[list[Pose2D]] — a list of detected persons per image,
sorted by descending mean confidence.  The pipeline selects the best person
per camera using temporal consistency before triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Core data type
# ---------------------------------------------------------------------------

@dataclass
class Pose2D:
    keypoints: np.ndarray  # (J, 2) xy in pixels
    confidences: np.ndarray  # (J,)
    detected: bool


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PoseBackend(Protocol):
    """Contract for all pose estimation backends. Output is always COCO-17."""

    name: str            # "yolo" | "mediapipe" | "rtmpose"
    num_joints: int      # always 17
    supports_batch: bool

    def process_batch(self, images: list[np.ndarray], imgsz: int = 320) -> list[list[Pose2D]]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# YOLO backend
# ---------------------------------------------------------------------------

_YOLO_VARIANTS: dict[str, str] = {
    "nano":  "yolov8n-pose.pt",
    "small": "yolov8s-pose.pt",
}


class YoloBackend:
    name = "yolo"
    num_joints = 17
    supports_batch = True

    def __init__(self, variant: str = "nano") -> None:
        from ultralytics import YOLO  # always installed
        import torch
        model_name = _YOLO_VARIANTS.get(variant, "yolov8n-pose.pt")
        self._model = YOLO(model_name)
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

    def process_batch(self, images: list[np.ndarray], imgsz: int = 320) -> list[list[Pose2D]]:
        if not images:
            return []
        results = self._model.predict(images, verbose=False, imgsz=imgsz, device=self._device)
        return [_yolo_result_to_all_poses(r) for r in results]

    def close(self) -> None:
        pass  # ultralytics handles its own cleanup


def _yolo_result_to_all_poses(result, num_joints: int = 17) -> list[Pose2D]:
    """Convert one ultralytics Result to a list of Pose2D (all persons, sorted by conf)."""
    if not result.keypoints or result.keypoints.xy.shape[0] == 0:
        return []
    xy = result.keypoints.xy.cpu().numpy()    # (N, J, 2)
    conf = result.keypoints.conf.cpu().numpy()  # (N, J)
    # Sort by mean confidence descending
    order = np.argsort(-conf.mean(axis=1))
    poses = []
    for idx in order:
        poses.append(Pose2D(
            keypoints=xy[idx].astype(np.float32),
            confidences=conf[idx].astype(np.float32),
            detected=True,
        ))
    return poses


# ---------------------------------------------------------------------------
# MediaPipe backend
# ---------------------------------------------------------------------------

# Mapping: (mediapipe_landmark_idx, coco17_joint_idx)
# Same landmark indices in both old BlazePose and new Tasks PoseLandmarker.
_MP_TO_COCO17: list[tuple[int, int]] = [
    (0,  0),   # nose
    (2,  1),   # L_eye
    (5,  2),   # R_eye
    (7,  3),   # L_ear
    (8,  4),   # R_ear
    (11, 5),   # L_shoulder
    (12, 6),   # R_shoulder
    (13, 7),   # L_elbow
    (14, 8),   # R_elbow
    (15, 9),   # L_wrist
    (16, 10),  # R_wrist
    (23, 11),  # L_hip
    (24, 12),  # R_hip
    (25, 13),  # L_knee
    (26, 14),  # R_knee
    (27, 15),  # L_ankle
    (28, 16),  # R_ankle
]

_MP_VARIANTS: list[str] = ["lite", "full", "heavy"]

_MP_MODEL_URLS: dict[str, str] = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

_MP_MODEL_DIR = Path.home() / ".livepose" / "models"


def _get_mp_model_path(variant: str) -> Path:
    """Return local path for the model, downloading if not already cached."""
    _MP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MP_MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        import urllib.request
        url = _MP_MODEL_URLS.get(variant, _MP_MODEL_URLS["full"])
        print(f"[MediaPipe] Downloading model ({variant}): {url}")
        urllib.request.urlretrieve(url, path)
        print(f"[MediaPipe] Saved to {path}")
    return path


class MediaPipeBackend:
    """MediaPipe PoseLandmarker backend (Tasks API, mediapipe ≥ 0.10)."""

    max_poses = 4
    name = "mediapipe"
    num_joints = 17
    supports_batch = False

    def __init__(self, variant: str = "full") -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker,
                PoseLandmarkerOptions,
                RunningMode,
            )
        except ImportError:
            raise ImportError(
                "mediapipe is not installed. "
                "Run: uv add mediapipe"
            )

        if variant not in _MP_MODEL_URLS:
            variant = "full"
        model_path = _get_mp_model_path(variant)

        import platform
        import torch

        delegate = BaseOptions.Delegate.CPU
        if torch.backends.mps.is_available() or torch.cuda.is_available() or (platform.system() == "Darwin" and platform.machine() == "arm64"):
            delegate = BaseOptions.Delegate.GPU

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=delegate
            ),
            running_mode=RunningMode.IMAGE,
            num_poses=self.max_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    def process_batch(self, images: list[np.ndarray], imgsz: int = 320) -> list[list[Pose2D]]:
        results = []
        for img in images:
            results.append(self._process_one(img))
        return results

    def _process_one(self, bgr: np.ndarray) -> list[Pose2D]:
        import cv2
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGBA, data=rgba
        )
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return []

        h, w = bgr.shape[:2]
        poses: list[Pose2D] = []
        for lms in result.pose_landmarks:
            kp = np.zeros((17, 2), dtype=np.float32)
            conf = np.zeros(17, dtype=np.float32)

            for mp_idx, coco_idx in _MP_TO_COCO17:
                lm = lms[mp_idx]
                kp[coco_idx] = [lm.x * w, lm.y * h]
                conf[coco_idx] = float(lm.visibility)

            poses.append(Pose2D(keypoints=kp, confidences=conf, detected=True))

        poses.sort(key=lambda pose: float(pose.confidences.mean()), reverse=True)
        return poses

    def close(self) -> None:
        self._landmarker.close()


# ---------------------------------------------------------------------------
# RTMPose backend (optional — requires rtmlib)
# ---------------------------------------------------------------------------

_RTMPOSE_VARIANTS: list[str] = ["body"]  # extend as rtmlib supports more


class RTMPoseBackend:
    name = "rtmpose"
    num_joints = 17
    supports_batch = False

    def __init__(self, variant: str = "body") -> None:
        try:
            from rtmlib import Body  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "rtmlib is not installed. "
                "Run: pip install 'livepose[rtmpose]' or uv add rtmlib"
            )
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self._body = Body(mode="balanced", to_openpose=False, device=device)

    def process_batch(self, images: list[np.ndarray], imgsz: int = 320) -> list[list[Pose2D]]:
        results = []
        for img in images:
            results.append(self._process_one(img))
        return results

    def _process_one(self, bgr: np.ndarray) -> list[Pose2D]:
        keypoints, scores = self._body(bgr)  # type: ignore[misc]
        if keypoints is None or len(keypoints) == 0:
            return []
        order = np.argsort(-np.asarray(scores).mean(axis=1))
        poses = []
        for idx in order:
            kp = np.array(keypoints[idx], dtype=np.float32)[:17]
            cf = np.array(scores[idx], dtype=np.float32)[:17]
            poses.append(Pose2D(keypoints=kp, confidences=cf, detected=True))
        return poses

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Registry of known backends (for availability reporting)
# ---------------------------------------------------------------------------

def _mediapipe_installed() -> bool:
    try:
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False


def _rtmlib_installed() -> bool:
    try:
        import rtmlib  # noqa: F401
        return True
    except ImportError:
        return False


def get_backend_registry() -> list[dict]:
    """Return backend list with live install checks (safe to call repeatedly)."""
    return [
        {
            "name": "yolo",
            "variants": list(_YOLO_VARIANTS.keys()),
            "supports_batch": True,
            "installed": True,  # always installed (ultralytics is a hard dep)
        },
        {
            "name": "mediapipe",
            "variants": _MP_VARIANTS,
            "supports_batch": False,
            "installed": _mediapipe_installed(),
        },
        {
            "name": "rtmpose",
            "variants": _RTMPOSE_VARIANTS,
            "supports_batch": False,
            "installed": _rtmlib_installed(),
        },
    ]


# Kept for backwards-compat imports; callers should prefer get_backend_registry().
BACKEND_REGISTRY: list[dict] = get_backend_registry()


def _make_backend(name: str, variant: str) -> PoseBackend:
    if name == "yolo":
        return YoloBackend(variant=variant)
    elif name == "mediapipe":
        return MediaPipeBackend(variant=variant)
    elif name == "rtmpose":
        return RTMPoseBackend(variant=variant)
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose from: yolo, mediapipe, rtmpose")


# ---------------------------------------------------------------------------
# PoseTracker — hot-swap proxy
# ---------------------------------------------------------------------------

class PoseTracker:
    """Thin proxy that holds an active backend and supports hot-swapping."""

    def __init__(self, model_size: str = "nano") -> None:
        self._backend: PoseBackend = YoloBackend(variant=model_size)

    @property
    def num_joints(self) -> int:
        return self._backend.num_joints

    @property
    def active_backend_name(self) -> str:
        return self._backend.name

    def swap_backend(self, name: str, variant: str = "nano") -> None:
        """Replace the active backend. Closes old one first."""
        old = self._backend
        self._backend = _make_backend(name, variant)
        old.close()

    def process_batch(self, images: list[np.ndarray], imgsz: int = 320) -> list[list[Pose2D]]:
        """Batched inference — returns list of detections per image."""
        if not images:
            return []
        return self._backend.process_batch(images, imgsz=imgsz)
