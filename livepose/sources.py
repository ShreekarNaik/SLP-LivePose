"""Frame sources: live cameras, video files, or anything that yields synced frames.

Design: a `FrameSource` is anything that yields `MultiFrame` (a dict of
camera_id -> Frame) at roughly the source's native FPS. Live USB cameras and
recorded videos use the same protocol so the rest of the pipeline doesn't care
which is which.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np


@dataclass
class Frame:
    cam_id: int
    image: np.ndarray  # BGR HxWx3
    timestamp: float   # seconds (perf_counter or video time)
    frame_index: int   # 0-based index within source


@dataclass
class MultiFrame:
    """A synchronized set of frames from N cameras at the same time-step."""

    frames: dict[int, Frame]
    sequence: int  # global step counter

    @property
    def cam_ids(self) -> list[int]:
        return sorted(self.frames.keys())


class FrameSource(Protocol):
    """Yields synchronized multi-camera frames."""

    fps: float
    cam_ids: list[int]

    def open(self) -> None: ...
    def read(self) -> MultiFrame | None:
        """Return next MultiFrame, or None when exhausted."""
        ...
    def close(self) -> None: ...
    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame (only meaningful for video sources)."""
        ...


@dataclass
class CameraSource:
    """Live USB / built-in cameras via OpenCV.

    Each device runs in its own thread; on `read()` we grab the latest frame
    from each. Synchronization is best-effort (driver-level), which is
    acceptable for non-fast motion. For frame-perfect sync, see skellycam's
    grab/retrieve protocol — out of scope for the lean MVP.
    """

    device_map: dict[int, int]  # cam_id -> OS device index
    fps: float = 30.0
    width: int = 1280
    height: int = 720

    _captures: dict[int, cv2.VideoCapture] = field(default_factory=dict)
    _latest: dict[int, Frame] = field(default_factory=dict)
    _threads: dict[int, threading.Thread] = field(default_factory=dict)
    _running: bool = False
    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def cam_ids(self) -> list[int]:
        return sorted(self.device_map.keys())

    def open(self) -> None:
        for cam_id, dev_idx in self.device_map.items():
            cap = cv2.VideoCapture(dev_idx)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera device {dev_idx} (cam_id={cam_id})")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._captures[cam_id] = cap
        self._running = True
        for cam_id in self._captures:
            t = threading.Thread(target=self._poll, args=(cam_id,), daemon=True)
            t.start()
            self._threads[cam_id] = t

    def _poll(self, cam_id: int) -> None:
        cap = self._captures[cam_id]
        idx = 0
        while self._running:
            ok, img = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            f = Frame(cam_id=cam_id, image=img, timestamp=time.perf_counter(), frame_index=idx)
            with self._lock:
                self._latest[cam_id] = f
            idx += 1

    def read(self) -> MultiFrame | None:
        # Wait until every camera has produced at least one frame
        while self._running and len(self._latest) < len(self.device_map):
            time.sleep(0.005)
        with self._lock:
            snapshot = dict(self._latest)
        if len(snapshot) < len(self.device_map):
            return None
        self._seq += 1
        return MultiFrame(frames=snapshot, sequence=self._seq)

    def close(self) -> None:
        self._running = False
        for t in self._threads.values():
            t.join(timeout=1.0)
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        self._latest.clear()
        self._threads.clear()

    def seek(self, frame_index: int) -> None:
        # No-op for live cameras
        pass


@dataclass
class VideoSource:
    """Plays back per-camera videos as if they were live, at native FPS.

    All videos must share frame count and FPS (or be close to it). We advance
    one frame across all videos per `read()` call, and pace ourselves to the
    target FPS unless `realtime` is False.
    """

    video_map: dict[int, Path]  # cam_id -> video file
    realtime: bool = True
    speed: float = 1.0
    loop: bool = False

    _captures: dict[int, cv2.VideoCapture] = field(default_factory=dict)
    _fps: float = 25.0
    _frame_idx: int = 0
    _total_frames: int = 0
    _last_emit: float = 0.0
    _seq: int = 0
    _paused: bool = False
    _pending_seek: int | None = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def cam_ids(self) -> list[int]:
        return sorted(self.video_map.keys())

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def current_frame(self) -> int:
        return self._frame_idx

    def open(self) -> None:
        for cam_id, path in self.video_map.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video {path} for cam_id={cam_id}")
            self._captures[cam_id] = cap
        # Take FPS and frame count from the first video
        first = next(iter(self._captures.values()))
        self._fps = float(first.get(cv2.CAP_PROP_FPS)) or 25.0
        self._total_frames = int(first.get(cv2.CAP_PROP_FRAME_COUNT))
        self._last_emit = time.perf_counter()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, min(speed, 8.0))

    def read(self) -> MultiFrame | None:
        if self._pending_seek is not None:
            self._frame_idx = max(0, min(self._pending_seek, self._total_frames - 1))
            for cap in self._captures.values():
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(self._frame_idx))
            self._last_emit = time.perf_counter()
            self._pending_seek = None

        if self._paused:
            time.sleep(0.02)
            return None

        if self._frame_idx >= self._total_frames:
            if self.loop:
                self.seek(0)
                # Next read will process the seek
                return None
            else:
                return None

        if self.realtime and self.speed > 0:
            target_dt = 1.0 / (self._fps * self.speed)
            elapsed = time.perf_counter() - self._last_emit
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
        self._last_emit = time.perf_counter()

        frames: dict[int, Frame] = {}
        for cam_id, cap in self._captures.items():
            ok, img = cap.read()
            if not ok:
                return None
            frames[cam_id] = Frame(
                cam_id=cam_id,
                image=img,
                timestamp=self._frame_idx / self._fps,
                frame_index=self._frame_idx,
            )

        self._seq += 1
        self._frame_idx += 1
        return MultiFrame(frames=frames, sequence=self._seq)

    def seek(self, frame_index: int) -> None:
        self._pending_seek = frame_index

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()


def list_available_cameras(max_devices: int = 8) -> list[int]:
    """Probe OS device indices and return ones that open successfully."""
    found = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found
