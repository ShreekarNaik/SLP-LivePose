"""FastAPI server: REST endpoints for import/control/export, WebSocket for streaming."""

from __future__ import annotations

import os
os.environ["OPENCV_FFMPEG_THREADS"] = "1"

import asyncio
import csv
import io
import json
import struct
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from livepose.core import (
    COCO17_JOINTS, COCO17_SKELETON,
    H36M17_JOINTS, H36M17_SKELETON,
    H36M_TO_COCO_SUBSET,
    Camera,
)
from livepose.tracker import BACKEND_REGISTRY
from livepose.importer import (
    DiscoveredDataset,
    Session,
    load_mpi_session,
    scan_for_datasets,
)
from livepose.pipeline import Pipeline, PipelineEvent
from livepose.sources import CameraSource, VideoSource, list_available_cameras


# ---- App state --------------------------------------------------------------

class AppState:
    pipeline: Optional[Pipeline] = None
    session: Optional[Session] = None
    discovered: dict[str, DiscoveredDataset] = {}  # path -> dataset
    export_buffer: list[PipelineEvent] = []        # rolling, capped
    export_capacity: int = 6000                     # ~4 min @ 25fps


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if state.pipeline is not None:
        state.pipeline.stop()


app = FastAPI(title="LivePose", lifespan=lifespan)


# ---- Pydantic schemas -------------------------------------------------------

class ScanRequest(BaseModel):
    path: str


class CameraInfoOut(BaseModel):
    cam_id: int
    K: list[list[float]]
    R: list[list[float]]
    t: list[float]
    width: int
    height: int


class SequencePreview(BaseModel):
    subject: str
    name: str
    display_name: str
    cam_ids: list[int]
    has_ground_truth: bool
    fps: float
    total_frames: int
    seq_path: str


class ScanResponse(BaseModel):
    kind: str
    root: str
    sequences: list[SequencePreview]


class ImportMpiRequest(BaseModel):
    seq_path: str
    include_ground_truth: bool = False


class SessionInfo(BaseModel):
    name: str
    fps: float
    total_frames: int
    cameras: list[CameraInfoOut]
    has_ground_truth: bool
    joint_names: list[str]
    skeleton: list[list[int]]


class StartLiveRequest(BaseModel):
    device_indices: list[int]
    width: int = 1280
    height: int = 720
    fps: int = 30


class EnabledCamerasRequest(BaseModel):
    enabled: list[int] | None = None  # null = all cameras


class SeekRequest(BaseModel):
    frame_index: int


class SpeedRequest(BaseModel):
    speed: float


class PauseRequest(BaseModel):
    paused: bool


class ProcessingRequest(BaseModel):
    smoothing_enabled: Optional[bool] = None
    min_cutoff: Optional[float] = None
    beta: Optional[float] = None
    imgsz: Optional[int] = None
    conf_threshold: Optional[float] = None
    backend: Optional[str] = None
    backend_variant: Optional[str] = None
    # Detection selection
    detection_max_distance_px: Optional[float] = None
    epipolar_threshold_px: Optional[float] = None
    # Bone IK
    bone_ik_enabled: Optional[bool] = None


# ---- Helpers ----------------------------------------------------------------

def _camera_to_out(cam: Camera) -> CameraInfoOut:
    return CameraInfoOut(
        cam_id=cam.cam_id,
        K=cam.K.tolist(),
        R=cam.R.tolist(),
        t=cam.t.tolist(),
        width=cam.width,
        height=cam.height,
    )


def _session_info(session: Session) -> SessionInfo:
    return SessionInfo(
        name=session.name,
        fps=session.fps,
        total_frames=session.total_frames,
        cameras=[_camera_to_out(c) for c in session.cameras.values()],
        has_ground_truth=session.ground_truth_loader is not None,
        joint_names=COCO17_JOINTS,
        skeleton=[list(e) for e in COCO17_SKELETON],
    )


def _stop_pipeline_if_running() -> None:
    if state.pipeline is not None:
        state.pipeline.stop()
        state.pipeline = None
        state.export_buffer.clear()


def _make_default_live_cameras(device_count: int) -> dict[int, Camera]:
    """Without calibration, create unit cameras spread on a circle.
    Triangulation won't be metric-accurate without real calibration, but the
    UI still has something to show. Real calibration would come from a charuco
    flow (out of scope for the MVP)."""
    cams: dict[int, Camera] = {}
    radius = 2000.0  # mm
    for i in range(device_count):
        angle = 2 * np.pi * i / max(1, device_count)
        cx, cy, cz = radius * np.cos(angle), 0.0, radius * np.sin(angle)
        # Look at origin
        forward = -np.array([cx, cy, cz])
        forward /= np.linalg.norm(forward)
        right = np.cross(np.array([0, 1, 0]), forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        R = np.stack([right, up, forward])
        t = -R @ np.array([cx, cy, cz])
        K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
        cams[i] = Camera(cam_id=i, K=K, R=R, t=t, width=1280, height=720)
    return cams


# ---- REST: discovery / import -----------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "session_loaded": state.session is not None}


@app.post("/api/scan", response_model=list[ScanResponse])
def scan(req: ScanRequest) -> list[ScanResponse]:
    p = Path(req.path).expanduser().resolve()
    if not p.exists():
        raise HTTPException(404, f"Path not found: {p}")
    discovered = scan_for_datasets(p)
    state.discovered.clear()
    out: list[ScanResponse] = []
    for ds in discovered:
        seqs: list[SequencePreview] = []
        for s in ds.sequences:
            # Cheap probe of fps/frames
            from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
            cap = VideoCapture(str(next(iter(s.video_map.values()))))
            fps = float(cap.get(CAP_PROP_FPS)) or 25.0
            n = int(cap.get(CAP_PROP_FRAME_COUNT))
            cap.release()
            seqs.append(SequencePreview(
                subject=s.subject,
                name=s.name,
                display_name=s.display_name,
                cam_ids=sorted(s.video_map.keys()),
                has_ground_truth=s.has_ground_truth,
                fps=fps,
                total_frames=n,
                seq_path=str(s.seq_path),
            ))
            state.discovered[str(s.seq_path)] = ds  # for later import
        out.append(ScanResponse(kind=ds.kind, root=str(ds.root), sequences=seqs))
    return out


@app.get("/api/scan/thumbnail/{cam_id}")
def thumbnail(cam_id: int):
    """Returns a JPEG thumbnail for the given cam in the currently scanned session.
    Currently uses the active session's preview frames (set by /import/preview)."""
    if state.session is None or cam_id not in state.session.sample_thumbnails:
        raise HTTPException(404, "No thumbnail available")
    return StreamingResponse(
        io.BytesIO(state.session.sample_thumbnails[cam_id]),
        media_type="image/jpeg",
    )


@app.post("/api/import/preview", response_model=SessionInfo)
def import_preview(req: ImportMpiRequest) -> SessionInfo:
    """Load thumbnails + camera info without starting the pipeline."""
    seq_path = req.seq_path
    ds = state.discovered.get(seq_path)
    if ds is None:
        raise HTTPException(400, "Sequence not in discovered set; call /api/scan first.")
    seq = next(s for s in ds.sequences if str(s.seq_path) == seq_path)
    state.session = load_mpi_session(seq, include_ground_truth=req.include_ground_truth)
    return _session_info(state.session)


@app.post("/api/import/start", response_model=SessionInfo)
async def import_start(req: ImportMpiRequest) -> SessionInfo:
    """Start the pipeline using the previously loaded (or now loaded) session."""
    info = import_preview(req)
    _stop_pipeline_if_running()
    assert state.session is not None
    src = VideoSource(video_map=state.session.video_map, realtime=True)
    state.pipeline = Pipeline(
        source=src,
        cameras=state.session.cameras,
        ground_truth_loader=state.session.ground_truth_loader,
    )
    state.pipeline.start(asyncio.get_running_loop())
    return info


# ---- REST: live cameras -----------------------------------------------------

@app.get("/api/live/devices")
def live_devices() -> list[int]:
    return list_available_cameras()


@app.post("/api/live/start", response_model=SessionInfo)
async def live_start(req: StartLiveRequest) -> SessionInfo:
    _stop_pipeline_if_running()
    device_map = {i: dev for i, dev in enumerate(req.device_indices)}

    # Prefer the most recently saved calibration if it covers the requested cams.
    cameras = _try_load_calibration(set(device_map.keys()))
    if cameras is None:
        cameras = _make_default_live_cameras(len(device_map))

    state.session = Session(
        name="Live",
        cameras=cameras,
        video_map={},
        fps=float(req.fps),
        total_frames=0,
        ground_truth_loader=None,
    )
    src = CameraSource(
        device_map=device_map,
        fps=float(req.fps),
        width=req.width,
        height=req.height,
    )
    state.pipeline = Pipeline(source=src, cameras=cameras, ground_truth_loader=None)
    state.pipeline.start(asyncio.get_running_loop())
    return _session_info(state.session)


# ---- REST: charuco calibration ---------------------------------------------

CALIBRATION_DIR = Path.home() / ".livepose" / "calibrations"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_CALIBRATION_PATH = CALIBRATION_DIR / "latest.json"


class StartCalibrationRequest(BaseModel):
    device_indices: list[int]
    width: int = 1280
    height: int = 720
    fps: int = 30
    squares_x: int = 5
    squares_y: int = 3
    square_length_mm: float = 80.0
    marker_length_mm: float = 60.0


class CalibrationProgressOut(BaseModel):
    per_cam_samples: dict[str, int]
    per_pair_shared: dict[str, int]
    last_corners_per_cam: dict[str, int]
    image_sizes: dict[str, list[int]]
    ready: bool
    cam_ids: list[int]


_calib_collector_state: dict = {"collector": None}


@app.post("/api/calibration/start")
def calibration_start(req: StartCalibrationRequest) -> dict:
    from livepose.calibration import (
        CalibrationCollector,
        CharucoBoardSpec,
    )
    _stop_pipeline_if_running()
    if _calib_collector_state["collector"] is not None:
        _calib_collector_state["collector"].stop()
        _calib_collector_state["collector"] = None

    device_map = {i: dev for i, dev in enumerate(req.device_indices)}
    src = CameraSource(
        device_map=device_map,
        fps=float(req.fps),
        width=req.width,
        height=req.height,
    )
    spec = CharucoBoardSpec(
        squares_x=req.squares_x,
        squares_y=req.squares_y,
        square_length_mm=req.square_length_mm,
        marker_length_mm=req.marker_length_mm,
    )
    collector = CalibrationCollector(source=src, board_spec=spec)
    collector.start()
    _calib_collector_state["collector"] = collector
    return {"ok": True, "cam_ids": list(device_map.keys())}


@app.get("/api/calibration/progress", response_model=CalibrationProgressOut)
def calibration_progress() -> CalibrationProgressOut:
    collector = _calib_collector_state["collector"]
    if collector is None:
        raise HTTPException(400, "No calibration in progress")
    p = collector.progress()
    return CalibrationProgressOut(
        per_cam_samples={str(k): v for k, v in p.per_cam_samples.items()},
        per_pair_shared={f"{a}_{b}": v for (a, b), v in p.per_pair_shared.items()},
        last_corners_per_cam={str(k): v for k, v in p.last_corners_per_cam.items()},
        image_sizes={str(k): list(v) for k, v in p.image_sizes.items()},
        ready=p.ready,
        cam_ids=sorted(p.per_cam_samples.keys()),
    )

@app.get("/api/calibration/board.png")
def calibration_board_png(
    squares_x: int = 5,
    squares_y: int = 3,
    square_length_mm: float = 80.0,
    marker_length_mm: float = 60.0,
):
    from livepose.calibration import CharucoBoardSpec, render_board_image
    spec = CharucoBoardSpec(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
    )
    img = render_board_image(spec)
    import cv2
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Failed to render board")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

@app.get("/api/calibration/preview/{cam_id}")
def calibration_preview(cam_id: int):
    collector = _calib_collector_state["collector"]
    if collector is None:
        raise HTTPException(400, "No calibration in progress")
    blob = collector.last_jpeg_per_cam.get(cam_id)
    if blob is None:
        raise HTTPException(404, "No frame yet for this cam")
    return StreamingResponse(io.BytesIO(blob), media_type="image/jpeg")


@app.post("/api/calibration/compute")
def calibration_compute() -> dict:
    from livepose.calibration import save_calibration
    collector = _calib_collector_state["collector"]
    if collector is None:
        raise HTTPException(400, "No calibration in progress")
    try:
        cameras = collector.compute()
    except Exception as e:
        raise HTTPException(400, f"Calibration failed: {e}")
    save_calibration(ACTIVE_CALIBRATION_PATH, cameras, board_spec=collector.spec)
    collector.stop()
    _calib_collector_state["collector"] = None
    return {
        "ok": True,
        "saved_to": str(ACTIVE_CALIBRATION_PATH),
        "cam_ids": sorted(cameras.keys()),
    }


@app.post("/api/calibration/cancel")
def calibration_cancel() -> dict:
    collector = _calib_collector_state["collector"]
    if collector is None:
        return {"ok": True}
    collector.stop()
    _calib_collector_state["collector"] = None
    return {"ok": True}


@app.get("/api/calibration/status")
def calibration_status() -> dict:
    """Whether a saved calibration exists, and which cam_ids it covers."""
    if not ACTIVE_CALIBRATION_PATH.exists():
        return {"exists": False, "cam_ids": []}
    from livepose.calibration import load_calibration
    cams, _ = load_calibration(ACTIVE_CALIBRATION_PATH)
    return {"exists": True, "cam_ids": sorted(cams.keys())}


def _try_load_calibration(requested_cam_ids: set[int]) -> Optional[dict[int, Camera]]:
    if not ACTIVE_CALIBRATION_PATH.exists():
        return None
    from livepose.calibration import calibration_dict_to_camera, load_calibration
    cams_dict, _ = load_calibration(ACTIVE_CALIBRATION_PATH)
    if not requested_cam_ids.issubset(set(cams_dict.keys())):
        return None
    return {cid: calibration_dict_to_camera(cid, cams_dict[cid]) for cid in requested_cam_ids}


# ---- REST: control ----------------------------------------------------------

@app.post("/api/control/cameras")
def control_cameras(req: EnabledCamerasRequest) -> dict:
    if state.pipeline is None:
        raise HTTPException(400, "No pipeline running")
    state.pipeline.enabled_cam_ids = set(req.enabled) if req.enabled is not None else None
    return {"ok": True, "enabled": req.enabled}


@app.post("/api/control/seek")
def control_seek(req: SeekRequest) -> dict:
    if state.pipeline is None:
        raise HTTPException(400, "No pipeline running")
    state.pipeline.source.seek(req.frame_index)
    state.pipeline.reset_filter()  # discard stale smoothing state
    return {"ok": True}


@app.post("/api/control/speed")
def control_speed(req: SpeedRequest) -> dict:
    if state.pipeline is None or not isinstance(state.pipeline.source, VideoSource):
        raise HTTPException(400, "Speed only applies to video sources")
    state.pipeline.source.set_speed(req.speed)
    return {"ok": True}


@app.post("/api/control/pause")
def control_pause(req: PauseRequest) -> dict:
    if state.pipeline is None or not isinstance(state.pipeline.source, VideoSource):
        raise HTTPException(400, "Pause only applies to video sources")
    state.pipeline.source.set_paused(req.paused)
    return {"ok": True}


@app.post("/api/control/stop")
def control_stop() -> dict:
    _stop_pipeline_if_running()
    state.session = None
    return {"ok": True}


@app.post("/api/control/reset-filters")
def control_reset_filters() -> dict:
    """Reset all estimator state (filters, bone model, tracking priors).

    Does not change the timeline position.  Clears the result cache so that
    re-observation repopulates the bone model from scratch.
    """
    if state.pipeline is None:
        raise HTTPException(400, "No pipeline running")
    state.pipeline.reset_filter()        # schedules _reset_all_filters() on next tick
    state.pipeline._result_cache.clear()  # force re-inference on scrub
    return {"ok": True}


# ---- REST: processing config ------------------------------------------------

@app.get("/api/control/backends")
def list_backends() -> list[dict]:
    """Return available pose backends with their variants and install status."""
    from livepose.tracker import get_backend_registry
    return get_backend_registry()


@app.get("/api/control/processing")
def get_processing() -> dict:
    """Return the current processing config (or defaults if no pipeline)."""
    if state.pipeline is None:
        from livepose.pipeline import ProcessingConfig
        cfg = ProcessingConfig()
        bone_phase = "init"
    else:
        cfg = state.pipeline.config
        bone_phase = state.pipeline._bone_model.phase
    return {
        "smoothing_enabled": cfg.smoothing_enabled,
        "min_cutoff": cfg.min_cutoff,
        "beta": cfg.beta,
        "imgsz": cfg.imgsz,
        "conf_threshold": cfg.conf_threshold,
        "backend": cfg.backend,
        "backend_variant": cfg.backend_variant,
        "detection_max_distance_px": cfg.detection_max_distance_px,
        "epipolar_threshold_px": cfg.epipolar_threshold_px,
        "bone_ik_enabled": cfg.bone_ik_enabled,
        "bone_phase": bone_phase,
    }


@app.post("/api/control/processing")
def set_processing(req: ProcessingRequest) -> dict:
    """Update processing config fields (partial update).
    If the pipeline is not running, the values are stored for the next session.
    """
    if state.pipeline is None:
        raise HTTPException(400, "No pipeline running")

    cfg = state.pipeline.config
    backend_changed = False

    if req.smoothing_enabled is not None:
        cfg.smoothing_enabled = req.smoothing_enabled
        if not req.smoothing_enabled:
            state.pipeline.reset_filter()
    if req.min_cutoff is not None:
        cfg.min_cutoff = max(0.01, req.min_cutoff)
    if req.beta is not None:
        cfg.beta = max(0.0, req.beta)
    if req.imgsz is not None:
        cfg.imgsz = int(req.imgsz)
    if req.conf_threshold is not None:
        cfg.conf_threshold = max(0.0, min(1.0, req.conf_threshold))
    # Detection selection
    if req.detection_max_distance_px is not None:
        cfg.detection_max_distance_px = max(1.0, req.detection_max_distance_px)
    if req.epipolar_threshold_px is not None:
        cfg.epipolar_threshold_px = max(1.0, req.epipolar_threshold_px)
    # Bone IK
    if req.bone_ik_enabled is not None:
        cfg.bone_ik_enabled = req.bone_ik_enabled

    if req.backend is not None and req.backend != cfg.backend:
        new_variant = req.backend_variant or cfg.backend_variant
        try:
            state.pipeline.swap_backend(req.backend, new_variant)
        except (ImportError, ValueError) as e:
            raise HTTPException(400, str(e))
        cfg.backend = req.backend
        cfg.backend_variant = new_variant
        backend_changed = True
    elif req.backend_variant is not None and req.backend_variant != cfg.backend_variant:
        try:
            state.pipeline.swap_backend(cfg.backend, req.backend_variant)
        except (ImportError, ValueError) as e:
            raise HTTPException(400, str(e))
        cfg.backend_variant = req.backend_variant
        backend_changed = True

    return {
        "ok": True,
        "backend_changed": backend_changed,
        "config": {
            "smoothing_enabled": cfg.smoothing_enabled,
            "min_cutoff": cfg.min_cutoff,
            "beta": cfg.beta,
            "imgsz": cfg.imgsz,
            "conf_threshold": cfg.conf_threshold,
            "backend": cfg.backend,
            "backend_variant": cfg.backend_variant,
            "detection_max_distance_px": cfg.detection_max_distance_px,
            "epipolar_threshold_px": cfg.epipolar_threshold_px,
            "bone_ik_enabled": cfg.bone_ik_enabled,
            "bone_phase": state.pipeline._bone_model.phase if state.pipeline else "init",
        },
    }


# ---- REST: CSV export -------------------------------------------------------

@app.get("/api/export/csv")
def export_csv():
    if not state.export_buffer:
        raise HTTPException(400, "No data buffered for export")

    buf = io.StringIO()
    w = csv.writer(buf)
    header = ["frame_index", "timestamp", "fps"]
    for j_name in COCO17_JOINTS:
        header += [f"{j_name}_x", f"{j_name}_y", f"{j_name}_z", f"{j_name}_valid"]
    w.writerow(header)

    for evt in state.export_buffer:
        row = [evt.frame_index, evt.timestamp, evt.fps]
        for j in range(len(COCO17_JOINTS)):
            x, y, z = evt.points_3d[j]
            v = bool(evt.valid_3d[j])
            row += [x, y, z, int(v)]
        w.writerow(row)

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="livepose_export.csv"'},
    )


# ---- WebSocket --------------------------------------------------------------

def _build_ws_message(evt: PipelineEvent) -> tuple[dict, list[bytes]]:
    """Build the JSON header + binary attachments for a WS message.

    Wire format:  4-byte big-endian header_length | header_json | jpeg_blobs concatenated
    Header includes per-camera blob offsets so client can slice out each JPEG.
    """
    blobs: list[bytes] = []
    cam_blob_offsets: list[dict] = []
    offset = 0
    for cid in sorted(evt.jpeg_per_cam.keys()):
        b = evt.jpeg_per_cam[cid]
        blobs.append(b)
        cam_blob_offsets.append({"cam_id": cid, "offset": offset, "length": len(b)})
        offset += len(b)

    header = {
        "type": "frame",
        "sequence": evt.sequence,
        "fps": evt.fps,
        "frame_index": evt.frame_index,
        "total_frames": evt.total_frames,
        "timestamp": evt.timestamp,
        "cam_ids": evt.cam_ids,
        "blobs": cam_blob_offsets,
        "poses_2d": {
            str(cid): {
                "kp": evt.poses_2d[cid].keypoints.tolist(),
                "conf": evt.poses_2d[cid].confidences.tolist(),
                "detected": evt.poses_2d[cid].detected,
            }
            for cid in evt.poses_2d
        },
        "points_3d": _nan_to_none(evt.points_3d).tolist() if evt.points_3d.size else [],
        "valid_3d": evt.valid_3d.tolist() if evt.valid_3d.size else [],
        "gt_3d": (_nan_to_none(evt.gt_3d).tolist() if evt.gt_3d is not None else None),
        "stage_timings_ms": evt.stage_timings_ms,
    }
    return header, blobs


def _nan_to_none(arr: np.ndarray) -> np.ndarray:
    """For JSON serialization: NaN -> None via tolist by replacing with None first."""
    out = arr.astype(object)
    out[~np.isfinite(arr.astype(float))] = None
    return out


def _build_meta_message() -> dict:
    """One-shot meta payload sent to each WS client on connect."""
    cameras_out: dict[str, Any] = {}
    if state.session is not None:
        for cam in state.session.cameras.values():
            cameras_out[str(cam.cam_id)] = {
                "K": cam.K.tolist(),
                "R": cam.R.tolist(),
                "t": cam.t.tolist(),
                "image_size": [cam.width, cam.height],
            }
    return {
        "type": "meta",
        "cameras": cameras_out,
        "skeletons": {
            "coco17": {
                "joints": COCO17_JOINTS,
                "edges": [list(e) for e in COCO17_SKELETON],
            },
            "h36m17": {
                "joints": H36M17_JOINTS,
                "edges": [list(e) for e in H36M17_SKELETON],
            },
        },
        "h36m_to_coco_subset": [list(p) for p in H36M_TO_COCO_SUBSET],
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if state.pipeline is None:
        await ws.send_json({"type": "error", "message": "No pipeline running"})
        await ws.close()
        return

    # Send one-shot calibration + skeleton meta immediately on connect
    await ws.send_json(_build_meta_message())

    queue = state.pipeline.subscribe()
    try:
        while True:
            evt = await queue.get()

            # Append to rolling export buffer
            state.export_buffer.append(evt)
            if len(state.export_buffer) > state.export_capacity:
                state.export_buffer = state.export_buffer[-state.export_capacity:]

            try:
                header, blobs = _build_ws_message(evt)
                header_bytes = json.dumps(header).encode()
                payload = struct.pack(">I", len(header_bytes)) + header_bytes + b"".join(blobs)
                await ws.send_bytes(payload)
            except WebSocketDisconnect:
                break
            except Exception:
                import traceback
                traceback.print_exc()
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if state.pipeline is not None:
            state.pipeline.unsubscribe(queue)


# ---- Static UI --------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "_static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")


@app.get("/")
def root_fallback():
    """Used when UI hasn't been built yet."""
    return {
        "message": "LivePose backend running. Build the UI: cd ui && npm run build",
        "docs": "/docs",
    }
