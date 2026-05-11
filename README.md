# LivePose

LivePose is a real-time multi-camera pose estimation app with 2D overlays, 3D visualization, dataset import, live camera input, and CSV export.

## Demo

[![LivePose demo](https://img.youtube.com/vi/Jv4HkbjrIFs/maxresdefault.jpg)](https://youtu.be/Jv4HkbjrIFs)

Click the preview above to watch the demo video.

Private live video request/access demo:

- Link: https://youtu.be/x0wtxJh0xg8
- Note: This video is private. Please request access to view it.

## Requirements

- Python 3.11+
- Node.js 18+
- A supported dataset or camera setup if you want to run sessions

## Install

Install the Python package first:

```bash
pip install -e .
```

Then install the UI dependencies:

```bash
cd ui
npm install
```

## Build The UI

The FastAPI server serves the built frontend from `livepose/_static`, so build the UI before starting the app:

```bash
cd ui
npm run build
```

## Run

Start the app from the project root:

```bash
python -m livepose
```

By default this starts the server on `http://127.0.0.1:8765` and opens the browser automatically. To use a different host or port:

```bash
python -m livepose --host 0.0.0.0 --port 8765
```

Add `--no-browser` if you do not want the browser to open automatically.

## How To Use

1. Open **Import Dataset** to scan a dataset folder or start a live camera session.
2. Choose a sequence or camera set and import it.
3. Watch the 2D feed grid and the 3D scene update in real time.
4. Use **Export CSV** to save the tracked output.

## Control Params

The top bar provides session controls:

- **RAW 2D**: show or hide the raw 2D detections.
- **REPROJ 3D**: show or hide 3D reprojections in the camera feeds.
- **GT**: show ground-truth overlays when the session includes them.
- **REPROJ GT**: show ground-truth reprojection overlays.
- **Import Dataset**: open the import dialog for dataset or live mode.
- **Export CSV**: download the current tracked results.
- **Stop**: stop the active session.

The processing panel exposes the main tuning parameters:

- **Backend / Variant**: choose the pose backend and model variant.
- **imgsz**: inference image size. Higher values can improve accuracy but cost more compute.
- **Conf**: minimum confidence threshold for detections.
- **Smoothing**: enables or disables temporal filtering.
- **min_cutoff**: smoothing strength. Higher values are less smooth.
- **beta**: response speed. Higher values react faster to motion.
- **Max Δ px**: maximum 2D distance used when matching detections.
- **Epipolar**: epipolar consistency threshold for multi-camera matching.
- **Bone IK**: enables or disables bone-based inverse kinematics refinement.

The timeline at the bottom also lets you:

- seek to a frame,
- pause or resume playback,
- change playback speed.

## Dataset Notes

The repository includes MPI-INF-3DHP data under `data/mpi_inf_3dhp` and a downscaled sample under `data/mpi_inf_3dhp_downscaled`.

## Project Layout

- `livepose/` - FastAPI backend, streaming pipeline, and import logic
- `ui/` - React/Vite frontend
- `data/` - dataset helpers and sample data
- `scripts/` - utility scripts
