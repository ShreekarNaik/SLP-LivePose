"""Create a downscaled copy of an MPI-INF-3DHP sequence using ffmpeg.

Scales videos to target_size, reduces FPS, adjusts calibration intrinsics,
and subsamples annotations to match the new frame rate.

Usage:
    python scripts/create_downscaled_dataset.py \
        --src data/mpi_inf_3dhp/S1/Seq1 \
        --dst data/mpi_inf_3dhp_downscaled/S1/Seq1 \
        --target-size 1024 --target-fps 18
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import scipy.io


def downscale_video_ffmpeg(src: Path, dst: Path, target_size: int, target_fps: float):
    """Downscale + re-FPS a single video using ffmpeg."""
    vf = f"fps={target_fps},scale={target_size}:{target_size}"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def adjust_calibration(src_cal: Path, dst_cal: Path, orig_size: int, target_size: int):
    """Scale intrinsic parameters for new resolution."""
    text = src_cal.read_text()
    scale = target_size / orig_size

    lines = text.splitlines()
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("size"):
            out_lines.append(f"  size        {target_size} {target_size}")
        elif stripped.startswith("intrinsic"):
            vals = [float(x) for x in stripped.split()[1:]]
            K = np.array(vals).reshape(4, 4)
            K[0, 0] *= scale  # fx
            K[0, 2] *= scale  # cx
            K[1, 1] *= scale  # fy
            K[1, 2] *= scale  # cy
            flat = " ".join(f"{v:.6g}" for v in K.flatten())
            out_lines.append(f"  intrinsic   {flat}")
        else:
            out_lines.append(line)

    dst_cal.write_text("\n".join(out_lines) + "\n")


def subsample_annotations(src_annot: Path, dst_annot: Path, orig_fps: float, target_fps: float):
    """Subsample annot.mat rows to match the new frame rate.

    ffmpeg's fps filter with drop picks frames nearest to the output timestamps,
    so output frame i corresponds to original frame round(i * orig_fps / target_fps).
    """
    mat = scipy.io.loadmat(str(src_annot))
    annot2 = mat["annot2"]  # (num_cams, 1), each entry (n_orig_frames, ...)
    annot3 = mat["annot3"]

    n_orig = annot3[0, 0].shape[0]
    n_new = int(n_orig * target_fps / orig_fps)

    # Build mapping: new frame i -> original frame index
    indices = np.array([min(round(i * orig_fps / target_fps), n_orig - 1) for i in range(n_new)])

    # Subsample each camera's annotations
    for cam_idx in range(annot3.shape[0]):
        if annot3[cam_idx, 0].shape[0] > 0:
            annot3[cam_idx, 0] = annot3[cam_idx, 0][indices]
        if annot2[cam_idx, 0].shape[0] > 0:
            annot2[cam_idx, 0] = annot2[cam_idx, 0][indices]

    scipy.io.savemat(str(dst_annot), {"annot2": annot2, "annot3": annot3})
    print(f"  {n_orig} -> {n_new} frames ({orig_fps}fps -> {target_fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="Create downscaled MPI-INF-3DHP sequence")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--target-fps", type=float, default=18.0)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    target = args.target_size

    img_src = src / "imageSequence"
    img_dst = dst / "imageSequence"
    img_dst.mkdir(parents=True, exist_ok=True)

    # Probe original FPS
    first_vid = sorted(img_src.glob("video_*.avi"))[0]
    cap = cv2.VideoCapture(str(first_vid))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    print(f"Source: {src} ({orig_w}x{orig_w} @ {orig_fps}fps)")
    print(f"Target: {dst} ({target}x{target} @ {args.target_fps}fps)")

    # 1. Downscale + reduce FPS
    videos = sorted(img_src.glob("video_*.avi"))
    print(f"\n--- Transcoding {len(videos)} videos ---")
    for vf in videos:
        print(f"  {vf.name} ...", end=" ", flush=True)
        downscale_video_ffmpeg(vf, img_dst / vf.name, target, args.target_fps)
        print("done")

    # 2. Adjust calibration
    print("\n--- Adjusting calibration ---")
    adjust_calibration(src / "camera.calibration", dst / "camera.calibration", orig_w, target)
    print(f"  intrinsics scaled: {orig_w} -> {target}")

    # 3. Subsample annotations to match new FPS
    annot_src = src / "annot.mat"
    if annot_src.exists():
        print("\n--- Subsampling annotations ---")
        subsample_annotations(annot_src, dst / "annot.mat", orig_fps, args.target_fps)

    print(f"\nDone! Dataset at: {dst}")


if __name__ == "__main__":
    main()
