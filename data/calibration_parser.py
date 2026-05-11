#!/usr/bin/env python3
"""
Parse MPI-INF-3DHP camera calibration files
"""
import numpy as np
from pathlib import Path

def parse_calibration_file(calib_file):
    """
    Parse Skeletool camera calibration file format.

    Returns dict with camera calibration matrices:
    - intrinsic: 3x3 K matrix (focal length, principal point)
    - extrinsic: 4x4 [R|t] matrix (rotation + translation)
    - size: (width, height) in pixels
    """
    cameras = {}

    with open(calib_file, 'r') as f:
        lines = f.readlines()

    current_cam = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('name'):
            current_cam = int(line.split()[-1])
            cameras[current_cam] = {}

        elif line.startswith('intrinsic'):
            values = [float(x) for x in line.split()[1:]]
            # Convert 4x4 to 3x3 (remove last row/col)
            K = np.array(values).reshape(4, 4)[:3, :3]
            cameras[current_cam]['intrinsic'] = K

        elif line.startswith('extrinsic'):
            values = [float(x) for x in line.split()[1:]]
            RT = np.array(values).reshape(4, 4)
            cameras[current_cam]['extrinsic'] = RT

        elif line.startswith('size'):
            w, h = int(line.split()[1]), int(line.split()[2])
            cameras[current_cam]['size'] = (w, h)

    return cameras

def project_3d_to_2d(point_3d, intrinsic, extrinsic):
    """
    Project 3D point to 2D image coordinates using camera parameters.

    Args:
        point_3d: (3,) array [X, Y, Z] in world coordinates (mm)
        intrinsic: (3,3) camera intrinsic matrix K
        extrinsic: (4,4) camera extrinsic matrix [R|t]

    Returns:
        (2,) array [u, v] in image coordinates (pixels)
    """
    # Transform to camera coordinates
    P_world = np.append(point_3d, 1)
    P_cam = extrinsic @ P_world

    # Project to image
    p = intrinsic @ P_cam[:3]
    u, v = p[0] / p[2], p[1] / p[2]

    return np.array([u, v])

if __name__ == '__main__':
    # Example usage
    calib_file = Path.home() / 'Documents/Projects/SLP/data/mpi_inf_3dhp/S1/Seq1/camera.calibration'

    cameras = parse_calibration_file(calib_file)

    print(f"Loaded {len(cameras)} cameras")
    print(f"\nCamera 0 intrinsic matrix K:")
    print(cameras[0]['intrinsic'])
    print(f"\nCamera 0 extrinsic [R|t]:")
    print(cameras[0]['extrinsic'])
    print(f"\nImage size: {cameras[0]['size']}")

    # Example: project a 3D point (at origin) to all cameras
    point_3d = np.array([0, 0, 0])  # mm
    print(f"\n3D point {point_3d} projects to image coordinates:")
    for cam_id in sorted(cameras.keys())[:3]:  # Show first 3 cameras
        proj = project_3d_to_2d(point_3d, cameras[cam_id]['intrinsic'], cameras[cam_id]['extrinsic'])
        print(f"  Camera {cam_id}: ({proj[0]:.1f}, {proj[1]:.1f})")
