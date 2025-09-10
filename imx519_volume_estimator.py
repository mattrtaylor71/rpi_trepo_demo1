#!/usr/bin/env python3
"""Volume estimation script using a Raspberry Pi 4B with an IMX519 camera.

This script captures images of an object placed on a table while the object is
rotated. It then reconstructs a dense point cloud using COLMAP (via the
``pycolmap`` Python bindings) and estimates the volume of the object after
scaling the model using an ArUco marker of known size.

Dependencies
------------
* OpenCV with contrib modules (for ``cv2.aruco``)
* pycolmap
* open3d

Typical usage
-------------
1. Print an ArUco marker (e.g. ``DICT_6X6_1000``) with a known side length.
   Place it flat on the table within the camera view.
2. Put the object on the table next to the marker.
3. Run ``python imx519_volume_estimator.py capture -o ./captures`` and rotate
   the object between captures. The script takes one frame per key press and
   saves the images in ``./captures``.
4. Run ``python imx519_volume_estimator.py reconstruct -i ./captures -m 0.035``
   where ``-m`` is the marker side length in meters.
5. The script outputs the estimated volume in cubic centimeters.

Note
----
The script orchestrates the pipeline but the heavy reconstruction is delegated
to COLMAP through ``pycolmap``. Running it on a Raspberry Pi may take a long
time depending on the number of images and resolution.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import pycolmap

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)


@dataclass
class CaptureConfig:
    """Configuration for image capture."""

    output_dir: pathlib.Path
    num_images: int
    camera_id: int = 0


def capture_images(cfg: CaptureConfig) -> None:
    """Capture ``num_images`` frames from the specified camera.

    The user should manually rotate the object between frames. Each frame is
    stored as ``img_XXX.jpg`` in ``output_dir``.
    """

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(cfg.camera_id)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("Press <space> to capture a frame, <q> to quit early.")
    idx = 0
    while idx < cfg.num_images:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            out_path = cfg.output_dir / f"img_{idx:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved {out_path}")
            idx += 1
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_colmap(image_dir: pathlib.Path, work_dir: pathlib.Path) -> pathlib.Path:
    """Run COLMAP via ``pycolmap`` and return path to the fused point cloud."""

    database_path = work_dir / "database.db"
    sparse_path = work_dir / "sparse"
    dense_path = work_dir / "dense"
    for p in (database_path, sparse_path, dense_path):
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    work_dir.mkdir(parents=True, exist_ok=True)

    pycolmap.extract_features(str(database_path), str(image_dir))
    pycolmap.match_features(str(database_path))
    maps = pycolmap.incremental_mapping(
        str(database_path), str(image_dir), str(sparse_path)
    )
    map_paths = list(maps.values())
    if not map_paths:
        raise RuntimeError("COLMAP failed to create a sparse model")

    # Densify
    model = map_paths[0]
    pycolmap.image_undistorter(
        str(image_dir), str(sparse_path), str(dense_path), model_ids=[model]
    )
    pycolmap.stereo(
        str(dense_path), str(dense_path / "stereo"), str(dense_path / "stereo"),
        cache_size=32
    )
    pycolmap.fuse(
        str(dense_path), str(dense_path / "fused.ply"),
        output_type="PLY"
    )
    return dense_path / "fused.ply"


def estimate_scale(image_path: pathlib.Path, marker_length: float) -> float:
    """Estimate metric scale using an ArUco marker of known size.

    Parameters
    ----------
    image_path: pathlib.Path
        Path to an image containing the marker.
    marker_length: float
        Length of the marker's side in meters.

    Returns
    -------
    float
        Conversion factor from COLMAP units to meters.
    """

    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco marker detected for scale estimation")

    # Use the first detected marker
    pixel_length = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
    return marker_length / pixel_length


def compute_volume(ply_path: pathlib.Path, scale: float) -> float:
    """Compute the volume in cubic centimeters from a fused point cloud."""

    pcd = o3d.io.read_point_cloud(str(ply_path))
    pcd.scale(scale, center=pcd.get_center())
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=0.02
    )
    hull, _ = mesh.compute_convex_hull()
    volume_m3 = hull.get_volume()
    return volume_m3 * 1e6  # convert m^3 to cm^3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap_p = sub.add_parser("capture", help="Capture images from camera")
    cap_p.add_argument("-o", "--output", type=pathlib.Path, required=True)
    cap_p.add_argument("-n", "--num-images", type=int, default=36)
    cap_p.add_argument("-c", "--camera-id", type=int, default=0)

    rec_p = sub.add_parser("reconstruct", help="Reconstruct and estimate volume")
    rec_p.add_argument("-i", "--images", type=pathlib.Path, required=True)
    rec_p.add_argument("-m", "--marker-length", type=float, required=True,
                       help="Marker side length in meters")

    args = parser.parse_args()
    if args.cmd == "capture":
        cfg = CaptureConfig(output_dir=args.output, num_images=args.num_images,
                             camera_id=args.camera_id)
        capture_images(cfg)
    elif args.cmd == "reconstruct":
        work_dir = args.images / "colmap"
        ply_path = run_colmap(args.images, work_dir)
        first_image = next(sorted(args.images.glob("*.jpg")))
        scale = estimate_scale(first_image, args.marker_length)
        volume_cc = compute_volume(ply_path, scale)
        print(f"Estimated volume: {volume_cc:.2f} cm^3")


if __name__ == "__main__":
    main()
