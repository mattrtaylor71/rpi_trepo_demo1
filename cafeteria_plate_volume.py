"""Simple object dimension estimation using an ArUco marker.

This script streams frames from a Raspberry Pi camera and watches for a new
object placed on a static background. Once the view stabilises the largest
foreground object is measured. The size of an ArUco marker (DICT_4X4_50, ID 0)
on the table is used to convert pixels to millimetres and the resulting width
and height of the object are drawn on the camera stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:  # pragma: no cover - Picamera2 only available on Raspberry Pi
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480
STABLE_FRAMES = 15
FG_AREA_ENTER = 5000  # pixels of foreground to trigger detection
MOTION_STABLE_THR = 2.0  # percent motion allowed when stable
ARUCO_MARKER_MM = 40  # physical size of the ID 0 marker


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    picam2: Optional[Picamera2]
    cap: Optional[cv2.VideoCapture]


def create_camera() -> Camera:
    """Create either a Picamera2 or OpenCV camera capture."""

    if Picamera2 is not None:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
        )
        picam2.configure(config)
        picam2.start()
        return Camera(picam2=picam2, cap=None)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return Camera(picam2=None, cap=cap)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def motion_percent(prev: np.ndarray, curr: np.ndarray) -> float:
    """Return percentage of pixels that changed between two frames."""

    gray_prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed = np.count_nonzero(thresh)
    return changed * 100.0 / thresh.size


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def mm_per_pixel(frame: np.ndarray) -> Optional[float]:
    """Return millimetres per pixel using the ID 0 ArUco marker."""

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if i == 0:
            side_px = np.mean(
                [
                    np.linalg.norm(c[0][0] - c[0][1]),
                    np.linalg.norm(c[0][1] - c[0][2]),
                    np.linalg.norm(c[0][2] - c[0][3]),
                    np.linalg.norm(c[0][3] - c[0][0]),
                ]
            )
            return ARUCO_MARKER_MM / side_px
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - relies on camera hardware
    cam = create_camera()
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )
    prev: Optional[np.ndarray] = None
    stable_count = 0
    armed = False
    overlay: Optional[np.ndarray] = None
    overlay_until = 0.0

    try:
        while True:
            if cam.picam2 is not None:
                frame = cam.picam2.capture_array("main")
            else:
                ok, frame_bgr = cam.cap.read()  # type: ignore[union-attr]
                if not ok:
                    break
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            fg_mask = bg_sub.apply(frame)

            if prev is None:
                prev = frame
                continue

            mp = motion_percent(prev, frame)
            prev = frame

            fg_area = np.count_nonzero(fg_mask)

            if time.time() > overlay_until:
                overlay = None

            if not armed and fg_area > FG_AREA_ENTER:
                armed = True
                stable_count = 0

            if armed:
                if mp < MOTION_STABLE_THR:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= STABLE_FRAMES:
                    contours, _ = cv2.findContours(
                        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(c)
                        scale = mm_per_pixel(frame)
                        output = frame.copy()
                        if scale is not None:
                            width_mm = w * scale
                            height_mm = h * scale
                            cv2.rectangle(
                                output, (x, y), (x + w, y + h), (0, 255, 0), 2
                            )
                            cv2.putText(
                                output,
                                f"{width_mm:.1f} x {height_mm:.1f} mm",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )
                        else:
                            cv2.putText(
                                output,
                                "Marker ID 0 not found",
                                (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 0, 255),
                                2,
                            )
                        overlay = output
                        overlay_until = time.time() + 3.0
                    armed = False
                    stable_count = 0

            display = overlay if overlay is not None else frame
            cv2.imshow("cafeteria", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == 27:
                break

    finally:
        if cam.picam2 is not None:
            cam.picam2.stop()
        if cam.cap is not None:
            cam.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    main()

