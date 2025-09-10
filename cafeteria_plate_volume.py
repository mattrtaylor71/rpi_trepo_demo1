"""Detect a stationary plate and capture its image.

This script streams frames from a Raspberry Pi camera and waits for a plate to
be placed beneath the camera. Once a circular plate is detected and remains
stationary for a short period, a still image is captured and written to disk.

The implementation is intentionally lightweight and reuses the Picamera2 setup
found in ``gesture_with_capture.py``. It only performs the detection and capture
step; further analysis of the plate contents can be added later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - Picamera2 is only available on Raspberry Pi
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480
STILL_W, STILL_H = 2304, 1296
STABLE_FRAMES = 15
MOVEMENT_THR = 5  # px tolerance for plate stability


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    picam2: Optional[Picamera2]
    video_cfg: Optional[object]
    still_cfg: Optional[object]
    cap: Optional[cv2.VideoCapture]


def create_camera() -> Camera:
    """Create either a Picamera2 or OpenCV camera capture."""

    if Picamera2 is not None:
        picam2 = Picamera2()
        video_cfg = picam2.create_video_configuration(
            main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
        )
        still_cfg = picam2.create_still_configuration(
            main={"format": "RGB888", "size": (STILL_W, STILL_H)}, display=None
        )
        picam2.configure(video_cfg)
        picam2.start()
        return Camera(picam2=picam2, video_cfg=video_cfg, still_cfg=still_cfg, cap=None)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return Camera(picam2=None, video_cfg=None, still_cfg=None, cap=cap)


# ---------------------------------------------------------------------------
# Plate detection
# ---------------------------------------------------------------------------


def find_plate(frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Return (x, y, r) for the largest detected circle (plate)."""

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype(int)
    x, y, r = max(circles, key=lambda c: c[2])
    return x, y, r


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main() -> None:
    cam = create_camera()

    prev_circle: Optional[Tuple[int, int, int]] = None
    stable = 0

    try:
        while True:
            if cam.picam2 is not None:
                frame = cam.picam2.capture_array("main")
            else:
                ok, frame_bgr = cam.cap.read()  # type: ignore[union-attr]
                if not ok:
                    break
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            circle = find_plate(frame)

            if circle is not None:
                x, y, r = circle
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                if prev_circle is not None:
                    dx = abs(x - prev_circle[0])
                    dy = abs(y - prev_circle[1])
                    dr = abs(r - prev_circle[2])
                    if dx < MOVEMENT_THR and dy < MOVEMENT_THR and dr < MOVEMENT_THR:
                        stable += 1
                    else:
                        stable = 0
                prev_circle = (x, y, r)

                if stable >= STABLE_FRAMES:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"plate_{ts}.jpg"
                    if cam.picam2 is not None:
                        cam.picam2.switch_mode(cam.still_cfg)
                        time.sleep(0.2)
                        still = cam.picam2.capture_array()
                        cam.picam2.switch_mode(cam.video_cfg)
                        cv2.imwrite(filename, cv2.cvtColor(still, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"Saved {filename}")
                    stable = 0
                    prev_circle = None
            else:
                stable = 0
                prev_circle = None

            cv2.imshow("plate", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
    finally:
        if cam.picam2 is not None:
            cam.picam2.stop()
        if cam.cap is not None:
            cam.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    main()

