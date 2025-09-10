
"""Cafeteria plate analysis with motion based capture.

This script streams frames from a Raspberry Pi camera (Picamera2 when
available, otherwise OpenCV) and watches for a new object entering the
scene. Once the view stabilises for a short period the current frame is
analysed: the plate contents are segmented, each food blob is classified via
an OpenAI vision model and the estimated volume is drawn over the camera
stream.

The table is expected to include a DICT_4X4_50 ArUco marker with ID 0 to
derive a millimetres‑per‑pixel scale for volume estimates.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
MOTION_ENTER_THR = 15.0  # percentage of pixels that must change to trigger
MOTION_STABLE_THR = 2.0   # percentage of pixels allowed to change when stable
ARUCO_MARKER_MM = 40      # physical size of the ID 0 marker
ASSUMED_HEIGHT_MM = 10    # assumed height of food for volume estimation


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
# Motion detection
# ---------------------------------------------------------------------------


def motion_percent(prev: np.ndarray, curr: np.ndarray) -> float:
    """Return percentage of pixels that changed between the two frames."""

    gray_prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed = np.count_nonzero(thresh)
    return changed * 100.0 / thresh.size


# ---------------------------------------------------------------------------
# Plate / food processing
# ---------------------------------------------------------------------------


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


def find_plate_mask(frame: np.ndarray) -> np.ndarray:
    """Approximate the plate as the largest contour."""

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gray)
    plate = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [plate], -1, 255, -1)
    return mask


def segment_food(frame: np.ndarray, plate_mask: np.ndarray) -> List[np.ndarray]:
    """Return contours for food blobs inside the plate."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    food = np.where(((s > 30) | (v < 200)) & (plate_mask > 0), 255, 0).astype(np.uint8)
    food = cv2.morphologyEx(food, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(food, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 500]


@dataclass
class BlobResult:
    label: str
    volume_ml: float
    bbox: Tuple[int, int, int, int]


def classify_blob(img: np.ndarray) -> str:
    """Classify a blob image using an OpenAI vision model."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "unknown"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return "unknown"
        b64 = base64.b64encode(buf).decode("utf-8")
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify the food in this image."},
                        {"type": "image", "image_base64": b64},
                    ],
                }
            ],
        )
        return resp.output[0].content[0].text.strip()
    except Exception:
        return "unknown"

def analyse_plate(frame: np.ndarray) -> Tuple[np.ndarray, List[BlobResult]]:
    """Analyse the plate and return an annotated frame and blob info."""

    output = frame.copy()
    results: List[BlobResult] = []

    scale = mm_per_pixel(frame)
    if scale is None:
        cv2.putText(
            output,
            "ArUco marker ID 0 not found",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return output, results

    mask = find_plate_mask(frame)
    blobs = segment_food(frame, mask)
    for cnt in blobs:
        area_px = cv2.contourArea(cnt)
        area_mm2 = area_px * (scale ** 2)
        volume_ml = (area_mm2 * ASSUMED_HEIGHT_MM) / 1000.0
        x, y, w, h = cv2.boundingRect(cnt)
        crop = frame[y : y + h, x : x + w]
        label = classify_blob(crop)
        results.append(BlobResult(label=label, volume_ml=volume_ml, bbox=(x, y, w, h)))
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{label}: {volume_ml:.1f} mL",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return output, results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    cam = create_camera()
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

            if prev is None:
                prev = frame
                continue

            mp = motion_percent(prev, frame)
            prev = frame

            if time.time() > overlay_until:
                overlay = None
            if not armed and mp > MOTION_ENTER_THR:
                armed = True
                stable_count = 0

            if armed:
                if mp < MOTION_STABLE_THR:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= STABLE_FRAMES:
                    annotated, infos = analyse_plate(frame)
                    for i, info in enumerate(infos, 1):
                        print(f"Blob {i}: {info.label} ~ {info.volume_ml:.1f} mL")
                    overlay = annotated
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
