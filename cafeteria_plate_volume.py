"""Cafeteria plate analysis script.

This script waits for a user to place a plate under an overhead camera. Once the
plate is stable on the table the plate region is segmented, blobs of food are
identified, their approximate volume is computed using an ArUco marker for
scale and the blobs are sent to ChatGPT for classification.

The script only provides a reference implementation and is designed to run on
Raspberry Pi hardware with a camera. Many parts are approximations and would
need calibration for a production environment.
"""
import os
import time
import base64
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - OpenAI might not be installed
    OpenAI = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Video capture settings. On embedded devices using the default OpenCV
# backend can exhaust memory. We try a low-resolution GStreamer pipeline first
# and fall back to the standard camera index with an explicit resolution.
CAMERA_INDEX = 0
GST_PIPELINE = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
)
MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_SIZE_MM = 50  # physical side length of the marker
STABLE_FRAMES = 10   # number of consecutive stable frames required
STABLE_THRESH = 2.0  # average pixel difference to consider stable
FOOD_HEIGHT_MM = 10  # assumed average height for volume estimation


@dataclass
class FoodBlob:
    label: str
    volume_ml: float


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_capture() -> cv2.VideoCapture:
    """Create a video capture object with memory-friendly settings.

    Tries a low-resolution GStreamer pipeline first. If that fails, falls back
    to the default camera index and explicitly sets a smaller resolution to
    avoid the "failed to allocate required memory" GStreamer error commonly
    seen on resource constrained devices.
    """
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def detect_marker(frame: np.ndarray) -> Optional[float]:
    """Return pixel-to-mm scale using a known ArUco marker.

    The function searches for a DICT_4X4_50 marker with ID 0. If found it
    computes the average length of its sides in pixels and returns the
    mm-per-pixel scale (millimeters per pixel).
    """
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if i == MARKER_ID:
            # compute side length in pixels
            pts = c.reshape(4, 2)
            side = np.mean([np.linalg.norm(pts[0] - pts[1]),
                            np.linalg.norm(pts[1] - pts[2]),
                            np.linalg.norm(pts[2] - pts[3]),
                            np.linalg.norm(pts[3] - pts[0])])
            return MARKER_SIZE_MM / side
    return None


def detect_plate(frame: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """Detect the plate via circle detection.

    Returns a mask with the plate region and the circle parameters
    (x, y, r)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype("int")
    # choose the largest circle assuming it's the plate
    x, y, r = max(circles, key=lambda c: c[2])
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.circle(mask, (x, y), r, 255, -1)
    return mask, (x, y, r)


def segment_foods(plate_img: np.ndarray, scale_mm_per_px: float) -> List[FoodBlob]:
    """Segment food blobs on the plate and estimate their volume."""
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    # Reduce noise and segment by color using k-means (K=3 arbitrary)
    Z = hsv.reshape((-1, 3))
    Z = np.float32(Z)
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented = centers[labels.flatten()].reshape(hsv.shape)
    segmented = cv2.cvtColor(segmented.astype("uint8"), cv2.COLOR_HSV2BGR)

    blobs = []
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    num, labels_im = cv2.connectedComponents(thresh)

    for idx in range(1, num):
        mask = (labels_im == idx).astype("uint8")
        area_px = cv2.countNonZero(mask)
        if area_px < 200:  # ignore tiny blobs
            continue
        area_mm2 = area_px * (scale_mm_per_px ** 2)
        volume_ml = area_mm2 * FOOD_HEIGHT_MM / 1000.0  # cubic mm to ml
        crop = cv2.bitwise_and(plate_img, plate_img, mask=mask)
        label = classify_food(crop)
        blobs.append(FoodBlob(label=label, volume_ml=volume_ml))
    return blobs


def classify_food(img: np.ndarray) -> str:
    """Classify a cropped food image using ChatGPT.

    If the OpenAI client is unavailable or the API key is missing, the
    function returns 'unknown'."""
    if OpenAI is None or not os.environ.get("OPENAI_API_KEY"):
        return "unknown"
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Identify the food shown."},
                    {"type": "input_image", "image_base64": b64},
                ],
            }],
        )
        return resp.output_text.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    cap = create_capture()
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    prev = None
    stable_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        scale = detect_marker(frame)
        if scale is None:
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        if prev is not None:
            diff = cv2.absdiff(frame, prev)
            mean_diff = diff.mean()
            if mean_diff < STABLE_THRESH:
                stable_count += 1
            else:
                stable_count = 0
        prev = frame.copy()

        if stable_count < STABLE_FRAMES:
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # plate stable - process once
        plate = detect_plate(frame)
        if plate is None:
            stable_count = 0
            continue
        mask, (x, y, r) = plate
        plate_img = cv2.bitwise_and(frame, frame, mask=mask)
        blobs = segment_foods(plate_img, scale)

        print("Detected foods:")
        for blob in blobs:
            print(f"  {blob.label}: {blob.volume_ml:.1f} ml")

        # reset for next plate
        stable_count = 0
        time.sleep(2)

        cv2.imshow("camera", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
