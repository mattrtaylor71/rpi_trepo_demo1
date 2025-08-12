import time, os, collections, threading, queue, base64, datetime
import numpy as np
import cv2
from picamera2 import Picamera2

# --------------------------
# OpenAI API (vision)
# --------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # we'll warn later

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # set to "gpt-5" if you have it
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# A tiny worker so API calls don't block the camera loop
work_q = queue.Queue(maxsize=4)
def api_worker():
    if OpenAI is None or not OPENAI_API_KEY:
        print("[WARN] OpenAI library or API key missing; skipping uploads.")
        return
    client = OpenAI(api_key=OPENAI_API_KEY)
    while True:
        item = work_q.get()
        if item is None:
            break
        tag, jpeg_bytes = item
        try:
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            msgs = [
                {"role": "system",
                 "content": "You are an expert product identifier. Describe the main item succinctly and name it if possible."},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": f"Identify the object in this photo. (trigger={tag})"},
                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                 ]},
            ]
            resp = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
            text = resp.choices[0].message.content
            print(f"[{tag}] Vision result -> {text}")
        except Exception as e:
            print(f"[{tag}] OpenAI error: {e}")
        finally:
            work_q.task_done()

api_thread = threading.Thread(target=api_worker, daemon=True)
api_thread.start()

# =========================
# Ultra-fast swipe via column/row motion energy (LR + UD)
# =========================
FRAME_W, FRAME_H = 640, 480
LO_W, LO_H       = 192, 108

ROI_Y0, ROI_Y1 = 0.25, 0.75
ROI_X0, ROI_X1 = 0.25, 0.75

CROSS_L, CROSS_R = 0.12, 0.88
HYST = 0.02
L_ARM, L_FIRE = CROSS_L - HYST, CROSS_L + HYST
R_ARM, R_FIRE = CROSS_R + HYST, CROSS_R - HYST

CROSS_T, CROSS_B = 0.12, 0.88
T_ARM, T_FIRE = CROSS_T - HYST, CROSS_T + HYST
B_ARM, B_FIRE = CROSS_B + HYST, CROSS_B - HYST

COOLDOWN_S      = 0.12
ABSENCE_RESET_S = 0.10

SPAN_WINDOW_S = 0.28
SPAN_THR_X, VEL_THR_X = 0.45, 2.2
SPAN_THR_Y, VEL_THR_Y = 0.45, 2.2

ENERGY_MIN_FRAC = 0.015
SMOOTH_COLS, SMOOTH_ROWS = 5, 5
POOL_FRAMES = 2

HEADLESS = os.environ.get("DISPLAY", "") == ""

# =========================
# Helpers
# =========================
def y_plane(yuv):
    if yuv.ndim == 2: return yuv[:LO_H, :LO_W]
    elif yuv.ndim == 3: return yuv[:, :, 0]
    raise ValueError(f"Unexpected lores shape {yuv.shape}")

def smooth1d(v, k):
    if k <= 0: return v
    ksz = 2*k + 1
    kernel = np.ones(ksz, dtype=np.float32) / ksz
    return np.convolve(v, kernel, mode="same")

def jpeg_bytes_from_rgb(rgb, quality=92):
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok: return None
    return jpg.tobytes()

def enqueue_openai(tag, rgb_frame):
    os.makedirs("captures", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"captures/{tag}_{ts}.jpg"
    jpg = jpeg_bytes_from_rgb(rgb_frame, quality=92)
    if jpg is None:
        print(f"[{tag}] JPEG encode failed")
        return
    with open(path, "wb") as f:
        f.write(jpg)
    print(f"[{tag}] saved {path}")
    try:
        work_q.put_nowait((tag, jpg))
    except queue.Full:
        print(f"[{tag}] queue full; skipping upload")

def laplacian_sharpness(gray_u8):
    # higher = sharper
    return cv2.Laplacian(gray_u8, cv2.CV_64F).var()

# =========================
# Camera (video + still modes)
# =========================
picam2 = Picamera2()

video_cfg = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)},
    lores={"format": "YUV420", "size": (LO_W, LO_H)},
    display="main",
)

# Choose a sensible still size for IMX708; 2304x1296 is fast & sharp.
# You can push higher (e.g., 4056x3040) but it’s slower & heavier.
STILL_W, STILL_H = 2304, 1296
still_cfg = picam2.create_still_configuration(
    main={"format": "RGB888", "size": (STILL_W, STILL_H)},
    display=None,
)

picam2.configure(video_cfg)

# Fast preview tuning
try:
    picam2.set_controls({
        "AeEnable": True, "AwbEnable": True,
        "AfMode": 2,                       # continuous AF for preview
        "FrameDurationLimits": (10000, 10000),  # ~100 fps (device may cap lower)
        "AnalogueGain": 12.0,
    })
except Exception:
    pass

picam2.start()

cam_lock = threading.Lock()  # protect mode switches & controls

def capture_high_quality(tag: str):
    """Switch to still mode, AF/AE settle, capture burst, pick sharpest, enqueue."""
    with cam_lock:
        try:
            # Give AF a kick to refocus at the object distance
            try:
                picam2.set_controls({"AfMode": 1})         # single-shot AF if supported
                picam2.set_controls({"AfTrigger": 1})      # start AF
            except Exception:
                # fall back to continuous
                picam2.set_controls({"AfMode": 2})

            # Switch to still mode
            picam2.switch_mode(still_cfg)
            # small settle for AE/AF in still mode
            time.sleep(0.18)

            # Burst capture and choose sharpest
            frames = []
            scores = []
            N = 3
            for _ in range(N):
                arr = picam2.capture_array("main")  # RGB888 at STILL_W x STILL_H
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                frames.append(arr)
                scores.append(laplacian_sharpness(gray))
                # brief pause lets AF/AE refine
                time.sleep(0.05)

            best_idx = int(np.argmax(scores))
            best = frames[best_idx]
            print(f"[{tag}] burst sharpness scores: {[round(s,1) for s in scores]} -> pick {best_idx}")

            enqueue_openai(tag, best)

        except Exception as e:
            print(f"[{tag}] capture error: {e}")

        finally:
            # Return to fast video mode and restore fast preview controls
            try:
                picam2.switch_mode(video_cfg)
                picam2.set_controls({
                    "AeEnable": True, "AwbEnable": True,
                    "AfMode": 2,
                    "FrameDurationLimits": (10000, 10000),
                    "AnalogueGain": 12.0,
                })
            except Exception:
                pass

def start_capture_thread(tag, rgb_preview_unused=None):
    # launch in background so the UI / swipe loop stays snappy
    t = threading.Thread(target=capture_high_quality, args=(tag,), daemon=True)
    t.start()

# =========================
# Swipe detection state
# =========================
prev_lo = None
mask_pool = collections.deque(maxlen=POOL_FRAMES)
trace_x = collections.deque()
trace_y = collections.deque()

state_x = "IDLE"
state_y = "IDLE"
last_fire = 0.0
last_seen_t_x = 0.0
last_seen_t_y = 0.0

sx, sy = FRAME_W / LO_W, FRAME_H / LO_H
y0 = int(ROI_Y0 * LO_H); y1 = int(ROI_Y1 * LO_H)
x0 = int(ROI_X0 * LO_W); x1 = int(ROI_X1 * LO_W)

try:
    while True:
        now = time.time()

        # Frames
        lo  = y_plane(picam2.capture_array("lores"))
        rgb = picam2.capture_array("main")   # fast preview
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        x_norm = None; y_norm = None

        if prev_lo is not None:
            diff = cv2.absdiff(lo, prev_lo)
            mask_pool.append(diff)
            pooled = mask_pool[0]
            for i in range(1, len(mask_pool)):
                pooled = cv2.bitwise_or(pooled, mask_pool[i])

            # Horizontal path
            band_h = pooled[y0:y1, :]
            col = band_h.astype(np.float32).sum(axis=0)
            if col.sum() >= ENERGY_MIN_FRAC * (255.0 * (y1 - y0) * LO_W):
                col_s = smooth1d(col, SMOOTH_COLS//2)
                xs = np.arange(LO_W, dtype=np.float32)
                wsum = col_s.sum()
                if wsum > 1e-3:
                    cx = float((col_s * xs).sum() / wsum)
                    x_norm = cx / LO_W
                    bar = (col_s / (col_s.max() + 1e-6) * 255.0).astype(np.uint8)
                    bar = np.tile(bar, (40,1))
                    bar = cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)
                    bar = cv2.resize(bar, (FRAME_W, 40))
                    dbg[0:40, 0:FRAME_W] = bar
                    cv2.line(dbg, (int(x_norm*FRAME_W), 40), (int(x_norm*FRAME_W), 70), (255,255,255), 2)

            # Vertical path
            band_v = pooled[:, x0:x1]
            row = band_v.astype(np.float32).sum(axis=1)
            if row.sum() >= ENERGY_MIN_FRAC * (255.0 * (x1 - x0) * LO_H):
                row_s = smooth1d(row, SMOOTH_ROWS//2)
                ys = np.arange(LO_H, dtype=np.float32)
                wsum = row_s.sum()
                if wsum > 1e-3:
                    cy = float((row_s * ys).sum() / wsum)
                    y_norm = cy / LO_H
                    barv = (row_s / (row_s.max() + 1e-6) * 255.0).astype(np.uint8)
                    barv = np.tile(barv[:, None], (1, 40))
                    barv = cv2.cvtColor(barv, cv2.COLOR_GRAY2BGR)
                    barv = cv2.resize(barv, (40, FRAME_H))
                    dbg[0:FRAME_H, FRAME_W-40:FRAME_W] = barv
                    cv2.line(dbg, (FRAME_W-40, int(y_norm*FRAME_H)), (FRAME_W, int(y_norm*FRAME_H)), (255,255,255), 2)

        prev_lo = lo

        # Windows
        while trace_x and (now - trace_x[0][0]) > SPAN_WINDOW_S: trace_x.popleft()
        while trace_y and (now - trace_y[0][0]) > SPAN_WINDOW_S: trace_y.popleft()
        if x_norm is not None: trace_x.append((now, x_norm)); last_seen_t_x = now
        if y_norm is not None: trace_y.append((now, y_norm)); last_seen_t_y = now

        # Spans / velocities
        span_x = vel_x = 0.0
        if len(trace_x) >= 2:
            xs = [p[1] for p in trace_x]; ts = [p[0] for p in trace_x]
            span_x = max(xs) - min(xs)
            dt = max(1e-3, ts[-1] - ts[0]); vel_x = (xs[-1] - xs[0]) / dt

        span_y = vel_y = 0.0
        if len(trace_y) >= 2:
            ys = [p[1] for p in trace_y]; ts2 = [p[0] for p in trace_y]
            span_y = max(ys) - min(ys)
            dt2 = max(1e-3, ts2[-1] - ts2[0]); vel_y = (ys[-1] - ys[0]) / dt2

        # Decide swipes
        gesture_text = ""
        can_fire = (now - last_fire) > COOLDOWN_S

        # Horizontal gates
        if x_norm is not None:
            if state_x == "IDLE":
                if x_norm <= L_ARM: state_x = "ARM_RIGHT"
                elif x_norm >= R_ARM: state_x = "ARM_LEFT"
            if can_fire:
                if state_x == "ARM_RIGHT" and x_norm >= R_FIRE:
                    gesture_text = "SWIPE_RIGHT"; last_fire = now; state_x = "IDLE"; trace_x.clear()
                    print("SWIPE_RIGHT")
                    start_capture_thread("swipe_right")
                elif state_x == "ARM_LEFT" and x_norm <= L_FIRE:
                    gesture_text = "SWIPE_LEFT";  last_fire = now; state_x = "IDLE"; trace_x.clear()
                    print("SWIPE_LEFT")
                    start_capture_thread("swipe_left")
        else:
            if state_x != "IDLE" and (now - last_seen_t_x) > ABSENCE_RESET_S:
                state_x = "IDLE"

        # Vertical gates (no action bound for now—kept for display)
        if y_norm is not None and gesture_text == "":
            if state_y == "IDLE":
                if y_norm <= T_ARM: state_y = "ARM_DOWN"
                elif y_norm >= B_ARM: state_y = "ARM_UP"
            if can_fire:
                if state_y == "ARM_DOWN" and y_norm >= B_FIRE:
                    gesture_text = "SWIPE_DOWN"; last_fire = now; state_y = "IDLE"; trace_y.clear()
                    print("SWIPE_DOWN")
                elif state_y == "ARM_UP" and y_norm <= T_FIRE:
                    gesture_text = "SWIPE_UP";   last_fire = now; state_y = "IDLE"; trace_y.clear()
                    print("SWIPE_UP")
        else:
            if state_y != "IDLE" and (now - last_seen_t_y) > ABSENCE_RESET_S:
                state_y = "IDLE"

        # Span + velocity fallbacks (horizontal only for actions)
        if can_fire and gesture_text == "" and span_x >= SPAN_THR_X and abs(vel_x) >= VEL_THR_X:
            if vel_x > 0:
                print("SWIPE_RIGHT (span/vel)")
                start_capture_thread("swipe_right")
                gesture_text = "SWIPE_RIGHT"
            else:
                print("SWIPE_LEFT (span/vel)")
                start_capture_thread("swipe_left")
                gesture_text = "SWIPE_LEFT"
            last_fire = now; state_x = "IDLE"; trace_x.clear()

        # HUD
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.rectangle(dbg, (0, int(ROI_Y0*FRAME_H)), (FRAME_W, int(ROI_Y1*FRAME_H)), (255,255,255), 1)
        cv2.line(dbg, (0, int(CROSS_T*FRAME_H)), (FRAME_W, int(CROSS_T*FRAME_H)), (255,255,255), 1)
        cv2.line(dbg, (0, int(CROSS_B*FRAME_H)), (FRAME_W, int(CROSS_B*FRAME_H)), 1)
        cv2.rectangle(dbg, (int(ROI_X0*FRAME_W), 0), (int(ROI_X1*FRAME_W), FRAME_H), (255,255,255), 1)

        x_txt = f"x={trace_x[-1][1]:.2f}" if trace_x else "x=--"
        y_txt = f"y={trace_y[-1][1]:.2f}" if trace_y else "y=--"
        if gesture_text:
            cv2.putText(dbg, gesture_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(dbg, f"{x_txt}  STATE_X={state_x}  spanX={span_x:.2f} velX={vel_x:.2f}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(dbg, f"{y_txt}  STATE_Y={state_y}  spanY={span_y:.2f} velY={vel_y:.2f}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Swipe (ultrafast LR+UD) + HQ capture", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.002)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
    # stop worker
    try:
        work_q.put_nowait(None)
    except queue.Full:
        pass
