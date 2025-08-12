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
    OpenAI = None

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # or "gpt-5"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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
                {"role":"system","content":"You are an expert product identifier. Be concise and name the item if possible."},
                {"role":"user","content":[
                    {"type":"text","text":f"Identify the object. (mode={tag})"},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]},
            ]
            resp = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
            print(f"[{tag}] Vision -> {resp.choices[0].message.content}")
        except Exception as e:
            print(f"[{tag}] OpenAI error: {e}")
        finally:
            work_q.task_done()

threading.Thread(target=api_worker, daemon=True).start()

# =========================
# Swipe detector config
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
POOL_FRAMES = 3                      # a bit more pooling for stability
HEADLESS = os.environ.get("DISPLAY", "") == ""

# =========================
# Stability / presence gating after a mode is set
# =========================
WAIT_AFTER_SWIPE_S    = 0.6        # shorter grace; user presents item
STABILITY_WINDOW_FR   = 5          # fewer frames to accept
MOTION_THR_FLOOR      = 0.0025     # absolute floor for motion (normalized)
MOTION_THR_SCALE      = 1.8        # dynamic: threshold = max(floor, EMA * scale)
PRESENCE_LAPLACE_MIN  = 18.0       # minimum edge detail
PRESENCE_LAPLACE_GAIN = 1.3        # dynamic: lap_thresh = max(min, baseline*gain)
ARM_TIMEOUT_S         = 8.0        # stop waiting if nothing happens
MOTION_EMA_ALPHA      = 0.25       # EMA smoothing for motion

# =========================
# Helpers
# =========================
def y_plane(yuv):
    if yuv.ndim == 2:  return yuv[:LO_H, :LO_W]
    if yuv.ndim == 3:  return yuv[:, :, 0]
    raise ValueError(f"Unexpected lores shape {yuv.shape}")

def smooth1d(v, k):
    if k <= 0: return v
    ksz = 2*k + 1
    return np.convolve(v, np.ones(ksz, np.float32)/ksz, mode="same")

def jpeg_bytes_from_rgb(rgb, quality=92):
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return jpg.tobytes() if ok else None

def enqueue_openai(tag, rgb_frame):
    os.makedirs("captures", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"captures/{tag}_{ts}.jpg"
    jpg = jpeg_bytes_from_rgb(rgb_frame, 92)
    if jpg is None:
        print(f"[{tag}] JPEG encode failed"); return
    with open(path, "wb") as f: f.write(jpg)
    print(f"[{tag}] saved {path}")
    try: work_q.put_nowait((tag, jpg))
    except queue.Full: print(f"[{tag}] queue full; skipping upload")

def laplacian_sharpness(gray_u8):
    return cv2.Laplacian(gray_u8, cv2.CV_64F).var()

def center_laplacian(bgr):
    h, w = bgr.shape[:2]
    cx0, cy0 = int(0.25*w), int(0.25*h)
    cx1, cy1 = int(0.75*w), int(0.75*h)
    crop = cv2.cvtColor(bgr[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
    return laplacian_sharpness(crop)

# =========================
# Camera (video + still modes)
# =========================
picam2 = Picamera2()

video_cfg = picam2.create_video_configuration(
    main={"format":"RGB888","size":(FRAME_W, FRAME_H)},
    lores={"format":"YUV420","size":(LO_W, LO_H)},
    display="main",
)

STILL_W, STILL_H = 2304, 1296
still_cfg = picam2.create_still_configuration(
    main={"format":"RGB888","size":(STILL_W, STILL_H)},
    display=None,
)

picam2.configure(video_cfg)
try:
    picam2.set_controls({
        "AeEnable": True, "AwbEnable": True,
        "AfMode": 2,
        "FrameDurationLimits": (10000, 10000),  # fast preview
        "AnalogueGain": 12.0,
    })
except Exception:
    pass
picam2.start()

cam_lock = threading.Lock()  # guards mode switches and frame grabs

def capture_high_quality(tag: str):
    with cam_lock:
        try:
            try:
                picam2.set_controls({"AfMode": 1})       # single-shot AF
                picam2.set_controls({"AfTrigger": 1})
            except Exception:
                picam2.set_controls({"AfMode": 2})

            picam2.switch_mode(still_cfg)
            time.sleep(0.22)  # AE/AF settle

            frames, scores = [], []
            for _ in range(4):
                arr = picam2.capture_array("main")
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                frames.append(arr)
                scores.append(laplacian_sharpness(gray))
                time.sleep(0.05)
            best = frames[int(np.argmax(scores))]
            print(f"[{tag}] sharpness: {[round(s,1) for s in scores]}")
            enqueue_openai(tag, best)
        except Exception as e:
            print(f"[{tag}] capture error: {e}")
        finally:
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

def start_capture_thread(tag):
    threading.Thread(target=capture_high_quality, args=(tag,), daemon=True).start()

# =========================
# Swipe + Mode state
# =========================
prev_lo_blur = None
mask_pool = collections.deque(maxlen=POOL_FRAMES)
trace_x, trace_y = collections.deque(), collections.deque()
state_x = state_y = "IDLE"
last_fire = 0.0
last_seen_t_x = last_seen_t_y = 0.0

y0, y1 = int(ROI_Y0 * LO_H), int(ROI_Y1 * LO_H)
x0, x1 = int(ROI_X0 * LO_W), int(ROI_X1 * LO_W)

MODE_MAP = {
    "SWIPE_RIGHT": "check_in",
    "SWIPE_LEFT":  "discard",
    "SWIPE_UP":    "opened",
    "SWIPE_DOWN":  "other",
}
current_mode = None
armed = False
arm_time = 0.0
stable_count = 0

# Adaptive stability thresholds
motion_ema = None
motion_thr_dyn = MOTION_THR_FLOOR
lap_baseline = 0.0
lap_thr_dyn  = PRESENCE_LAPLACE_MIN

def set_mode_from(gesture: str, now_ts: float, bgr_for_baseline=None):
    """Set mode, arm stability, and seed dynamic thresholds from current scene."""
    global current_mode, armed, arm_time, stable_count
    global motion_thr_dyn, lap_baseline, lap_thr_dyn
    m = MODE_MAP.get(gesture)
    if not m: return
    current_mode = m
    armed = True
    arm_time = now_ts
    stable_count = 0
    # derive laplacian baseline from current frame's center crop
    if bgr_for_baseline is not None:
        lap = center_laplacian(bgr_for_baseline)
        lap_baseline = lap
        lap_thr_dyn  = max(PRESENCE_LAPLACE_MIN, lap * PRESENCE_LAPLACE_GAIN)
    print(f"[mode] {current_mode} (armed)  lap_base={lap_baseline:.1f}  lap_thr={lap_thr_dyn:.1f}")

try:
    while True:
        now = time.time()

        # Skip reads during still capture
        if not cam_lock.acquire(blocking=False):
            time.sleep(0.005)
            continue
        try:
            lo  = y_plane(picam2.capture_array("lores"))
            rgb = picam2.capture_array("main")
        finally:
            cam_lock.release()

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        x_norm = y_norm = None

        # --- Preprocess lores for robust motion: blur + diff + pool ---
        lo_blur = cv2.GaussianBlur(lo, (3,3), 0)

        if prev_lo_blur is not None:
            diff = cv2.absdiff(lo_blur, prev_lo_blur)
            mask_pool.append(diff)
            pooled = mask_pool[0]
            for i in range(1, len(mask_pool)):
                pooled = cv2.bitwise_or(pooled, mask_pool[i])

            # Horizontal (columns) in vertical band
            band_h = pooled[y0:y1, :]
            col = band_h.astype(np.float32).sum(axis=0)
            if col.sum() >= ENERGY_MIN_FRAC * (255.0 * (y1 - y0) * LO_W):
                col_s = smooth1d(col, SMOOTH_COLS//2)
                xs = np.arange(LO_W, dtype=np.float32)
                wsum = col_s.sum()
                if wsum > 1e-3:
                    cx = float((col_s * xs).sum() / wsum)
                    x_norm = cx / LO_W
                    bar = (col_s / (col_s.max()+1e-6) * 255.0).astype(np.uint8)
                    dbg[0:40, 0:FRAME_W] = cv2.resize(cv2.cvtColor(np.tile(bar,(40,1)), cv2.COLOR_GRAY2BGR),(FRAME_W,40))
                    cv2.line(dbg, (int(x_norm*FRAME_W), 40), (int(x_norm*FRAME_W), 70), (255,255,255), 2)

            # Vertical (rows) in horizontal band
            band_v = pooled[:, x0:x1]
            row = band_v.astype(np.float32).sum(axis=1)
            if row.sum() >= ENERGY_MIN_FRAC * (255.0 * (x1 - x0) * LO_H):
                row_s = smooth1d(row, SMOOTH_ROWS//2)
                ys = np.arange(LO_H, dtype=np.float32)
                wsum = row_s.sum()
                if wsum > 1e-3:
                    cy = float((row_s * ys).sum() / wsum)
                    y_norm = cy / LO_H
                    barv = (row_s / (row_s.max()+1e-6) * 255.0).astype(np.uint8)
                    barv = np.tile(barv[:,None], (1,40))
                    barv = cv2.cvtColor(barv, cv2.COLOR_GRAY2BGR)
                    barv = cv2.resize(barv, (40, FRAME_H))
                    dbg[0:FRAME_H, FRAME_W-40:FRAME_W] = barv
                    cv2.line(dbg, (FRAME_W-40, int(y_norm*FRAME_H)), (FRAME_W, int(y_norm*FRAME_H)), (255,255,255), 2)

            # --- Motion in the central band only (less background noise) ---
            center_band = pooled[int(LO_H*0.20):int(LO_H*0.80), int(LO_W*0.20):int(LO_W*0.80)]
            motion_now = float(center_band.sum()) / (255.0 * center_band.size)

            # EMA smoothing
            if motion_ema is None:
                motion_ema = motion_now
            else:
                motion_ema = MOTION_EMA_ALPHA * motion_now + (1.0 - MOTION_EMA_ALPHA) * motion_ema

            # show motion bar
            m_norm = np.clip(motion_ema / 0.02, 0.0, 1.0)  # visualize vs 2% motion
            cv2.rectangle(dbg, (20, 50), (20 + int(200*(1.0 - m_norm)), 65), (255,255,255), -1)
            cv2.putText(dbg, f"MOTION ema={motion_ema:.4f} thr={motion_thr_dyn:.4f}",
                        (230, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        prev_lo_blur = lo_blur

        # ---- Maintain swipe traces for gates/fallbacks ----
        # (same as before)
        # Keep short windows for LR/UD swipes
        SPAN_WINDOW_S = 0.28
        # Deques exist already: trace_x / trace_y
        now_ts = now  # alias

        # Compute x_norm / y_norm appended for span/vel
        # (We already set x_norm / y_norm above)
        while trace_x and (now - trace_x[0][0]) > SPAN_WINDOW_S: trace_x.popleft()
        while trace_y and (now - trace_y[0][0]) > SPAN_WINDOW_S: trace_y.popleft()
        if x_norm is not None: trace_x.append((now, x_norm)); last_seen_t_x = now
        if y_norm is not None: trace_y.append((now, y_norm)); last_seen_t_y = now

        span_x = vel_x = 0.0
        if len(trace_x) >= 2:
            xs = [p[1] for p in trace_x]; ts = [p[0] for p in trace_x]
            span_x = max(xs) - min(xs); dt = max(1e-3, ts[-1]-ts[0]); vel_x = (xs[-1]-xs[0])/dt
        span_y = vel_y = 0.0
        if len(trace_y) >= 2:
            ys = [p[1] for p in trace_y]; ts2 = [p[0] for p in trace_y]
            span_y = max(ys) - min(ys); dt2 = max(1e-3, ts2[-1]-ts2[0]); vel_y = (ys[-1]-ys[0])/dt2

        # ---- Swipes -> set MODE (no immediate capture) ----
        gesture_text = ""
        can_fire = (now - last_fire) > COOLDOWN_S

        def set_mode_and_seed(gesture: str):
            global motion_thr_dyn
            motion_thr_dyn = max(
                MOTION_THR_FLOOR,
                (motion_ema or MOTION_THR_FLOOR) * MOTION_THR_SCALE
            )
            set_mode_from(gesture, now_ts, bgr_for_baseline=bgr)
            cv2.putText(dbg, "ARMED", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


        # Horizontal gates
        if x_norm is not None:
            if state_x == "IDLE":
                if x_norm <= L_ARM: state_x = "ARM_RIGHT"
                elif x_norm >= R_ARM: state_x = "ARM_LEFT"
            if can_fire:
                if state_x == "ARM_RIGHT" and x_norm >= R_FIRE:
                    gesture_text = "SWIPE_RIGHT"; last_fire = now; state_x = "IDLE"; trace_x.clear()
                    print("SWIPE_RIGHT"); set_mode_and_seed("SWIPE_RIGHT")
                elif state_x == "ARM_LEFT" and x_norm <= L_FIRE:
                    gesture_text = "SWIPE_LEFT";  last_fire = now; state_x = "IDLE"; trace_x.clear()
                    print("SWIPE_LEFT");  set_mode_and_seed("SWIPE_LEFT")
        else:
            if state_x != "IDLE" and (now - last_seen_t_x) > ABSENCE_RESET_S: state_x = "IDLE"

        # Vertical gates
        if y_norm is not None and gesture_text == "":
            if state_y == "IDLE":
                if y_norm <= T_ARM: state_y = "ARM_DOWN"
                elif y_norm >= B_ARM: state_y = "ARM_UP"
            if can_fire:
                if state_y == "ARM_DOWN" and y_norm >= B_FIRE:
                    gesture_text = "SWIPE_DOWN"; last_fire = now; state_y = "IDLE"; trace_y.clear()
                    print("SWIPE_DOWN"); set_mode_and_seed("SWIPE_DOWN")
                elif state_y == "ARM_UP" and y_norm <= T_FIRE:
                    gesture_text = "SWIPE_UP";   last_fire = now; state_y = "IDLE"; trace_y.clear()
                    print("SWIPE_UP");   set_mode_and_seed("SWIPE_UP")
        else:
            if state_y != "IDLE" and (now - last_seen_t_y) > ABSENCE_RESET_S: state_y = "IDLE"

        # Span/velocity fallback
        if can_fire and gesture_text == "" and span_x >= SPAN_THR_X and abs(vel_x) >= VEL_THR_X:
            g = "SWIPE_RIGHT" if vel_x > 0 else "SWIPE_LEFT"
            print(f"{g} (span/vel)"); set_mode_and_seed(g); last_fire = now; state_x = "IDLE"; trace_x.clear()

        # ---------------------------
        # Armed: wait-for-stability (adaptive)
        # ---------------------------
        if armed:
            if (now - arm_time) > ARM_TIMEOUT_S:
                print("[mode] timeout; disarming without capture")
                armed = False
            elif (now - arm_time) < WAIT_AFTER_SWIPE_S:
                stable_count = 0
            else:
                lap_c = center_laplacian(bgr)
                # "stable" if motion EMA below dynamic threshold and enough detail in center
                is_stable = (motion_ema is not None and motion_ema < motion_thr_dyn) and (lap_c >= lap_thr_dyn)
                stable_count = stable_count + 1 if is_stable else 0
                if stable_count >= STABILITY_WINDOW_FR:
                    tag = current_mode or "unknown_mode"
                    print(f"[mode] stable -> capturing ({tag})  motion={motion_ema:.4f} thr={motion_thr_dyn:.4f} lap={lap_c:.1f} thr={lap_thr_dyn:.1f}")
                    start_capture_thread(tag)
                    armed = False

                # HUD for stability
                # bar shows how close we are to motion threshold
                closeness = 1.0 - np.clip((motion_ema or 0.0) / (motion_thr_dyn or 1e-6), 0.0, 1.0)
                cv2.rectangle(dbg, (20, 50), (20 + int(200*closeness), 65), (255,255,255), -1)
                cv2.putText(dbg, f"STABLE {stable_count}/{STABILITY_WINDOW_FR}  lap={int(lap_c)}>={int(lap_thr_dyn)}",
                            (230, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # HUD
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.rectangle(dbg, (0, int(ROI_Y0*FRAME_H)), (FRAME_W, int(ROI_Y1*FRAME_H)), (255,255,255), 1)
        cv2.line(dbg, (0, int(CROSS_T*FRAME_H)), (FRAME_W, int(CROSS_T*FRAME_H)), (255,255,255), 1)
        cv2.line(dbg, (0, int(CROSS_B*FRAME_H)), (FRAME_W, int(CROSS_B*FRAME_H)), (255,255,255), 1)
        cv2.rectangle(dbg, (int(ROI_X0*FRAME_W), 0), (int(ROI_X1*FRAME_W), FRAME_H), (255,255,255), 1)

        mode_txt = current_mode if current_mode else "--"
        cv2.putText(dbg, f"MODE: {mode_txt}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if gesture_text:
            cv2.putText(dbg, gesture_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Swipe -> Mode -> Stable capture (adaptive)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.002)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
    try: work_q.put_nowait(None)
    except queue.Full: pass
