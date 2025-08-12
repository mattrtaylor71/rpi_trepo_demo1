import time, os, collections, threading, queue, base64, datetime
import numpy as np
import cv2
from picamera2 import Picamera2

# ─────────────────────────────────────────────────────────────────────────────
# EPAPER UI (2.13" tri-color V4). Non-blocking, background-rendered.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# EPAPER UI (2.13" tri-color V4) — no flash except when going to MAIN
# ─────────────────────────────────────────────────────────────────────────────
import threading as _thr
from queue import Queue as _Q
try:
    from waveshare_epd import epd2in13b_V4 as _EPD
    from PIL import Image as _Image, ImageDraw as _Draw, ImageFont as _Font
except Exception:
    _EPD = None
    _Image = _Draw = _Font = None

class EpaperUI:
    """
    - Full refresh ONLY for main screen (and on first boot).
    - Mode + captured screens use partial refresh without re-init’ing each time.
    - No `Clear()` outside of boot.
    """

    def __init__(self):
        self.enabled = (_EPD is not None and _Image is not None)
        self.epd = None
        self.W = self.H = None
        self.font_big = self.font_md = self.font_sm = None
        self.q = _Q(maxsize=8)
        self.cur_mode = None
        self.last_screen = None
        self.monochrome = True

        # runtime state
        self._partial_supported = False
        self._lut = None              # 'full' | 'partial' | None
        self.prevB = None
        self.prevR = None

        self._worker = _thr.Thread(target=self._run, daemon=True)
        if self.enabled:
            self._worker.start()

    # PUBLIC API
    def show_main(self):            self._post(("main", None))
    def show_mode_prompt(self, m):  self.cur_mode = m; self._post(("mode", m))
    def show_captured(self, m, t):  self.cur_mode = m; self._post(("captured", (m, t)))
    def show_timeout(self):         self.cur_mode = None; self._post(("timeout", None))

    # INTERNALS
    def _post(self, msg):
        if not self.enabled: return
        if msg == self.last_screen: return
        try:
            self.q.put_nowait(msg)
            self.last_screen = msg
        except: pass

    def _safe_init(self):
        if self.epd is not None: return
        try:
            self.epd = _EPD.EPD()
            self.epd.init()          # full LUT at boot
            self.epd.Clear()         # ONLY here

            # Panel coordinates
            self.baseW, self.baseH = self.epd.width, self.epd.height
            if self.baseH > self.baseW:
                self.W, self.H = self.baseH, self.baseW
                self.rotate_deg = 90
            else:
                self.W, self.H = self.baseW, self.baseH
                self.rotate_deg = 0
            self.rotate_180 = False

            # Fonts
            base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            font1 = os.path.join(base, "font", "Font.ttc")
            if os.path.exists(font1):
                self.font_big = _Font.truetype(font1, 26)
                self.font_md  = _Font.truetype(font1, 18)
                self.font_sm  = _Font.truetype(font1, 14)
            else:
                f = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                self.font_big = _Font.truetype(f, 26)
                self.font_md  = _Font.truetype(f, 18)
                self.font_sm  = _Font.truetype(f, 14)

            # detect partial support once
            self._partial_supported = any([
                hasattr(self.epd, "displayPartial"),
                hasattr(self.epd, "display_Partial"),
                hasattr(self.epd, "DisplayPartial"),
                hasattr(self.epd, "init_fast"),
                hasattr(self.epd, "Init_Partial"),
                hasattr(self.epd, "init_Partial"),
            ])
            self._lut = 'full'  # we just did a full init
        except Exception as e:
            print(f"[EPD] init failed: {e}")
            self.enabled = False

    def _run(self):
        while True:
            msg = self.q.get()
            try:
                self._safe_init()
                if not self.enabled:
                    continue
                kind, payload = msg
                if kind == "main":
                    self._draw_main()       # FULL
                elif kind == "mode":
                    self._draw_mode(payload)  # PARTIAL
                elif kind == "captured":
                    m, ok = payload
                    self._draw_captured(m, ok)   # PARTIAL
                elif kind == "timeout":
                    self._draw_main()       # FULL
            except Exception as e:
                print(f"[EPD] worker error: {e}")
            finally:
                self.q.task_done()

    # DRAW HELPERS
    def _new_layers(self):
        black = _Image.new('1', (self.W, self.H), 255)
        red   = _Image.new('1', (self.W, self.H), 255)
        db = _Draw.Draw(black)
        dr = db if getattr(self, "monochrome", False) else _Draw.Draw(red)
        return black, red, db, dr

    def _rotate_for_panel(self, img):
        img_out = img
        if getattr(self, "rotate_deg", 0) in (90, 270):
            img_out = img_out.rotate(self.rotate_deg, expand=True)
        elif getattr(self, "rotate_deg", 0) == 180:
            img_out = img_out.rotate(180)
        if getattr(self, "rotate_180", False):
            img_out = img_out.rotate(180)
        return img_out

    # LUT mode switchers — never re-init more than necessary
    def _enter_full(self):
        if self._lut == 'full': return
        try:
            self.epd.init()
            self._lut = 'full'
        except Exception as e:
            print(f"[EPD] enter_full failed: {e}")

    def _enter_partial(self):
        if self._lut == 'partial': return
        try:
            if   hasattr(self.epd, "init_fast"):     self.epd.init_fast()
            elif hasattr(self.epd, "Init_Partial"):  self.epd.Init_Partial()
            elif hasattr(self.epd, "init_Partial"):  self.epd.init_Partial()
            else:                                    self.epd.init()  # fallback
            self._lut = 'partial'
        except Exception as e:
            print(f"[EPD] enter_partial failed: {e}")

    def _buffers_equal(self, a, b):
        if a is None or b is None: return False
        try:
            return a.tobytes() == b.tobytes()
        except Exception:
            return False

    def _push_full(self, black, red):
        self._enter_full()
        imgB = self._rotate_for_panel(black)
        imgR = self._rotate_for_panel(red)
        # Skip if identical (rare for main, but cheap)
        if self._buffers_equal(imgB, self.prevB) and self._buffers_equal(imgR, self.prevR):
            return
        try:
            self.epd.display(self.epd.getbuffer(imgB), self.epd.getbuffer(imgR))
        except TypeError:
            self.epd.display(self.epd.getbuffer(imgB))
        self.prevB, self.prevR = imgB, imgR

    def _push_partial(self, black, red):
        self._enter_partial()
        imgB = self._rotate_for_panel(black)
        imgR = self._rotate_for_panel(red)
        if self._buffers_equal(imgB, self.prevB) and self._buffers_equal(imgR, self.prevR):
            return  # no-op; avoid any flash

        pushed = False
        try:
            if   hasattr(self.epd, "displayPartial"):
                try:
                    self.epd.displayPartial(self.epd.getbuffer(imgB), self.epd.getbuffer(imgR))
                except TypeError:
                    self.epd.displayPartial(self.epd.getbuffer(imgB))
                pushed = True
            elif hasattr(self.epd, "display_Partial"):
                self.epd.display_Partial(self.epd.getbuffer(imgB))
                pushed = True
            elif hasattr(self.epd, "DisplayPartial"):
                self.epd.DisplayPartial(self.epd.getbuffer(imgB))
                pushed = True
        except Exception as e:
            print(f"[EPD] displayPartial failed: {e}")

        if not pushed:
            # Fallback: regular display WITHOUT Clear() and WITHOUT LUT re-init.
            try:
                self.epd.display(self.epd.getbuffer(imgB), self.epd.getbuffer(imgR))
            except TypeError:
                self.epd.display(self.epd.getbuffer(imgB))

        self.prevB, self.prevR = imgB, imgR

    # SCREENS
    def _draw_main(self):
        b, r, db, dr = self._new_layers()
        db.text((8, 8), "Swipe to choose mode", font=self.font_md, fill=0)
        y = int(self.H * 0.60); arrow_sz = 20
        self._arrow(db, x=int(self.W * 0.30), y=y, size=arrow_sz, direction="left")
        db.text((int(self.W * 0.24), y + 18), "discard", font=self.font_sm, fill=0)
        self._arrow(dr, x=int(self.W * 0.70), y=y, size=arrow_sz, direction="right")
        dr.text((int(self.W * 0.64), y + 18), "check-in", font=self.font_sm, fill=0)
        db.rectangle((0, 0, self.W - 1, self.H - 1), outline=0, width=1)
        self._push_full(b, r)  # <- full refresh only here

    def _draw_mode(self, mode):
        b, r, db, dr = self._new_layers()
        title = "Mode: " + {"discard":"DISCARD","check_in":"CHECK-IN","opened":"OPENED","other":"OTHER"}.get(mode, mode or "--").upper()
        db.text((8, 8), title, font=self.font_big, fill=0)
        prompt = "Hold item still ~1ft from camera"
        (dr if mode == "check_in" else db).text((8, 36), prompt, font=self.font_md, fill=0)
        y = int(self.H * 0.60); arrow_sz = 20
        if mode == "discard":
            self._arrow(db, x=int(self.W * 0.30), y=y, size=arrow_sz, direction="left")
        elif mode == "check_in":
            self._arrow(dr, x=int(self.W * 0.70), y=y, size=arrow_sz, direction="right")
        else:
            db.rectangle((int(self.W*0.45), y-10, int(self.W*0.55), y+10), outline=0, width=2)
        db.rectangle((0, 0, self.W - 1, self.H - 1), outline=0, width=1)
        self._push_partial(b, r)  # <- partial (no flash)

    def _draw_captured(self, mode, ok_text):
        b, r, db, dr = self._new_layers()
        title = "Mode: " + {"discard":"DISCARD","check_in":"CHECK-IN","opened":"OPENED","other":"OTHER"}.get(mode, mode or "--").upper()
        db.text((8, 8), title, font=self.font_big, fill=0)
        x0, y0, x1, y1 = 8, 36, self.W - 8, 76
        dr.rectangle((x0, y0, x1, y1), outline=0, fill=255)
        dr.text((x0 + 10, y0 + 6), ok_text, font=self.font_md, fill=0)
        db.text((8, 84), "Hold item still ~1ft from camera", font=self.font_md, fill=0)
        y = int(self.H * 0.60); arrow_sz = 20
        if mode == "discard":
            self._arrow(db, x=int(self.W * 0.30), y=y, size=arrow_sz, direction="left")
        elif mode == "check_in":
            self._arrow(dr, x=int(self.W * 0.70), y=y, size=arrow_sz, direction="right")
        db.rectangle((0, 0, self.W - 1, self.H - 1), outline=0, width=1)
        self._push_partial(b, r)  # <- partial (no flash)

    def _arrow(self, draw, x, y, size=24, direction="left"):
        s = size
        if direction == "left":
            draw.polygon([(x-s, y), (x, y-s), (x, y+s)], outline=0, fill=0)
            draw.rectangle((x, y-4, x+s, y+4), outline=0, fill=0)
        elif direction == "right":
            draw.polygon([(x+s, y), (x, y-s), (x, y+s)], outline=0, fill=0)
            draw.rectangle((x-s, y-4, x, y+4), outline=0, fill=0)


# Create a singleton UI (safe even if epaper libs absent)
EPD_UI = EpaperUI()

# --------------------------
# OpenAI API (vision)
# --------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

work_q = queue.Queue(maxsize=4)

# --------------------------
# OpenAI API (vision) + worker
# --------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

work_q = queue.Queue(maxsize=4)
WORKER_ALIVE = False  # set True once the worker enters its loop

def api_worker():
    global WORKER_ALIVE
    if OpenAI is None:
        print("[OpenAI] Worker not started: openai SDK not importable.")
        return
    if not OPENAI_API_KEY:
        print("[OpenAI] Worker not started: OPENAI_API_KEY is empty.")
        return
    if OPENAI_MODEL.startswith("gpt-5"):
        print("[OpenAI][WARN] gpt-5 with chat.completions may fail. Use Responses API or gpt-4o/4o-mini.")

    print(f"[OpenAI] Worker starting. model={OPENAI_MODEL}")
    client = OpenAI(api_key=OPENAI_API_KEY)
    WORKER_ALIVE = True

    while True:
        item = work_q.get()
        if item is None:
            print("[OpenAI] Worker shutting down.")
            break
        tag, jpeg_bytes = item
        try:
            print(f"[OpenAI] Dequeued tag='{tag}', size={len(jpeg_bytes)} bytes (qsize after dequeue={work_q.qsize()})")
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            msgs = [
                {"role":"system","content":"You are an expert product identifier. Be concise and name the item if possible."},
                {"role":"user","content":[
                    {"type":"text","text":f"Identify the object. (mode={tag})"},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]},
            ]
            t0 = time.perf_counter()
            print(f"[OpenAI] -> chat.completions.create (model={OPENAI_MODEL}, tag='{tag}')")
            resp = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
            dt_ms = (time.perf_counter() - t0) * 1000
            print(f"[OpenAI] <- response (tag='{tag}', {dt_ms:.0f} ms)")
            text = (resp.choices[0].message.content or "").strip()
            print(f"[{tag}] Vision -> {text if text else '<empty content>'}")
        except Exception as e:
            print(f"[OpenAI ERROR] tag='{tag}': {type(e).__name__}: {e}")
        finally:
            work_q.task_done()

def enqueue_openai(tag, rgb_frame):
    os.makedirs("captures", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"captures/{tag}_{ts}.jpg"
    jpg = jpeg_bytes_from_rgb(rgb_frame, 92)
    if jpg is None:
        print(f"[enqueue] {tag}: JPEG encode failed")
        return
    with open(path, "wb") as f:
        f.write(jpg)
    print(f"[enqueue] {tag}: saved {path} ({len(jpg)} bytes)")
    if not WORKER_ALIVE:
        print("[enqueue][WARN] OpenAI worker not running. Check OPENAI_API_KEY and import. Skipping enqueue.")
        return
    try:
        work_q.put_nowait((tag, jpg))
        print(f"[enqueue] {tag}: enqueued. qsize={work_q.qsize()}")
    except queue.Full:
        print(f"[enqueue] {tag}: queue full, skipped")

# spin up worker (do this once after defs)
threading.Thread(target=api_worker, daemon=True).start()

# Optional: tiny watchdog to show queue health every ~5s
def q_watchdog():
    while True:
        time.sleep(5)
        print(f"[q-watch] alive={WORKER_ALIVE} qsize={work_q.qsize()}")
threading.Thread(target=q_watchdog, daemon=True).start()

EPD_UI.show_main()

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
POOL_FRAMES = 3
HEADLESS = os.environ.get("DISPLAY", "") == ""

CAPTURE_COOLDOWN_S = 0.8
last_capture_t = 0.0

# =========================
# Stability / presence gating after a mode is set
# =========================
WAIT_AFTER_SWIPE_S   = 0.9
STABILITY_WINDOW_FR  = 8
MOTION_EMA_ALPHA     = 0.15
MOTION_THR_SCALE     = 2.3
MOTION_THR_FLOOR     = 0.004
PRESENCE_LAPLACE_MIN = 28.0
PRESENCE_LAPLACE_GAIN = 1.3

ARM_TIMEOUT_S         = 8.0
MOTION_EMA_ALPHA      = 0.25

# Stability hysteresis / dwell
ENTER_RELAX = 1.00
EXIT_RELAX  = 1.20
MIN_STABLE_S = 0.35
CONFIRM_FR   = 2

stable_since = None
confirm_left = 0

# Require removal (low-detail) before re-arming
CLEAR_LAPLACE_FRAC = 0.65
CLEAR_WINDOW_FR    = 6

need_clear  = False
clear_count = 0

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

FLIP_X = True   # flips left/right interpretation
FLIP_Y = False  # set True if up/down feel reversed

def set_mode_from(gesture: str, now_ts: float, bgr_for_baseline=None):
    global current_mode, armed, arm_time, stable_count
    global motion_thr_dyn, lap_baseline, lap_thr_dyn
    global need_clear
    m = MODE_MAP.get(gesture)
    if not m: return
    current_mode = m
    armed = True
    need_clear = False
    arm_time = now_ts
    stable_count = 0
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
                    cx_raw = float((col_s * xs).sum() / wsum) / LO_W
                    x_norm = (1.0 - cx_raw) if FLIP_X else cx_raw

                    # debug bar across the top
                    bar = (col_s / (col_s.max()+1e-6) * 255.0).astype(np.uint8)
                    if FLIP_X:
                        bar = bar[::-1]  # mirror the debug histogram too
                    dbg[0:40, 0:FRAME_W] = cv2.resize(
                        cv2.cvtColor(np.tile(bar, (40, 1)), cv2.COLOR_GRAY2BGR),
                        (FRAME_W, 40)
                    )
                    cv2.line(dbg, (int(x_norm*FRAME_W), 40), (int(x_norm*FRAME_W), 70), (255,255,255), 2)

            # Vertical (rows) in horizontal band
            band_v = pooled[:, x0:x1]
            row = band_v.astype(np.float32).sum(axis=1)
            if row.sum() >= ENERGY_MIN_FRAC * (255.0 * (x1 - x0) * LO_H):
                row_s = smooth1d(row, SMOOTH_ROWS//2)
                ys = np.arange(LO_H, dtype=np.float32)
                wsum = row_s.sum()
                if wsum > 1e-3:
                    cy_raw = float((row_s * ys).sum() / wsum) / LO_H
                    y_norm = (1.0 - cy_raw) if FLIP_Y else cy_raw

                    # debug bar on the right edge
                    barv = (row_s / (row_s.max()+1e-6) * 255.0).astype(np.uint8)
                    barv = np.tile(barv[:, None], (1, 40))
                    if FLIP_Y:
                        barv = barv[::-1, :]
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
        SPAN_WINDOW_S = 0.28
        now_ts = now

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
            EPD_UI.show_mode_prompt(current_mode)
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
            if (now - last_capture_t) < CAPTURE_COOLDOWN_S:
                stable_count = 0
            elif (now - arm_time) > ARM_TIMEOUT_S:
                print("[mode] timeout; disarming without capture")
                EPD_UI.show_timeout()
                armed = False
            elif (now - arm_time) < WAIT_AFTER_SWIPE_S:
                stable_count = 0
            else:
                lap_c = center_laplacian(bgr)

                thr_enter = motion_thr_dyn * ENTER_RELAX
                thr_exit  = motion_thr_dyn * EXIT_RELAX

                below_enter = (motion_ema is not None) and (motion_ema < thr_enter)
                above_exit  = (motion_ema is not None) and (motion_ema > thr_exit)
                sharp_enough = (lap_c >= lap_thr_dyn)

                if above_exit or not sharp_enough:
                    stable_count = 0
                    stable_since = None
                    confirm_left = 0
                elif below_enter and sharp_enough:
                    stable_count += 1
                    if stable_since is None:
                        stable_since = now
                    if (now - stable_since) >= MIN_STABLE_S and stable_count >= STABILITY_WINDOW_FR:
                        if confirm_left == 0:
                            confirm_left = CONFIRM_FR
                        else:
                            confirm_left -= 1
                            if confirm_left == 0:
                                tag = current_mode or "unknown_mode"
                                print(f"[mode] stable -> capturing ({tag})  motion={motion_ema:.4f} thr={motion_thr_dyn:.4f} lap={lap_c:.1f} thr={lap_thr_dyn:.1f}")
                                start_capture_thread(tag)
                                EPD_UI.show_captured(tag, "checked in!" if tag == "check_in" else "discarded!")
                                last_capture_t = now
                                arm_time = now
                                stable_count = 0
                                stable_since = None

                                need_clear = True
                                armed = False
                                clear_count = 0
                                print("[mode] captured; waiting for item removal to re-arm")
                else:
                    confirm_left = 0

                # HUD (optional)
                closeness = 1.0 - np.clip((motion_ema or 0.0) / (motion_thr_dyn or 1e-6), 0.0, 1.0)
                cv2.rectangle(dbg, (20, 50), (20 + int(200*closeness), 65), (255,255,255), -1)
                cv2.putText(dbg, f"STABLE {stable_count}/{STABILITY_WINDOW_FR} dwell>={MIN_STABLE_S:.2f}s lap={int(lap_c)}>={int(lap_thr_dyn)}",
                            (230, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Re-arm when item is removed (center detail goes low for a few frames)
        if need_clear:
            lap_c = center_laplacian(bgr)
            clear_thr = max(PRESENCE_LAPLACE_MIN * 0.8, lap_thr_dyn * CLEAR_LAPLACE_FRAC)
            if lap_c < clear_thr:
                clear_count += 1
            else:
                clear_count = 0

            if clear_count >= CLEAR_WINDOW_FR:
                need_clear = False
                armed = True
                arm_time = now
                print("[mode] scene cleared; re-armed for next item")

            cv2.putText(dbg, "REMOVE ITEM", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

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
