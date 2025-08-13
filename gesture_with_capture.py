import time, os, collections, threading, queue, base64, datetime
import numpy as np
import cv2
from picamera2 import Picamera2

# ─────────────────────────────────────────────────────────────────────────────
# EPAPER UI (2.13" mono B/W V4) — PARTIAL-ONLY AFTER BOOT (no flashing)
# ─────────────────────────────────────────────────────────────────────────────
import threading as _thr
from queue import Queue as _Q
try:
    from waveshare_epd import epd2in13_V4 as _EPD   # ✅ mono driver
    from PIL import Image as _Image, ImageDraw as _Draw, ImageFont as _Font
except Exception:
    _EPD = None
    _Image = _Draw = _Font = None

class EpaperUI:
    HARD_REFRESH_PERIOD_S = 0  # set >0 to occasionally scrub ghosting

    def __init__(self):
        self.enabled = (_EPD is not None and _Image is not None)
        self.epd = None
        self.W = self.H = None
        self.font_big = self.font_md = self.font_sm = None
        self.font_italic = None
        self.q = _Q(maxsize=8)
        self.cur_mode = None
        self.last_screen = None

        self.prev = None           # last pushed PIL 1-bit (post-rotation)
        self.rotate_deg = 0
        self.rotate_180 = False
        self._last_hard = 0.0

        if self.enabled:
            self._worker = _thr.Thread(target=self._run, daemon=True)
            self._worker.start()

    # Public
    def show_main(self):            self._post(("main", None))
    def show_mode_prompt(self, m, frac=1.0):
        self.cur_mode = m
        self._post(("mode", (m, frac)))
    def show_captured(self, m, t, frac=1.0):
        self.cur_mode = m
        self._post(("captured", (m, t, frac)))
    def show_expiry_prompt(self):
        self._post(("expiry", None))
    def show_timeout(self):
        self.cur_mode = None
        self._post(("timeout", None))

    # Internals
    def _post(self, msg):
        if not self.enabled: return
        if msg == self.last_screen: return
        try:
            self.q.put_nowait(msg)
            self.last_screen = msg
        except: pass

    def _safe_init(self):
        if self.epd is not None:
            return
        try:
            self.epd = _EPD.EPD()

            # FULL once to clear (unavoidable flash)
            if hasattr(self.epd, "FULL_UPDATE"):
                self.epd.init(self.epd.FULL_UPDATE)
            else:
                self.epd.init()
            try:
                self.epd.Clear(0xFF)
            except Exception:
                pass

            # Geometry
            self.baseW, self.baseH = getattr(self.epd, "width", 0), getattr(self.epd, "height", 0)
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
            italic_fb = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
            regular_fb = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            if os.path.exists(font1):
                self.font_big = _Font.truetype(font1, 26)
                self.font_md  = _Font.truetype(font1, 18)
                self.font_sm  = _Font.truetype(font1, 14)
                try:
                    self.font_italic = _Font.truetype(font1, 16)
                except Exception:
                    self.font_italic = _Font.truetype(italic_fb, 16) if os.path.exists(italic_fb) else _Font.truetype(regular_fb, 16)
            else:
                self.font_big = _Font.truetype(regular_fb, 26)
                self.font_md  = _Font.truetype(regular_fb, 18)
                self.font_sm  = _Font.truetype(regular_fb, 14)
                self.font_italic = _Font.truetype(italic_fb, 16) if os.path.exists(italic_fb) else _Font.truetype(regular_fb, 16)

            # Enter PARTIAL and set a white base image
            self._enter_partial_mode()
            white = _Image.new('1', (self.W, self.H), 255)
            white = self._rotate(white)
            try:
                self.epd.displayPartBaseImage(self._buf(white))
            except Exception:
                self.epd.displayPartial(self._buf(white))
            self.prev = white
            self._last_hard = time.time()

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

                # Optional scheduled hard refresh
                if self.HARD_REFRESH_PERIOD_S > 0 and (time.time() - self._last_hard) >= self.HARD_REFRESH_PERIOD_S:
                    self._hard_refresh()

                kind, payload = msg
                if kind == "main":            self._draw_main()
                elif kind == "mode":
                    m, frac = payload
                    self._draw_mode(m, frac)
                elif kind == "captured":
                    m, ok, frac = payload
                    self._draw_captured(m, ok, frac)
                elif kind == "expiry":       self._draw_expiry()
                elif kind == "timeout":       self._draw_main()
            except Exception as e:
                print(f"[EPD] worker error: {e}")
            finally:
                self.q.task_done()

    # Low-level helpers
    def _new_layer(self):
        img = _Image.new('1', (self.W, self.H), 255)
        d   = _Draw.Draw(img)
        return img, d

    def _rotate(self, img):
        if self.rotate_deg in (90, 270):
            img = img.rotate(self.rotate_deg, expand=True)
        elif self.rotate_deg == 180:
            img = img.rotate(180)
        if self.rotate_180:
            img = img.rotate(180)
        return img

    def _buf(self, img):
        return self.epd.getbuffer(img)

    def _enter_partial_mode(self):
        try:
            if hasattr(self.epd, "PART_UPDATE"):
                self.epd.init(self.epd.PART_UPDATE)
            elif hasattr(self.epd, "init_fast"):
                self.epd.init_fast()
            else:
                if   hasattr(self.epd, "Init_Partial"):  self.epd.Init_Partial()
                elif hasattr(self.epd, "init_Partial"):  self.epd.init_Partial()
                else:                                     self.epd.init()
        except Exception as e:
            print(f"[EPD] enter PART_UPDATE failed: {e}")

    def _push_partial(self, img):
        out = self._rotate(img)
        if self.prev is not None:
            try:
                if out.tobytes() == self.prev.tobytes():
                    return
            except Exception:
                pass
        try:
            if hasattr(self.epd, "displayPartial"):
                self.epd.displayPartial(self._buf(out))
            elif hasattr(self.epd, "display_Partial"):
                self.epd.display_Partial(self._buf(out))
            else:
                self.epd.display(self._buf(out))
        except Exception as e:
            print(f"[EPD] partial draw failed: {e}")
        self.prev = out

    def _hard_refresh(self):
        try:
            if hasattr(self.epd, "FULL_UPDATE"):
                self.epd.init(self.epd.FULL_UPDATE)
            else:
                self.epd.init()
            white = _Image.new('1', (self.W, self.H), 255)
            white = self._rotate(white)
            self.epd.display(self._buf(white))
            try:
                self.epd.Clear(0xFF)
            except Exception:
                pass
            self._enter_partial_mode()
            try:
                self.epd.displayPartBaseImage(self._buf(white))
            except Exception:
                self.epd.displayPartial(self._buf(white))
            self.prev = white
            self._last_hard = time.time()
        except Exception as e:
            print(f"[EPD] hard refresh failed: {e}")

    # Utility: centered label
    def _centered_text(self, d, y, text, font):
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (self.W - tw) // 2
        d.text((x, y), text, font=font, fill=0)
        return (x, y, x + tw, y + th)

    # Screens
    def _draw_main(self):
        img, d = self._new_layer()
        self._centered_text(d, 6, "Swipe to choose mode", self.font_md)
        y = int(self.H * 0.60); s = 20
        left_x = int(self.W * 0.30)
        right_x = int(self.W * 0.70)
        self._arrow(d, x=left_x, y=y, size=s, direction="left")
        bbox = d.textbbox((0, 0), "Discard", font=self.font_sm)
        tw = bbox[2] - bbox[0]
        d.text((left_x - tw // 2, y + 18), "Discard", font=self.font_sm, fill=0)
        self._arrow(d, x=right_x, y=y, size=s, direction="right")
        bbox = d.textbbox((0, 0), "Check-in", font=self.font_sm)
        tw = bbox[2] - bbox[0]
        d.text((right_x - tw // 2, y + 18), "Check-in", font=self.font_sm, fill=0)
        self._push_partial(img)

    def _draw_mode(self, mode, frac):
        img, d = self._new_layer()
        title = {
            "discard": "Discard",
            "check_in": "Check-in",
            "opened": "Opened",
            "other": "Other",
            "expiry": "Checking expiry...",
        }.get(mode, mode or "--")
        self._centered_text(d, 6, title, self.font_big)
        line1 = "hold items"
        line2 = "1–2ft away from camera"
        bbox1 = d.textbbox((0, 0), line1, font=self.font_italic)
        bbox2 = d.textbbox((0, 0), line2, font=self.font_italic)
        tw1, th1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        tw2, th2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        gap = 4
        total_h = th1 + gap + th2
        y0 = (self.H - total_h) // 2
        d.text(((self.W - tw1) // 2, y0), line1, font=self.font_italic, fill=0)
        d.text(((self.W - tw2) // 2, y0 + th1 + gap), line2, font=self.font_italic, fill=0)

        # countdown bar
        bar_h = 8
        bar_y0 = self.H - bar_h - 4
        d.rectangle((0, bar_y0, self.W, bar_y0 + bar_h), outline=0, fill=255)
        d.rectangle((0, bar_y0, int(self.W * max(0.0, min(1.0, frac))), bar_y0 + bar_h), outline=0, fill=0)

        self._push_partial(img)

    def _draw_captured(self, mode, ok_text, frac):
        img, d = self._new_layer()
        banner = ok_text.strip()
        bbox = d.textbbox((0, 0), banner, font=self.font_big)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        x = (self.W - tw)//2; y = (self.H - th)//2

        d.text((x, y), banner, font=self.font_big, fill=0)

        # countdown bar
        bar_h = 8
        bar_y0 = self.H - bar_h - 4
        d.rectangle((0, bar_y0, self.W, bar_y0 + bar_h), outline=0, fill=255)
        d.rectangle((0, bar_y0, int(self.W * max(0.0, min(1.0, frac))), bar_y0 + bar_h), outline=0, fill=0)

        self._push_partial(img)

    def _draw_expiry(self):
        img, d = self._new_layer()
        self._centered_text(d, 6, "Log expiration date?", self.font_md)
        self._push_partial(img)

    def _arrow(self, draw, x, y, size=24, direction="left"):
        s = size
        if direction == "left":
            draw.polygon([(x-s, y), (x, y-s), (x, y+s)], outline=0, fill=0)
            draw.rectangle((x, y-4, x+s, y+4), outline=0, fill=0)
        elif direction == "right":
            draw.polygon([(x+s, y), (x, y-s), (x, y+s)], outline=0, fill=0)
            draw.rectangle((x-s, y-4, x, y+4), outline=0, fill=0)

# Singleton UI
EPD_UI = EpaperUI()

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI (optional worker)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
work_q = queue.Queue(maxsize=4)
WORKER_ALIVE = False

def api_worker():
    global WORKER_ALIVE
    if OpenAI is None:
        print("[OpenAI] Worker not started: openai SDK not importable."); return
    if not OPENAI_API_KEY:
        print("[OpenAI] Worker not started: OPENAI_API_KEY is empty."); return
    if OPENAI_MODEL.startswith("gpt-5"):
        print("[OpenAI][WARN] gpt-5 with chat.completions may fail. Use Responses API or gpt-4o/4o-mini.")
    print(f"[OpenAI] Worker starting. model={OPENAI_MODEL}")
    client = OpenAI(api_key=OPENAI_API_KEY)
    WORKER_ALIVE = True
    while True:
        item = work_q.get()
        if item is None:
            print("[OpenAI] Worker shutting down."); break
        tag, jpeg_bytes = item
        try:
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            if tag == "expiry":
                user_text = "This is a food item. The expiration date is shown. Extract the expiration or best-by date."
                system_text = "You read expiration dates from product photos. Respond with the date only."
            else:
                user_text = f"Identify the object. (mode={tag})"
                system_text = "You are an expert product identifier. Be concise."
            msgs = [
                {"role":"system","content":system_text},
                {"role":"user","content":[
                    {"type":"text","text":user_text},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]},
            ]
            t0 = time.perf_counter()
            resp = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
            dt_ms = (time.perf_counter() - t0) * 1000
            text = (resp.choices[0].message.content or "").strip()
            print(f"[OpenAI] ({tag}) {dt_ms:.0f} ms -> {text or '<empty>'}")
        except Exception as e:
            print(f"[OpenAI ERROR] tag='{tag}': {type(e).__name__}: {e}")
        finally:
            work_q.task_done()

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
        print(f"[enqueue] {tag}: JPEG encode failed"); return
    with open(path, "wb") as f:
        f.write(jpg)
    print(f"[enqueue] {tag}: saved {path} ({len(jpg)} bytes)")
    if not WORKER_ALIVE:
        print("[enqueue][WARN] OpenAI worker not running; skipping enqueue."); return
    try:
        work_q.put_nowait((tag, jpg))
        print(f"[enqueue] {tag}: enqueued. qsize={work_q.qsize()}")
    except queue.Full:
        print(f"[enqueue] {tag}: queue full, skipped")

threading.Thread(target=api_worker, daemon=True).start()
def q_watchdog():
    while True:
        time.sleep(5)
        print(f"[q-watch] alive={WORKER_ALIVE} qsize={work_q.qsize()}")
threading.Thread(target=q_watchdog, daemon=True).start()

EPD_UI.show_main()

# ─────────────────────────────────────────────────────────────────────────────
# Gesture + capture
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

CAPTURE_COOLDOWN_S = 2.5
last_capture_t = 0.0
countdown_last_sec = -1
EXPIRY_WAIT_S = 8.0

# Stability / thresholds
WAIT_AFTER_SWIPE_S   = 1.0
STABILITY_WINDOW_FR  = 12
MOTION_EMA_ALPHA     = 0.25              # single source of truth

PRESENCE_LAPLACE_MIN  = 10.0
PRESENCE_LAPLACE_GAIN = 1.35
MAX_LAPLACE_THR       = 95.0
LAPLACE_MARGIN        = 3.0

MOTION_THR_SCALE = 1.5
MOTION_THR_FLOOR = 0.004
MAX_MOTION_THR   = 0.040

STEADY_OVERRIDE_AFTER_S = 2.8            # safety valve

ARM_TIMEOUT_S    = 8.0
ENTER_RELAX      = 1.00
EXIT_RELAX       = 1.20
MIN_STABLE_S     = 0.50
CONFIRM_FR       = 2

stable_since = None
confirm_left = 0
presence_dwell_start = None

CLEAR_LAPLACE_FRAC = 0.75
CLEAR_WINDOW_FR    = 10
MIN_CLEAR_S        = 1.2
PRESENCE_DWELL_S   = 0.60
need_clear  = False
clear_count = 0

# Helpers
def y_plane(yuv):
    if yuv.ndim == 2:  return yuv[:LO_H, :LO_W]
    if yuv.ndim == 3:  return yuv[:, :, 0]
    raise ValueError(f"Unexpected lores shape {yuv.shape}")

def smooth1d(v, k):
    if k <= 0: return v
    ksz = 2*k + 1
    return np.convolve(v, np.ones(ksz, np.float32)/ksz, mode="same")

def laplacian_sharpness(gray_u8):
    return cv2.Laplacian(gray_u8, cv2.CV_64F).var()

def center_laplacian(bgr):
    h, w = bgr.shape[:2]
    cx0, cy0 = int(0.25*w), int(0.25*h)
    cx1, cy1 = int(0.75*w), int(0.75*h)
    crop = cv2.cvtColor(bgr[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
    return laplacian_sharpness(crop)

# Camera
picam2 = Picamera2()
video_cfg = picam2.create_video_configuration(
    main={"format":"RGB888","size":(FRAME_W, FRAME_H)},
    lores={"format":"YUV420","size":(LO_W, LO_H)},
    display="main",
)
STILL_W, STILL_H = 2304, 1296
still_cfg = picam2.create_still_configuration(
    main={"format":"RGB888","size":(STILL_W, STILL_H)}, display=None,
)
picam2.configure(video_cfg)
try:
    picam2.set_controls({
        "AeEnable": True, "AwbEnable": True,
        "AfMode": 2,
        "FrameDurationLimits": (10000, 10000),
        "AnalogueGain": 12.0,
    })
except Exception:
    pass
picam2.start()
cam_lock = threading.Lock()

def capture_high_quality(tag: str):
    with cam_lock:
        try:
            try:
                picam2.set_controls({"AfMode": 1}); picam2.set_controls({"AfTrigger": 1})
            except Exception:
                picam2.set_controls({"AfMode": 2})
            picam2.switch_mode(still_cfg)
            time.sleep(0.22)
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

# State
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
awaiting_expiry = False
expiry_prompt_time = 0.0

motion_ema = None
motion_thr_dyn = MOTION_THR_FLOOR
lap_baseline = 0.0
lap_thr_dyn  = PRESENCE_LAPLACE_MIN

FLIP_X = True
FLIP_Y = False

def set_mode_from(gesture: str, now_ts: float, bgr_for_baseline=None):
    global current_mode, armed, arm_time, stable_count
    global motion_thr_dyn, lap_baseline, lap_thr_dyn
    global need_clear, stable_since, confirm_left, presence_dwell_start
    global countdown_last_sec, awaiting_expiry, expiry_prompt_time

    m = MODE_MAP.get(gesture)
    if not m: return
    current_mode = m
    armed = True
    need_clear = False
    awaiting_expiry = False
    arm_time = now_ts
    expiry_prompt_time = 0.0
    countdown_last_sec = -1
    stable_count = 0
    stable_since = None
    confirm_left = 0
    presence_dwell_start = None

    if bgr_for_baseline is not None:
        lap_baseline = center_laplacian(bgr_for_baseline)
        lap_thr_dyn  = max(PRESENCE_LAPLACE_MIN, lap_baseline * PRESENCE_LAPLACE_GAIN)
        lap_thr_dyn  = min(lap_thr_dyn, MAX_LAPLACE_THR)
    else:
        lap_baseline = 0.0
        lap_thr_dyn  = PRESENCE_LAPLACE_MIN

    print(f"[mode] {current_mode} (armed)  lap_base={lap_baseline:.1f}  lap_thr={lap_thr_dyn:.1f}")

try:
    while True:
        now = time.time()

        if not cam_lock.acquire(blocking=False):
            time.sleep(0.005); continue
        try:
            lo  = y_plane(picam2.capture_array("lores"))
            rgb = picam2.capture_array("main")
        finally:
            cam_lock.release()

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        x_norm = y_norm = None
        lo_blur = cv2.GaussianBlur(lo, (3,3), 0)

        if prev_lo_blur is not None:
            diff = cv2.absdiff(lo_blur, prev_lo_blur)
            mask_pool.append(diff)
            pooled = mask_pool[0]
            for i in range(1, len(mask_pool)):
                pooled = cv2.bitwise_or(pooled, mask_pool[i])

            band_h = pooled[y0:y1, :]
            col = band_h.astype(np.float32).sum(axis=0)
            if col.sum() >= ENERGY_MIN_FRAC * (255.0 * (y1 - y0) * LO_W):
                col_s = smooth1d(col, SMOOTH_COLS//2)
                xs = np.arange(LO_W, dtype=np.float32)
                wsum = col_s.sum()
                if wsum > 1e-3:
                    cx_raw = float((col_s * xs).sum() / wsum) / LO_W
                    x_norm = (1.0 - cx_raw) if FLIP_X else cx_raw
                    bar = (col_s / (col_s.max()+1e-6) * 255.0).astype(np.uint8)
                    if FLIP_X: bar = bar[::-1]
                    dbg[0:40, 0:FRAME_W] = cv2.resize(cv2.cvtColor(np.tile(bar, (40, 1)), cv2.COLOR_GRAY2BGR),(FRAME_W, 40))
                    cv2.line(dbg, (int(x_norm*FRAME_W), 40), (int(x_norm*FRAME_W), 70), (255,255,255), 2)

            band_v = pooled[:, x0:x1]
            row = band_v.astype(np.float32).sum(axis=1)
            if row.sum() >= ENERGY_MIN_FRAC * (255.0 * (x1 - x0) * LO_H):
                row_s = smooth1d(row, SMOOTH_ROWS//2)
                ys = np.arange(LO_H, dtype=np.float32)
                wsum = row_s.sum()
                if wsum > 1e-3:
                    cy_raw = float((row_s * ys).sum() / wsum) / LO_H
                    y_norm = (1.0 - cy_raw) if FLIP_Y else cy_raw
                    barv = (row_s / (row_s.max()+1e-6) * 255.0).astype(np.uint8)
                    barv = np.tile(barv[:, None], (1, 40))
                    if FLIP_Y: barv = barv[::-1, :]
                    barv = cv2.cvtColor(barv, cv2.COLOR_GRAY2BGR)
                    barv = cv2.resize(barv, (40, FRAME_H))
                    dbg[0:FRAME_H, FRAME_W-40:FRAME_W] = barv
                    cv2.line(dbg, (FRAME_W-40, int(y_norm*FRAME_H)), (FRAME_W, int(y_norm*FRAME_H)), (255,255,255), 2)

            # Motion in central band
            center_band = pooled[int(LO_H*0.20):int(LO_H*0.80), int(LO_W*0.20):int(LO_W*0.80)]
            motion_now = float(center_band.sum()) / (255.0 * center_band.size)
            motion_ema = motion_now if motion_ema is None else (MOTION_EMA_ALPHA * motion_now + (1.0 - MOTION_EMA_ALPHA) * motion_ema)

            # HUD motion bar
            m_norm = np.clip(motion_ema / 0.02, 0.0, 1.0)
            cv2.rectangle(dbg, (20, 50), (20 + int(200*(1.0 - m_norm)), 65), (255,255,255), -1)
            cv2.putText(dbg, f"MOTION ema={motion_ema:.4f} thr={motion_thr_dyn:.4f}", (230, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        prev_lo_blur = lo_blur

        # Maintain traces
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

        # Mode setting
        gesture_text = ""
        can_fire = (now - last_fire) > COOLDOWN_S

        def set_mode_and_seed(gesture: str):
            global motion_thr_dyn
            ema = (motion_ema if motion_ema is not None else MOTION_THR_FLOOR)
            motion_thr_dyn = max(MOTION_THR_FLOOR, min(MAX_MOTION_THR, ema * MOTION_THR_SCALE))
            set_mode_from(gesture, now_ts, bgr_for_baseline=bgr)
            EPD_UI.show_mode_prompt(current_mode, 1.0)
            cv2.putText(dbg, "ARMED", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

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

        if can_fire and gesture_text == "" and span_x >= SPAN_THR_X and abs(vel_x) >= VEL_THR_X:
            g = "SWIPE_RIGHT" if vel_x > 0 else "SWIPE_LEFT"
            print(f"{g} (span/vel)"); set_mode_and_seed(g); last_fire = now; state_x = "IDLE"; trace_x.clear()

        # Expiration prompt timeout
        if awaiting_expiry and (now - expiry_prompt_time) > EXPIRY_WAIT_S:
            awaiting_expiry = False
            need_clear = True
            armed = False
            clear_count = 0
            print("[expiry] timeout; awaiting item removal")

        # ---------------------------
        # Armed: wait-for-stability
        # ---------------------------
        if armed:
            total = EXPIRY_WAIT_S if awaiting_expiry else ARM_TIMEOUT_S
            remaining = max(0.0, total - (now - arm_time))
            if (now - last_capture_t) < CAPTURE_COOLDOWN_S:
                stable_count = 0
                presence_dwell_start = None
            elif (now - arm_time) > total:
                if awaiting_expiry:
                    print("[expiry] timeout; awaiting item removal")
                    awaiting_expiry = False
                    need_clear = True
                    armed = False
                    clear_count = 0
                    last_capture_t = now
                    stable_count = 0
                    stable_since = None
                    confirm_left = 0
                    presence_dwell_start = None
                    countdown_last_sec = -1
                    EPD_UI.show_captured("expiry", "NO EXPIRY", 1.0)
                else:
                    print("[mode] timeout; disarming without capture")
                    EPD_UI.show_timeout()
                    armed = False
                    presence_dwell_start = None
                    countdown_last_sec = -1
            elif (now - arm_time) < WAIT_AFTER_SWIPE_S:
                stable_count = 0
                presence_dwell_start = None
            else:
                # update countdown bar roughly each second
                if (now - last_capture_t) > 0.5:
                    sec = int(remaining)
                    if sec != countdown_last_sec:
                        countdown_last_sec = sec
                        display_mode = "expiry" if awaiting_expiry else current_mode
                        EPD_UI.show_mode_prompt(display_mode, remaining / total)

                lap_c = center_laplacian(bgr)
                thr_enter = motion_thr_dyn * ENTER_RELAX
                thr_exit  = motion_thr_dyn * EXIT_RELAX

                sharp_enough = (lap_c >= (lap_thr_dyn + LAPLACE_MARGIN))
                below_enter  = (motion_ema is not None) and (motion_ema < thr_enter)
                above_exit   = (motion_ema is not None) and (motion_ema > thr_exit)

                # Presence dwell: start timing only when BOTH gates are met
                if below_enter and sharp_enough:
                    if presence_dwell_start is None:
                        presence_dwell_start = now
                else:
                    presence_dwell_start = None

                # Safety valve (optional): very steady for a while → let sharpness slide
                if (not sharp_enough) and below_enter and (now - arm_time) > STEADY_OVERRIDE_AFTER_S:
                    sharp_enough = True
                    if presence_dwell_start is None:
                        presence_dwell_start = now  # begin dwell from now
                    print("[stable?] overriding sharpness due to sustained steadiness")

                # Debug
                if int(time.time() * 5) % 5 == 0:
                    print(f"[stable?] mo={motion_ema:.4f} < {thr_enter:.4f} lap={lap_c:.1f} >= {lap_thr_dyn+LAPLACE_MARGIN:.1f} "
                          f"dwell={(0 if presence_dwell_start is None else now-presence_dwell_start):.2f}/{PRESENCE_DWELL_S:.2f}")

                if above_exit or not sharp_enough:
                    stable_count = 0
                    stable_since = None
                    confirm_left = 0
                else:
                    if below_enter and sharp_enough and presence_dwell_start and (now - presence_dwell_start) >= PRESENCE_DWELL_S:
                        # classic frame-count requirement too
                        stable_count += 1
                        if stable_since is None:
                            stable_since = now
                        if (now - stable_since) >= MIN_STABLE_S and stable_count >= STABILITY_WINDOW_FR:
                            if confirm_left == 0:
                                confirm_left = CONFIRM_FR
                            else:
                                confirm_left -= 1
                                if confirm_left == 0:
                                    tag = "expiry" if awaiting_expiry else (current_mode or "unknown_mode")
                                    print(f"[mode] stable -> capturing ({tag})  mo={motion_ema:.4f}/{motion_thr_dyn:.4f} lap={lap_c:.1f}/{lap_thr_dyn+LAPLACE_MARGIN:.1f}")
                                    start_capture_thread(tag)
                                    if tag == "check_in":
                                        msg = "\u2713 CHECKED IN!"
                                    elif tag == "expiry":
                                        msg = "EXPIRY SAVED"
                                    else:
                                        msg = "\u2717 DISCARDED!"
                                    EPD_UI.show_captured(tag, msg, 1.0)
                                    last_capture_t = now
                                    arm_time = now
                                    stable_count = 0
                                    stable_since = None
                                    presence_dwell_start = None
                                    if awaiting_expiry:
                                        awaiting_expiry = False
                                        need_clear = True
                                        armed = False
                                        clear_count = 0
                                        print("[expiry] captured; waiting for item removal")
                                    elif tag == "check_in":
                                        awaiting_expiry = True
                                        expiry_prompt_time = now
                                        armed = True
                                        need_clear = False
                                        clear_count = 0
                                        countdown_last_sec = -1
                                        EPD_UI.show_mode_prompt("expiry", 1.0)
                                        print("[expiry] checking for expiration date")
                                    else:
                                        need_clear = True
                                        armed = False
                                        clear_count = 0
                                        print("[mode] captured; waiting for item removal to re-arm")

        # Re-arm when item is removed (center detail low for a few frames + min time)
        if need_clear:
            lap_c = center_laplacian(bgr)
            clear_thr = max(PRESENCE_LAPLACE_MIN * 0.8, lap_thr_dyn * CLEAR_LAPLACE_FRAC)
            if lap_c < clear_thr:
                clear_count += 1
            else:
                clear_count = 0

            ready_by_time = (now - last_capture_t) >= MIN_CLEAR_S
            if ready_by_time and clear_count >= CLEAR_WINDOW_FR:
                need_clear = False
                armed = True
                arm_time = now
                countdown_last_sec = -1
                presence_dwell_start = None
                print("[mode] scene cleared; re-armed")
                if current_mode:
                    EPD_UI.show_mode_prompt(current_mode, 1.0)

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
    try:
        picam2.stop()
    except Exception:
        pass
    if not HEADLESS:
        cv2.destroyAllWindows()
    try:
        work_q.put_nowait(None)
    except queue.Full:
        pass
