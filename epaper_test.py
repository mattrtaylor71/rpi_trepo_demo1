#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import os
picdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pic')
libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)

import logging
from waveshare_epd import epd2in13b_V4
try:
    from waveshare_epd import epdconfig  # for module_exit on Ctrl-C
except Exception:
    epdconfig = None

import time
from PIL import Image, ImageDraw, ImageFont
import traceback

logging.basicConfig(level=logging.INFO)

def _pick_font(size):
    """Prefer repo font/Font.ttc, fallback to system DejaVu, else default."""
    fontdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'font')
    candidates = [
        os.path.join(fontdir, 'Font.ttc'),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def main():
    try:
        logging.info("epd2in13b_V4 Demo")

        epd = epd2in13b_V4.EPD()
        logging.info("init and Clear")
        epd.init()
        epd.Clear()

        W, H = epd.height, epd.width  # NOTE: Waveshare often swaps these
        # If your text appears rotated, swap the assignments:
        # W, H = epd.width, epd.height

        # Create two 1-bit images: one for black, one for red
        black = Image.new('1', (W, H), 255)  # 255: white
        red   = Image.new('1', (W, H), 255)

        draw_b = ImageDraw.Draw(black)
        draw_r = ImageDraw.Draw(red)

        # Fonts
        font24 = _pick_font(24)
        font18 = _pick_font(18)
        font12 = _pick_font(12)

        # --- Draw on black layer ---
        draw_b.rectangle((0, 0, W-1, H-1), outline=0)  # border
        draw_b.text((8, 6), "Trepo • e-paper", font=font24, fill=0)
        draw_b.text((8, 36), "2.13\" tri-color V4", font=font18, fill=0)

        # Simple “barcode” lines (black)
        x = 8
        for i in range(32):
            if i % 2 == 0:
                draw_b.line((x+i, 60, x+i, 110), fill=0, width=1)

        # --- Draw on red layer ---
        draw_r.rectangle((6, 120, W-6, H-6), outline=0, fill=255)  # frame only
        draw_r.text((12, 124), "Hello from Raspberry Pi!", font=font18, fill=0)
        draw_r.text((12, 146), time.strftime("Time: %H:%M:%S"), font=font12, fill=0)

        logging.info("Display")
        epd.display(epd.getbuffer(black), epd.getbuffer(red))

        time.sleep(2)

        logging.info("sleep")
        epd.sleep()

    except KeyboardInterrupt:
        logging.info("ctrl + c:")
        if epdconfig and hasattr(epdconfig, "module_exit"):
            epdconfig.module_exit()
        sys.exit()

    except Exception as e:
        logging.error(e)
        logging.debug(traceback.format_exc())
        # Try to put display to sleep to avoid ghosting if something failed mid-draw
        try:
            epd.sleep()
        except Exception:
            pass

if __name__ == "__main__":
    main()
