#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
from PIL import Image, ImageDraw, ImageFont

# Import the e-Paper driver
libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)

import epd2in13_V2  # Waveshare driver for the 2.13" V2 display

def main():
    try:
        epd = epd2in13_V2.EPD()
        epd.init(epd.FULL_UPDATE)
        epd.Clear(0xFF)

        # Create a blank image for drawing
        image = Image.new('1', (epd.height, epd.width), 255)  # 255: clear the frame
        draw = ImageDraw.Draw(image)

        # Draw some text
        font = ImageFont.load_default()
        draw.text((10, 10), 'Hello, ePaper!', font=font, fill=0)
        draw.text((10, 30), 'Waveshare 2.13"', font=font, fill=0)

        # Draw a rectangle
        draw.rectangle((5, 50, 120, 100), outline=0, fill=255)
        draw.line((5, 50, 120, 100), fill=0)
        draw.line((5, 100, 120, 50), fill=0)

        # Display image on e-paper
        epd.display(epd.getbuffer(image))
        time.sleep(2)

        # Sleep the display
        epd.sleep()

    except IOError as e:
        print(e)

    except KeyboardInterrupt:
        print("Ctrl+C detected, exiting...")
        epd2in13_V2.epdconfig.module_exit()
        exit()

if __name__ == '__main__':
    main()