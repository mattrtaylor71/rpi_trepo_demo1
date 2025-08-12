#!/bin/bash
set -euo pipefail

VENV=".venv"

echo "==> Updating system"
sudo apt update
sudo apt full-upgrade -y

echo "==> Installing OS-level deps"
# picamera2 comes from apt; OpenCV GUI deps only matter if you use cv2.imshow
sudo apt install -y \
  python3-full python3-venv python3-pip \
  python3-picamera2 \
  libatlas-base-dev libjpeg-dev libopenexr-dev \
  libgstreamer1.0-dev libgtk-3-0 libgl1

echo "==> Creating virtual environment at ${VENV}"
python3 -m venv "${VENV}"

echo "==> Upgrading pip/setuptools/wheel in venv"
"${VENV}/bin/pip" install --upgrade pip setuptools wheel

# Decide OpenCV flavor: headless if no display; GUI build otherwise
OPENCV_PKG="opencv-python"
if [ -z "${DISPLAY:-}" ]; then
  OPENCV_PKG="opencv-python-headless"
fi
echo "==> Using OpenCV package: ${OPENCV_PKG}"

echo "==> Installing Python packages in venv"
# Notes:
# - numpy 1.26.x is reliable on Pi (aarch64) wheels
# - OpenCV 4.8.1.78 has stable ARM wheels; newer may work but this is solid
# - absl-py fixes your 'No module named absl' error
# - openai is the official SDK (v1+)
# - python-dotenv optional: loads .env locally if you want
"${VENV}/bin/pip" install \
  "numpy==1.26.4" \
  "${OPENCV_PKG}==4.8.1.78" \
  "absl-py>=1.4.0" \
  "openai>=1.40.0" \
  "python-dotenv>=1.0.1"

echo "==> Done."

# Helpful next steps
cat <<'EOM'

To run your script:

  # 1) Activate the venv
  source .venv/bin/activate

  # 2) Set your API key (or put it in ~/.bashrc/.zshrc)
  export OPENAI_API_KEY="sk-***"

  # 3) Run your program
  python gesture_with_capture.py

(Optionally) create a .env and auto-load it in Python with python-dotenv:
  echo 'OPENAI_API_KEY=sk-***' > .env

EOM
