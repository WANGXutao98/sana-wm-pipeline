#!/bin/bash
# Set up the Pi3 + MoGe side of the camera-estimation environment.
# VIPE is installed separately by setup_vipe.sh.

PIP_CMD="python3 -m pip install --user"

ROOT="${SANA_WM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TP="$ROOT/third_party"
WT="$ROOT/weights"
export HF_HOME="$WT/hf"
mkdir -p "$TP" "$WT" "$HF_HOME"

echo "=== [0/5] system ffmpeg ==="
(apt-get update && apt-get install -y ffmpeg) >/tmp/apt_ffmpeg.log 2>&1 \
  && echo "system ffmpeg OK" || echo "WARN: apt ffmpeg failed"

echo "=== [1/5] camera runtime deps ==="
$PIP_CMD decord huggingface_hub safetensors einops "opencv-python-headless<4.10" imageio-ffmpeg || exit 11

echo "=== [2/5] clone Pi3 ==="
[ -d "$TP/Pi3" ] || git clone --depth 1 https://github.com/yyfz/Pi3.git "$TP/Pi3" || exit 12
[ -f "$TP/Pi3/requirements.txt" ] && $PIP_CMD -r "$TP/Pi3/requirements.txt"

echo "=== [3/5] install MoGe ==="
$PIP_CMD "git+https://github.com/microsoft/MoGe.git" || exit 13

echo "=== [4/5] download Pi3 weights ==="
python3 -c "from huggingface_hub import snapshot_download as s; s('yyfz233/Pi3', local_dir='$WT/pi3')" || exit 14

echo "=== [5/5] download MoGe-2 weights ==="
python3 -c "from huggingface_hub import snapshot_download as s; s('Ruicheng/moge-2-vitl-normal', local_dir='$WT/moge2')" || exit 15

echo "=== [fix] pin numpy<2 ==="
# The deps above may pull numpy 2.x into --user, which
# SHADOWS the NGC base numpy 1.26.4 and is ABI-incompatible with the container's
# torch 2.8.0a0 -> `torch.from_numpy` raises "Numpy is not available", breaking
# the Pi3/MoGe depth precompute. Force numpy<2 in --user to match the NGC torch.
$PIP_CMD "numpy<2" || exit 17
python3 -c "import torch,numpy as np; print('numpy',np.__version__); torch.from_numpy(np.zeros((2,2)))" || exit 18

# NOTE: Pi3 is a cloned repo (not pip-installed) — callers must put it on
# PYTHONPATH (export PYTHONPATH=$ROOT/third_party/Pi3:$PYTHONPATH) before running
# precompute_fused_depth.py / the pose stage under system python3.

echo "CAMERA_SETUP_DONE"
touch "$ROOT/.camera_setup_done"
