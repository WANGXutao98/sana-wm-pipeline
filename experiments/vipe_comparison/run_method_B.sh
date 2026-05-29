#!/usr/bin/env bash
# Method B: VIPE with Pi3X + MoGe-2 depth backend (SANA-WM enhanced)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEQ="${REPO_ROOT}/experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk"
OUT="${REPO_ROOT}/experiments/vipe_comparison/results/method_B"
VIDEO="${SEQ}/video.mp4"

if [ ! -f "${VIDEO}" ]; then
  echo "[error] video.mp4 not found. Run prepare_tum.py first."
  exit 1
fi

mkdir -p "${OUT}"

export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

echo "[method_B] Running VIPE with Pi3X + MoGe-2 backend (SANA-WM)..."
echo "[method_B] PI3X weights: ${SANA_WM_PI3X_WEIGHTS}"
echo "[method_B] MoGe2 weights: ${SANA_WM_MOGE2_WEIGHTS}"
echo "[method_B] Input:  ${VIDEO}"
echo "[method_B] Output: ${OUT}"

source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

vipe infer "${VIDEO}" \
    --output "${OUT}" \
    --pipeline sana_wm_pi3x_moge2

echo "[method_B] Done."
echo "[method_B] Pose artifact: ${OUT}/pose/video.npz"

python3 - <<'PYEOF'
import numpy as np, sys
from pathlib import Path

npz = Path("experiments/vipe_comparison/results/method_B/pose/video.npz")
if not npz.exists():
    print(f"[error] {npz} not found"); sys.exit(1)
d = np.load(npz)
print(f"[check] poses shape={d['data'].shape}, inds shape={d['inds'].shape}")
print(f"[check] first pose:\n{d['data'][0]}")
PYEOF
