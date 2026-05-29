#!/usr/bin/env bash
# Method C: VIPE with full-video Pi3X batch + MoGe-2 EMA fusion (SANA-WM Plan A)
# SLAM uses unidepth-l; post uses VideoPi3XDepthProcessor (all frames at once).
# Expected runtime on H100: ~20 min (Pi3X 75 chunks × 8s + MoGe-2 613 frames × 0.1s)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEQ="${REPO_ROOT}/experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk"
OUT="${REPO_ROOT}/experiments/vipe_comparison/results/method_C"
VIDEO="${SEQ}/video.mp4"

if [ ! -f "${VIDEO}" ]; then
  echo "[error] video.mp4 not found. Run prepare_tum.py first."
  exit 1
fi

mkdir -p "${OUT}"

export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

echo "[method_C] Running VIPE with full-video Pi3X+MoGe-2 batch (Plan A)..."
echo "[method_C] PI3X weights: ${SANA_WM_PI3X_WEIGHTS}"
echo "[method_C] MoGe2 weights: ${SANA_WM_MOGE2_WEIGHTS}"
echo "[method_C] Input:  ${VIDEO}"
echo "[method_C] Output: ${OUT}"

source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

vipe infer "${VIDEO}" \
    --output "${OUT}" \
    --pipeline sana_wm_pi3x_moge2_full

echo "[method_C] Done."
echo "[method_C] Pose artifact: ${OUT}/pose/video.npz"

python3 - <<'PYEOF'
import numpy as np, sys
from pathlib import Path

npz = Path("experiments/vipe_comparison/results/method_C/pose/video.npz")
if not npz.exists():
    print(f"[error] {npz} not found"); sys.exit(1)
d = np.load(npz)
print(f"[check] poses shape={d['data'].shape}, inds shape={d['inds'].shape}")
print(f"[check] first pose:\n{d['data'][0]}")
PYEOF
