#!/usr/bin/env bash
# Method A: VIPE with unidepth-l depth backend (vanilla)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEQ="${REPO_ROOT}/experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk"
OUT="${REPO_ROOT}/experiments/vipe_comparison/results/method_A"
VIDEO="${SEQ}/video.mp4"

if [ ! -f "${VIDEO}" ]; then
  echo "[error] video.mp4 not found. Run prepare_tum.py first."
  exit 1
fi

mkdir -p "${OUT}"

echo "[method_A] Running VIPE with unidepth-l backend..."
echo "[method_A] Input:  ${VIDEO}"
echo "[method_A] Output: ${OUT}"

source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

vipe infer "${VIDEO}" \
    --output "${OUT}" \
    --pipeline sana_wm_pose_only

echo "[method_A] Done."
echo "[method_A] Pose artifact: ${OUT}/pose/video.npz"

python3 - <<'PYEOF'
import numpy as np, sys
from pathlib import Path

npz = Path("experiments/vipe_comparison/results/method_A/pose/video.npz")
if not npz.exists():
    print(f"[error] {npz} not found"); sys.exit(1)
d = np.load(npz)
print(f"[check] poses shape={d['data'].shape}, inds shape={d['inds'].shape}")
print(f"[check] first pose:\n{d['data'][0]}")
PYEOF
