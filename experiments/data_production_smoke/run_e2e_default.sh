#!/usr/bin/env bash
# End-to-end smoke test: DL3DV Default mode (Stage 01→02_default→05→06)
# Stage 04 (filter) is skipped: DL3DV smoke lacks UniMatch/DOVER deps.
#
# Usage: bash experiments/data_production_smoke/run_e2e_default.sh [DATA_DIR]
# Default DATA_DIR: /mnt/afs/davidwang/workspace/data/dl3dv_smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 环境 ──────────────────────────────────────────────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

DATA_DIR="${1:-/mnt/afs/davidwang/workspace/data/dl3dv_smoke}"
OUT_DIR="/mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default"
mkdir -p "${OUT_DIR}"

cd "${PROJECT_ROOT}"

# ── 找到所有场景 ──────────────────────────────────────────────────────────────
# DL3DV smoke 布局：DATA_DIR/1K/<scene_id>/ 或 DATA_DIR/<scene_id>/
mapfile -t SCENES < <(
  ls -d "${DATA_DIR}"/1K/*/  2>/dev/null ||
  ls -d "${DATA_DIR}"/*/     2>/dev/null ||
  true
)

if [ "${#SCENES[@]}" -eq 0 ]; then
  echo "ERROR: No scene directories found under ${DATA_DIR}"
  exit 1
fi
echo "Found ${#SCENES[@]} scenes in ${DATA_DIR}"

for SCENE_DIR in "${SCENES[@]}"; do
  SCENE_DIR="${SCENE_DIR%/}"   # strip trailing slash
  SCENE_ID="$(basename "${SCENE_DIR}")"
  echo ""
  echo "===== Scene: ${SCENE_ID} ====="

  WORK_DIR="${OUT_DIR}/work/${SCENE_ID}"
  mkdir -p "${WORK_DIR}"

  # ── Step 0: prepare (images/ + transforms.json → video.mp4 + gt_poses.npy) ──
  if [ ! -f "${SCENE_DIR}/video.mp4" ]; then
    echo "  [Step 0] Preparing scene (images → video.mp4 + gt poses)..."
    python "${SCRIPT_DIR}/prepare_dl3dv.py" "${SCENE_DIR}"
  else
    echo "  [Step 0] video.mp4 already exists, skipping prepare."
    # Ensure gt_poses.npy is present (needed for gt-pose mode; default mode doesn't
    # use it, but prepare may still produce it harmlessly).
    if [ ! -f "${SCENE_DIR}/gt_poses.npy" ]; then
      python "${SCRIPT_DIR}/prepare_dl3dv.py" "${SCENE_DIR}"
    fi
  fi

  # ── Step 1: Normalize (→ 1280×720 @ 16fps) ───────────────────────────────
  NORM_VIDEO="${WORK_DIR}/normalized.mp4"
  if [ ! -f "${NORM_VIDEO}" ]; then
    echo "  [Step 1] Normalizing to 1280×720 @ 16fps..."
    python - <<PYEOF
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
from pathlib import Path
info = normalize_video(Path("${SCENE_DIR}/video.mp4"), Path("${NORM_VIDEO}"))
print(f"  Normalized: {info.n_frames} frames @ {info.fps} fps  ({info.width}x{info.height})")
PYEOF
  else
    echo "  [Step 1] normalized.mp4 already exists, skipping."
  fi

  # ── Step 2: Default pose estimation (Pi3X + MoGe-2 + VIPE) ──────────────
  POSE_DIR="${WORK_DIR}/pose_work"
  POSE_ARTIFACT="${WORK_DIR}/pose_artifact.npz"
  if [ ! -f "${POSE_ARTIFACT}" ]; then
    echo "  [Step 2] Default pose estimation (Pi3X + MoGe-2 + VIPE)..."
    python - <<PYEOF
from sana_wm_pipeline.stage02_pose import mode_default
from pathlib import Path
import numpy as np
art = mode_default.run_default(Path("${NORM_VIDEO}"), Path("${POSE_DIR}"))
np.savez_compressed(
    "${POSE_ARTIFACT}",
    poses_c2w=art.poses_c2w,
    intrinsics=art.intrinsics,
    scale_per_frame=art.scale_per_frame,
)
T = art.poses_c2w.shape[0]
print(f"  Pose artifact saved: T={T} frames  -> ${POSE_ARTIFACT}")
PYEOF
  else
    echo "  [Step 2] pose_artifact.npz already exists, skipping."
  fi

  # ── Step 3 (Stage 04): Filter — SKIPPED for DL3DV smoke ──────────────────
  echo "  [Step 3/Stage04] Filter skipped (no UniMatch/DOVER in smoke env)."

  # ── Step 4 (Stage 05): Caption — stub fallback ───────────────────────────
  CAPTION_FILE="${WORK_DIR}/caption.txt"
  if [ ! -f "${CAPTION_FILE}" ]; then
    echo "  [Step 4/Stage05] Caption (stub fallback, no GPU VLM in smoke)..."
    python - <<PYEOF
from sana_wm_pipeline.stage05_caption.qwen35_vl_runner import CAPTION_FALLBACK
from pathlib import Path
Path("${CAPTION_FILE}").write_text(CAPTION_FALLBACK, encoding="utf-8")
print(f"  Caption written: {CAPTION_FALLBACK}")
PYEOF
  else
    echo "  [Step 4/Stage05] caption.txt already exists, skipping."
  fi

  # ── Step 5 (Stage 06): Pack WebDataset shard ─────────────────────────────
  echo "  [Step 5/Stage06] Packing WebDataset shard (strict_frames=False)..."
  python - <<PYEOF
import numpy as np
from pathlib import Path
from sana_wm_pipeline.stage06_pack.schema import Sample
from sana_wm_pipeline.stage06_pack.webdataset_writer import ShardWriter

data       = np.load("${POSE_ARTIFACT}")
poses_c2w  = data["poses_c2w"].astype(np.float32)
intrinsics = data["intrinsics"].astype(np.float32)
scale      = data["scale_per_frame"].astype(np.float32)
caption    = Path("${CAPTION_FILE}").read_text(encoding="utf-8").strip()

sample = Sample(
    sample_id="${SCENE_ID}",
    video_path="${NORM_VIDEO}",
    poses_c2w=poses_c2w,
    intrinsics_NVD=intrinsics,
    scale_per_frame=scale,
    caption=caption,
    meta={
        "source": "DL3DV",
        "pose_mode": "default",
        "scene_id": "${SCENE_ID}",
    },
)
with ShardWriter("${OUT_DIR}", samples_per_shard=100, strict_frames=False) as w:
    w.write(sample)
print(f"  Shard written to ${OUT_DIR}")
PYEOF

  echo "  [DONE] ${SCENE_ID}"
done

echo ""
echo "===== All scenes complete ====="
echo "Shards at: ${OUT_DIR}"
