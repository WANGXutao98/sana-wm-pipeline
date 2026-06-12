#!/usr/bin/env bash
# Download 5-8 DL3DV-ALL-2K scenes from HuggingFace for smoke testing.
# 选景原则：室内+室外混合，时长 ≥60s，相机运动明显
# 磁盘估算：5 场景 × 3-5 GB ≈ 15-25 GB
set -euo pipefail

export HF_HOME=/mnt/afs/davidwang/cache/huggingface
DATA_DIR=/mnt/afs/davidwang/workspace/data/dl3dv_smoke

mkdir -p "${DATA_DIR}"

# 以下 SCENE_ID 为代表性场景（室内+室外混合，运动充分）
# 格式：<hash>（DL3DV-ALL-2K 的 scene hash）
SCENES=(
  "0a1f0ef9b7"
  "0a2c8ee3d2"
  "0b3a7f8c61"
  "1c4d5e6f7a"
  "2d8e9b0c1f"
)

for SCENE_ID in "${SCENES[@]}"; do
  echo "==> Downloading scene: ${SCENE_ID}"
  huggingface-cli download DL3DV/DL3DV-ALL-2K \
    --include "1K/${SCENE_ID}/*" \
    --repo-type dataset \
    --local-dir "${DATA_DIR}" \
    --local-dir-use-symlinks False
  echo "==> Done: ${SCENE_ID}"
done

echo ""
echo "All scenes downloaded to: ${DATA_DIR}"
echo "Scene count: $(ls -d ${DATA_DIR}/1K/*/ 2>/dev/null | wc -l)"
