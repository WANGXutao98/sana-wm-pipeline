#!/bin/bash
# Sekai-Real-Walking-HQ 冒烟测试（单样本，Default模式）
set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────────────────────
export PROJ_DIR="/mnt/afs/davidwang/workspace/sana_wm_pipeline"
export ENV_DIR="/mnt/afs/davidwang/miniconda3/envs/sana_wm"
export OUT_BASE="/mnt/afs/davidwang/workspace/sana_test_data/smoke_result"
#export VIDEO_PATH="/mnt/afs/davidwang/workspace/sana_test_data/sekai-real-walking/sekai-real-walking-hq__FP8j6WfkTY_0085528_0087328.mp4"
export VIDEO_PATH="/mnt/afs/davidwang/workspace/sana_test_data/sekai-real-walking/sekai-real-walking-hq_3aFIYNiOBlg_0081182_0082982.mp4"

#export CAMERA_PATH="/mnt/afs/davidwang/workspace/sana_test_data/sekai-real-walking/sekai-real-walking-hq__FP8j6WfkTY_0085528_0087328.camera.npz"
export CAMERA_PATH="/mnt/afs/davidwang/workspace/sana_test_data/sekai-real-walking/sekai-real-walking-hq_3aFIYNiOBlg_0081182_0082982.camera.npz"


# ── 模型权重 ──────────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="/mnt/afs/davidwang/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="/mnt/afs/davidwang/models/moge2"

# ── 离线模式 + 缓存 ───────────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface

# ── 激活环境 ──────────────────────────────────────────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm
export LD_LIBRARY_PATH="$ENV_DIR/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "=== Sekai-Real-Walking-HQ Smoke Test ==="
echo "视频: $VIDEO_PATH"
echo "GT: $CAMERA_PATH"
echo "输出: $OUT_BASE"
echo ""

# ── 预检 ──────────────────────────────────────────────────────────────────────
python -c "
import sana_wm_pipeline; print('sana_wm_pipeline ✓')
import vipe_ext; print('vipe_ext ✓')
import vipe; print('vipe ✓')
import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} ✓')
"

# ── 准备样本目录 ─────────────────────────────────────────────────────────────
EXTRACT_DIR="$OUT_BASE/raw_samples"
mkdir -p "$EXTRACT_DIR"

# ponytail: 单样本，直接复制不需要tar解包
SAMPLE_ID="$(basename "$VIDEO_PATH" .mp4)"
cp "$VIDEO_PATH" "$EXTRACT_DIR/"
cp "$CAMERA_PATH" "$EXTRACT_DIR/"

echo "样本目录: $EXTRACT_DIR"
echo ""

# ── 处理（复用 smoke_test_batch.py 逻辑） ────────────────────────────────────
cd "$PROJ_DIR"
python << PYEOF
import sys
sys.path.insert(0, "$PROJ_DIR/src")
from pathlib import Path

# ponytail: 直接import process_sample，避免重复代码
exec(open("$PROJ_DIR/scripts/smoke_test_batch.py").read().split("def main()")[0])

sample_id = "$SAMPLE_ID"
extract_dir = Path("$EXTRACT_DIR")
output_dir = Path("$OUT_BASE")

# 单样本处理
success = process_sample(sample_id, 0, extract_dir, output_dir)
sys.exit(0 if success else 1)
PYEOF

echo ""
echo "=== 测试完成 ==="
