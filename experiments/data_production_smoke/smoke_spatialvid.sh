#!/bin/bash
# SpatialVID-hq 冒烟测试（3个最短样本，Default模式）
# 用法：bash experiments/data_production_smoke/smoke_spatialvid.sh
set -euo pipefail

# ── 路径配置（AFS开发机） ─────────────────────────────────────────────────────
export PROJ_DIR="/mnt/afs/davidwang/workspace/sana_wm_pipeline"
export ENV_DIR="/mnt/afs/davidwang/miniconda3/envs/sana_wm"
export OUT_BASE="/mnt/afs/davidwang/workspace/sana_test_data/smoke_result"
export TAR_PATH="/mnt/afs/davidwang/workspace/sana_test_data/SpatialVID-hq/SpatialVID-hq-000000.tar"

# ── 模型权重 ──────────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="/mnt/afs/davidwang/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="/mnt/afs/davidwang/models/moge2"

# ── 离线模式 + 缓存目录 ───────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface

# ── 激活环境 ──────────────────────────────────────────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

# 添加PyTorch库路径（修复libc10.so加载问题）
export LD_LIBRARY_PATH="$ENV_DIR/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "=== SpatialVID-hq Smoke Test ==="
echo "输出目录: $OUT_BASE"
echo "Tar包: $TAR_PATH"
echo ""

# ── 预检导入 ──────────────────────────────────────────────────────────────────
python -c "
import sana_wm_pipeline; print('sana_wm_pipeline ✓')
import vipe_ext;          print('vipe_ext ✓')
import vipe;              print('vipe ✓')
import torch;             print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} ✓')
"

# ── 选择3个最短样本 ───────────────────────────────────────────────────────────
SAMPLES_FILE="$OUT_BASE/selected_samples.txt"
mkdir -p "$OUT_BASE"

if [ ! -f "$SAMPLES_FILE" ]; then
    echo "=== 选择最短样本 ==="
    python "$PROJ_DIR/scripts/select_shortest_samples.py" \
        "$TAR_PATH" --num-samples 3 --output "$SAMPLES_FILE"
fi

echo ""
echo "=== 选中的样本 ==="
cat "$SAMPLES_FILE"
echo ""

# ── 解包选中的样本 ────────────────────────────────────────────────────────────
EXTRACT_DIR="$OUT_BASE/raw_samples"
mkdir -p "$EXTRACT_DIR"

echo "=== 解包样本到 $EXTRACT_DIR ==="
while IFS=$'\t' read -r sample_id n_frames; do
    echo "  解包: $sample_id ($n_frames 帧)"
    tar -xf "$TAR_PATH" -C "$EXTRACT_DIR" \
        "${sample_id}.mp4" "${sample_id}.camera.npz" 2>/dev/null || true
done < "$SAMPLES_FILE"

echo ""
ls -lh "$EXTRACT_DIR"
echo ""

# ── 批量处理（单进程，模型只加载一次） ─────────────────────────────────────
echo "=== 启动批量处理 ==="
cd "$PROJ_DIR" && python "$PROJ_DIR/scripts/smoke_test_batch.py" \
    --samples "$SAMPLES_FILE" \
    --extract-dir "$EXTRACT_DIR" \
    --output-dir "$OUT_BASE"

EXIT_CODE=$?
exit $EXIT_CODE
