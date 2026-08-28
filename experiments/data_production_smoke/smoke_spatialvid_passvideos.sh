#!/bin/bash
# SpatialVID pass_videos 全量冒烟测试（246个视频，Default模式）
# 用法：bash experiments/data_production_smoke/smoke_spatialvid_passvideos.sh
set -euo pipefail

# ── 路径配置（AFS开发机） ─────────────────────────────────────────────────────
export PROJ_DIR="/mnt/afs/davidwang/workspace/sana_wm_pipeline"
export ENV_DIR="/mnt/afs/davidwang/miniconda3/envs/sana_wm"
export OUT_BASE="/mnt/afs/davidwang/workspace/sana_test_data/spatialvid_passvideos_smoke"
export VIDEO_DIR="/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/pass_videos"

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

echo "=== SpatialVID pass_videos 全量冒烟测试 ==="
echo "输出目录: $OUT_BASE"
echo "视频目录: $VIDEO_DIR"
echo ""

# ── 预检导入 ──────────────────────────────────────────────────────────────────
python -c "
import sana_wm_pipeline; print('sana_wm_pipeline ✓')
import vipe_ext;          print('vipe_ext ✓')
import vipe;              print('vipe ✓')
import torch;             print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} ✓')
"

# ── 扫描所有mp4文件 ───────────────────────────────────────────────────────────
SAMPLES_FILE="$OUT_BASE/selected_samples.txt"
mkdir -p "$OUT_BASE"

if [ ! -f "$SAMPLES_FILE" ]; then
    echo "=== 扫描视频文件 ==="
    python3 << 'EOF'
import cv2
from pathlib import Path
import sys

video_dir = Path("/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/pass_videos")
output_file = Path("/mnt/afs/davidwang/workspace/sana_test_data/spatialvid_passvideos_smoke/selected_samples.txt")

videos = sorted(video_dir.glob("*.mp4"))
print(f"找到 {len(videos)} 个视频")

with open(output_file, 'w') as f:
    for vpath in videos:
        cap = cv2.VideoCapture(str(vpath))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        sample_id = vpath.stem
        f.write(f"{sample_id}\t{frame_count}\n")

print(f"样本列表已写入: {output_file}")
EOF
fi

echo ""
echo "=== 选中的样本（前10个）==="
head -10 "$SAMPLES_FILE"
echo "..."
echo "总计: $(wc -l < $SAMPLES_FILE) 个样本"
echo ""

# ── 批量处理（单进程，模型只加载一次） ─────────────────────────────────────
echo "=== 启动批量处理 ==="
cd "$PROJ_DIR" && python "$PROJ_DIR/scripts/smoke_test_batch_passvideos.py" \
    --samples "$SAMPLES_FILE" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUT_BASE"

EXIT_CODE=$?
exit $EXIT_CODE
