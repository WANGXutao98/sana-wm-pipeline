#!/bin/bash
# CMCC 冒烟测试脚本（Pass Videos）
# 用途：验证 CMCC 环境部署成功，确保核心功能可用
# 用法：bash experiments/data_production_smoke/smoke_cmcc_pass.sh
set -euo pipefail

# ── 路径配置（CMCC）──────────────────────────────────────────────────────────
export NEW_BASE="/root/work/david_work"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"
export VIDEO_DIR="$NEW_BASE/smoke_pass_videos"
export OUT_BASE="$NEW_BASE/smoke_pass_results"

# ── GPU设置 ───────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ── 模型权重 ──────────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"
export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"

# ── 离线模式 + 优化 ───────────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 激活环境 ──────────────────────────────────────────────────────────────────
ENV_DIR="$NEW_BASE/sana_wm_env"
source "$ENV_DIR/bin/activate"

# 添加项目到 PYTHONPATH
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "CMCC 冒烟测试 - Pass Videos"
echo "========================================"
echo "项目目录: $PROJ_DIR"
echo "视频目录: $VIDEO_DIR"
echo "输出目录: $OUT_BASE"
echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo ""

# ── 预检：环境验证 ────────────────────────────────────────────────────────────
echo "=== [1/6] 环境预检 ==="
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')

import sana_wm_pipeline
print('✓ sana_wm_pipeline')

import vipe
from pi3 import Pi3X
from moge.model.v2 import MoGeModel
print('✓ vipe, pi3, moge')

import numpy as np, cv2
print(f'✓ numpy {np.__version__}, opencv {cv2.__version__}')
"

if [ $? -ne 0 ]; then
    echo "✗ 环境预检失败，请检查依赖"
    exit 1
fi

echo ""

# ── 检查视频目录 ──────────────────────────────────────────────────────────────
if [ ! -d "$VIDEO_DIR" ]; then
    echo "✗ 视频目录不存在: $VIDEO_DIR"
    exit 1
fi

VIDEO_FILES=($(find "$VIDEO_DIR" -name "*.mp4" | sort))
VIDEO_COUNT=${#VIDEO_FILES[@]}

if [ $VIDEO_COUNT -eq 0 ]; then
    echo "✗ 视频目录为空: $VIDEO_DIR"
    exit 1
fi

echo "=== [2/6] 发现视频文件 ==="
echo "视频数量: $VIDEO_COUNT"
for i in "${!VIDEO_FILES[@]}"; do
    VIDEO="${VIDEO_FILES[$i]}"
    BASENAME=$(basename "$VIDEO" .mp4)
    SIZE=$(du -sh "$VIDEO" | cut -f1)
    echo "  [$((i+1))/$VIDEO_COUNT] $BASENAME ($SIZE)"
done
echo ""

# ── 创建输出目录 ──────────────────────────────────────────────────────────────
mkdir -p "$OUT_BASE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUT_BASE/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "=== [3/6] 输出目录 ==="
echo "本次运行: $RUN_DIR"
echo ""

# ── 批量处理视频 ──────────────────────────────────────────────────────────────
cd "$PROJ_DIR"

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_VIDEOS=()

for i in "${!VIDEO_FILES[@]}"; do
    VIDEO="${VIDEO_FILES[$i]}"
    BASENAME=$(basename "$VIDEO" .mp4)
    SCENE_DIR="$RUN_DIR/$BASENAME"
    mkdir -p "$SCENE_DIR"

    echo "=== [4/6] 处理样本 [$((i+1))/$VIDEO_COUNT]: $BASENAME ==="

    # 复制原始视频到工作目录
    cp "$VIDEO" "$SCENE_DIR/video.mp4"

    # Stage 1: 归一化
    echo "  [Stage 1] 视频归一化..."
    NORM_VIDEO="$SCENE_DIR/normalized.mp4"
    python -c "
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
try:
    info = normalize_video(Path('$SCENE_DIR/video.mp4'), Path('$NORM_VIDEO'))
    print(f'  ✓ 归一化完成: {info.n_frames} 帧 @ {info.fps}fps ({info.width}x{info.height})')
except Exception as e:
    print(f'  ✗ 归一化失败: {e}')
    exit(1)
" 2>&1 | tee "$SCENE_DIR/stage1.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 1 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 1)")
        continue
    fi

    # Stage 2: VIPE SLAM
    echo "  [Stage 2] VIPE SLAM (Pi3X + MoGe-2)..."
    VIPE_WORK="$SCENE_DIR/vipe_work_default"
    ARTIFACT_JSON="$VIPE_WORK/pose_artifact_default.json"
    mkdir -p "$VIPE_WORK"

    python -c "
import json
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_default import run_default

try:
    art = run_default(Path('$NORM_VIDEO'), Path('$VIPE_WORK'))
    print(f'  ✓ SLAM 完成: poses {art.poses_c2w.shape}, intrinsics {art.intrinsics.shape}')

    # 保存结果
    Path('$ARTIFACT_JSON').write_text(json.dumps({
        'poses_c2w': art.poses_c2w.tolist(),
        'intrinsics': art.intrinsics.tolist(),
        'scale_per_frame': art.scale_per_frame.tolist(),
    }))
except Exception as e:
    print(f'  ✗ SLAM 失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" </dev/null 2>&1 | tee "$SCENE_DIR/stage2.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 2 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 2)")
        continue
    fi

    # Stage 6: 打包 WebDataset
    echo "  [Stage 6] 打包 WebDataset shard..."
    SHARDS_DIR="$SCENE_DIR/shards"
    mkdir -p "$SHARDS_DIR"
    SHARD="$SHARDS_DIR/$BASENAME.tar"

    python - <<PYEOF
import io, json, numpy as np, tarfile
from pathlib import Path

try:
    scene_id = "$BASENAME"
    art = json.loads(Path("$ARTIFACT_JSON").read_text())
    poses = np.array(art["poses_c2w"], np.float32)
    intr = np.array(art["intrinsics"], np.float32)
    scale = np.array(art["scale_per_frame"], np.float32)

    def add_npy(tf, key, arr):
        b = io.BytesIO()
        np.save(b, arr)
        raw = b.getvalue()
        ti = tarfile.TarInfo(f"{scene_id}.{key}")
        ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))

    with tarfile.open("$SHARD", "w") as tf:
        # 视频
        vb = Path("$NORM_VIDEO").read_bytes()
        ti = tarfile.TarInfo(f"{scene_id}.mp4")
        ti.size = len(vb)
        tf.addfile(ti, io.BytesIO(vb))

        # Pose 数据
        add_npy(tf, "poses_c2w.npy", poses)
        add_npy(tf, "intrinsics.npy", intr)
        add_npy(tf, "scale.npy", scale)

        # Caption（占位）
        cap = "CMCC smoke test video"
        cb = cap.encode()
        ti = tarfile.TarInfo(f"{scene_id}.caption.txt")
        ti.size = len(cb)
        tf.addfile(ti, io.BytesIO(cb))

        # Metadata
        meta = json.dumps({
            "scene_id": scene_id,
            "T": len(poses),
            "mode": "default",
            "dataset": "cmcc_smoke_pass",
            "group": "smoke-test"
        }).encode()
        ti = tarfile.TarInfo(f"{scene_id}.meta.json")
        ti.size = len(meta)
        tf.addfile(ti, io.BytesIO(meta))

    print(f'  ✓ Shard 打包完成: {len(poses)} 帧')
except Exception as e:
    print(f'  ✗ 打包失败: {e}')
    exit(1)
PYEOF

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 6 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 6)")
        continue
    fi

    echo "  ✓ 样本处理完成: $BASENAME"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo ""
done

# ── 生成测试报告 ──────────────────────────────────────────────────────────────
echo "=== [5/6] 生成测试报告 ==="

REPORT_FILE="$RUN_DIR/smoke_test_report.txt"
cat > "$REPORT_FILE" <<EOF
CMCC 冒烟测试报告
==================

运行时间: $(date)
输出目录: $RUN_DIR

测试结果
--------
总样本数: $VIDEO_COUNT
成功: $SUCCESS_COUNT
失败: $FAIL_COUNT

EOF

if [ $FAIL_COUNT -gt 0 ]; then
    echo "失败样本:" >> "$REPORT_FILE"
    for failed in "${FAILED_VIDEOS[@]}"; do
        echo "  - $failed" >> "$REPORT_FILE"
    done
    echo "" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" <<EOF
详细日志
--------
每个样本的详细日志保存在对应目录下:
  - stage1.log: 视频归一化日志
  - stage2.log: VIPE SLAM 日志

EOF

cat "$REPORT_FILE"

# ── 最终结果 ──────────────────────────────────────────────────────────────────
echo ""
echo "=== [6/6] 测试完成 ==="
echo "报告文件: $REPORT_FILE"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓✓✓ 冒烟测试全部通过 ($SUCCESS_COUNT/$VIDEO_COUNT) ✓✓✓"
    exit 0
else
    echo "✗✗✗ 冒烟测试部分失败 (成功: $SUCCESS_COUNT, 失败: $FAIL_COUNT) ✗✗✗"
    exit 1
fi
