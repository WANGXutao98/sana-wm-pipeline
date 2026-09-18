#!/bin/bash
# DL3DV GT-Pose 模式冒烟测试（自动扫描数据目录）
# 用法：bash experiments/data_production_smoke/smoke_dl3dv.sh
set -euo pipefail

# ── 路径配置（AFS开发机） ─────────────────────────────────────────────────────
export PROJ_DIR="/mnt/afs/davidwang/workspace/sana_wm_pipeline"
export ENV_DIR="/mnt/afs/davidwang/miniconda3/envs/sana_wm"
export OUT_BASE="/mnt/afs/davidwang/workspace/sana_test_data/smoke_result_dl3dv"
export DL3DV_DIR="/mnt/afs/davidwang/workspace/sana_test_data/smoke_dl3dv_test_sample"

# ── 模型权重 ──────────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="/mnt/afs/davidwang/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="/mnt/afs/davidwang/models/moge2"

# ── 性能配置 ──────────────────────────────────────────────────────────────────
# 官方默认：SANA_WM_MAX_FRAMES=64
# Pi3X 只在 64 帧子集上运行（节省 GPU 内存），GT poses 保持完整帧数
export SANA_WM_MAX_FRAMES=64

# ── 环境激活 ──────────────────────────────────────────────────────────────────
echo "=== 激活环境 ==="
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm
cd "$PROJ_DIR"

# ── 验证数据目录 ──────────────────────────────────────────────────────────────
echo "=== 验证数据目录 ==="
if [[ ! -d "$DL3DV_DIR" ]]; then
    echo "❌ 错误：数据目录不存在: $DL3DV_DIR"
    exit 1
fi

echo "数据目录: $DL3DV_DIR"
echo ""

# ── 自动扫描样本 ──────────────────────────────────────────────────────────────
echo "=== 扫描数据目录中的样本 ==="

SAMPLES_FILE="$OUT_BASE/samples.tsv"
mkdir -p "$OUT_BASE"

# 临时文件用于收集样本
TEMP_SAMPLES=$(mktemp)

# 扫描所有 .mp4 文件
for mp4_file in "$DL3DV_DIR"/*.mp4; do
    # 检查文件是否存在（处理通配符未匹配的情况）
    [[ -f "$mp4_file" ]] || continue

    # 提取 sample_id（去掉路径和 .mp4 扩展名）
    sample_id=$(basename "$mp4_file" .mp4)

    # 检查对应的 camera.npz 是否存在
    camera_file="$DL3DV_DIR/${sample_id}.camera.npz"
    if [[ ! -f "$camera_file" ]]; then
        echo "  ⚠️  跳过 $sample_id (缺少 camera.npz)"
        continue
    fi

    # 使用 Python 读取 camera.npz 获取帧数
    n_frames=$(python3 -c "
import numpy as np
import sys
try:
    data = np.load('$camera_file')
    print(data['c2w'].shape[0])
except Exception as e:
    print('0', file=sys.stderr)
    sys.exit(1)
")

    # 验证帧数是否有效
    if [[ "$n_frames" =~ ^[0-9]+$ ]] && [[ "$n_frames" -gt 0 ]]; then
        echo "$sample_id	$n_frames" >> "$TEMP_SAMPLES"
        echo "  ✅ $sample_id ($n_frames 帧)"
    else
        echo "  ⚠️  跳过 $sample_id (无法读取帧数)"
    fi
done

# 检查是否找到有效样本
if [[ ! -s "$TEMP_SAMPLES" ]]; then
    echo ""
    echo "❌ 错误：未在 $DL3DV_DIR 中找到有效样本"
    echo "   每个样本需要："
    echo "     - {sample_id}.mp4"
    echo "     - {sample_id}.camera.npz (包含 'c2w' 数据)"
    rm -f "$TEMP_SAMPLES"
    exit 1
fi

# 移动到最终位置
mv "$TEMP_SAMPLES" "$SAMPLES_FILE"

# 统计信息
SAMPLE_COUNT=$(wc -l < "$SAMPLES_FILE")
echo ""
echo "共发现 $SAMPLE_COUNT 个有效样本"
echo "样本列表已保存到: $SAMPLES_FILE"
echo ""

# ── 验证样本文件 ──────────────────────────────────────────────────────────────
echo "=== 验证样本文件完整性 ==="
MISSING_COUNT=0
while IFS=$'\t' read -r sample_id n_frames; do
    VIDEO_FILE="$DL3DV_DIR/${sample_id}.mp4"
    CAMERA_FILE="$DL3DV_DIR/${sample_id}.camera.npz"

    if [[ -f "$VIDEO_FILE" && -f "$CAMERA_FILE" ]]; then
        VIDEO_SIZE=$(du -h "$VIDEO_FILE" | cut -f1)
        CAMERA_SIZE=$(du -h "$CAMERA_FILE" | cut -f1)
        echo "  ✅ $sample_id ($n_frames 帧)"
        echo "     video: $VIDEO_SIZE, camera: $CAMERA_SIZE"
    else
        echo "  ❌ $sample_id (缺失文件)"
        [[ ! -f "$VIDEO_FILE" ]] && echo "     缺失: ${sample_id}.mp4"
        [[ ! -f "$CAMERA_FILE" ]] && echo "     缺失: ${sample_id}.camera.npz"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done < "$SAMPLES_FILE"

echo ""
if [[ $MISSING_COUNT -gt 0 ]]; then
    echo "❌ 错误：有 $MISSING_COUNT 个样本文件缺失"
    exit 1
fi

echo "✅ 所有样本文件验证通过"
echo ""

# ── 显示测试配置 ──────────────────────────────────────────────────────────────
echo "=== 测试配置 ==="
echo "数据目录: $DL3DV_DIR"
echo "输出目录: $OUT_BASE"
echo "样本数量: $SAMPLE_COUNT"
echo "Pi3X 子采样: $SANA_WM_MAX_FRAMES 帧"
echo ""

# ── 执行批处理 ────────────────────────────────────────────────────────────────
echo "=== 启动批量处理 ==="
python "$PROJ_DIR/scripts/smoke_batch_gtpose.py" \
    --samples "$SAMPLES_FILE" \
    --dl3dv-dir "$DL3DV_DIR" \
    --output-dir "$OUT_BASE"

EXIT_CODE=$?

# ── 测试报告 ──────────────────────────────────────────────────────────────────
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "="*60
    echo "✅ DL3DV 冒烟测试通过"
    echo "="*60
    echo "数据目录: $DL3DV_DIR"
    echo "输出目录: $OUT_BASE"
    echo "样本数量: $SAMPLE_COUNT"
    echo ""
    echo "查看结果:"
    echo "  摘要: cat $OUT_BASE/smoke_test_summary.json"
    echo "  详细: ls -lh $OUT_BASE/*/"
    echo ""
    echo "验证要点:"
    echo "  1. 所有样本 status: success"
    echo "  2. Scale 值在合理范围（通常 1-10）"
    echo "  3. 输出帧数 = GT 帧数"
    echo "  4. 使用 GT intrinsics（不是 seed）"
else
    echo "="*60
    echo "❌ DL3DV 冒烟测试失败"
    echo "="*60
    echo "请查看日志排查问题"
fi

exit $EXIT_CODE
