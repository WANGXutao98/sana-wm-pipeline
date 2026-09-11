#!/bin/bash
# 移除 -e 选项，手动检查每个命令的返回值以便捕获详细错误
set -uo pipefail

# ====================================================================
# DOVER + UniMatch 冒烟测试脚本 (CMCC 环境) - 增强错误诊断版本
# ====================================================================
# 用途: 验证 CMCC 环境下 Stage3 视频质量筛选流水线是否正常工作
# 创建日期: 2026-09-02
# 版本: v2 - 增强错误捕获
# ====================================================================

# ── 路径配置 ──
export NEW_BASE="/root/work/david_work"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"
export VIDEO_DIR="$NEW_BASE/smoke_pass_videos"
export OUT_DIR="$NEW_BASE/stage3_smoke_results"

# ── 环境路径 ──
export ENV_QC="$NEW_BASE/envs/sana_qc_cmcc"
export PYTHON="$ENV_QC/bin/python3"

# ── GPU 设置 ──
export CUDA_VISIBLE_DEVICES=0

# ── 模型权重与源码路径 ──
export DOVER_PATH="$PROJ_DIR/models/DOVER"
export UNIMATCH_PATH="$PROJ_DIR/models/unimatch"
export DOVER_WEIGHTS="$DOVER_PATH/pretrained_weights/DOVER.pth"
export UNIMATCH_WEIGHTS="$UNIMATCH_PATH/pretrained/gmflow-scale2-regrefine6-mixdata.pth"

# ── 缓存路径（必须！让 DOVER 使用本地 ConvNeXt 权重）──
export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"

# ── 离线模式 ──
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── PYTHONPATH（必须包含 DOVER 和 UniMatch）──
export PYTHONPATH="$PROJ_DIR/src:$DOVER_PATH:$UNIMATCH_PATH${PYTHONPATH:+:$PYTHONPATH}"

# ── 创建输出目录 ──
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "DOVER + UniMatch 冒烟测试 (CMCC) - v2"
echo "=========================================="
echo "环境: $ENV_QC"
echo "DOVER 路径: $DOVER_PATH"
echo "UniMatch 路径: $UNIMATCH_PATH"
echo "视频目录: $VIDEO_DIR"
echo "输出目录: $OUT_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# ── 环境预检 ──
echo "=== [1/6] 环境预检 ==="
$PYTHON -c "
import torch
print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')

import sys
sys.path.insert(0, '$DOVER_PATH')
sys.path.insert(0, '$UNIMATCH_PATH')

from dover.models import DOVER
print('✓ DOVER')

from unimatch.unimatch import UniMatch
print('✓ UniMatch')

import decord
print(f'✓ decord {decord.__version__}')

import cv2
print(f'✓ opencv {cv2.__version__}')
"

if [ $? -ne 0 ]; then
    echo "✗ 环境预检失败"
    exit 1
fi

# ── 检查模型权重 ──
echo ""
echo "=== [2/6] 检查模型权重 ==="
if [ ! -f "$DOVER_WEIGHTS" ]; then
    echo "✗ DOVER 权重不存在: $DOVER_WEIGHTS"
    exit 1
fi
echo "✓ DOVER 权重: $(ls -lh $DOVER_WEIGHTS | awk '{print $5}')"

if [ ! -f "$UNIMATCH_WEIGHTS" ]; then
    echo "✗ UniMatch 权重不存在: $UNIMATCH_WEIGHTS"
    exit 1
fi
echo "✓ UniMatch 权重: $(ls -lh $UNIMATCH_WEIGHTS | awk '{print $5}')"

# ── 发现视频文件 ──
echo ""
echo "=== [3/6] 发现视频文件 ==="
VIDEO_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -type f -name "*.mp4" | wc -l)
echo "视频数量: $VIDEO_COUNT"

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "✗ 未找到测试视频"
    exit 1
fi

echo "测试样本（前3个）:"
find "$VIDEO_DIR" -maxdepth 1 -type f -name "*.mp4" | head -n 3 | while read video; do
    echo "  - $(basename $video)"
done

# ── 检查 Python 脚本是否存在 ──
echo ""
echo "=== [预检] 检查 Python 脚本 ==="
SCRIPT_PATH="$PROJ_DIR/scripts/stage3_batch_minimal_cmcc.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "✗ Python 脚本不存在: $SCRIPT_PATH"
    exit 1
fi
echo "✓ Python 脚本存在: $SCRIPT_PATH"
ls -lh "$SCRIPT_PATH"

# ── 测试脚本是否可以导入（不执行，只检查语法）──
echo ""
echo "=== [预检] 检查脚本语法 ==="
$PYTHON -m py_compile "$SCRIPT_PATH"
if [ $? -ne 0 ]; then
    echo "✗ Python 脚本语法错误"
    exit 1
fi
echo "✓ 脚本语法正确"

# ── 运行 stage3 脚本（使用 CMCC 适配版本）──
echo ""
echo "=== [4/6] 运行 DOVER+UniMatch 筛选 ==="
echo "切换到项目目录: $PROJ_DIR"
cd "$PROJ_DIR"

echo "执行命令:"
echo "$PYTHON scripts/stage3_batch_minimal_cmcc.py \\"
echo "  --input_dir \"$VIDEO_DIR\" \\"
echo "  --output \"$OUT_DIR/stage3_smoke_results.jsonl\" \\"
echo "  --dover_path \"$DOVER_PATH\" \\"
echo "  --unimatch_path \"$UNIMATCH_PATH\" \\"
echo "  --resume"
echo ""

# 使用 set +e 临时允许命令失败，以便捕获错误
set +e
$PYTHON scripts/stage3_batch_minimal_cmcc.py \
  --input_dir "$VIDEO_DIR" \
  --output "$OUT_DIR/stage3_smoke_results.jsonl" \
  --dover_path "$DOVER_PATH" \
  --unimatch_path "$UNIMATCH_PATH" \
  --resume 2>&1 | tee "$OUT_DIR/python_output.log"

PYTHON_EXIT_CODE=$?
set -e

echo ""
echo "Python 脚本退出码: $PYTHON_EXIT_CODE"

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "✗ Stage3 脚本运行失败（退出码: $PYTHON_EXIT_CODE）"
    echo ""
    echo "最后 50 行输出:"
    tail -n 50 "$OUT_DIR/python_output.log"
    echo ""
    echo "完整日志: $OUT_DIR/python_output.log"
    exit 1
fi

echo "✓ Stage3 脚本执行成功"

# ── 检查输出结果 ──
echo ""
echo "=== [5/6] 检查输出结果 ==="
if [ -f "$OUT_DIR/stage3_smoke_results.jsonl" ]; then
    LINE_COUNT=$(wc -l < "$OUT_DIR/stage3_smoke_results.jsonl")
    echo "✓ 输出文件存在: $LINE_COUNT 条结果"
    echo ""
    echo "前 3 条结果:"
    head -n 3 "$OUT_DIR/stage3_smoke_results.jsonl" | python3 -m json.tool || head -n 3 "$OUT_DIR/stage3_smoke_results.jsonl"

    # 验证分数范围
    echo ""
    echo "分数统计:"
    $PYTHON -c "
import json
import sys

results = []
with open('$OUT_DIR/stage3_smoke_results.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

if results:
    dover_scores = [r['dover_fused'] for r in results if 'dover_fused' in r and r['dover_fused'] is not None]
    unimatch_scores = [r['unimatch_flow'] for r in results if 'unimatch_flow' in r and r['unimatch_flow'] is not None]

    if dover_scores:
        print(f'  DOVER fused: min={min(dover_scores):.3f}, max={max(dover_scores):.3f}, avg={sum(dover_scores)/len(dover_scores):.3f}')
        if max(dover_scores) < 0.3:
            print('  ⚠️  警告: DOVER 分数异常低（应在 0.3-1.0 范围）')
    else:
        print('  ⚠️  警告: 无有效 DOVER 分数')

    if unimatch_scores:
        print(f'  UniMatch flow: min={min(unimatch_scores):.2f}, max={max(unimatch_scores):.2f}, avg={sum(unimatch_scores)/len(unimatch_scores):.2f}')
    else:
        print('  ⚠️  警告: 无有效 UniMatch 分数')

    pass_count = sum(1 for r in results if r.get('verdict') == 'pass')
    error_count = sum(1 for r in results if r.get('verdict') == 'error')
    print(f'  通过: {pass_count}/{len(results)} ({pass_count*100/len(results):.1f}%)')
    if error_count > 0:
        print(f'  错误: {error_count}/{len(results)} ({error_count*100/len(results):.1f}%)')
"
else
    echo "✗ 输出文件不存在: $OUT_DIR/stage3_smoke_results.jsonl"
    exit 1
fi

# ── 测试完成 ──
echo ""
echo "=== [6/6] 测试完成 ==="
echo "✓✓✓ DOVER+UniMatch 冒烟测试通过 ✓✓✓"
echo ""
echo "输出文件: $OUT_DIR/stage3_smoke_results.jsonl"
echo "Python 日志: $OUT_DIR/python_output.log"
echo ""
echo "下一步："
echo "  1. 检查分数范围是否合理（DOVER: 0.3-1.0, UniMatch: 0-100）"
echo "  2. 如果通过，可以启动全量生产任务"
