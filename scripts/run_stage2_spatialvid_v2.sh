#!/bin/bash
# Stage 2 V2 - SpatialVID 单独执行脚本
# 使用修复后的索引性能版本

set -e

echo "========================================"
echo "Stage 2 V2 - SpatialVID 执行"
echo "开始时间: $(date)"
echo "========================================"
echo ""

# ==================== 配置 ====================
DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"
OUTPUT_DIR="/root/work/david_work/qc_output_new/wds-SpatialVID-hq"
S1_JSONL="$OUTPUT_DIR/stage1_results.jsonl"
S2_JSONL="$OUTPUT_DIR/stage2_results_full_v2.jsonl"
LOG_FILE="$OUTPUT_DIR/stage2_run_full_v2.log"

SAMPLE_FRAC=1.0
N_WORKERS=32

# ==================== 环境检查 ====================
echo "[检查] 数据根目录: $DATA_ROOT"
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] 数据根目录不存在"
    exit 1
fi

echo "[检查] Stage 1 结果: $S1_JSONL"
if [ ! -f "$S1_JSONL" ]; then
    echo "[ERROR] Stage 1 结果不存在"
    exit 1
fi

echo "[检查] Python 环境"
if [ -z "$PYTHONPATH" ]; then
    echo "[WARNING] PYTHONPATH 未设置，尝试自动设置..."
    export PYTHONPATH="/root/work/david_work/sana_wm_qc/src:$PYTHONPATH"
fi

python -c "from sana_wm_pipeline.qc.stage2_deep_extracted_v2 import run_stage2_extracted_v2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Stage 2 V2 模块导入失败"
    echo "请确认："
    echo "  1. 已部署 stage2_deep_extracted_v2.py"
    echo "  2. PYTHONPATH 包含 src 目录"
    exit 1
fi

echo "[检查] 输出目录: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "[INFO] 配置参数:"
echo "  数据根目录: $DATA_ROOT"
echo "  Stage 1 输入: $S1_JSONL"
echo "  Stage 2 输出: $S2_JSONL"
echo "  采样率: $SAMPLE_FRAC (全量)"
echo "  并发数: $N_WORKERS"
echo "  日志文件: $LOG_FILE"
echo ""

# ==================== 执行 Stage 2 ====================
echo "========================================"
echo "开始处理 wds-SpatialVID-hq"
echo "========================================"
echo ""

python -u << PYEOF 2>&1 | tee "$LOG_FILE"
from pathlib import Path
from sana_wm_pipeline.qc.stage2_deep_extracted_v2 import run_stage2_extracted_v2
import time

print("[stage2-v2] 开始执行")
print(f"[stage2-v2] Stage 1 输入: $S1_JSONL")
print(f"[stage2-v2] Stage 2 输出: $S2_JSONL")
print(f"[stage2-v2] 数据根目录: $DATA_ROOT")
print()

start_time = time.time()

try:
    n = run_stage2_extracted_v2(
        Path("$S1_JSONL"),
        Path("$S2_JSONL"),
        Path("$DATA_ROOT"),
        sample_frac=$SAMPLE_FRAC,
        n_workers=$N_WORKERS
    )

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60
    elapsed_hr = elapsed_min / 60

    print()
    print("=" * 60)
    print(f"✅ Stage 2 完成！")
    print(f"   处理样本数: {n}")
    print(f"   耗时: {elapsed_hr:.2f} 小时 ({elapsed_min:.1f} 分钟)")
    print(f"   输出文件: $S2_JSONL")
    print("=" * 60)

except Exception as e:
    print()
    print("=" * 60)
    print(f"❌ Stage 2 执行失败")
    print(f"   错误: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    exit(1)

PYEOF

exit_code=$?

# ==================== 结果统计 ====================
echo ""
echo "========================================"
echo "执行结果统计"
echo "========================================"

if [ -f "$S2_JSONL" ]; then
    total_lines=$(wc -l < "$S2_JSONL")
    echo "输出样本数: $total_lines"

    # 统计有问题的样本
    if command -v jq &> /dev/null; then
        flagged=$(jq -r 'select(.stage2.reasons | length > 0) | .sample_id' "$S2_JSONL" 2>/dev/null | wc -l)
        echo "标记问题样本: $flagged"

        if [ $flagged -gt 0 ]; then
            echo ""
            echo "问题类型分布（前 10 种）:"
            jq -r '.stage2.reasons[]' "$S2_JSONL" 2>/dev/null | sort | uniq -c | sort -rn | head -10
        fi
    fi

    file_size=$(du -h "$S2_JSONL" | cut -f1)
    echo "文件大小: $file_size"
else
    echo "❌ 输出文件不存在"
fi

echo ""
echo "========================================"
echo "执行完成"
echo "结束时间: $(date)"
echo "退出码: $exit_code"
echo "========================================"

exit $exit_code
