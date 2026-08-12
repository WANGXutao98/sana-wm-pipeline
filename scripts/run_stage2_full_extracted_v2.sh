#!/bin/bash
# Stage 2 全量执行脚本 v2 - 修复索引性能问题
# 使用 stage2_deep_extracted_v2.py（预构建索引）

set -e

# ==================== 配置 ====================
DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"
OUTPUT_ROOT="/root/work/david_work/qc_output_new"
SAMPLE_FRAC=1.0
N_WORKERS=32

# 数据集列表
DATASETS=(
    "wds-sekai-game-drone"
    "wds-sekai-game-walking"
    "wds-OmniWorld-Game"
    "wds-DL3DV-ALL-2K"
    "wds-sekai-real-walking-hq"
    "wds-RealEstate10K-360p"
    "wds-SpatialVID-hq"
)

# ==================== 环境检查 ====================
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] 数据根目录不存在: $DATA_ROOT"
    exit 1
fi

if [ -z "$PYTHONPATH" ]; then
    echo "[WARNING] PYTHONPATH 未设置，尝试自动设置..."
    export PYTHONPATH="/root/work/david_work/sana_wm_qc/src:$PYTHONPATH"
fi

# 测试模块导入
python -c "from sana_wm_pipeline.qc.stage2_deep_extracted_v2 import run_stage2_extracted_v2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Stage 2 v2 模块导入失败，请检查 PYTHONPATH"
    exit 1
fi

# ==================== 主循环 ====================
LOG_FILE="$OUTPUT_ROOT/../stage2_batch_full_v2_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] 主日志: $LOG_FILE"
echo "Stage 2 v2 全量执行开始" > "$LOG_FILE"
echo "时间: $(date)" >> "$LOG_FILE"
echo "数据根目录: $DATA_ROOT" >> "$LOG_FILE"
echo "采样率: $SAMPLE_FRAC" >> "$LOG_FILE"
echo "并发数: $N_WORKERS" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

for group in "${DATASETS[@]}"; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "处理数据集: $group" | tee -a "$LOG_FILE"
    echo "开始时间: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    group_dir="$OUTPUT_ROOT/$group"
    s1_jsonl="$group_dir/stage1_results.jsonl"
    s2_jsonl="$group_dir/stage2_results_full_v2.jsonl"
    run_log="$group_dir/stage2_run_full_v2.log"

    mkdir -p "$group_dir"

    if [ ! -f "$s1_jsonl" ]; then
        echo "[SKIP] Stage 1 结果不存在: $s1_jsonl" | tee -a "$LOG_FILE"
        continue
    fi

    python -u -c "
from pathlib import Path
from sana_wm_pipeline.qc.stage2_deep_extracted_v2 import run_stage2_extracted_v2

n = run_stage2_extracted_v2(
    Path('$s1_jsonl'),
    Path('$s2_jsonl'),
    Path('$DATA_ROOT'),
    sample_frac=$SAMPLE_FRAC,
    n_workers=$N_WORKERS
)
print(f'[完成] 处理了 {n} 个样本')
    " 2>&1 | tee "$run_log" | tee -a "$LOG_FILE"

    echo "完成时间: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "========================================" | tee -a "$LOG_FILE"
echo "Stage 2 v2 全量执行完成" | tee -a "$LOG_FILE"
echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
