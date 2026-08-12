#!/bin/bash
set -eo pipefail

# ========== 环境配置 ==========
cd /root/work/david_work/sana_wm_qc
source /root/work/david_work/sana_wm_qc_env/bin/activate
export PYTHONPATH=/root/work/david_work/sana_wm_qc/src:$PYTHONPATH

# ========== 前置校验 ==========
if ! python -c "from sana_wm_pipeline.qc.stage2_deep_extracted import run_stage2_extracted" &> /dev/null; then
    echo "[ERROR] Stage 2 解压版本模块导入失败，请检查代码与环境"
    exit 1
fi

# ========== 配置项 ==========
OUTPUT_ROOT="/root/work/david_work/qc_output_new"
DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"
N_WORKERS=32
SAMPLE_FRAC=1.0  # 全量处理 Pass + Flag 样本

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="/root/work/david_work/stage2_batch_full_${TIMESTAMP}.log"

declare -a DATASETS=(
    "wds-sekai-game-drone"
    "wds-sekai-game-walking"
    "wds-OmniWorld-Game"
    "wds-DL3DV-ALL-2K"
    "wds-sekai-real-walking-hq"
    "wds-RealEstate10K-360p"
    "wds-SpatialVID-hq"
)

# ========== 启动日志 ==========
SAMPLE_PCT=$(echo "scale=1; $SAMPLE_FRAC * 100" | bc)
echo "==========================================" | tee -a "$LOG_FILE"
echo "Stage 2 全量深度检查（解压数据版本）" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "采样率: ${SAMPLE_FRAC} (${SAMPLE_PCT}%)" | tee -a "$LOG_FILE"
echo "并发数: ${N_WORKERS}" | tee -a "$LOG_FILE"
echo "数据根目录: ${DATA_ROOT}" | tee -a "$LOG_FILE"
echo "模式: 从解压目录直接读取文件（无需 tar）" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# ========== 验证数据根目录 ==========
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] 数据根目录不存在: $DATA_ROOT" | tee -a "$LOG_FILE"
    exit 1
fi

# 检查是否有解压目录
if ! ls "$DATA_ROOT"/final_wds-* > /dev/null 2>&1; then
    echo "[ERROR] 数据根目录下没有 final_wds-* 目录" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[OK] 数据根目录验证通过" | tee -a "$LOG_FILE"

# ========== 主循环 ==========
SUCCESS_COUNT=0
FAIL_COUNT=0

for idx in "${!DATASETS[@]}"; do
    group="${DATASETS[$idx]}"
    echo "" | tee -a "$LOG_FILE"
    echo "==================== [ $((idx+1)) / ${#DATASETS[@]} ] $group ====================" | tee -a "$LOG_FILE"

    output_dir="$OUTPUT_ROOT/$group"
    s1_jsonl="$output_dir/stage1_results.jsonl"
    s2_jsonl="$output_dir/stage2_results_full.jsonl"  # 新文件名，避免覆盖旧结果
    mkdir -p "$output_dir"

    # 检查 Stage1 结果
    if [ ! -f "$s1_jsonl" ]; then
        echo "[ERROR] Stage 1 结果不存在: $s1_jsonl" | tee -a "$LOG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # 统计 Stage 1 样本数
    s1_total=$(wc -l < "$s1_jsonl" | tr -d ' ')
    s1_pass=$(grep -c '"verdict": "pass"' "$s1_jsonl" || echo 0)
    s1_flag=$(grep -c '"verdict": "flag"' "$s1_jsonl" || echo 0)
    s1_fail=$(grep -c '"verdict": "fail"' "$s1_jsonl" || echo 0)

    echo "[INFO] Stage 1 统计:" | tee -a "$LOG_FILE"
    echo "      总样本数: $s1_total" | tee -a "$LOG_FILE"
    echo "      Pass: $s1_pass | Flag: $s1_flag | Fail: $s1_fail" | tee -a "$LOG_FILE"

    # 检查 Stage2 是否已存在
    if [ -f "$s2_jsonl" ]; then
        existing=$(wc -l < "$s2_jsonl" | tr -d ' ')
        echo "[WARN] 已存在 Stage 2 结果: $existing 条" | tee -a "$LOG_FILE"

        if [ -t 0 ]; then
            read -p "是否覆盖？[y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "[SKIP] 跳过 $group" | tee -a "$LOG_FILE"
                continue
            fi
        else
            echo "[SKIP] 非交互模式，自动跳过 $group" | tee -a "$LOG_FILE"
            continue
        fi

        # 备份旧结果
        backup="${s2_jsonl}.bak.$(date +%Y%m%d_%H%M%S)"
        mv "$s2_jsonl" "$backup"
        echo "[INFO] 旧结果已备份: $backup" | tee -a "$LOG_FILE"
    fi

    # 执行 Stage 2（解压版本）
    start_time=$(date +%s)
    echo "[INFO] 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

    # 使用新的解压版本函数
    if python -c '
import sys
from pathlib import Path
from sana_wm_pipeline.qc.stage2_deep_extracted import run_stage2_extracted

s1_path = Path(sys.argv[1])
s2_path = Path(sys.argv[2])
data_root = Path(sys.argv[3])
sample_frac = float(sys.argv[4])
n_workers = int(sys.argv[5])

print(f"[stage2] Stage 1 输入: {s1_path}", flush=True)
print(f"[stage2] Stage 2 输出: {s2_path}", flush=True)
print(f"[stage2] 数据根目录: {data_root}", flush=True)
print(f"[stage2] 采样率: {sample_frac}", flush=True)
print(f"[stage2] 并发数: {n_workers}", flush=True)

n = run_stage2_extracted(s1_path, s2_path, data_root, sample_frac=sample_frac, n_workers=n_workers)
print(f"[stage2] ✅ 成功处理 {n} 个样本 → {s2_path}", flush=True)
' "$s1_jsonl" "$s2_jsonl" "$DATA_ROOT" "$SAMPLE_FRAC" "$N_WORKERS" 2>&1 | tee -a "$output_dir/stage2_run_full.log"
    then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        elapsed_min=$(echo "scale=2; $elapsed / 60" | bc)

        if [ -f "$s2_jsonl" ]; then
            total=$(wc -l < "$s2_jsonl" | tr -d ' ')
            if [ "$total" -gt 0 ]; then
                echo "[OK] 成功: $group" | tee -a "$LOG_FILE"
                echo "     深度检查样本数: $total" | tee -a "$LOG_FILE"
                echo "     耗时: ${elapsed}s (${elapsed_min} 分钟)" | tee -a "$LOG_FILE"

                # 统计跳过数量（预期处理数 - 实际处理数）
                expected=$((s1_pass + s1_flag))
                skipped=$((expected - total))
                if [ "$skipped" -gt 0 ]; then
                    echo "     跳过（文件不存在）: $skipped" | tee -a "$LOG_FILE"
                fi

                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo "[ERROR] 结果文件为空: $group" | tee -a "$LOG_FILE"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        else
            echo "[ERROR] 未生成结果文件: $group" | tee -a "$LOG_FILE"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "[ERROR] 执行异常: $group" | tee -a "$LOG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo "==========================================" | tee -a "$LOG_FILE"
done

# ========== 最终汇总 ==========
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Stage 2 全量批量检查完成（解压版本）" | tee -a "$LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "成功: $SUCCESS_COUNT | 失败: $FAIL_COUNT | 总数: ${#DATASETS[@]}" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "完整日志: $LOG_FILE" | tee -a "$LOG_FILE"

# ========== 生成汇总报告 ==========
echo "" | tee -a "$LOG_FILE"
echo "生成各数据集统计..." | tee -a "$LOG_FILE"

for group in "${DATASETS[@]}"; do
    output_dir="$OUTPUT_ROOT/$group"
    s2_jsonl="$output_dir/stage2_results_full.jsonl"

    if [ -f "$s2_jsonl" ]; then
        total=$(wc -l < "$s2_jsonl" | tr -d ' ')
        issues=$(grep -c '"reasons": \[' "$s2_jsonl" | grep -v '\[\]' || echo 0)
        echo "  $group: $total 样本，$issues 个问题" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "✅ 全部完成！" | tee -a "$LOG_FILE"
