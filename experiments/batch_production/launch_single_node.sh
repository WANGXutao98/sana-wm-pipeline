#!/bin/bash
# 用法: bash launch_single_node.sh <GROUP> [NODE_RANK=0] [NUM_NODES=1]
# 日志增强版 + 完全对齐smoke脚本环境
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

GROUP="${1:?用法: $0 <GROUP>}"
NODE_RANK="${2:-0}"
NUM_NODES="${3:-1}"
NUM_GPUS="${NUM_GPUS:-8}"  # 支持通过环境变量临时修改GPU数量

# ==================== 总日志配置 ====================
LOG_DIR="$OUT_BASE/$GROUP/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/node${NODE_RANK}_main_${TIMESTAMP}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "========================================"
echo "SANA-WM 单节点批量生产启动"
echo "启动时间: $(date)"
echo "运行命令: $0 $*"
echo "节点编号: $NODE_RANK/$NUM_NODES"
echo "GPU数量: $NUM_GPUS"
echo "数据集组: $GROUP"
echo "数据根目录: $DATA_ROOT"
echo "输出根目录: $OUT_BASE"
echo "总日志文件: $MAIN_LOG"
echo "========================================"

# ==================== ✅ 完全对齐smoke脚本的环境初始化 ====================
# 1. 全局强制PYTHONNOUSERSITE（CMCC平台最关键！阻止系统库干扰）
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# 2. 环境验证（和smoke脚本完全一致）
echo "=== 环境验证 ==="
python -c "
import sana_wm_pipeline; print('sana_wm_pipeline ✓')
import vipe_ext;          print('vipe_ext ✓')
import vipe;              print('vipe ✓')
import torch;             print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} ✓')
print(f'Torch CUDA版本: {torch.version.cuda}')
"
# ======================================================================

SHARD_DIR="$DATA_ROOT/$GROUP/shards"
[[ -d "$SHARD_DIR" ]] || { echo "[ERROR] 不存在: $SHARD_DIR"; exit 1; }
SHARDS=($(ls "$SHARD_DIR"/*.tar 2>/dev/null | sort))
TOTAL_SHARDS=${#SHARDS[@]}
[[ $TOTAL_SHARDS -gt 0 ]] || { echo "[ERROR] 没有 .tar 文件"; exit 1; }

GLOBAL_WORKERS=$((NUM_NODES * NUM_GPUS))
echo "GROUP=$GROUP  TOTAL_SHARDS=$TOTAL_SHARDS  全局Worker总数: $GLOBAL_WORKERS"

mkdir -p "$OUT_BASE/$GROUP/logs"
PIDS=()
for LOCAL_GPU in $(seq 0 $((NUM_GPUS - 1))); do
    GLOBAL_WORKER=$((NODE_RANK * NUM_GPUS + LOCAL_GPU))
    INDICES=""
    for IDX in $(seq $GLOBAL_WORKER $GLOBAL_WORKERS $((TOTAL_SHARDS - 1))); do
        INDICES="${INDICES}${IDX},"
    done
    INDICES="${INDICES%,}"
    [[ -z "$INDICES" ]] && { echo "  GPU $LOCAL_GPU: 无 shard，跳过"; continue; }
    LOG="$OUT_BASE/$GROUP/logs/node${NODE_RANK}_gpu${LOCAL_GPU}.log"
    echo "  GPU $LOCAL_GPU → global_worker=$GLOBAL_WORKER  shards=[$INDICES]  日志: $LOG"
    
    CUDA_VISIBLE_DEVICES=$LOCAL_GPU \
    python -u "$PROJ_DIR/experiments/batch_production/run_worker.py" \
        --group "$GROUP" \
        --data-root "$DATA_ROOT" \
        --out-base "$OUT_BASE" \
        --worker-id $GLOBAL_WORKER \
        --shard-indices "$INDICES" \
        --samples-per-shard "${SAMPLES_PER_OUTPUT_SHARD:-200}" \
        >> "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "等待 ${#PIDS[@]} 个 worker 完成..."
START_TIME=$SECONDS
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "  worker $i 失败"; FAILED=$((FAILED+1)); }
done
TOTAL_ELAPSED=$((SECONDS - START_TIME))

DONE_CNT=$(ls "$OUT_BASE/$GROUP/progress/"*.done 2>/dev/null | wc -l) || true
N_OK_TOTAL=$(grep -ho '"n_ok": [0-9]*' "$OUT_BASE/$GROUP/progress/"*.done 2>/dev/null | awk -F': ' '{s+=$2} END{print s+0}') || true
N_FAIL_TOTAL=$(grep -ho '"n_fail": [0-9]*' "$OUT_BASE/$GROUP/progress/"*.done 2>/dev/null | awk -F': ' '{s+=$2} END{print s+0}') || true

echo "========================================"
echo "节点 $NODE_RANK 运行完成"
echo "完成时间: $(date)"
echo "总耗时: $((TOTAL_ELAPSED/3600))h$((TOTAL_ELAPSED%3600/60))m$((TOTAL_ELAPSED%60))s"
echo "Worker失败数: $FAILED"
echo "Shard进度: $DONE_CNT/$TOTAL_SHARDS ($((DONE_CNT*100/TOTAL_SHARDS))%)"
echo "成功样本数: $N_OK_TOTAL"
echo "失败样本数: $N_FAIL_TOTAL"
echo "总日志文件: $MAIN_LOG"
echo "========================================"

exit $FAILED
