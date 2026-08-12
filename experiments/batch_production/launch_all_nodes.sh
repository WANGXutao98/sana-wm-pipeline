#!/bin/bash
# 多节点批量生产总入口（CMCC VC 调度器，无 SLURM/torchrun 自动 rank）
#
# 用法:
#   bash launch_all_nodes.sh [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>
# 示例:
#   bash launch_all_nodes.sh wds-sekai-real-walking-hq \
#        /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_sekai_realwalking_hq \
#        /root/work/david_work/sana-wm-48-hostfiles/hostfile
#
# 设计要点（见 docs/.../2026-06-18-cmcc-batch-production-hardening.md 根因诊断）：
#   - 远程环境唯一真源 = source config.sh（含 activate + offline + alloc），
#     绝不向远程注入 master 本机 PATH/LD_LIBRARY_PATH（旧 bug：单节点 CUDA 失效）
#   - source config.sh 后 unset CUDA_VISIBLE_DEVICES（节点镜像预置为空串=0卡可见）
#   - pip install -e 在预检阶段每节点串行装一次（不在 worker 热路径并发装）
#   - rank = 存活节点稠密下标；NUM_NODES = 存活节点数
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

CHECK_ONLY=0
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=1 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
    shift
done

DATASET="${1:?用法: $0 [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>}"
OUT_PATH="${2:?用法: $0 [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>}"
HOSTFILE="${3:?用法: $0 [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>}"
[[ -f "$HOSTFILE" ]] || { echo "[ERROR] hostfile 不存在: $HOSTFILE"; exit 1; }

# 本次运行的输出根：覆盖 config.sh 默认，并向所有远程 SSH 透传
export OUT_BASE="$OUT_PATH"

echo "========================================"
echo "数据集(group): $DATASET"
echo "输出根 OUT_BASE: $OUT_BASE"
echo "hostfile: $HOSTFILE"
echo "数据根 DATA_ROOT: $DATA_ROOT"
echo "========================================"

# ── 解析 hostfile（防 CRLF / 注释 / 空行 / 纯数字误填）────────────────────────
mapfile -t RAW_LINES < "$HOSTFILE"
LINES=()
for line in "${RAW_LINES[@]}"; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    NODE_ID=$(awk '{print $1}' <<< "$line")
    if [[ "$NODE_ID" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] 行节点名 '$NODE_ID' 是纯数字，会被 SSH 当无效 IP，请修 hostfile"; exit 1
    fi
    LINES+=("$line")
done
ORIGINAL_NUM_NODES=${#LINES[@]}
[[ $ORIGINAL_NUM_NODES -ge 1 ]] || { echo "[ERROR] hostfile 为空: $HOSTFILE"; exit 1; }

DRIVER_LOG_DIR="$OUT_BASE/driver_logs"
mkdir -p "$DRIVER_LOG_DIR"

##############################################################################
# 阶段1/2：预检（每节点串行：装包一次 + 环境/CUDA/GPU数 校验），剔除坏点
##############################################################################
echo -e "\n===== 阶段1/2：预检 $ORIGINAL_NUM_NODES 节点 ====="
VALID_LINES=()
for i in "${!LINES[@]}"; do
    NODE_ID=$(awk '{print $1}' <<< "${LINES[$i]}")
    SLOTS=$(awk -F'slots=' 'NF>1{print $2}' <<< "${LINES[$i]}"); SLOTS="${SLOTS:-8}"
    echo -e "\n--- 探测节点: $NODE_ID (期望 $SLOTS 卡) ---"

    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" "
        set -e
        # 非交互 SSH 默认不 source 这两个文件；节点的 CUDA forward-compat 路径常在此设置，
        # 缺了会导致 cu130 torch 在旧驱动(12040)上报 'driver too old / CUDA 不可用'。
        # 注意：这是在远程读远程的文件，与已删除的 master env 注入（master 端展开）无关。
        source /etc/profile 2>/dev/null || true
        source ~/.bashrc 2>/dev/null || true
        export OUT_BASE='$OUT_BASE'
        export DATA_ROOT='$DATA_ROOT'
        source '$PROJ_DIR/experiments/batch_production/config.sh'
        unset CUDA_VISIBLE_DEVICES

        test -x '$ENV_DIR/bin/python' || { echo '[FAIL] ENV_DIR 缺失'; exit 1; }
        test -f '$PROJ_DIR/experiments/batch_production/run_worker.py' || { echo '[FAIL] 代码缺失'; exit 1; }
        test -d '$DATA_ROOT/$DATASET/shards' || { echo '[FAIL] 数据集 shards 不存在: $DATASET'; exit 1; }
        mkdir -p '$OUT_BASE' || { echo '[FAIL] OUT_BASE 不可写'; exit 1; }

        # 每节点串行安装一次（不在 worker 热路径并发装，避免 editable-install 竞态）
        pip install --no-user -e '$PROJ_DIR' --no-deps --no-build-isolation --quiet

        python -c \"
import sana_wm_pipeline, vipe_ext, vipe, torch
assert torch.cuda.is_available(), 'CUDA 不可用'
n = torch.cuda.device_count()
assert n == $SLOTS, f'GPU 数量={n}，期望={$SLOTS}'
print(f'[OK] cuda={torch.version.cuda} gpus={n}')
\"
        echo '[OK] 节点就绪'
    "; then
        VALID_LINES+=("${LINES[$i]}")
    else
        echo "[WARN] $NODE_ID 预检未通过，已从本次任务池剔除"
    fi
done

LINES=("${VALID_LINES[@]}")
NUM_NODES=${#LINES[@]}
echo -e "\n===== 预检汇总：健康 $NUM_NODES / 原始 $ORIGINAL_NUM_NODES（剔除 $((ORIGINAL_NUM_NODES-NUM_NODES))）====="
[[ $NUM_NODES -ge 1 ]] || { echo "[ERROR] 无可用节点，中止"; exit 1; }

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "（--check-only：预检完成，不拉起任务）"; exit 0
fi

##############################################################################
# 阶段2/2：稠密 rank 拉起
##############################################################################
echo -e "\n===== 阶段2/2：用 $NUM_NODES 个健康节点拉起 ====="
for i in "${!LINES[@]}"; do
    NODE_ID=$(awk '{print $1}' <<< "${LINES[$i]}")
    LOG="$DRIVER_LOG_DIR/node${i}_driver.log"

    RUNNING=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
        "pgrep -f 'run_groups_sequential.sh|run_worker.py' | wc -l" 2>/dev/null) || RUNNING=0
    if [[ "${RUNNING:-0}" -gt 0 ]]; then
        echo "[WARN] rank $i ($NODE_ID) 已有任务在跑，先清理旧进程"
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
            "pkill -9 -f 'run_groups_sequential.sh|run_worker.py' 2>/dev/null || true" || true
        sleep 2
    fi

    echo "Rank $i → $NODE_ID (driver: $LOG)"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" "
        set -e
        source /etc/profile 2>/dev/null || true
        source ~/.bashrc 2>/dev/null || true
        export OUT_BASE='$OUT_BASE'
        export DATA_ROOT='$DATA_ROOT'
        source '$PROJ_DIR/experiments/batch_production/config.sh'
        unset CUDA_VISIBLE_DEVICES
        cd '$PROJ_DIR'
        {
            nohup bash experiments/batch_production/run_groups_sequential.sh --groups $DATASET $i $NUM_NODES
        } > '$LOG' 2>&1 < /dev/null &
        disown
    " || echo "[WARN] rank $i ($NODE_ID) SSH 启动失败"
done

echo -e "\n===== 拉起完毕：$NUM_NODES 节点 ====="
echo "监控: bash $SCRIPT_DIR/watch_progress.sh $DATASET $OUT_BASE"
echo "停止: bash $SCRIPT_DIR/stop_all_nodes.sh $HOSTFILE"
