#!/bin/bash
# 用法: bash run_groups_sequential.sh [--batch1-only | --groups G1,G2,...] [NODE_RANK=0] [NUM_NODES=1]
# OUT_BASE/DATA_ROOT 通过调用前 export 覆盖（见 launch_all_nodes.sh 双参数透传）。
#
# --groups：显式指定本次处理的 group 列表，跳过 BATCH1_GROUPS + 自动发现"剩余全部"的逻辑。
# 多个物理节点组（多个独立 launch_all_nodes.sh 提交）同时跑同一个 DATA_ROOT 时，
# 必须用 --groups 给每组节点分配互斥的 group 列表，否则两组节点各自的"剩余全部"
# 自动发现都会覆盖全量 group，必然重复处理、并在 worker 输出目录上发生写冲突
# （worker_id 命名只看 NODE_RANK*8+LOCAL_GPU，不含任何 job 标识）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

BATCH1_ONLY=0
CUSTOM_GROUPS=""
if [[ "${1:-}" == "--batch1-only" ]]; then
    BATCH1_ONLY=1; shift
elif [[ "${1:-}" == "--groups" ]]; then
    CUSTOM_GROUPS="${2:?用法: --groups G1,G2,...}"; shift 2
fi
NODE_RANK="${1:-0}"
NUM_NODES="${2:-1}"

# 多节点必须显式指定 group（避免两组节点各自"发现剩余全部"造成 worker 目录写冲突）
if [[ $NUM_NODES -gt 1 && -z "$CUSTOM_GROUPS" && $BATCH1_ONLY -eq 0 ]]; then
    echo "[ERROR] 多节点(NUM_NODES=$NUM_NODES)必须用 --groups 或 --batch1-only 显式指定 group，禁止自动发现全部"
    exit 1
fi

# 确定处理队列
if [[ -n "$CUSTOM_GROUPS" ]]; then
    IFS=',' read -ra GROUPS_TO_RUN <<< "$CUSTOM_GROUPS"
else
    GROUPS_TO_RUN=("${BATCH1_GROUPS[@]}")
    if [[ $BATCH1_ONLY -eq 0 ]]; then
        ALL=($(ls "$DATA_ROOT/" 2>/dev/null | grep "^wds-" | sort))
        BATCH1_SET=" ${BATCH1_GROUPS[*]} "
        for G in "${ALL[@]}"; do
            [[ "$BATCH1_SET" == *" $G "* ]] || GROUPS_TO_RUN+=("$G")
        done
    fi
fi

echo "=== 串行处理 ${#GROUPS_TO_RUN[@]} 个 group ==="
printf '  %s\n' "${GROUPS_TO_RUN[@]}"

OVERALL_FAIL=0
for GROUP in "${GROUPS_TO_RUN[@]}"; do
    echo ""
    echo "══ 开始: $GROUP  $(date) ══"
    START=$SECONDS
    bash "$SCRIPT_DIR/launch_single_node.sh" "$GROUP" "$NODE_RANK" "$NUM_NODES" \
        || { echo "[WARN] $GROUP 有 worker 失败，继续下一个"; OVERALL_FAIL=$((OVERALL_FAIL+1)); }
    ELAPSED=$((SECONDS-START))
    echo "  $GROUP 耗时: $((ELAPSED/3600))h$((ELAPSED%3600/60))m"
done

echo "=== 全部完成 | 失败 group: $OVERALL_FAIL ==="
exit $OVERALL_FAIL

