#!/bin/bash
set -euo pipefail

[[ $# -ge 1 ]] || { echo "用法: $0 <HOSTFILE> [<HOSTFILE2> ...]"; exit 1; }
KILL_PATTERN="run_worker\.py|run_groups_sequential\.sh|launch_single_node\.sh"

for HOSTFILE in "$@"; do
    [[ -f "$HOSTFILE" ]] || { echo "ERROR: hostfile 不存在: $HOSTFILE"; exit 1; }
    echo -e "\n===== 停止 hostfile: $HOSTFILE ====="
    while read -r NODE_ID _; do
        NODE_ID="${NODE_ID%$'\r'}"
        [[ -z "$NODE_ID" || "$NODE_ID" =~ ^# ]] && continue
        echo "--- 节点 $NODE_ID ---"
        COUNT=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
            "pgrep -f '$KILL_PATTERN' | wc -l" 2>/dev/null) || COUNT=0
        if [[ "${COUNT:-0}" -gt 0 ]]; then
            echo "[1/2] $COUNT 进程 → SIGTERM"
            ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
                "pkill -TERM -f '$KILL_PATTERN' 2>/dev/null || true" || true
            sleep 2
            REMAIN=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
                "pgrep -f '$KILL_PATTERN' | wc -l" 2>/dev/null) || REMAIN=0
            if [[ "${REMAIN:-0}" -gt 0 ]]; then
                echo "[2/2] 残留 $REMAIN → SIGKILL"
                ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
                    "pkill -KILL -f '$KILL_PATTERN' 2>/dev/null || true" || true
            fi
            echo "节点 $NODE_ID 已停 ✅"
        else
            echo "节点 $NODE_ID 无进程，跳过 ✅"
        fi
    done < "$HOSTFILE"
done
echo -e "\n===== 全部 hostfile 处理完成 ====="
