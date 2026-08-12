#!/bin/bash
# 镜像不能重新发布时的热更新工具：把本地（已改好）的 batch_production 控制脚本
# scp 同步到一个或多个 hostfile 列出的所有节点，避免逐节点手动 vim 改容易漏改/改错。
#
# 用法:
#   bash sync_to_nodes.sh <HOSTFILE> [<HOSTFILE2> ...]
# 示例（一次同步到 6 节点 + 4 节点全部 10 台）:
#   bash sync_to_nodes.sh /root/work/david_work/sana-wm-v4_hostfiles/hostfiles \
#                         /root/work/david_work/sana-wm-v4-48/hostfiles
#
# 前提：本脚本要从一台已经拿到最新代码的节点上执行（比如你准备拿来跑 launch_all_nodes.sh
# 的那台 master），它需要能 SSH root 免密到 hostfile 里列出的所有节点（包括它自己，重复
# 同步给自己是无害的）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

[[ $# -ge 1 ]] || { echo "用法: $0 <HOSTFILE> [<HOSTFILE2> ...]"; exit 1; }

REMOTE_DIR="$PROJ_DIR/experiments/batch_production"
# 同步整个目录的脚本文件，不只是这次改的 2 个——下次再迭代任何一个文件都不用改这里
FILES=("$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.py)

FAIL=0
for HOSTFILE in "$@"; do
    [[ -f "$HOSTFILE" ]] || { echo "[ERROR] hostfile 不存在: $HOSTFILE"; exit 1; }

    mapfile -t RAW_LINES < "$HOSTFILE"
    for line in "${RAW_LINES[@]}"; do
        line="${line%$'\r'}"   # 防御 CRLF 文件
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        NODE_ID=$(awk '{print $1}' <<< "$line")

        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
            "mkdir -p '$REMOTE_DIR'" || { echo "[FAIL] $NODE_ID 无法连接/创建目录"; FAIL=1; continue; }

        if scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no -q \
            "${FILES[@]}" root@"$NODE_ID":"$REMOTE_DIR/"; then
            echo "[OK]   $NODE_ID  (hostfile: $(basename "$HOSTFILE"))"
        else
            echo "[FAIL] $NODE_ID  scp 失败"
            FAIL=1
        fi
    done
done

if [[ $FAIL -ne 0 ]]; then
    echo -e "\n[ERROR] 部分节点同步失败，请检查上面 [FAIL] 行后重跑（脚本可重复执行，已成功的节点重新覆盖无副作用）"
    exit 1
fi
echo -e "\n全部节点同步完成"
