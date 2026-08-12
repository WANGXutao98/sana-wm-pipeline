#!/bin/bash
# 用法: bash watch_progress.sh [GROUP]
# 实时监控某个 group 的批量生产进度：已完成 input shard 数 / GPU 显存利用率 / 各 worker 最新日志
# 故意不开 -e：这是无限轮询循环，任何一次取数失败（progress/ 还为空、nvidia-smi 缺失等）
# 都应跳过本轮继续显示，而不是杀掉整个监控会话
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

GROUP="${1:-wds-sekai-real-walking-hq}"
OUT_BASE="${2:-$OUT_BASE}"   # 可指定自定义输出路径对应的任务组

while true; do
    clear
    echo "=== SANA-WM 批量生产进度  $(date) ==="
    echo "GROUP: $GROUP"

    TOTAL=$(ls "$DATA_ROOT/$GROUP/shards/"*.tar 2>/dev/null | wc -l) || true
    DONE=$(ls "$OUT_BASE/$GROUP/progress/"*.done 2>/dev/null | wc -l) || true
    [[ $TOTAL -gt 0 ]] && PCT=$((DONE * 100 / TOTAL)) || PCT=0
    echo "输入 shard: $DONE / $TOTAL  ($PCT%)"

    N_OK=$(grep -ho '"n_ok": [0-9]*' "$OUT_BASE/$GROUP/progress/"*.done 2>/dev/null \
        | awk -F': ' '{s+=$2} END{print s+0}') || true
    N_FAIL=$(grep -ho '"n_fail": [0-9]*' "$OUT_BASE/$GROUP/progress/"*.done 2>/dev/null \
        | awk -F': ' '{s+=$2} END{print s+0}') || true
    echo "样本: ok=$N_OK  fail=$N_FAIL"

    echo ""
    echo "=== GPU 状态 ==="
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null | \
    awk -F',' '{printf "  GPU%s: %sMiB已用 %sMiB空闲  利用率%s%%\n",$1,$2,$3,$4}'

    echo ""
    echo "=== 各 worker 最新日志（末3行）==="
    for LOG in "$OUT_BASE/$GROUP/logs/"node*_gpu*.log; do
        [[ -f "$LOG" ]] || continue
        echo "  [$(basename "$LOG" .log)]"
        tail -3 "$LOG" 2>/dev/null | sed 's/^/    /'
    done

    sleep 30
done
