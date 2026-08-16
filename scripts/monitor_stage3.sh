#!/bin/bash
# 实时监控 Stage3 批量处理进度

OUTPUT_FILE="/mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results.jsonl"
LOG_FILE="/mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_run.log"

echo "=== Stage3 批量处理进度监控 ==="
echo ""

while true; do
    if [ -f "$OUTPUT_FILE" ]; then
        PROCESSED=$(wc -l < "$OUTPUT_FILE")
        PASS=$(grep -c '"verdict": "pass"' "$OUTPUT_FILE" 2>/dev/null || echo 0)
        FAIL=$(grep -c '"verdict": "fail"' "$OUTPUT_FILE" 2>/dev/null || echo 0)
        ERROR=$(grep -c '"verdict": "error"' "$OUTPUT_FILE" 2>/dev/null || echo 0)

        PASS_PCT=$(awk "BEGIN {printf \"%.1f\", $PASS*100/$PROCESSED}")
        PROGRESS_PCT=$(awk "BEGIN {printf \"%.1f\", $PROCESSED*100/5000}")

        clear
        echo "=== Stage3 批量处理进度 ==="
        echo ""
        echo "总进度: $PROCESSED / 5000 ($PROGRESS_PCT%)"
        echo "  Pass:  $PASS ($PASS_PCT%)"
        echo "  Fail:  $FAIL"
        echo "  Error: $ERROR"
        echo ""

        if [ -f "$LOG_FILE" ]; then
            echo "最近日志:"
            tail -3 "$LOG_FILE" | grep -E "Stage3|Processing" || echo "(等待输出...)"
        fi

        # 检查是否完成
        if [ "$PROCESSED" -eq 5000 ]; then
            echo ""
            echo "✅ 批量处理完成！"
            break
        fi
    else
        echo "等待输出文件生成..."
    fi

    sleep 10
done
