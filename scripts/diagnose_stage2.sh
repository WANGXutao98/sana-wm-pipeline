#!/bin/bash
# Stage 2 执行状态诊断脚本

echo "=========================================="
echo "Stage 2 执行状态诊断"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 1. 检查进程是否还在运行
echo "1. 检查 Python 进程状态"
echo "----------------------------------------"
ps aux | grep -E "stage2|python.*sana_wm" | grep -v grep
if [ $? -eq 0 ]; then
    echo "✅ Python 进程正在运行"
else
    echo "❌ 没有找到 Python 进程"
fi
echo ""

# 2. 检查 CPU 和内存使用
echo "2. CPU 和内存使用情况"
echo "----------------------------------------"
ps aux | grep -E "stage2|python.*sana_wm" | grep -v grep | awk '{print "CPU: " $3 "%, MEM: " $4 "%, PID: " $2 ", CMD: " $11 " " $12}'
echo ""

# 3. 检查输出文件状态
echo "3. 当前输出文件状态"
echo "----------------------------------------"
OUTPUT_FILE="/root/work/david_work/qc_output_new/wds-RealEstate10K-360p/stage2_results_full.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    current_count=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
    file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
    last_modified=$(stat -c %y "$OUTPUT_FILE" | cut -d. -f1)
    echo "文件: $OUTPUT_FILE"
    echo "  当前行数: $current_count / 68821 ($(echo "scale=2; $current_count * 100 / 68821" | bc)%)"
    echo "  文件大小: $file_size"
    echo "  最后修改: $last_modified"

    # 检查最近是否有更新（最近 5 分钟）
    current_time=$(date +%s)
    file_time=$(stat -c %Y "$OUTPUT_FILE")
    time_diff=$((current_time - file_time))

    if [ $time_diff -lt 300 ]; then
        echo "  状态: ✅ 文件最近有更新（${time_diff}秒前）"
    else
        minutes=$((time_diff / 60))
        echo "  状态: ⚠️  文件 ${minutes} 分钟未更新"
    fi
else
    echo "❌ 输出文件不存在: $OUTPUT_FILE"
fi
echo ""

# 4. 检查日志文件最后几行
echo "4. 执行日志最后 10 行"
echo "----------------------------------------"
LOG_DIR="/root/work/david_work/qc_output_new/wds-RealEstate10K-360p"
if [ -f "$LOG_DIR/stage2_run_full.log" ]; then
    tail -10 "$LOG_DIR/stage2_run_full.log"
else
    echo "日志文件不存在: $LOG_DIR/stage2_run_full.log"
fi
echo ""

# 5. 预估剩余时间
echo "5. 进度预估"
echo "----------------------------------------"
if [ -f "$OUTPUT_FILE" ]; then
    current_count=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
    total=68821

    # 从主日志获取开始时间
    start_time_str=$(grep -A 1 "wds-RealEstate10K-360p" /root/work/david_work/stage2_batch_full_*.log | grep "开始时间" | tail -1 | cut -d: -f2- | xargs)

    if [ ! -z "$start_time_str" ]; then
        start_epoch=$(date -d "$start_time_str" +%s 2>/dev/null || echo 0)
        current_epoch=$(date +%s)
        elapsed=$((current_epoch - start_epoch))

        if [ $current_count -gt 0 ] && [ $elapsed -gt 0 ]; then
            rate=$(echo "scale=2; $current_count / $elapsed" | bc)
            remaining=$((total - current_count))
            eta_seconds=$(echo "scale=0; $remaining / $rate" | bc)
            eta_minutes=$((eta_seconds / 60))
            eta_hours=$((eta_minutes / 60))

            echo "已处理: $current_count / $total ($(echo "scale=2; $current_count * 100 / $total" | bc)%)"
            echo "已耗时: $((elapsed / 60)) 分钟"
            echo "处理速度: $(echo "scale=2; $rate * 60" | bc) 样本/分钟"
            echo "预计剩余: ${eta_hours} 小时 $((eta_minutes % 60)) 分钟"
            echo "预计完成: $(date -d "+${eta_seconds} seconds" '+%Y-%m-%d %H:%M:%S')"
        fi
    fi
fi
echo ""

# 6. 检查磁盘 I/O
echo "6. 磁盘 I/O 状态"
echo "----------------------------------------"
iostat -x 1 2 | tail -20
echo ""

# 7. 检查错误日志
echo "7. 检查最近错误"
echo "----------------------------------------"
if [ -f "$LOG_DIR/stage2_run_full.log" ]; then
    error_count=$(grep -c -i "error\|exception\|traceback" "$LOG_DIR/stage2_run_full.log" || echo 0)
    echo "错误计数: $error_count"
    if [ $error_count -gt 0 ]; then
        echo "最近错误:"
        grep -i "error\|exception" "$LOG_DIR/stage2_run_full.log" | tail -5
    fi
else
    echo "无法检查错误（日志文件不存在）"
fi
echo ""

# 8. 给出建议
echo "8. 诊断建议"
echo "----------------------------------------"

if ps aux | grep -E "stage2|python.*sana_wm" | grep -v grep > /dev/null; then
    if [ -f "$OUTPUT_FILE" ]; then
        current_time=$(date +%s)
        file_time=$(stat -c %Y "$OUTPUT_FILE")
        time_diff=$((current_time - file_time))

        if [ $time_diff -lt 600 ]; then
            echo "✅ 进程正常运行，文件正在更新"
            echo "建议: 继续等待"
        else
            echo "⚠️  进程在运行，但文件长时间未更新"
            echo "建议: 可能卡住，考虑以下操作："
            echo "  1. 检查 CPU 利用率（应该较高）"
            echo "  2. 查看详细日志: tail -f $LOG_DIR/stage2_run_full.log"
            echo "  3. 如果确认卡住，可以重启（会从断点续传）"
        fi
    else
        echo "⚠️  进程在运行，但输出文件不存在"
        echo "建议: 可能刚开始或有问题，等待 5 分钟后重新检查"
    fi
else
    echo "❌ 进程未运行"
    echo "建议: 检查是否异常退出，查看主日志"
fi

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="
