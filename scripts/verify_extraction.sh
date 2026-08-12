#!/bin/bash
set -euo pipefail
# scripts/verify_extraction.sh
# 验证解压结果，统计样本完整性

BASE_DIR="${1:-/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output}"
OUTPUT_CSV="${BASE_DIR}/sample_completeness.csv"
TEMP_CSV="${OUTPUT_CSV}.tmp"

echo "=== Extraction Verification ==="
echo "Scanning: $BASE_DIR"
echo ""

# 统计总样本数（每个样本应该有 5 个文件）
TOTAL_SAMPLES=0
COMPLETE_SAMPLES=0    # 5 个文件
PARTIAL_SAMPLES=0     # 1-4 个文件

# 生成 CSV 头
echo "sample_id,file_count,complete" > "$TEMP_CSV"

# 遍历所有解压目录
find "$BASE_DIR" -type d -name "shard-*" 2>/dev/null | while read -r extracted_dir; do
    [ -z "$extracted_dir" ] && continue

    # 获取该目录中所有样本的 base name（去掉扩展名）
    find "$extracted_dir" -type f -name "*.mp4" 2>/dev/null | while read -r mp4_file; do
        sample_id=$(basename "$mp4_file" .mp4)

        # 统计该样本的文件数
        file_count=0
        [ -f "$extracted_dir/${sample_id}.mp4" ] && file_count=$((file_count + 1))
        [ -f "$extracted_dir/${sample_id}.caption.txt" ] && file_count=$((file_count + 1))
        [ -f "$extracted_dir/${sample_id}.poses_c2w.npy" ] && file_count=$((file_count + 1))
        [ -f "$extracted_dir/${sample_id}.intrinsics.npy" ] && file_count=$((file_count + 1))
        [ -f "$extracted_dir/${sample_id}.scale.npy" ] && file_count=$((file_count + 1))

        if [ $file_count -eq 5 ]; then
            echo "${sample_id},5,true" >> "$TEMP_CSV"
        else
            echo "${sample_id},${file_count},false" >> "$TEMP_CSV"
        fi
    done
done

# Count results from temp CSV file (not inside loop to avoid subshell scope issues)
# Using grep + wc -l approach: counts lines from temp file after loop completes
# Alternative (loop-internal counters) loses state due to subshell in pipe context
TOTAL_SAMPLES=$(tail -n +2 "$TEMP_CSV" | wc -l)
COMPLETE_SAMPLES=$(grep ",5,true$" "$TEMP_CSV" | wc -l)
PARTIAL_SAMPLES=$(grep ",.*,false$" "$TEMP_CSV" | wc -l)

# Atomic move: temp file → final CSV ensures consistency if script interrupts
mv "$TEMP_CSV" "$OUTPUT_CSV"

# 输出统计
echo "=== Verification Results ==="
echo "Total samples: $TOTAL_SAMPLES"

if [ $TOTAL_SAMPLES -gt 0 ]; then
    COMPLETE_PCT=$(awk "BEGIN {printf \"%.1f\", $COMPLETE_SAMPLES*100.0/$TOTAL_SAMPLES}")
    PARTIAL_PCT=$(awk "BEGIN {printf \"%.1f\", $PARTIAL_SAMPLES*100.0/$TOTAL_SAMPLES}")
    echo "Complete samples (5 files): $COMPLETE_SAMPLES (${COMPLETE_PCT}%)"
    echo "Partial samples (1-4 files): $PARTIAL_SAMPLES (${PARTIAL_PCT}%)"
else
    echo "Complete samples (5 files): 0 (0.0%)"
    echo "Partial samples (1-4 files): 0 (0.0%)"
    echo "Warning: No samples found in $BASE_DIR"
fi

echo ""
echo "Detailed report: $OUTPUT_CSV"
