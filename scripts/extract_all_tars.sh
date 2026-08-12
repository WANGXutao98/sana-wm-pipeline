#!/bin/bash
# scripts/extract_all_tars.sh
# 批量提取所有 tar 文件

set -euo pipefail

# 默认配置
BASE_DIR="${BASE_DIR:-/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output}"
PARALLEL_JOBS="${PARALLEL_JOBS:-16}"
DRY_RUN=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_SINGLE="$SCRIPT_DIR/extract_single_tar.sh"

# 使用说明
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --base-dir PATH      Base directory containing tar files (default: $BASE_DIR)
    --parallel N         Number of parallel jobs (default: $PARALLEL_JOBS)
    --dry-run            List tar files without extracting
    --help               Show this help message

Example:
    $0
    $0 --parallel 8
    $0 --dry-run
EOF
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# 主逻辑
echo "=== Tar Extraction System ==="
echo "Base directory: $BASE_DIR"
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

# 扫描所有 tar 文件
echo "Scanning for tar files..."
TAR_LIST_FILE="/tmp/tar_list_$$.txt"
find "$BASE_DIR" -name "*.tar" -type f > "$TAR_LIST_FILE"
TOTAL_TARS=$(wc -l < "$TAR_LIST_FILE")
echo "Found $TOTAL_TARS tar files"

# 过滤已完成的（断点恢复）
echo "Filtering completed tars..."
SUCCESS_LIST_FILE="/tmp/success_list_$$.txt"
find "$BASE_DIR" -name "*.SUCCESS" -type f | sed 's/.SUCCESS$/.tar/' > "$SUCCESS_LIST_FILE"
COMPLETED_TARS=$(wc -l < "$SUCCESS_LIST_FILE")

TAR_TODO_FILE="/tmp/tar_todo_$$.txt"
if [ -s "$SUCCESS_LIST_FILE" ]; then
    grep -v -F -f "$SUCCESS_LIST_FILE" "$TAR_LIST_FILE" > "$TAR_TODO_FILE"
else
    cp "$TAR_LIST_FILE" "$TAR_TODO_FILE"
fi
TODO_TARS=$(wc -l < "$TAR_TODO_FILE")

echo "Already completed: $COMPLETED_TARS"
echo "Remaining to process: $TODO_TARS"
echo ""

# Dry-run 模式：只列出待处理的 tar
if [ "$DRY_RUN" = true ]; then
    echo "=== Dry-run mode: listing first 20 tar files to process ==="
    head -20 "$TAR_TODO_FILE"
    echo "..."
    echo "Total to process: $TODO_TARS"

    # 清理临时文件
    rm -f "$TAR_LIST_FILE" "$SUCCESS_LIST_FILE" "$TAR_TODO_FILE"
    exit 0
fi

# 生成主日志
LOG_FILE="$BASE_DIR/extraction.log"
{
    echo "=== Tar Extraction Started ==="
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Total tar files: $TOTAL_TARS"
    echo "Already completed: $COMPLETED_TARS"
    echo "To process: $TODO_TARS"
    echo "Parallel jobs: $PARALLEL_JOBS"
    echo "============================"
} > "$LOG_FILE"

echo "Extraction log: $LOG_FILE"

# 检查 GNU parallel 是否可用
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for extraction..."
    cat "$TAR_TODO_FILE" | parallel -j "$PARALLEL_JOBS" --halt never bash "$EXTRACT_SINGLE" {}
else
    echo "GNU parallel not found, using xargs..."
    cat "$TAR_TODO_FILE" | xargs -P "$PARALLEL_JOBS" -I {} bash "$EXTRACT_SINGLE" {}
fi

# 统计最终结果
FINAL_SUCCESS=$(find "$BASE_DIR" -name "*.SUCCESS" -type f | wc -l)
FINAL_FAILED=$(find "$BASE_DIR" -name "*.FAILED" -type f | wc -l)

{
    echo ""
    echo "=== Extraction Completed ==="
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Successful: $FINAL_SUCCESS"
    echo "Failed: $FINAL_FAILED"
    echo "============================"
} >> "$LOG_FILE"

echo ""
echo "=== Final Statistics ==="
echo "Successful: $FINAL_SUCCESS / $TOTAL_TARS"
echo "Failed: $FINAL_FAILED / $TOTAL_TARS"
echo "See full log: $LOG_FILE"

# 清理临时文件
rm -f "$TAR_LIST_FILE" "$SUCCESS_LIST_FILE" "$TAR_TODO_FILE"
