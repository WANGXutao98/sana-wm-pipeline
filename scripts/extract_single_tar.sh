#!/bin/bash
# scripts/extract_single_tar.sh
# 提取单个 tar 文件，最大化样本恢复

set -euo pipefail

# 使用说明
if [ $# -ne 1 ]; then
    echo "Usage: $0 <tar_file_path>"
    exit 1
fi

TAR_FILE="$1"
TAR_DIR="$(dirname "$TAR_FILE")"
TAR_NAME="$(basename "$TAR_FILE" .tar)"
EXTRACTED_DIR="${TAR_DIR}/${TAR_NAME}"
SUCCESS_MARKER="${TAR_DIR}/${TAR_NAME}.SUCCESS"
FAILED_MARKER="${TAR_DIR}/${TAR_NAME}.FAILED"

echo "Processing: $TAR_FILE"

# 断点恢复：跳过已完成的
if [ -f "$SUCCESS_MARKER" ]; then
    echo "[SKIP] Already extracted: $TAR_NAME"
    exit 0
fi

# 清理部分失败的（有目录但无 SUCCESS 标记）
if [ -d "$EXTRACTED_DIR" ] && [ ! -f "$SUCCESS_MARKER" ]; then
    echo "[CLEANUP] Removing incomplete extraction: $EXTRACTED_DIR"
    rm -rf "$EXTRACTED_DIR"
fi

# 移除旧的 FAILED 标记（如果重新运行）
rm -f "$FAILED_MARKER"

# 创建解压目录
mkdir -p "$EXTRACTED_DIR"

# 第一次尝试：标准容错提取
echo "[ATTEMPT 1] Standard extraction with fault tolerance"
EXTRACTION_SUCCESS=false
if tar -xf "$TAR_FILE" \
       --ignore-failed-read \
       --warning=no-timestamp \
       -C "$EXTRACTED_DIR" \
       2>/dev/null; then
    EXTRACTION_SUCCESS=true
fi

# 第二次尝试：dd 修复 + 容错提取
if [ "$EXTRACTION_SUCCESS" = false ]; then
    echo "[ATTEMPT 2] dd recovery + extraction"
    if dd if="$TAR_FILE" bs=512 conv=noerror,sync 2>/dev/null | \
       tar -x --ignore-zeros -C "$EXTRACTED_DIR" 2>/dev/null; then
        EXTRACTION_SUCCESS=true
    fi
fi

# 验证结果：检查解压目录是否有文件
FILE_COUNT=$(find "$EXTRACTED_DIR" -type f | wc -l)

if [ "$FILE_COUNT" -gt 0 ]; then
    # 成功：有文件提取出来
    touch "$SUCCESS_MARKER"
    echo "[SUCCESS] Extracted $FILE_COUNT files from $TAR_NAME"
    exit 0
else
    # 失败：目录为空
    touch "$FAILED_MARKER"
    rmdir "$EXTRACTED_DIR" 2>/dev/null || true
    echo "[FAILED] No files extracted from $TAR_NAME"
    exit 1
fi
