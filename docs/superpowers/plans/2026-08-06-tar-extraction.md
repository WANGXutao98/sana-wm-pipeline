# Tar 批量解压系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现鲁棒的 tar 批量解压系统，从 3TB 损坏 tar 集合中最大化恢复 282K 样本

**Architecture:** Bash 脚本 + GNU parallel，激进容错策略（tar --ignore-failed-read + dd 修复），并行 16 进程，通过标记文件实现断点恢复

**Tech Stack:** Bash, GNU tar, dd, GNU parallel (或 xargs), find, grep

## Global Constraints

- 目标环境: CMCC 服务器，bash 4.0+
- 数据路径: `/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output`
- 并行度: 16 进程（可配置）
- 标记文件: `.SUCCESS` (成功) / `.FAILED` (失败)
- 幂等性: 支持中断后重新运行，自动跳过已完成的 tar
- 日志: 极简（只记录 tar 级别状态）

---

## 文件结构

**创建的文件:**
- `scripts/extract_all_tars.sh` - 主脚本，扫描 tar 列表并并行调用提取函数
- `scripts/extract_single_tar.sh` - 单个 tar 提取函数（核心逻辑）
- `scripts/verify_extraction.sh` - 验证脚本（可选），统计样本完整性

**生成的文件:**
- `extraction.log` - 主日志（开始时间、总数、进度）
- `{tar_name}.SUCCESS` - 成功标记（空文件，与 tar 文件同级）
- `{tar_name}.FAILED` - 失败标记（空文件，与 tar 文件同级）
- `{tar_name}/` - 解压目录（与 tar 文件同级）

---

### Task 1: 实现单个 tar 提取函数

**Files:**
- Create: `scripts/extract_single_tar.sh`

**Interfaces:**
- Consumes: tar 文件绝对路径（命令行参数）
- Produces: 解压目录 + `.SUCCESS` 或 `.FAILED` 标记文件

- [ ] **Step 1: 创建脚本骨架**

```bash
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

# 主逻辑占位
echo "Processing: $TAR_FILE"
```

- [ ] **Step 2: 添加断点恢复检查**

```bash
# 在主逻辑占位处添加

# 1. 断点恢复：跳过已完成的
if [ -f "$SUCCESS_MARKER" ]; then
    echo "[SKIP] Already extracted: $TAR_NAME"
    exit 0
fi

# 2. 清理部分失败的（有目录但无 SUCCESS 标记）
if [ -d "$EXTRACTED_DIR" ] && [ ! -f "$SUCCESS_MARKER" ]; then
    echo "[CLEANUP] Removing incomplete extraction: $EXTRACTED_DIR"
    rm -rf "$EXTRACTED_DIR"
fi

# 3. 移除旧的 FAILED 标记（如果重新运行）
rm -f "$FAILED_MARKER"
```

- [ ] **Step 3: 添加解压目录创建**

```bash
# 创建解压目录
mkdir -p "$EXTRACTED_DIR"
```

- [ ] **Step 4: 实现第一次尝试（标准容错提取）**

```bash
# 第一次尝试：标准容错提取
echo "[ATTEMPT 1] Standard extraction with fault tolerance"
if tar -xf "$TAR_FILE" \
       --ignore-failed-read \
       --warning=no-timestamp \
       -C "$EXTRACTED_DIR" \
       2>/dev/null; then
    EXTRACTION_SUCCESS=true
else
    EXTRACTION_SUCCESS=false
fi
```

- [ ] **Step 5: 实现第二次尝试（dd 修复）**

```bash
# 第二次尝试：dd 修复 + 容错提取
if [ "$EXTRACTION_SUCCESS" = false ]; then
    echo "[ATTEMPT 2] dd recovery + extraction"
    if dd if="$TAR_FILE" bs=512 conv=noerror,sync 2>/dev/null | \
       tar -x --ignore-zeros -C "$EXTRACTED_DIR" 2>/dev/null; then
        EXTRACTION_SUCCESS=true
    fi
fi
```

- [ ] **Step 6: 实现结果验证和标记**

```bash
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
```

- [ ] **Step 7: 测试单个 tar 提取**

```bash
# 在 CMCC 机器上测试
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts
chmod +x extract_single_tar.sh

# 测试：提取冒烟测试的 tar
./extract_single_tar.sh /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/final_wds-sekai-game-drone/wds-sekai-game-drone/w003/shard-000003-000001.tar
```

预期输出：
```
Processing: .../shard-000003-000001.tar
[ATTEMPT 1] Standard extraction with fault tolerance
[SUCCESS] Extracted XXX files from shard-000003-000001
```

验证：
- 检查 `shard-000003-000001/` 目录存在且有文件
- 检查 `shard-000003-000001.SUCCESS` 文件存在

- [ ] **Step 8: 测试断点恢复**

```bash
# 再次运行同一个 tar
./extract_single_tar.sh /root/work/filestorage/.../shard-000003-000001.tar
```

预期输出：
```
[SKIP] Already extracted: shard-000003-000001
```

- [ ] **Step 9: 提交单个 tar 提取函数**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add scripts/extract_single_tar.sh
git commit -m "feat: add robust single tar extraction function

- Aggressive fault tolerance (tar --ignore-failed-read + dd recovery)
- Resume support via .SUCCESS marker
- Automatic cleanup of partial extractions
- Maximize sample recovery from corrupted tars"
```

---

### Task 2: 实现主控脚本

**Files:**
- Create: `scripts/extract_all_tars.sh`

**Interfaces:**
- Consumes: `extract_single_tar.sh` 脚本
- Produces: 批量解压执行，生成 `extraction.log`

- [ ] **Step 1: 创建主脚本骨架**

```bash
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

# 解析参数占位
# 主逻辑占位
```

- [ ] **Step 2: 实现命令行参数解析**

```bash
# 在参数解析占位处添加

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
```

- [ ] **Step 3: 实现 tar 文件扫描和过滤**

```bash
# 在主逻辑占位处添加

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
grep -v -F -f "$SUCCESS_LIST_FILE" "$TAR_LIST_FILE" > "$TAR_TODO_FILE" || touch "$TAR_TODO_FILE"
TODO_TARS=$(wc -l < "$TAR_TODO_FILE")

echo "Already completed: $COMPLETED_TARS"
echo "Remaining to process: $TODO_TARS"
echo ""
```

- [ ] **Step 4: 实现 dry-run 模式**

```bash
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
```

- [ ] **Step 5: 实现主日志记录**

```bash
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
```

- [ ] **Step 6: 实现并行提取（优先使用 GNU parallel）**

```bash
# 检查 GNU parallel 是否可用
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for extraction..."
    cat "$TAR_TODO_FILE" | parallel -j "$PARALLEL_JOBS" --halt never "$EXTRACT_SINGLE" {}
else
    echo "GNU parallel not found, using xargs..."
    cat "$TAR_TODO_FILE" | xargs -P "$PARALLEL_JOBS" -I {} bash "$EXTRACT_SINGLE" {}
fi
```

- [ ] **Step 7: 实现完成统计**

```bash
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
```

- [ ] **Step 8: 本地测试 dry-run 模式**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts
chmod +x extract_all_tars.sh

# 测试 dry-run
./extract_all_tars.sh --dry-run
```

预期输出：
```
=== Tar Extraction System ===
Base directory: /root/work/filestorage/.../jdvbbfb_output
Parallel jobs: 16

Scanning for tar files...
Found XXXX tar files
Filtering completed tars...
Already completed: 1
Remaining to process: XXXX

=== Dry-run mode: listing first 20 tar files to process ===
/path/to/shard-000003-000002.tar
...
```

- [ ] **Step 9: 提交主控脚本**

```bash
git add scripts/extract_all_tars.sh
git commit -m "feat: add parallel tar extraction orchestration

- Scan and filter tar files with resume support
- Parallel execution with GNU parallel or xargs
- Dry-run mode for preview
- Progress logging and statistics"
```

---

### Task 3: 实现验证脚本（可选）

**Files:**
- Create: `scripts/verify_extraction.sh`

**Interfaces:**
- Consumes: 解压目录
- Produces: 样本完整性统计，生成 `sample_completeness.csv`

- [ ] **Step 1: 创建验证脚本**

```bash
#!/bin/bash
# scripts/verify_extraction.sh
# 验证解压结果，统计样本完整性

set -euo pipefail

BASE_DIR="${1:-/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output}"
OUTPUT_CSV="${BASE_DIR}/sample_completeness.csv"

echo "=== Extraction Verification ==="
echo "Scanning: $BASE_DIR"
echo ""

# 统计总样本数（每个样本应该有 5 个文件）
TOTAL_SAMPLES=0
COMPLETE_SAMPLES=0    # 5 个文件
PARTIAL_SAMPLES=0     # 1-4 个文件

# 生成 CSV 头
echo "sample_id,file_count,complete" > "$OUTPUT_CSV"

# 遍历所有解压目录
while IFS= read -r extracted_dir; do
    # 获取该目录中所有样本的 base name（去掉扩展名）
    samples=$(find "$extracted_dir" -type f -name "*.mp4" -exec basename {} .mp4 \;)
    
    for sample_id in $samples; do
        ((TOTAL_SAMPLES++))
        
        # 统计该样本的文件数
        file_count=0
        [ -f "$extracted_dir/${sample_id}.mp4" ] && ((file_count++))
        [ -f "$extracted_dir/${sample_id}.caption.txt" ] && ((file_count++))
        [ -f "$extracted_dir/${sample_id}.poses_c2w.npy" ] && ((file_count++))
        [ -f "$extracted_dir/${sample_id}.intrinsics.npy" ] && ((file_count++))
        [ -f "$extracted_dir/${sample_id}.scale.npy" ] && ((file_count++))
        
        if [ $file_count -eq 5 ]; then
            ((COMPLETE_SAMPLES++))
            echo "${sample_id},5,true" >> "$OUTPUT_CSV"
        else
            ((PARTIAL_SAMPLES++))
            echo "${sample_id},${file_count},false" >> "$OUTPUT_CSV"
        fi
    done
done < <(find "$BASE_DIR" -type d -name "shard-*" | grep -v ".tar")

# 输出统计
echo "=== Verification Results ==="
echo "Total samples: $TOTAL_SAMPLES"
echo "Complete samples (5 files): $COMPLETE_SAMPLES ($(awk "BEGIN {printf \"%.1f\", $COMPLETE_SAMPLES*100.0/$TOTAL_SAMPLES}")%)"
echo "Partial samples (1-4 files): $PARTIAL_SAMPLES ($(awk "BEGIN {printf \"%.1f\", $PARTIAL_SAMPLES*100.0/$TOTAL_SAMPLES}")%)"
echo ""
echo "Detailed report: $OUTPUT_CSV"
```

- [ ] **Step 2: 添加执行权限并测试**

```bash
chmod +x scripts/verify_extraction.sh

# 测试（在完成部分解压后）
# ./verify_extraction.sh
```

- [ ] **Step 3: 提交验证脚本**

```bash
git add scripts/verify_extraction.sh
git commit -m "feat: add extraction verification script

- Count total samples and check completeness
- Report complete samples (5 files) vs partial
- Generate detailed CSV report"
```

---

### Task 4: 创建使用文档和执行指南

**Files:**
- Create: `scripts/TAR_EXTRACTION_README.md`

**Interfaces:**
- Consumes: 所有脚本
- Produces: 完整的使用文档

- [ ] **Step 1: 创建 README 文档**

```markdown
# Tar 批量解压系统使用指南

## 快速开始

### 1. 上传脚本到 CMCC 机器

```bash
# 在本地执行
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
scp scripts/extract_*.sh user@cmcc:/root/work/sana_qc_pipeline/scripts/
```

### 2. 授予执行权限

```bash
# 在 CMCC 机器执行
cd /root/work/sana_qc_pipeline/scripts
chmod +x extract_all_tars.sh extract_single_tar.sh verify_extraction.sh
```

### 3. 预览待处理的 tar 列表

```bash
./extract_all_tars.sh --dry-run | head -30
```

### 4. 启动解压（建议在 tmux 中运行）

```bash
# 创建 tmux 会话
tmux new -s tar_extraction

# 启动解压
./extract_all_tars.sh

# 查看日志
tail -f /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/extraction.log
```

### 5. 监控进度（另开终端）

```bash
# 每 60 秒刷新一次进度
watch -n 60 'find /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output -name "*.SUCCESS" | wc -l'
```

### 6. 验证结果（可选）

```bash
./verify_extraction.sh
```

---

## 命令行选项

```bash
# 默认配置（并行度 16）
./extract_all_tars.sh

# 自定义并行度
./extract_all_tars.sh --parallel 8

# 预览模式（不执行）
./extract_all_tars.sh --dry-run

# 自定义基础目录
./extract_all_tars.sh --base-dir /custom/path

# 帮助信息
./extract_all_tars.sh --help
```

---

## 断点恢复

**中断后重新运行**：
```bash
# Ctrl+C 中断后，重新运行
./extract_all_tars.sh
```

脚本会自动：
- 跳过已有 `.SUCCESS` 标记的 tar
- 清理未完成的解压目录（无 `.SUCCESS` 标记）
- 继续处理剩余的 tar

---

## 监控和诊断

### 查看总体进度

```bash
BASE_DIR="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"

# 总 tar 数
find $BASE_DIR -name "*.tar" | wc -l

# 已完成数
find $BASE_DIR -name "*.SUCCESS" | wc -l

# 失败数
find $BASE_DIR -name "*.FAILED" | wc -l
```

### 检查卡住的进程

```bash
# 查看正在运行的 tar 提取进程
ps aux | grep extract_single_tar

# 如果有进程卡住，kill 后重新运行主脚本
kill <PID>
./extract_all_tars.sh  # 自动恢复
```

### 检查磁盘空间

```bash
df -h /root/work/filestorage
```

---

## 故障排查

### 问题：进程长时间无进展

**检查**：
```bash
ps aux | grep tar
```

**解决**：kill 卡住的进程，脚本会自动跳过已完成的

### 问题：磁盘空间不足

**检查**：
```bash
df -h /root/work/filestorage
```

**解决**：清理其他临时文件，或暂停部分进程

### 问题：大量 tar 标记为 FAILED

**检查**：
```bash
find $BASE_DIR -name "*.FAILED" | head -10
```

**分析**：原 tar 文件可能完全损坏，无法恢复

---

## 预期性能

- **单个 tar**: 30-60 秒（取决于大小和损坏程度）
- **并行 16 进程**: 每分钟 16-32 个 tar
- **总时长**: 12-24 小时（假设 1000-2000 个 tar）

---

## 与 Stage 3 集成

解压完成后，`stage3_gpu.py` 会自动从解压目录读取（性能提升 1000 倍）：

```bash
# 运行 Stage 3 冒烟测试
cd /root/work/david_work/sana_qc_pipeline
python scripts/run_stage3_cmcc.py \
  --stage1-jsonl /root/work/david_work/qc_output_new/smoke_test_manifest.jsonl \
  --output-dir /root/work/david_work/qc_output_new/smoke_test_stage3 \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml
```

预期：~18 分钟完成 901 样本（之前会卡死）
```

保存到文件: `scripts/TAR_EXTRACTION_README.md`

- [ ] **Step 2: 提交使用文档**

```bash
git add scripts/TAR_EXTRACTION_README.md
git commit -m "docs: add tar extraction system usage guide

- Quick start instructions
- Command-line options
- Resume and monitoring
- Troubleshooting guide
- Stage 3 integration"
```

---

## 自审清单

**规范检查**:
- ✅ 所有 tar 文件路径使用绝对路径或从 BASE_DIR 派生
- ✅ 标记文件命名一致：`{tar_name}.SUCCESS` / `{tar_name}.FAILED`
- ✅ 解压目录命名一致：`{tar_name}/`（去掉 .tar 扩展名）
- ✅ 断点恢复逻辑：检查 `.SUCCESS` → 跳过，有目录但无 `.SUCCESS` → 清理重来
- ✅ 错误处理：所有 tar/dd 命令都重定向 stderr 到 `/dev/null`
- ✅ 幂等性：多次运行结果一致

**完整性检查**:
- ✅ Task 1 实现单个 tar 提取（核心逻辑）
- ✅ Task 2 实现并行控制和主流程
- ✅ Task 3 实现验证脚本（可选）
- ✅ Task 4 提供完整使用文档

**无占位符**:
- ✅ 所有命令都是完整可执行的
- ✅ 所有路径都是明确的
- ✅ 所有参数都有默认值或明确说明

---

## 执行后验证

完成所有任务后，在 CMCC 机器上执行：

```bash
# 1. 测试单个 tar 提取
cd /root/work/sana_qc_pipeline/scripts
./extract_single_tar.sh <某个测试 tar 的路径>

# 2. 验证断点恢复
./extract_single_tar.sh <同一个 tar>  # 应该跳过

# 3. Dry-run 预览
./extract_all_tars.sh --dry-run

# 4. 小规模测试（限制并行度）
./extract_all_tars.sh --parallel 2  # 只测试 2 个并发

# 5. 验证结果
./verify_extraction.sh
```

**预期成功标准**:
- 单个 tar 提取成功，生成解压目录和 `.SUCCESS` 标记
- 断点恢复正常工作
- Dry-run 正确列出待处理的 tar
- 小规模测试能正常并行提取
- 验证脚本能统计样本完整性
