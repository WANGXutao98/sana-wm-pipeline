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
