# 中移动机器使用指南 - WebDataset 统计分析工具

## 🎯 你的使用场景

**目标路径**: `~/work/filestorage/shangaoooooo/davidwang/repair_done/final_wds-SpatialVID-hq`

**目录结构**:
```
final_wds-SpatialVID-hq/
├── driver_logs/
└── wds-SpatialVID-hq/
    ├── logs/
    ├── progress/
    ├── w000/
    │   ├── shard-000000-000000.tar
    │   ├── shard-000000-000001.tar
    │   └── ... (更多 tar 文件)
    ├── w001/
    ├── w002/
    └── ... (w000-w047, 共48个worker)
```

✅ 这个结构完全符合工具要求！

---

## 🚀 使用步骤

### 步骤 1: 上传工具到中移动机器

在你的**本地机器或当前 H100 机器**上执行：

```bash
# 上传主程序到中移动机器
scp -P <端口> analyze_wds_stats.py root@<中移动机器IP>:~/work/

# 或者如果你有配置好的 SSH alias
scp analyze_wds_stats.py cmcc:~/work/
```

### 步骤 2: SSH 登录中移动机器

```bash
ssh root@<中移动机器IP> -p <端口>
# 或使用 alias
ssh cmcc
```

### 步骤 3: 运行分析（单个数据集）

```bash
cd ~/work

# 方法1: 直接指定 final_wds-SpatialVID-hq 的父目录
python3 analyze_wds_stats.py \
  --input-dir ~/work/filestorage/shangaoooooo/davidwang/repair_done \
  --output-dir ~/work/analysis_results \
  --verbose
```

**这个命令会**:
- ✅ 扫描 `final_wds-SpatialVID-hq` 目录
- ✅ 统计所有 48 个 worker (w000-w047) 的数据
- ✅ 计算总样本数和数据量
- ✅ 生成 CSV/JSON/Markdown 三种报告

### 步骤 4: 查看结果

```bash
# 查看终端输出摘要（已经显示）

# 查看 CSV 报告（每个 shard 详情）
cat ~/work/analysis_results/shard_statistics.csv

# 查看 JSON 报告（完整统计）
cat ~/work/analysis_results/dataset_statistics.json

# 查看 Markdown 报告（易读格式）
cat ~/work/analysis_results/comparison_report.md
```

---

## 📊 预期输出示例

### 终端输出
```
================================================================================
WebDataset Statistics Analyzer
================================================================================

Input directory: ~/work/filestorage/shangaoooooo/davidwang/repair_done
Output directory: ~/work/analysis_results
Scan started at: 2026-07-28 10:00:00

Found 1 dataset(s):
  - final_wds-SpatialVID-hq

Processing final_wds-SpatialVID-hq...
  ✓ Scanned XXX shards

✓ Scan completed in X.X seconds

================================================================================
数据集处理统计报告
================================================================================

总样本数: 365,362
总数据量: 1,030.42 GB
总 Shard 数: 714
数据集数量: 1

================================================================================
各数据集统计
================================================================================

【final_wds-SpatialVID-hq】
  样本数: 365,362
  数据量: 1,030.42 GB
  Shard 数: 714
  Worker 数: 48
  平均每 shard: 512.0 samples
```

### JSON 输出（关键字段）
```json
{
  "scan_time": "2026-07-28T10:00:00",
  "total_samples": 365362,
  "total_bytes": 1106561245184,
  "total_shards": 714,
  "datasets": {
    "final_wds-SpatialVID-hq": {
      "total_samples": 365362,
      "total_bytes": 1106561245184,
      "total_shards": 714,
      "workers": 48,
      "avg_samples_per_shard": 512.0,
      "size_gb": 1030.42
    }
  }
}
```

---

## 🔍 如果有多个数据集

如果你的 `repair_done` 目录下有多个数据集：

```
repair_done/
├── final_wds-SpatialVID-hq/
├── final_wds-DL3DV-ALL-2K/
└── final_wds-OmniWorld-Game/
```

工具会**自动扫描所有**数据集：

```bash
python3 analyze_wds_stats.py \
  --input-dir ~/work/filestorage/shangaoooooo/davidwang/repair_done \
  --output-dir ~/work/analysis_results \
  --verbose
```

输出会包含所有数据集的统计。

---

## 📋 与原始数据对比（可选）

如果你想对比处理完成度，创建原始统计文件：

```bash
# 在中移动机器上创建 original_stats.json
cat > ~/work/original_stats.json << 'EOF'
{
  "datasets": {
    "SpatialVID-hq": {
      "total_samples": 365362
    }
  }
}
EOF

# 运行带对比的分析
python3 analyze_wds_stats.py \
  --input-dir ~/work/filestorage/shangaoooooo/davidwang/repair_done \
  --output-dir ~/work/analysis_results \
  --original-stats ~/work/original_stats.json \
  --verbose
```

这样会在报告中显示：
```
数据集                  已处理      原始       完成率
SpatialVID-hq      365,362    365,362    100.0% ✓
```

---

## ⚡ 性能预估

基于你的数据规模（48 workers × 约15 shards/worker = ~720 shards）：

- **扫描时间**: 约 1-3 分钟
- **内存占用**: < 500 MB
- **输出文件**: 约 100-200 KB（三个报告文件总计）

---

## 🛠️ 故障排除

### 问题1: Python 版本
```bash
# 检查版本（需要 3.7+）
python3 --version

# 如果版本过低，使用 conda
conda activate base  # 或其他 Python 3.7+ 环境
```

### 问题2: 权限问题
```bash
# 检查目录权限
ls -ld ~/work/filestorage/shangaoooooo/davidwang/repair_done

# 如果需要，添加读权限
chmod -R u+r ~/work/filestorage/shangaoooooo/davidwang/repair_done
```

### 问题3: 磁盘空间
```bash
# 检查可用空间（输出报告需要少量空间）
df -h ~/work
```

---

## 📥 快速命令总结

```bash
# 1. 上传工具（从 H100 机器）
scp analyze_wds_stats.py cmcc:~/work/

# 2. SSH 登录中移动机器
ssh cmcc

# 3. 运行分析
cd ~/work
python3 analyze_wds_stats.py \
  --input-dir ~/work/filestorage/shangaoooooo/davidwang/repair_done \
  --output-dir ~/work/analysis_results \
  --verbose

# 4. 查看结果
cat ~/work/analysis_results/comparison_report.md
```

---

## 💡 提示

1. **首次使用建议加 `--verbose`**: 可以看到扫描进度
2. **输出目录会自动创建**: 不需要手动 mkdir
3. **可以多次运行**: 每次会覆盖之前的报告
4. **JSON 文件可用于后续分析**: 例如用 Python/jq 进一步处理

---

## 🎯 你的具体命令

基于你提供的路径，**完整命令**是：

```bash
# SSH 登录中移动机器后
cd ~/work

# 运行分析
python3 analyze_wds_stats.py \
  --input-dir ~/work/filestorage/shangaoooooo/davidwang/repair_done \
  --output-dir ~/work/analysis_results \
  --verbose

# 查看 Markdown 报告（最易读）
cat ~/work/analysis_results/comparison_report.md

# 或查看 JSON（如果需要机器处理）
cat ~/work/analysis_results/dataset_statistics.json | python3 -m json.tool
```

---

**准备好了吗？现在就可以上传工具到中移动机器并开始使用！** 🚀
