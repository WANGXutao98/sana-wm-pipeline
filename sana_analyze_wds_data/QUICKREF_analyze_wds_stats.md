# WebDataset 统计分析工具 - 快速参考

## 🚀 快速开始

```bash
# 1. 基础用法
python3 analyze_wds_stats.py --input-dir /path/to/jdvbbfb_output

# 2. 查看帮助
python3 analyze_wds_stats.py --help

# 3. 运行测试
python3 test_analyze_wds_stats.py
```

## 📋 参数速查

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `--input-dir` | ✅ | 输入目录 | `/path/to/jdvbbfb_output` |
| `--output-dir` | ❌ | 输出目录（默认当前） | `./reports` |
| `--original-stats` | ❌ | 原始统计文件 | `original_stats.json` |
| `--verbose` | ❌ | 详细输出 | - |

## 📊 输出文件

```
output_dir/
├── shard_statistics.csv      # Shard 级别详细数据
├── dataset_statistics.json   # 完整统计（JSON）
└── comparison_report.md      # 对比报告（Markdown）
```

## 🎯 常用命令

### 基础分析
```bash
python3 analyze_wds_stats.py --input-dir ./jdvbbfb_output
```

### 完整分析（带对比）
```bash
python3 analyze_wds_stats.py \
  --input-dir ./jdvbbfb_output \
  --original-stats original_stats.json \
  --output-dir ./analysis_results \
  --verbose
```

### 仅特定数据集
```bash
# 先创建符号链接
mkdir temp_analysis
ln -s /path/to/jdvbbfb_output/final_wds-SpatialVID-hq temp_analysis/
python3 analyze_wds_stats.py --input-dir ./temp_analysis
```

## 📝 原始统计文件格式

创建 `original_stats.json`:

```json
{
  "datasets": {
    "SpatialVID-hq": {"total_samples": 365362},
    "DL3DV-ALL-2K": {"total_samples": 9993},
    "RealEstate10K-360p": {"total_samples": 73165},
    "OmniWorld-Game": {"total_samples": 6576},
    "sekai-real-walking-hq": {"total_samples": 18208},
    "sekai-game-walking": {"total_samples": 1618},
    "sekai-game-drone": {"total_samples": 932},
    "Context-as-Memory": {"total_samples": 100}
  }
}
```

## 🔍 目录结构要求

```
input_dir/
├── final_wds-{dataset1}/
│   └── wds-{dataset1}/
│       ├── w000/
│       │   ├── shard-000000-000000.tar
│       │   └── ...
│       └── w001/...
├── final_wds-{dataset2}/
└── ...
```

## ⚡ 性能指标

- **扫描速度**: ~0.1 秒/tar
- **1,353 shards**: ~2-5 分钟
- **内存占用**: < 500 MB

## 🛠️ 故障排除

### 问题：权限错误
```bash
# 检查目录权限
ls -ld /path/to/jdvbbfb_output

# 添加读权限
chmod -R u+r /path/to/jdvbbfb_output
```

### 问题：找不到数据集
```bash
# 检查目录结构
ls /path/to/jdvbbfb_output/
# 应该看到 final_wds-* 目录

# 检查子目录
ls /path/to/jdvbbfb_output/final_wds-*/
# 应该看到 wds-* 目录
```

### 问题：Python 版本
```bash
# 检查版本（需要 3.7+）
python3 --version

# 如果版本过低，使用 conda
conda create -n wds_stats python=3.9
conda activate wds_stats
```

## 📚 相关文档

- **完整文档**: `README_analyze_wds_stats.md`
- **代码分析**: `ANALYSIS_SUMMARY.md`
- **使用示例**: `example_usage.sh`
- **测试脚本**: `test_analyze_wds_stats.py`

## 💡 使用技巧

### 1. 后台运行大规模扫描
```bash
nohup python3 analyze_wds_stats.py \
  --input-dir /path/to/large_dataset \
  --verbose > scan.log 2>&1 &

# 查看进度
tail -f scan.log
```

### 2. 定期监控
```bash
# 每天 0 点运行
# 添加到 crontab -e
0 0 * * * cd /workspace && python3 analyze_wds_stats.py \
  --input-dir ./jdvbbfb_output \
  --output-dir ./daily_reports/$(date +\%Y-\%m-\%d)
```

### 3. 对比两次扫描结果
```bash
# 第一次扫描
python3 analyze_wds_stats.py \
  --input-dir ./data \
  --output-dir ./scan_v1

# 第二次扫描
python3 analyze_wds_stats.py \
  --input-dir ./data \
  --output-dir ./scan_v2 \
  --original-stats ./scan_v1/dataset_statistics.json
```

### 4. 只看特定数据集统计
```bash
# 扫描后用 jq 过滤 JSON
cat dataset_statistics.json | \
  jq '.datasets["final_wds-SpatialVID-hq"]'
```

## 🎓 核心原理

1. **不解压扫描**: 只读 tar 目录结构，不提取文件
2. **样本识别**: 从文件名提取 key（`{key}.{ext}`）
3. **去重统计**: 用 `set()` 统计唯一样本数
4. **分层汇总**: Shard → Dataset → Global

## ✅ 测试验证

```bash
# 运行快速测试（~1秒）
python3 test_analyze_wds_stats.py

# 预期输出
✓ Test passed! The analyzer works correctly.
```

---

**版本**: 1.0.0  
**更新**: 2026-07-28  
**作者**: Claude (Opus 4.8)
