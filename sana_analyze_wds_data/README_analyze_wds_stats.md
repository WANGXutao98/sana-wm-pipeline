# WebDataset 统计分析工具

快速扫描 WebDataset tar 文件并生成统计报告，专为分析 sana-wm 数据管线输出设计。

## 功能特性

- ✅ 快速扫描（不解压 tar 文件）
- ✅ 统计每个 shard 的样本数
- ✅ 生成 CSV、JSON、Markdown 报告
- ✅ 与原始数据集对比
- ✅ 只依赖 Python 标准库

## 安装

### 方法 1: 直接使用（推荐）

无需安装额外依赖，Python 3.7+ 即可运行：

```bash
python3 analyze_wds_stats.py --help
```

### 方法 2: 安装可选依赖（进度条）

如果需要进度条显示，可以安装 tqdm：

**使用阿里云镜像（国内推荐）：**
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ tqdm
```

**使用清华镜像：**
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
```

## 使用方法

### 基础用法

```bash
python3 analyze_wds_stats.py --input-dir /path/to/jdvbbfb_output
```

### 与原始数据对比

```bash
python3 analyze_wds_stats.py \
  --input-dir /path/to/jdvbbfb_output \
  --original-stats original_stats.json \
  --output-dir ./analysis_results
```

### 详细输出

```bash
python3 analyze_wds_stats.py \
  --input-dir /path/to/jdvbbfb_output \
  --verbose
```

## 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--input-dir` | ✅ | 输入目录（包含 final_wds-* 子目录） |
| `--output-dir` | ❌ | 输出目录（默认：当前目录） |
| `--original-stats` | ❌ | 原始统计 JSON 文件（用于对比） |
| `--verbose` | ❌ | 显示详细进度信息 |

## 输出文件

脚本会生成以下报告文件：

- **`shard_statistics.csv`** - 每个 shard 的详细信息
- **`dataset_statistics.json`** - 完整统计数据（JSON 格式）
- **`comparison_report.md`** - 对比报告（Markdown 格式）

## 输入目录结构

预期的输入目录结构：

```
jdvbbfb_output/
├── final_wds-DL3DV-ALL-2K/
│   └── wds-DL3DV-ALL-2K/
│       ├── w000/
│       │   ├── shard-000000-000000.tar
│       │   ├── shard-000000-000001.tar
│       │   └── ...
│       ├── w001/
│       └── ... (w002-w039)
├── final_wds-SpatialVID-hq/
├── final_wds-OmniWorld-Game/
└── ...
```

## 示例输出

### 终端输出

```
================================================================================
数据集处理统计报告
================================================================================

总样本数: 475,954
总数据量: 2,127.26 GB (2.08 TB)
总 Shard 数: 1,353
数据集数量: 8

================================================================================
各数据集统计
================================================================================

【final_wds-SpatialVID-hq】
  样本数: 365,362
  数据量: 1,030.42 GB
  Shard 数: 714
  Worker 数: 40
  平均每 shard: 512.0 samples

...

================================================================================
与原始数据对比
================================================================================
数据集                         已处理         原始      完成率
--------------------------------------------------------------------------------
SpatialVID-hq              365,362    365,362      100.0% ✓
DL3DV-ALL-2K                 9,993      9,993      100.0% ✓
...
```

### CSV 输出示例

```csv
dataset,worker,shard,samples,size_bytes,size_mb,size_gb
final_wds-SpatialVID-hq,w000,shard-000000-000000.tar,512,1610612736,1536.00,1.5000
final_wds-SpatialVID-hq,w000,shard-000000-000001.tar,512,1610612736,1536.00,1.5000
...
```

## 性能

- **扫描速度**: 约 2-5 分钟（1,353 个 shard）
- **内存占用**: < 500 MB
- **单个 tar 扫描**: < 0.1 秒

## 原理

1. 读取 tar 文件的目录结构（不解压内容）
2. 从文件名提取样本 key（WebDataset 约定：`{key}.{ext}`）
3. 统计唯一 key 的数量
4. 汇总所有 shard 的统计信息

## 故障排除

### 权限错误

如果遇到权限问题，确保对输入目录有读权限：

```bash
ls -ld /path/to/jdvbbfb_output
```

### 损坏的 tar 文件

损坏的 tar 文件会被自动跳过，并在终端显示警告信息。

### Python 版本

需要 Python 3.7 或更高版本：

```bash
python3 --version
```

## 远程使用

### 在 GPU 服务器上使用

如果在远程 GPU 服务器上使用，先上传脚本：

```bash
# 从本地上传到服务器
scp -P 10523 analyze_wds_stats.py root@180.184.148.133:/mnt/afs/davidwang/workspace/
```

然后在服务器上运行：

```bash
cd /mnt/afs/davidwang/workspace
python3 analyze_wds_stats.py --input-dir ./jdvbbfb_output
```

## 生成原始统计文件

如果需要创建原始统计文件用于对比，可以手动创建 JSON 文件：

```json
{
  "datasets": {
    "SpatialVID-hq": {
      "total_samples": 365362
    },
    "DL3DV-ALL-2K": {
      "total_samples": 9993
    },
    "RealEstate10K-360p": {
      "total_samples": 73165
    },
    "OmniWorld-Game": {
      "total_samples": 6576
    },
    "sekai-real-walking-hq": {
      "total_samples": 18208
    },
    "sekai-game-walking": {
      "total_samples": 1618
    },
    "sekai-game-drone": {
      "total_samples": 932
    },
    "Context-as-Memory": {
      "total_samples": 100
    }
  }
}
```

保存为 `original_stats.json` 后使用 `--original-stats` 参数。

## 快速测试

运行内置测试脚本验证工具是否正常工作：

```bash
python3 test_analyze_wds_stats.py
```

## 许可证

MIT License

## 作者

Claude (Opus 4.8)

## 版本

1.0.0 - 2026-07-28
