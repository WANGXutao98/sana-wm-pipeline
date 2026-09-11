# run_worker.py 多模式使用指南

## 概述

`run_worker.py` 现在支持三种标注模式，可以在运行时通过 `--mode` 参数选择：

| 模式 | 适用数据集 | GT数据要求 |
|------|-----------|-----------|
| `default` | 互联网视频、SpatialVID-HQ | 无 |
| `gt_depth` | OmniWorld | GT深度图 (`depth.npy`) |
| `gt_pose` | Sekai-Game、DL3DV | GT poses (`poses.npy`) |

## 使用示例

### 模式1: Default（默认，向后兼容）

用于没有GT数据的互联网视频。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/batch_production/run_worker.py \
  --group wds-spatialvid-hq \
  --data-root /path/to/jdvbbfb-v3-full \
  --out-base /path/to/output \
  --worker-id 0 \
  --shard-indices 0,8,16,24 \
  --samples-per-shard 200
```

### 模式2: GT-Depth

用于有GT深度图的数据集（如OmniWorld）。

**GT数据目录结构**:
```
/path/to/gt_data/
├── sample_001/
│   └── depth.npy       # (T, H, W) GT深度
├── sample_002/
│   └── depth.npy
└── ...
```

**命令**:
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/batch_production/run_worker.py \
  --group wds-omniworld \
  --data-root /path/to/jdvbbfb-v3-full \
  --out-base /path/to/output \
  --worker-id 0 \
  --shard-indices 0,8,16 \
  --samples-per-shard 200 \
  --mode gt_depth \
  --gt-data-dir /path/to/gt_data
```

### 模式3: GT-Pose

用于有GT相机轨迹的数据集（如Sekai-Game、DL3DV）。

**GT数据目录结构**:
```
/path/to/gt_data/
├── sample_001/
│   └── poses.npy       # (T, 4, 4) GT c2w poses
├── sample_002/
│   └── poses.npy
└── ...
```

**命令**:
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/batch_production/run_worker.py \
  --group wds-sekai-real-walking-hq \
  --data-root /path/to/jdvbbfb-v3-full \
  --out-base /path/to/output \
  --worker-id 0 \
  --shard-indices 0,8,16 \
  --samples-per-shard 200 \
  --mode gt_pose \
  --gt-data-dir /path/to/gt_data
```

## 错误处理

### 错误1: 缺少 --gt-data-dir

```
[ERROR] --mode gt_depth 需要 --gt-data-dir 参数
```

**解决**: 添加 `--gt-data-dir /path/to/gt_data`

### 错误2: GT数据目录不存在

```
[ERROR] GT数据目录不存在: /path/to/gt_data
```

**解决**: 检查路径是否正确，确保目录存在

### 错误3: GT文件缺失

```
[FAIL] sample_001: GT depth not found: /path/to/gt_data/sample_001/depth.npy
```

**解决**: 
1. 检查GT数据是否完整
2. 确认文件命名符合规范（`depth.npy` 或 `poses.npy`）

## 输出格式

输出的 WebDataset tar 文件中，`Sample.meta` 会包含实际使用的模式：

```json
{
  "scene_id": "sample_001",
  "T": 121,
  "mode": "gt_pose",
  "dataset": "jdvbbfb-v3-full",
  "group": "wds-sekai-real-walking-hq",
  "source_shard": "sekai-real-walking-hq-000000.tar"
}
```

## 性能建议

- **Default模式**: 最慢（需要Pi3X+MoGe推理），~30-60s/样本
- **GT-Depth模式**: 中等（只需MoGe推理），~20-40s/样本
- **GT-Pose模式**: 最快（只需Pi3X推理），~10-20s/样本

建议使用NVMe SSD作为工作目录以减少IO开销。

## 参考文档

- [架构影响评估](./ARCHITECTURE_IMPACT_ASSESSMENT.md)
- [模式对齐分析](./MODE_ALIGNMENT_CRITICAL_ANALYSIS.md)
- [SANA-WM论文 Appendix B.1](../2605.15178v1.md)
