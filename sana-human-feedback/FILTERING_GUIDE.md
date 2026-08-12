# SANA-WM 人工反馈数据筛选 - 操作指南

> **文档版本**：v1.0  
> **生成日期**：2026-08-01  
> **维护者**：数据集处理团队

---

## 一、任务执行摘要

### 1.1 筛选结果

✅ **筛选完成**：从 9 个人工标注文件中成功筛选出 **1,980 条**高质量训练样本

| 指标 | 数值 |
|------|------|
| **源文件总样本数** | 6,698 |
| **筛选通过样本数** | 1,980 |
| **通过率** | 29.56% |

### 1.2 输出文件

| 文件 | 路径 | 用途 |
|------|------|------|
| **筛选结果** | `sana-human-feedback/filtered_training_samples.jsonl` | CMCC 数据回溯匹配清单 |
| **统计报表** | `sana-human-feedback/filter_statistics.txt` | 详细统计数据 |
| **筛选脚本** | `scripts/filter_human_feedback.py` | 可重复执行的筛选工具 |

---

## 二、筛选条件说明

### 2.1 硬性筛选规则（双重条件，缺一不可）

#### 条件 1：数据集归属限定

**仅保留以下 3 个数据集**：
- `RealEstate10K`（室内场景，YouTube 视频源）
- `SpatialVID-hq`（高质量空间视频）
- `sekai-real-walking-hq`（真实街景行走）

**排除数据集**（已过滤）：
- `OmniWorld-Game`（游戏合成数据）
- `sekai-game-drone`（游戏无人机视角）
- `sekai-game-walking`（游戏角色行走）
- `DL3DV-ALL-2K`（3D 重建数据集）

#### 条件 2：质量评分限定

**保留评级**：
- `good`（良好）
- `excellent`（优秀）

**排除评级**（已过滤）：
- `average`（中等）
- `poor`（较差）

### 2.2 数据集分布统计

| 数据集 | Good | Excellent | 总计 | 占比 |
|--------|------|-----------|------|------|
| **SpatialVID-hq** | 499 | 482 | 981 | 49.5% |
| **sekai-real-walking-hq** | 182 | 516 | 698 | 35.3% |
| **RealEstate10K** | 278 | 23 | 301 | 15.2% |
| **总计** | 959 | 1,021 | 1,980 | 100% |

**关键发现**：
- ✅ **SpatialVID-hq** 样本质量最高，excellent 占比 49.1%
- ✅ **sekai-real-walking-hq** 优质样本占比最大（excellent 73.9%）
- ⚠️ **RealEstate10K** excellent 样本较少（仅 23 条），主要为 good 评级

---

## 三、筛选结果文件说明

### 3.1 文件格式

**文件类型**：JSONL（JSON Lines）  
**编码格式**：UTF-8  
**行数**：1,980 行（每行一条样本）

### 3.2 数据 Schema

每条样本包含以下字段（**完全继承原始标注数据，无删减**）：

```json
{
  "sample_id": "SpatialVID-hq_783abd4b-a070-5a25-b95f-c831d49d8a0e",
  "quality_rating": "excellent",
  "use_for_training": true,
  "issues": [],
  "notes": "n_jumps=32 较高，但视频看起来很流畅",
  "annotator": "yn"
}
```

**字段说明**：

| 字段 | 类型 | 说明 | 回溯用途 |
|------|------|------|---------|
| `sample_id` | string | **样本唯一标识符** | ✅ 用于在 CMCC 原始数据集中匹配源文件 |
| `quality_rating` | string | 质量评分（good/excellent） | 训练数据分级依据 |
| `use_for_training` | boolean | 是否用于训练标记 | 二次确认字段 |
| `issues` | array | 标注人员发现的问题列表 | 质量分析参考 |
| `notes` | string | 标注备注 | 边界案例说明 |
| `annotator` | string | 标注人员 ID | 溯源与质量审计 |

### 3.3 sample_id 命名规则

#### 格式 1：SpatialVID-hq
```
SpatialVID-hq_<uuid>
示例：SpatialVID-hq_783abd4b-a070-5a25-b95f-c831d49d8a0e
```

#### 格式 2：sekai-real-walking-hq
```
sekai-real-walking-hq_<youtube_id>_<start_frame>_<end_frame>
示例：sekai-real-walking-hq_a4xbZ7ogoVM_0070350_0072150
```

#### 格式 3：RealEstate10K
```
RealEstate10K-360p_<split>__<unique_id>
示例：RealEstate10K-360p_train__11034893f72fe474
```

---

## 四、CMCC 数据回溯操作指南

### 4.1 前置条件

**CMCC 原始数据集路径**：
```bash
/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/
├── wds-RealEstate10K-360p/
├── wds-SpatialVID-hq/
└── wds-sekai-real-walking-hq/
```

**每个样本包含的文件**：
```
<sample_id>.video.mp4          # 视频文件（720p, 16fps）
<sample_id>.poses_c2w.npy      # 相机轨迹
<sample_id>.intrinsics.npy     # 内参
<sample_id>.scale.npy          # 尺度
<sample_id>.caption.txt        # 文字描述
```

### 4.2 回溯匹配脚本

创建 `scripts/extract_training_data_from_filtered.py`：

```python
#!/usr/bin/env python3
"""
从 CMCC 原始数据集中提取筛选后的训练样本

输入：filtered_training_samples.jsonl
输出：打包后的训练数据集
"""

import json
import shutil
from pathlib import Path

# 配置
FILTERED_LIST = "/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filtered_training_samples.jsonl"
CMCC_DATA_ROOT = "/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output"
OUTPUT_DIR = "/root/work/filestorage/shangaoooooo/davidwang/training_data_filtered_v1"

# 数据集路径映射
DATASET_MAPPING = {
    "RealEstate10K": "wds-RealEstate10K-360p",
    "SpatialVID-hq": "wds-SpatialVID-hq",
    "sekai-real-walking-hq": "wds-sekai-real-walking-hq"
}

def extract_dataset_name(sample_id):
    """从 sample_id 提取数据集名称"""
    for dataset in DATASET_MAPPING.keys():
        if sample_id.startswith(dataset):
            return dataset
    return None

def find_sample_files(sample_id, dataset_name):
    """在 CMCC 原始数据集中查找样本文件"""
    dataset_dir = Path(CMCC_DATA_ROOT) / DATASET_MAPPING[dataset_name]
    
    # 搜索所有 worker 目录
    for worker_dir in dataset_dir.glob("w*"):
        # 5 个必需文件
        required_files = [
            worker_dir / f"{sample_id}.video.mp4",
            worker_dir / f"{sample_id}.poses_c2w.npy",
            worker_dir / f"{sample_id}.intrinsics.npy",
            worker_dir / f"{sample_id}.scale.npy",
            worker_dir / f"{sample_id}.caption.txt"
        ]
        
        # 检查文件是否存在
        if all(f.exists() for f in required_files):
            return required_files
    
    return None

def main():
    print("=" * 70)
    print("CMCC 训练数据回溯提取任务")
    print("=" * 70)
    
    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取筛选后的样本列表
    with open(FILTERED_LIST, 'r') as f:
        filtered_samples = [json.loads(line) for line in f]
    
    print(f"\n📋 待提取样本数：{len(filtered_samples)}")
    
    # 统计
    success_count = 0
    missing_count = 0
    missing_samples = []
    
    # 逐条提取
    for i, sample in enumerate(filtered_samples, 1):
        sample_id = sample['sample_id']
        dataset_name = extract_dataset_name(sample_id)
        
        if i % 100 == 0:
            print(f"⏳ 进度：{i}/{len(filtered_samples)}")
        
        # 查找源文件
        source_files = find_sample_files(sample_id, dataset_name)
        
        if source_files:
            # 复制文件到输出目录
            for src_file in source_files:
                dst_file = output_path / src_file.name
                shutil.copy2(src_file, dst_file)
            success_count += 1
        else:
            missing_count += 1
            missing_samples.append(sample_id)
            print(f"  ⚠️  找不到样本：{sample_id}")
    
    # 输出统计
    print("\n" + "=" * 70)
    print("提取完成统计")
    print("=" * 70)
    print(f"✅ 成功提取：{success_count}")
    print(f"❌ 缺失样本：{missing_count}")
    
    if missing_samples:
        print("\n缺失样本列表：")
        for sample_id in missing_samples:
            print(f"  - {sample_id}")
    
    print(f"\n📦 输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

### 4.3 执行步骤

```bash
# 1. 登录 CMCC 机器
ssh cmcc_server

# 2. 激活环境
source /root/work/david_work/activate_sana_wm.sh

# 3. 执行提取脚本
python3 /path/to/extract_training_data_from_filtered.py

# 4. 验证提取结果
ls -lh /root/work/filestorage/shangaoooooo/davidwang/training_data_filtered_v1/ | wc -l
# 预期：1980 × 5 = 9900 个文件

# 5. 打包为 WebDataset tar 格式（可选）
python3 /path/to/pack_to_webdataset.py \
  --input /root/work/filestorage/shangaoooooo/davidwang/training_data_filtered_v1/ \
  --output /root/work/filestorage/shangaoooooo/davidwang/training_shards_filtered_v1/ \
  --samples_per_shard 100

# 6. 备份到持久存储
rsync -av --progress \
  /root/work/filestorage/shangaoooooo/davidwang/training_data_filtered_v1/ \
  /root/work/filestorage/shangaoooooo/davidwang/backup/training_data_filtered_v1_$(date +%Y%m%d)/
```

---

## 五、数据质量分析

### 5.1 评分分布

| 评分 | 数量 | 占比 |
|------|------|------|
| **Excellent** | 1,021 | 51.6% |
| **Good** | 959 | 48.4% |

**分析**：
- ✅ 超过半数样本为 excellent 评级，整体质量优秀
- ✅ good 评级样本占比接近半数，可作为补充训练数据

### 5.2 数据集质量对比

| 数据集 | Excellent 占比 | 推荐用途 |
|--------|---------------|---------|
| **sekai-real-walking-hq** | 73.9% | ⭐⭐⭐ 核心训练集 |
| **SpatialVID-hq** | 49.1% | ⭐⭐ 主力训练集 |
| **RealEstate10K** | 7.6% | ⭐ 多样性补充 |

**建议**：
1. **优先级排序**：sekai-real-walking-hq > SpatialVID-hq > RealEstate10K
2. **训练策略**：可按 excellent 和 good 分两批训练，或混合采样
3. **数据增强**：RealEstate10K 样本较少，可考虑适度增强

### 5.3 拒绝原因分析

| 拒绝原因 | 数量 | 占比 |
|---------|------|------|
| **数据集不匹配** | 3,880 | 57.9% |
| **评分不达标** | 838 | 12.5% |

**分析**：
- 游戏类数据（OmniWorld / sekai-game）占比较高但被全部过滤
- 评分不达标主要为 average 和 poor 评级
- 筛选策略保守，确保训练数据高质量

---

## 六、后续迭代调整指引

### 6.1 调整筛选条件

**如需放宽条件**，编辑 `scripts/filter_human_feedback.py`：

```python
# 增加数据集
TARGET_DATASETS = {
    "RealEstate10K",
    "SpatialVID-hq",
    "sekai-real-walking-hq",
    "DL3DV-ALL-2K"  # 新增
}

# 放宽评分标准
TARGET_RATINGS = {"good", "excellent", "average"}  # 新增 average
```

重新执行：
```bash
python3 scripts/filter_human_feedback.py
```

### 6.2 增量筛选

**如有新的标注文件**，直接放入 `sana-human-feedback/` 目录，脚本会自动处理所有 `annotation_results_*.jsonl` 文件。

### 6.3 数据去重

当前版本 **不进行去重**（保留所有符合条件的样本）。

如需去重，在 `filter_human_feedback.py` 中添加：
```python
# 在 filter_samples() 函数中
seen_sample_ids = set()

for sample in samples:
    sample_id = sample['sample_id']
    if sample_id in seen_sample_ids:
        continue  # 跳过重复样本
    seen_sample_ids.add(sample_id)
    filtered_samples.append(sample)
```

---

## 七、注意事项与最佳实践

### 7.1 数据完整性保障

✅ **已实施**：
- 完整保留原始字段，不删改任何标注信息
- 保留 `notes` 和 `issues` 字段，用于质量分析
- 保留 `annotator` 字段，支持溯源审计

### 7.2 回溯匹配风险

⚠️ **潜在风险**：
1. **样本缺失**：CMCC 原始数据集可能部分样本损坏或丢失
2. **路径变更**：数据集目录结构调整导致匹配失败
3. **文件不完整**：5 个必需文件（video/poses/intrinsics/scale/caption）缺失

**缓解措施**：
- 提取前先验证 CMCC 数据集完整性
- 生成缺失样本清单，向数据团队反馈
- 备份原始数据到持久存储

### 7.3 版本管理

**建议命名规范**：
```
filtered_training_samples_v1.0_20260801.jsonl
training_data_filtered_v1.0_20260801/
```

**版本记录**：
- v1.0 (2026-08-01)：初始版本，1,980 样本
- v1.1 (TBD)：增加 average 评级，预计 +XXX 样本
- v2.0 (TBD)：整合新标注批次

---

## 八、快速参考

### 8.1 关键文件路径

```bash
# 筛选结果
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filtered_training_samples.jsonl

# 统计报表
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filter_statistics.txt

# 筛选脚本
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/filter_human_feedback.py

# CMCC 原始数据
/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/
```

### 8.2 常用命令

```bash
# 查看筛选结果样本数
wc -l filtered_training_samples.jsonl

# 统计各数据集样本数
grep -o '"sample_id": "[^"]*"' filtered_training_samples.jsonl | \
  cut -d'"' -f4 | cut -d'_' -f1 | sort | uniq -c

# 统计各评级样本数
grep -o '"quality_rating": "[^"]*"' filtered_training_samples.jsonl | \
  cut -d'"' -f4 | sort | uniq -c

# 查看特定数据集样本
grep "SpatialVID-hq" filtered_training_samples.jsonl | head -n 5
```

---

## 九、联系与支持

**维护团队**：数据集处理团队  
**技术负责人**：David Wang  
**文档版本**：v1.0  
**最后更新**：2026-08-01

**相关文档**：
- QC 系统文档：`docs/03-QC_SYSTEM.md`
- 数据集说明：`docs/reference/DATASETS.md`
- 部署指南：`docs/04-DEPLOYMENT.md`

---

**附录：筛选口径留存**

```yaml
筛选版本: v1.0
筛选日期: 2026-08-01
源文件数: 9
源样本总数: 6698
筛选通过数: 1980
通过率: 29.56%

筛选条件:
  数据集白名单:
    - RealEstate10K
    - SpatialVID-hq
    - sekai-real-walking-hq
  
  评分白名单:
    - good
    - excellent
  
  逻辑关系: AND（两个条件必须同时满足）

数据集分布:
  SpatialVID-hq: 981 (49.5%)
  sekai-real-walking-hq: 698 (35.3%)
  RealEstate10K: 301 (15.2%)

评分分布:
  excellent: 1021 (51.6%)
  good: 959 (48.4%)
```
