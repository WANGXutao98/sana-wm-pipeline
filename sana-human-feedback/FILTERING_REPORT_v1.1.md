# SANA-WM 人工反馈数据筛选任务 v1.1 - 执行报告

> **执行日期**：2026-08-01  
> **任务状态**：✅ 成功完成  
> **版本**：v1.1（新增 acceptable 评级）

---

## 一、版本说明

### v1.1 主要变更

**筛选条件放宽**：
- v1.0：仅保留 `good` + `excellent` 评级
- v1.1：新增 `acceptable`（可接受）评级

**评级说明**：
- `excellent`（优秀）：最高质量，无明显问题
- `good`（良好）：质量良好，轻微瑕疵
- `acceptable`（可接受）：可用于训练，但存在一定问题（如轻微模糊、运动不够流畅等）

---

## 二、核心数据统计

### 2.1 总体统计

| 指标 | v1.0 | v1.1 | 增量 |
|------|------|------|------|
| **筛选通过数** | 1,980 | **2,651** | **+671** |
| **通过率** | 29.56% | **39.59%** | **+10.03%** |
| **增长率** | - | - | **+33.9%** |

### 2.2 评分分布

| 评分 | 数量 | 占比 | 质量等级 |
|------|------|------|---------|
| **Excellent** | 1,021 | 38.5% | ⭐⭐⭐ |
| **Good** | 959 | 36.2% | ⭐⭐ |
| **Acceptable** | 671 | 25.3% | ⭐ |
| **总计** | **2,651** | **100%** | - |

---

## 三、按数据集分类统计

### 3.1 详细分布

| 数据集 | Excellent | Good | Acceptable | 总计 | 占比 |
|--------|-----------|------|------------|------|------|
| **SpatialVID-hq** | 482 | 499 | 632 | **1,613** | **60.8%** |
| **RealEstate10K** | 23 | 278 | 20 | **321** | **12.1%** |
| **sekai-real-walking-hq** | 516 | 182 | 19 | **717** | **27.1%** |
| **总计** | **1,021** | **959** | **671** | **2,651** | **100%** |

### 3.2 版本对比（各数据集增量）

| 数据集 | v1.0 | v1.1 | 增量 | 增长率 |
|--------|------|------|------|--------|
| **SpatialVID-hq** | 981 | 1,613 | **+632** | **+64.4%** |
| **RealEstate10K** | 301 | 321 | +20 | +6.6% |
| **sekai-real-walking-hq** | 698 | 717 | +19 | +2.7% |
| **总计** | **1,980** | **2,651** | **+671** | **+33.9%** |

**关键发现**：
- ✅ **SpatialVID-hq** 是 acceptable 样本的主要来源（632 条，占新增样本的 94.2%）
- ✅ **sekai-real-walking-hq** 和 **RealEstate10K** 的 acceptable 样本较少，说明这两个数据集整体质量更高

---

## 四、数据质量分析

### 4.1 各数据集质量对比

| 数据集 | Excellent 占比 | Good 占比 | Acceptable 占比 | 质量评价 |
|--------|---------------|-----------|----------------|---------|
| **sekai-real-walking-hq** | 72.0% | 25.4% | 2.6% | ⭐⭐⭐ 优质 |
| **SpatialVID-hq** | 29.9% | 30.9% | 39.2% | ⭐⭐ 中等偏上 |
| **RealEstate10K** | 7.2% | 86.6% | 6.2% | ⭐⭐ 中等 |

**分析**：
- **sekai-real-walking-hq**：高质量样本占比最高（97.4% 为 good 或以上），acceptable 样本极少
- **SpatialVID-hq**：质量分布较为均衡，acceptable 样本占比较高（39.2%）
- **RealEstate10K**：以 good 评级为主（86.6%），excellent 和 acceptable 样本都较少

### 4.2 Acceptable 样本特征分析

**示例样本**：
```json
{
  "sample_id": "RealEstate10K-360p_train__a856be05423bf9f5",
  "quality_rating": "acceptable",
  "use_for_training": true,
  "issues": ["运动流畅但画面模糊"],
  "notes": "",
  "annotator": "yn"
}
```

**常见问题**：
- 画面模糊（但运动流畅）
- 轻微抖动
- 光照不均
- n_jumps 略高但可接受

---

## 五、输出产物

### 5.1 文件清单

| 文件 | 路径 | 大小 | 行数 | 用途 |
|------|------|------|------|------|
| **v1.1 筛选结果** | `filtered_training_samples_v1.1_with_acceptable.jsonl` | ~232 KB | 2,651 | CMCC 数据回溯 |
| **v1.1 统计报表** | `filter_statistics_v1.1_corrected.txt` | ~2 KB | - | 详细统计 |
| **v1.1 筛选脚本** | `scripts/filter_human_feedback_v1.1_corrected.py` | ~8 KB | - | 可重复执行 |
| **v1.0 筛选结果** | `filtered_training_samples.jsonl` | ~173 KB | 1,980 | 高质量基线 |

### 5.2 快速验证

```bash
# 验证 v1.1 文件行数
wc -l filtered_training_samples_v1.1_with_acceptable.jsonl
# 输出：2651

# 统计各评级样本数
grep -o '"quality_rating": "[^"]*"' filtered_training_samples_v1.1_with_acceptable.jsonl | \
  cut -d'"' -f4 | sort | uniq -c
# 输出：
#  671 acceptable
# 1021 excellent
#  959 good

# 查看 acceptable 样本示例
grep '"acceptable"' filtered_training_samples_v1.1_with_acceptable.jsonl | head -n 1 | python3 -m json.tool
```

---

## 六、训练建议

### 6.1 推荐训练策略

#### 策略 A：分层训练（推荐）

**阶段 1**：仅使用 excellent + good 样本（v1.0）
- 样本数：1,980
- 质量：高
- 用途：模型基础训练，确保基线质量

**阶段 2**：加入 acceptable 样本（v1.1）
- 样本数：2,651
- 质量：中等偏上
- 用途：增强模型鲁棒性，提升泛化能力

#### 策略 B：质量加权混合训练

**采样权重**：
- excellent：权重 1.0
- good：权重 0.8
- acceptable：权重 0.5

**优点**：一次训练，按质量加权采样

#### 策略 C：数据集优先级训练

**优先级排序**：
1. sekai-real-walking-hq（高质量，717 样本）
2. SpatialVID-hq（中等质量，样本量大，1,613 样本）
3. RealEstate10K（中等质量，321 样本）

### 6.2 使用建议

**推荐场景**：
- ✅ 需要更多训练样本时使用 v1.1
- ✅ 希望提升模型对中等质量视频的鲁棒性
- ✅ 数据增强后质量评估

**不推荐场景**：
- ❌ 追求最高训练数据质量（使用 v1.0）
- ❌ 计算资源有限，只能训练少量样本（优先 v1.0）

---

## 七、CMCC 数据回溯指引

### 7.1 回溯脚本更新

**修改提取脚本**以支持 v1.1：

```python
# 将 FILTERED_LIST 路径修改为 v1.1 文件
FILTERED_LIST = "/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filtered_training_samples_v1.1_with_acceptable.jsonl"
OUTPUT_DIR = "/root/work/filestorage/shangaoooooo/davidwang/training_data_filtered_v1.1"
```

### 7.2 预期产出

**文件数量**：
- v1.1：2,651 样本 × 5 个文件 = **13,255 个文件**
- v1.0：1,980 样本 × 5 个文件 = 9,900 个文件
- 增量：+3,355 个文件

**存储空间预估**：
- 单样本平均大小：~50 MB（video + poses + intrinsics + scale + caption）
- v1.1 总大小：~133 GB
- v1.0 总大小：~99 GB
- 增量：~34 GB

---

## 八、版本选择指南

### 8.1 版本对比

| 指标 | v1.0 | v1.1 | 差异 |
|------|------|------|------|
| **样本数** | 1,980 | 2,651 | +671 (+33.9%) |
| **质量标准** | 严格（good+excellent） | 宽松（+acceptable） | 放宽 |
| **Excellent 占比** | 51.6% | 38.5% | -13.1% |
| **Good 占比** | 48.4% | 36.2% | -12.2% |
| **Acceptable 占比** | 0% | 25.3% | +25.3% |
| **推荐用途** | 基础训练 | 鲁棒性训练 | - |

### 8.2 选择建议

**使用 v1.0（严格筛选）**：
- ✅ 追求最高训练数据质量
- ✅ 模型基础训练阶段
- ✅ 计算资源有限

**使用 v1.1（放宽筛选）**：
- ✅ 需要更多训练样本
- ✅ 提升模型鲁棒性
- ✅ 数据增强后二次训练

**混合使用（推荐）**：
1. 先用 v1.0 训练基础模型
2. 再用 v1.1 进行微调
3. 观察 acceptable 样本对模型的影响

---

## 九、质量监控建议

### 9.1 训练时监控指标

**按评级监控损失**：
```python
# 训练脚本中添加
train_loss_by_rating = {
    'excellent': [],
    'good': [],
    'acceptable': []
}

for batch in dataloader:
    loss = model(batch)
    rating = batch['quality_rating']
    train_loss_by_rating[rating].append(loss.item())
```

**定期生成报告**：
- Excellent 样本平均损失
- Good 样本平均损失
- Acceptable 样本平均损失

**决策标准**：
- 如果 acceptable 样本损失显著高于 excellent/good，考虑降低其采样权重
- 如果三者损失接近，说明 acceptable 样本质量可接受

### 9.2 模型评估

**验证集选择**：
- 仅使用 excellent 样本作为验证集
- 确保评估标准统一

**A/B 测试**：
- 模型 A：仅用 v1.0 训练
- 模型 B：用 v1.1 训练
- 对比两者在高质量验证集上的表现

---

## 十、快速参考

### 10.1 关键路径

```bash
# v1.0 筛选结果（高质量）
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filtered_training_samples.jsonl

# v1.1 筛选结果（包含 acceptable）
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filtered_training_samples_v1.1_with_acceptable.jsonl

# v1.1 统计报表
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filter_statistics_v1.1_corrected.txt

# v1.1 筛选脚本
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/filter_human_feedback_v1.1_corrected.py
```

### 10.2 常用命令

```bash
# 对比两个版本的差异
comm -13 \
  <(cut -d'"' -f4 filtered_training_samples.jsonl | sort) \
  <(cut -d'"' -f4 filtered_training_samples_v1.1_with_acceptable.jsonl | sort) \
  > v1.1_new_samples.txt

# 查看新增样本数
wc -l v1.1_new_samples.txt
# 输出：671

# 统计新增样本的数据集分布
for sample in $(cat v1.1_new_samples.txt); do
  echo $sample | cut -d'_' -f1
done | sort | uniq -c
```

---

## 十一、后续迭代

### 11.1 v1.2 规划（可选）

**潜在改进方向**：
1. **进一步放宽**：包含部分 poor 样本中的可用案例
2. **增加数据集**：纳入游戏数据（如需多样性）
3. **智能筛选**：基于训练反馈动态调整阈值

### 11.2 迭代流程

```bash
# 1. 修改筛选条件
vim scripts/filter_human_feedback_v1.2.py

# 2. 执行筛选
python3 scripts/filter_human_feedback_v1.2.py

# 3. 生成新版本文件
# filtered_training_samples_v1.2.jsonl

# 4. 对比版本差异
# v1.0 → v1.1 → v1.2
```

---

## 十二、联系与支持

**维护团队**：数据集处理团队  
**技术负责人**：David Wang  
**文档版本**：v1.1  
**最后更新**：2026-08-01

**相关文档**：
- v1.0 操作指南：`sana-human-feedback/FILTERING_GUIDE.md`
- QC 系统文档：`docs/03-QC_SYSTEM.md`
- 数据集说明：`docs/reference/DATASETS.md`

---

## 附录：筛选口径留存

```yaml
筛选版本: v1.1
筛选日期: 2026-08-01
源文件数: 9
源样本总数: 6698
筛选通过数: 2651
通过率: 39.59%

筛选条件:
  数据集白名单:
    - RealEstate10K
    - SpatialVID-hq
    - sekai-real-walking-hq
  
  评分白名单:
    - excellent
    - good
    - acceptable  # v1.1 新增
  
  逻辑关系: AND

数据集分布:
  SpatialVID-hq: 1613 (60.8%)
  sekai-real-walking-hq: 717 (27.1%)
  RealEstate10K: 321 (12.1%)

评分分布:
  excellent: 1021 (38.5%)
  good: 959 (36.2%)
  acceptable: 671 (25.3%)

版本对比:
  v1.0 → v1.1 增量: +671 样本 (+33.9%)
  主要来源: SpatialVID-hq (+632)
```

---

## ✅ 任务完成确认

- [x] v1.1 筛选脚本开发完成
- [x] 数据筛选执行完成（2,651 条样本）
- [x] 评级修正（acceptable 而非 average）
- [x] 统计报表生成完成
- [x] 版本对比分析完成
- [x] 训练建议文档编写完成

**两个版本均可直接交付训练团队使用！** 🚀

**推荐训练路径**：先用 v1.0 建立基线 → 再用 v1.1 提升鲁棒性
