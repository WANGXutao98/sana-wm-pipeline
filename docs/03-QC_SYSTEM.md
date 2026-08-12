# SANA-WM QC 质检系统完整文档

> **文档状态**：当前有效  
> **最后更新**：2026-08-01  
> **维护者**：David Wang

---

## 目录

1. [系统概述](#一系统概述)
2. [Stage 1 快速扫描](#二stage-1-快速扫描2-分钟)
3. [Stage 2 深度检测](#三stage-2-深度检测4-10-小时-已完成)
4. [Stage 3 GPU 评估](#四stage-3-gpu-评估6-7-小时-待优化)
5. [人工审查](#五人工审查8-12-天-已产出结果)
6. [QC 结果汇总与数据提取](#六qc-结果汇总与数据提取-当前待办)
7. [性能与成本分析](#七性能与成本分析)
8. [论文对齐验证](#八论文对齐验证)
9. [故障排查](#九故障排查)
10. [未来优化方向](#十未来优化方向)

---

## 一、系统概述

### 1.1 设计目标

- **自动化覆盖率**：96%+（Stage 1+2+3）
- **人工审查比例**：3.6%（边界样本收尾）
- **误拒率目标**：< 2%
- **漏检率目标**：< 5%

### 1.2 三阶段架构

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: 快速扫描 (2 min, 11 项检查)                   │
│  ├─ SO(3) 旋转有效性                                    │
│  ├─ 首帧归零                                            │
│  ├─ FoV 范围                                            │
│  ├─ 焦距一致性                                          │
│  ├─ 轨迹跳变 (差异化阈值)                               │
│  ├─ Caption 长度                                        │
│  ├─ Caption 相机词                                      │
│  ├─ 颜色饱和度                                          │
│  └─ ...                                                 │
│  输出: stage1_results.jsonl                            │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: 深度检测 (4-10 h, 4 项检查) ✅               │
│  ├─ 黑帧比例检测                                        │
│  ├─ 场景切换检测                                        │
│  ├─ 轨迹冻结检测                                        │
│  └─ Caption 深度验证                                    │
│  输出: stage2_results.jsonl                            │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: GPU 评估 (6-7 h, 3 项检查) ⏳                │
│  ├─ UniMatch 光流连续性 ✅                              │
│  ├─ DOVER 视频质量 ⚠️ CPU 模式慢                       │
│  └─ Qwen Caption 改写 ⚠️ 需更换模型                    │
│  输出: stage3_results.jsonl                            │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  人工审查: 边界样本 (8-12 天, 6,702 样本) ✅            │
│  ├─ Pass 样本: 3,000 (验证漏检)                         │
│  ├─ Flag 样本: 3,076 (重点审查)                         │
│  └─ Fail 样本: 626 (确认误拒)                           │
│  输出: human_review_results.jsonl                      │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  最终汇总: 生成训练数据列表 ⏳                          │
│  └─ pass_final.txt (~14 万条样本)                       │
└─────────────────────────────────────────────────────────┘
```

### 1.3 当前项目状态（强制更新 2026-08-01）

| 模块 | 状态 | 备注 |
|------|------|------|
| **Stage 1 快速扫描** | ✅ 已完成 | 代码已实现，55 测试全过 |
| **Stage 2 深度检测** | ✅ 已完成 | 已全量执行，产出 jsonl |
| **Stage 3 GPU 评估** | ⏳ 待优化 | DOVER CPU 慢 + Qwen-27B 需更换 |
| **人工审查** | ✅ 已产出结果 | 6,702 样本标注完成，jsonl 格式 |
| **数据提取** | ⏳ 当前待办 | 本次文档整理完成后启动 |

---

## 二、Stage 1 快速扫描（2 分钟）

### 2.1 检查项完整清单（11 项）

| # | 检查项 | 阈值 | 用途 | 实现函数 |
|---|--------|------|------|---------|
| 1 | **SO(3) 旋转有效性** | \|det(R) - 1\| < 0.01 | 检测非法旋转矩阵 | `metrics.py:check_so3_validity` |
| 2 | **首帧归零** | ∥t₀∥ < 0.1m | 确保相对坐标系 | `metrics.py:check_first_frame_origin` |
| 3 | **FoV 范围** | 30° < FoV < 120° | 排除畸变/鱼眼镜头 | `metrics.py:check_fov_range` |
| 4 | **焦距一致性** | CV(fx) < 0.05 | 检测内参突变 | `metrics.py:check_focal_consistency` |
| 5 | **焦距差异** | \|fx - fy\| / fx < 0.1 | 检测非方形像素 | `metrics.py:check_focal_difference` |
| 6 | **轨迹跳变** | n_jumps < 30/50/80 | 检测传送/闪现（差异化） | `metrics.py:count_trajectory_jumps` |
| 7 | **Caption 长度** | len ≥ 50 字符 | 避免过于笼统 | `metrics.py:check_caption_length` |
| 8 | **Caption 相机词** | 无 camera/pan/zoom 等 | 确保 scene-static | `metrics.py:check_camera_words` |
| 9 | **颜色饱和度** | 0.1 < mean_saturation < 0.9 | 排除灰度/过饱和 | `metrics.py:check_color_saturation` |
| 10 | **视频时长** | 50s < duration < 70s | 确保足够训练帧数 | `metrics.py:check_video_duration` |
| 11 | **轨迹总长度** | total_distance > 1m | 排除定点拍摄 | `metrics.py:check_trajectory_length` |

### 2.2 差异化配置策略

**核心洞察**：游戏数据 vs 真实世界数据的物理规律不同，需要不同的判定标准。

| Group 类型 | max_jumps_fail | check_camera_words | caption_min_len | 理由 |
|-----------|---------------|-------------------|----------------|------|
| **OmniWorld-Game** | 50 | False | 50 | 游戏允许瞬移，caption 含框架词 |
| **sekai-game-walking** | 50 | False | 50 | 同上 |
| **sekai-game-drone** | 80 | False | 50 | 无人机高速移动，跳变更频繁 |
| **DL3DV-ALL-2K** | 30 | True | 0 | 真实场景，但无 caption（数据源问题）|
| **SpatialVID-hq** | 30 | True | 50 | 真实场景，高质量 caption |
| **RealEstate10K** | 30 | True | 50 | 真实室内场景 |
| **sekai-real-walking-hq** | 30 | True | 50 | 真实街景 |

### 2.3 执行命令

```bash
# 单个 group 执行
python -m sana_wm_pipeline.qc.stage1_fast \
  --input_dir /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/wds-SpatialVID-hq/ \
  --output stage1_spatialvid_results.jsonl \
  --config_name wds-SpatialVID-hq

# 全部 7 个 group 批量执行
for group in wds-DL3DV-ALL-2K wds-OmniWorld-Game wds-SpatialVID-hq \
             wds-RealEstate10K-360p wds-sekai-real-walking-hq \
             wds-sekai-game-drone wds-sekai-game-walking; do
  python -m sana_wm_pipeline.qc.stage1_fast \
    --input_dir /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/$group/ \
    --output qc_output/${group}_stage1.jsonl \
    --config_name $group
done
```

### 2.4 输出格式

```json
{
  "sample_id": "wds-SpatialVID-hq/w003/shard_012/sample_00456",
  "checks": {
    "so3_valid": true,
    "first_frame_origin": true,
    "fov_range": true,
    "fov_value": 65.3,
    "focal_consistency": true,
    "focal_cv": 0.012,
    "focal_difference": true,
    "n_jumps": 12,
    "caption_len": 127,
    "has_camera_words": false,
    "color_saturation": 0.45,
    "video_duration": 60.1,
    "trajectory_length": 8.7
  },
  "stage1_pass": true,
  "flags": [],
  "config_name": "wds-SpatialVID-hq"
}
```

---

## 三、Stage 2 深度检测（4-10 小时）✅ 已完成

### 3.1 检查项清单（4 项）

| # | 检查项 | 阈值 | 用途 | 单样本耗时 |
|---|--------|------|------|-----------|
| 1 | **黑帧比例** | < 5% | 检测编码错误/遮挡 | ~0.5s |
| 2 | **场景切换** | < 3 次 | 检测拼接视频 | ~1.0s |
| 3 | **轨迹冻结** | 连续静止 < 5s | 检测定点拍摄 | ~0.2s |
| 4 | **Caption 深度验证** | Qwen-mini VLM | 检测无效描述 | ~2.0s |

### 3.2 执行状态

✅ **已完成**：
- 全部 7 个 group 已执行完毕
- 产出 7 个 `*_stage2.jsonl` 文件
- Pass 率：77.1% (141,550 / 183,413)

---

## 四、Stage 3 GPU 评估（6-7 小时）⏳ 待优化

### 4.1 检查项清单（3 项）

| # | 检查项 | 模型 | 阈值 | 用途 | 单样本耗时 | 状态 |
|---|--------|------|------|------|-----------|------|
| 1 | **光流连续性** | UniMatch | mean_flow ∈ [3, 100] | 检测运动合理性 | ~3s | ✅ 实现完成 |
| 2 | **视频质量评分** | DOVER | score > 0.5 | 检测模糊/抖动 | ~5s (CPU) | ⚠️ CPU 模式慢 |
| 3 | **Caption 改写** | Qwen-27B | 自动改写相机词 | 生成 scene-static | ~2s | ⚠️ 需更换模型 |

### 4.2 当前问题与待办

#### 问题 1：DOVER CPU 模式性能瓶颈

**现状**：
- H100 GPU 模式不兼容（PyTorch 版本冲突）
- 强制使用 CPU 模式：`export DOVER_DEVICE=cpu`
- 性能：~5s/样本（CPU）vs ~0.5s/样本（GPU 理论值）

**待办**：
1. 验证 DOVER GPU 兼容性修复方案
2. 如无法修复，考虑替代模型（BRISQUE / NIQE）
3. 或接受 CPU 模式，优化多进程并行

#### 问题 2：Qwen-27B 模型效率

**现状**：
- 推理耗时：~2s/样本
- 显存占用：45GB
- 模型路径：`/root/work/david_work/models/Qwen2.5-VL-27B/`

**待办**：
- **选型更高效模型**（核心任务）
- 候选方案：
  1. Qwen2-14B-VL-INT4（量化版本）
  2. Qwen2-7B-VL（更小模型）
  3. Llama-3.2-11B-Vision
  4. InternVL2-8B

**目标**：
- 推理时间 < 1s/样本
- 显存 < 20GB
- Caption 改写质量不降低

### 4.3 执行命令（CMCC 48 GPU）

```bash
# Stage 3 需要 GPU，使用 Slurm 并行调度
python scripts/run_stage3_cmcc.py \
  --stage2_results qc_output/*/stage2_results.jsonl \
  --video_root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
  --output_dir /root/work/david_work/qc_output/ \
  --num_gpus 48 \
  --model_unimatch /root/work/david_work/models/UniMatch/ \
  --model_dover pip \
  --model_qwen /root/work/david_work/models/Qwen2.5-VL-27B/  # 待更换

# 预计耗时：6-7 小时（18 万样本 × 5.5s / 48 GPU）
```

### 4.4 输出格式

```json
{
  "sample_id": "wds-SpatialVID-hq/w003/shard_012/sample_00456",
  "stage1_pass": true,
  "stage2_pass": true,
  "checks": {
    "unimatch_flow": {
      "pass": true,
      "mean_flow": 15.3,
      "max_flow": 87.2
    },
    "dover_quality": {
      "pass": true,
      "score": 0.72
    },
    "caption_rewrite": {
      "original": "The camera follows a person walking through...",
      "rewritten": "A person walks through a modern office space...",
      "had_camera_words": true
    }
  },
  "stage3_pass": true,
  "processing_time_seconds": 5.8,
  "gpu_id": 12
}
```

---

## 五、人工审查（8-12 天）✅ 已产出结果

### 5.1 采样策略

| 类别 | 采样数量 | 采样策略 | 目的 |
|------|---------|---------|------|
| **Pass 样本** | 3,000 | 随机采样 | 验证漏检率（假阴性） |
| **Flag 样本** | 3,076 | 近阈值优先 | 重点审查边界案例 |
| **Fail 样本** | 626 | 全部采样 | 确认误拒率（假阳性） |
| **总计** | 6,702 | 3.6% 全量 | 最终质量把关 |

### 5.2 审查流程

1. **培训**（30 分钟）：阅读 `review_system/00-从这里开始.md`
2. **熟悉工具**（30 分钟）：运行 `review_helper.py`，练习 20 个黄金样本
3. **正式标注**（8-12 天）：每天 150-250 样本，5-10 人并行

### 5.3 当前状态

✅ **已完成**：
- 培训文档：6 个文件（review_system/）
- 黄金样本：20 个（golden_samples_with_data/）
- 审查批次：9 个批次打包（总计 ~110 GB）
- **标注结果**：已产出 jsonl 格式文件

### 5.4 结果格式

```json
{
  "sample_id": "wds-OmniWorld-Game/w002/shard_005/sample_00123",
  "human_label": "pass",
  "reason_code": "trajectory_minor_jump",
  "notes": "游戏场景跳变12次，属于正常范围",
  "reviewer": "reviewer_03",
  "timestamp": "2026-07-15T14:23:11Z"
}
```

**Label 枚举**：
- `pass`: 通过，可用于训练
- `flag`: 需进一步讨论
- `reject`: 拒绝，不可用

---

## 六、QC 结果汇总与数据提取 ⏳ 当前待办

### 6.1 三阶段结果合并

```python
def merge_qc_results(stage1_path, stage2_path, stage3_path, human_path):
    """合并四个阶段的 QC 结果，生成最终 pass 列表"""
    
    # 1. 加载四个阶段结果
    s1 = load_jsonl(stage1_path)
    s2 = load_jsonl(stage2_path)
    s3 = load_jsonl(stage3_path) if stage3_path else {}
    human = load_jsonl(human_path)
    
    # 2. 按 sample_id 合并
    results = {}
    for item in s1:
        sid = item['sample_id']
        results[sid] = {'stage1': item}
    
    for item in s2:
        results[item['sample_id']]['stage2'] = item
    
    for item in s3:
        results[item['sample_id']]['stage3'] = item
    
    for item in human:
        results[item['sample_id']]['human'] = item
    
    # 3. 最终判定逻辑
    pass_list = []
    reject_list = []
    
    for sid, data in results.items():
        # 人工标注优先级最高
        if 'human' in data:
            if data['human']['label'] == 'pass':
                pass_list.append(sid)
            elif data['human']['label'] == 'reject':
                reject_list.append(sid)
            # flag 样本需进一步决策
        
        # 自动化三阶段全 pass
        elif (data.get('stage1', {}).get('pass', False) and 
              data.get('stage2', {}).get('pass', False) and 
              data.get('stage3', {}).get('pass', True)):  # Stage 3 可选
            pass_list.append(sid)
        
        else:
            reject_list.append(sid)
    
    return pass_list, reject_list
```

### 6.2 数据提取脚本

```bash
# 1. 合并 QC 结果
python scripts/merge_qc_results.py \
  --stage1 qc_output/*/stage1_results.jsonl \
  --stage2 qc_output/*/stage2_results.jsonl \
  --stage3 qc_output/*/stage3_results.jsonl \
  --human qc_output/human_review_results.jsonl \
  --output qc_final_output/

# 输出：
#   pass_final.txt          # ~14 万条样本 ID
#   reject_final.txt        # ~4 万条样本 ID
#   qc_summary_report.html  # 可视化报告

# 2. 提取训练数据
python scripts/extract_training_data.py \
  --pass_list qc_final_output/pass_final.txt \
  --source_dir /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
  --output_dir /root/work/filestorage/shangaoooooo/davidwang/training_data_final/

# 3. 验证完整性
python scripts/verify_training_data.py \
  --data_dir /root/work/filestorage/shangaoooooo/davidwang/training_data_final/ \
  --expected_count 141550

# 4. 生成训练清单
python scripts/generate_training_manifest.py \
  --data_dir /root/work/filestorage/shangaoooooo/davidwang/training_data_final/ \
  --output training_manifest.json
```

### 6.3 最终输出

```
qc_final_output/
├── pass_final.txt             # 最终通过样本列表（~14 万条）
├── reject_final.txt           # 拒绝样本列表
├── qc_summary_report.html     # 可视化报告
└── qc_merged_results.jsonl    # 完整结果（调试用）

training_data_final/
├── 000000.tar                 # 训练 shard
├── 000001.tar
├── ...
└── training_manifest.json     # 元数据清单
```

---

## 七、性能与成本分析

### 7.1 QC 系统整体性能

| 阶段 | 耗时 | 计算资源 | 吞吐量 |
|------|------|---------|--------|
| Stage 1 | 2 min | 32 核 CPU | 全量 (18 万) |
| Stage 2 | 4-10 h | 32 核 CPU | Stage 1 pass (~4 万) |
| Stage 3 | 6-7 h | 48 H100 | 全量 (18 万) |
| 人工审查 | 8-12 天 | 5-10 人 | 6,702 样本 |
| **总计** | ~13 天 | 48 GPU + 10 人 | 18 万样本 |

### 7.2 Pass 率演进

| 优化阶段 | Pass 率 | 可用样本数 | 提升 |
|---------|---------|-----------|------|
| 初始版本 | 31.1% | 57,130 | - |
| Stage 1 优化 | 77.1% | 141,550 | +84,420 |
| Stage 3 优化（预期） | 80%+ | 146,000+ | +4,450+ |

### 7.3 成本优化建议

1. **Stage 3 加速**：TensorRT + 模型量化 → 耗时减半
2. **人工审查优化**：主动学习采样 → 样本数减半
3. **增量 QC**：新增样本只跑增量，避免全量重跑

---

## 八、论文对齐验证

### 8.1 论文 Table 6 过滤体系覆盖状态

| 论文指标 | 本项目实现 | 覆盖率 |
|---------|-----------|--------|
| VMAF Motion | UniMatch 光流（替代） | ✅ 100% |
| UniMatch Flow | UniMatch | ✅ 100% |
| DOVER Quality | DOVER | ✅ 100% |
| Caption Length | Stage 1 检查 | ✅ 100% |
| Camera Words | Stage 1 + Qwen 改写 | ✅ 100% |
| FoV Range | Stage 1 检查 | ✅ 100% |
| Trajectory Jumps | Stage 1 检查（差异化阈值） | ✅ 100% |

**结论**：✅ 本项目 QC 系统完整覆盖论文 App. B.3 所有过滤项。

---

## 九、故障排查

### 9.1 Stage 1 常见错误

**错误 1**：`KeyError: 'poses_c2w'`  
**原因**：输入目录缺少 Stage 2 输出  
**解决**：确认 `--input_dir` 包含 `poses_c2w.npy`

**错误 2**：`AssertionError: det(R) != 1`  
**原因**：旋转矩阵非法（数值精度问题）  
**解决**：放宽容差 `abs(det(R) - 1) < 1e-3`

### 9.2 Stage 3 常见错误

**错误 1**：`CUDA out of memory (DOVER)`  
**原因**：H100 GPU 模式不兼容  
**解决**：`export DOVER_DEVICE=cpu`

**错误 2**：`Qwen model type mismatch`  
**原因**：AutoModelForCausalLM 不适用于 VLM  
**解决**：改用 `AutoModel.from_pretrained(trust_remote_code=True)`

---

## 十、未来优化方向

### 10.1 短期（1 个月）

1. ✅ **QC 结果落地**：提取最终训练数据集
2. ⏳ **Stage 3 优化**：DOVER GPU 适配 + Qwen 模型更换
3. ⏳ **性能基准测试**：产出详细 benchmark 报告

### 10.2 中期（3 个月）

1. ⏳ **主动学习**：根据人工审查反馈动态调整阈值
2. ⏳ **增量 QC**：支持断点续传和增量样本评估
3. ⏳ **可视化看板**：实时监控 QC 进度和通过率

### 10.3 长期（6 个月+）

1. ⏳ **自适应阈值**：基于训练反馈自动优化 QC 参数
2. ⏳ **多模态融合**：结合音频特征进行质量评估
3. ⏳ **联邦 QC**：多机构协作质检（数据不出域）

---

**相关文档**：
- 系统架构：`01-ARCHITECTURE.md`
- 管线详解：`02-PIPELINE_STAGES.md`
- 部署指南：`04-DEPLOYMENT.md`
- 人工审查：`../review_system/00-从这里开始.md`
