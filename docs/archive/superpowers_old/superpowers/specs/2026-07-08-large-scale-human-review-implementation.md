# 大规模人工审查系统落地方案

**日期**: 2026-07-08  
**状态**: 设计已批准  
**负责人**: David Wang  
**审查规模**: 6,000-7,000 样本  
**人力**: 5 名测试人员  
**时间**: 3 周

---

## 概述

本文档描述 SANA-WM 视频质量检测系统在 Stage 1+2 和 Stage 3 之间的大规模人工审查落地方案。目标是在 1 周内完成 6,000-7,000 个样本的人工审查，优化自动化筛选准确性，为 Stage 3（GPU 密集型）提供高质量输入。

---

## 目标

### 主要目标
- 5 名测试人员在 1 周（5 个工作日）内完成 6,000-7,000 个样本的审查
- 优化 Stage 1+2 的自动化筛选结果
- 为 Stage 3 提供人工验证的高质量样本清单

### 次要目标
- 收集结构化反馈，识别自动化阈值问题
- 发现 Stage 1+2 检测逻辑的盲区
- 为未来规则优化提供量化数据

---

## 约束条件

### 人力资源
- **人员**: 5 名测试人员（混合背景：部分有视频 QA 经验，部分新手）
- **时间**: 1 周（5 个工作日 × 8 小时 = 200 人时）
- **工作模式**: 分批并行审查，自学+答疑

### 数据规模
- **总样本**: ~140,000 个（7 个 group）
- **审查目标**: 6,000-7,000 个（~5% 覆盖率）
- **单样本时间**: 1.5-2 分钟

### 技术栈
- Export 脚本：支持 balanced/flag_driven 采样
- Import 脚本：验证格式、生成 disagreement 分析
- Apply 脚本：合并人工决策到最终清单
- 已通过端到端测试验证

---

## 总体架构

### 核心原则

1. **Flag 比例驱动采样**
   - 根据 Stage 1+2 实际结果动态分配配额
   - Flag 率高的 group 获得更多审查资源

2. **分批分发，灵活调整**
   - 分 2 批：Batch 1（3,500）+ Batch 2（2,500-3,500）
   - 可根据进度动态平衡工作量

3. **轻量质控，自学为主**
   - 黄金样本训练（15-20 个）
   - 每日 15 分钟站会
   - 随机抽查 5%

4. **不重跑 Stage 1+2**
   - 人工审查结果直接应用
   - 阈值优化建议记录到文档，留待下次迭代

### 三周时间线

```
Week 1 (准备): Stage 1+2 全量 → 分析结果 → 生成 Batch 1
Week 2 (审查): Batch 1 审查 → Batch 2 生成+审查
Week 3 (决策): 导入分析 → 决策 → Stage 3 开始
```

---

## 采样策略详细设计

### 总样本量

**目标**: 6,000-7,000 个样本

**理由**:
- 5 人 × 5 天 × 8 小时 = 200 人时
- 单样本 1.5-2 分钟 → 理论容量 6,000-8,000
- 保守估计 6,000-7,000（留缓冲）

### 分配公式

基于 Stage 1+2 实际结果动态分配：

```python
group_quota = min(
    max(500, flag_count * 0.15),  # 至少500个，flag的15%
    total_samples * 0.10           # 不超过该group总数的10%
)
```

**约束**:
- 每个 group 最低 500 个（保证统计显著性）
- Flag 样本采样率 15%（最需要人工判断）
- 单 group 不超过总数的 10%（避免过度采样小 group）

### 采样优先级（每个 group 内部）

1. **边界样本（40%）**: auto_fail 但接近阈值
   - n_jumps = threshold + 1 或 +2
   - caption_len = threshold - 5
   - 目的：发现阈值是否过严

2. **多问题样本（20%）**: flag_reasons ≥ 2 个
   - 复杂情况，需要人工权衡
   - 目的：理解多因素交互

3. **Pass 验证（30%）**: 随机抽取 pass 样本
   - 检查是否有漏检（false negative）
   - 目的：验证自动化的保守程度

4. **随机 Fail（10%）**: 明显失败的样本
   - 校准测试人员的判断标准
   - 目的：建立"底线"共识

### 分批策略

#### Batch 1（3,500 样本，3 天）

**目标**: 快速覆盖高风险区域

**包含**:
- Flag 率 > 30% 的 group
- 预计：sekai-real-walking (1,800) + SpatialVID (900) + 其他高 flag group (800)

**分配**:
- Person 1: 1,200 样本
- Person 2: 1,200 样本
- Person 3: 1,100 样本

**时间**: Day 1-3 (Week 2)

#### Batch 2（2,500-3,500 样本，2 天）

**目标**: 补充覆盖，平衡工作量

**包含**:
- 剩余 group
- Batch 1 的补充样本（如果有 group 需要更多）

**分配**:
- Person 4: 1,200-1,500 样本
- Person 5: 1,200-1,500 样本
- Batch 1 完成快的人: 帮忙（动态分配）

**时间**: Day 4-5 (Week 2)

---

## 测试人员操作手册设计

### 手册结构

**目标读者**:
- 主要：5 名测试人员（混合背景）
- 次要：项目管理人员

**设计原则**:
- 分层结构：快速开始 + 详细指南
- 大量示例和截图
- 易于查阅的 FAQ

### 第一部分：快速开始（5 分钟上手）

#### 1.1 审查目标

在 Stage 1+2 自动筛选之后、Stage 3 GPU 评估之前，人工审查视频样本，判断是否应该通过质检。

**你的任务**:
- 播放视频（10-20 秒）
- 判断质量是否合格
- 填写 CSV 文件

#### 1.2 必需工具

1. **VLC Media Player** - 视频播放
   - 下载：https://www.videolan.org/
   - 快捷键：空格（播放/暂停）、方向键（快进/快退）

2. **Excel / LibreOffice Calc** - 填写审查结果
   - 推荐 Excel 或 LibreOffice（保证 CSV 兼容性）

#### 1.3 三步流程

**步骤 1**: 下载你的审查包
- 解压 `personX_batchY.zip`
- 包含：`videos/`（视频文件）+ `review_list.csv` + `decisions_template.csv`

**步骤 2**: 逐个审查样本
- 用 VLC 播放 `videos/` 中的视频
- 参考 `review_list.csv` 中的自动化判断和指标
- 观察视频质量、轨迹、caption 匹配度

**步骤 3**: 填写 `decisions_template.csv`
- **human_verdict**: `pass` 或 `fail`（不确定可留空）
- **primary_issue**: 主要问题类型（从枚举值中选择）
- 保存为 `personX_batchY_filled.csv` 提交

### 第二部分：详细指南

#### 2.1 审查标准详解

**轨迹质量**：相机运动是否平滑？

- **Good（好）**: 相机运动流畅，无明显跳跃
  - 示例：缓慢平移、稳定前进、平滑转向

- **Acceptable（可接受）**: 有小的不连续，但整体可用
  - 示例：偶尔小跳跃（< 0.5m），但大部分流畅
  - **关键**：如果大部分帧都连续，小瑕疵可接受

- **Poor（差）**: 明显跳跃、抖动、传送
  - 示例：突然从室内跳到室外、相机位置瞬移
  - **判断**：如果跳跃 > 3 次，通常是 poor

**视频质量**：画面是否清晰？

- **Good**: 清晰、无模糊、无伪影
- **Acceptable**: 轻微模糊或压缩伪影，但可辨识
- **Poor**: 严重模糊、大量伪影、黑屏、卡顿

**Caption 匹配**：文字描述是否准确？

- **Good**: 准确描述视频内容
  - 示例：caption "person walking in a park"，视频确实是人在公园走
  
- **Acceptable**: 大致正确但不精确
  - 示例：caption "indoor scene"，视频是室内但描述太泛

- **Poor**: 明显不匹配
  - 示例：caption "outdoor"，视频明显是室内

#### 2.2 字段填写规范

**必填字段**:

1. **human_verdict**（你的判断）
   - `pass`: 该样本应该通过质检，进入 Stage 3
   - `fail`: 该样本应该被淘汰
   - **留空**: 不确定时可留空，系统将使用自动化判断

2. **primary_issue**（主要问题）
   
   从以下 11 种问题类型中选择：
   
   ```
   trajectory_minor          小的轨迹跳变（可接受范围内）
   trajectory_major          大的轨迹跳变（明显不连续）
   video_blurry              视频模糊
   video_artifacts           视频伪影/卡顿
   caption_mismatch          Caption与内容不匹配
   caption_too_vague         Caption过于泛化
   black_frames              包含黑屏
   scene_cut_abrupt          突兀的场景切换
   multiple_issues           多个问题同时存在
   no_issue                  没有明显问题（用于pass样本）
   other                     其他（请在notes说明）
   ```

**可选字段**（建议填写）:

3. **video_quality**: `good` / `acceptable` / `poor`

4. **trajectory_quality**: `good` / `acceptable` / `poor`

5. **notes**: 自由文本备注
   - 用于记录特殊情况或不确定的地方
   - 示例："轨迹有3个小跳跃但整体流畅"

#### 2.3 边界情况处理

**情况 1**: 轨迹有小跳跃，但整体流畅
- **判断**: Pass
- **填写**: 
  - human_verdict = `pass`
  - trajectory_quality = `acceptable`
  - primary_issue = `trajectory_minor`
  - notes = "3个小跳跃但可接受"

**情况 2**: 视频质量差，但轨迹完美
- **判断**: 看严重程度
  - 如果只是轻微模糊 → Pass（video_quality = acceptable）
  - 如果严重模糊无法辨识 → Fail
- **原则**: 轨迹和视频质量都需要达标

**情况 3**: Caption 不匹配，但视频质量好
- **判断**: Fail（Caption 是关键信息）
- **填写**:
  - human_verdict = `fail`
  - primary_issue = `caption_mismatch`
  - notes = 说明不匹配的具体内容

**情况 4**: 自动化判 fail，但你觉得 pass
- **判断**: 按你的判断填 pass
- **这正是人工审查的价值**！记录你的理由到 notes

**情况 5**: 完全不确定
- **判断**: 留空 human_verdict
- **建议**: 先标记，审查完其他样本后回头看
- **或**: 在每日站会提出讨论

#### 2.4 工作节奏建议

**推荐节奏**（基于 1,200 样本/人）:
- 每天目标：400 样本
- 每小时：~50 样本
- 每 100 个样本休息 5-10 分钟（防止疲劳）

**质量优先于速度**:
- 如果感到疲劳，休息
- 不确定的样本可以留空
- 完成 80-90% 已经很好

### 第三部分：黄金样本训练

#### 3.1 训练目标

在正式审查前，先审查 15-20 个"标准答案"样本，统一判断标准。

**要求**: 与标准答案一致性 ≥ 80%（12/15 或 16/20）

**如果未达标**: 
- 重新学习操作手册
- 与项目负责人讨论分歧样本
- 重新测试

#### 3.2 黄金样本清单

**样本 1-5: 明显 Pass**
- 完美轨迹 + 清晰视频 + 准确 caption
- 目的：建立"优秀样本"的基准

**样本 6-10: 明显 Fail**
- 严重跳跃、模糊、或 caption 完全错误
- 目的：建立"底线"标准

**样本 11-15: 边界样本**
- 需要仔细权衡的情况
- 目的：对齐边界判断标准

**样本 16-20: 复杂样本**
- 多个小问题同时存在
- 目的：学习如何选择 primary_issue

*(具体黄金样本由项目负责人生成，附带标准答案)*

#### 3.3 自测清单

审查黄金样本后，自检：
- [ ] 我理解 good/acceptable/poor 的区别吗？
- [ ] 我能快速识别主要问题类型吗？
- [ ] 我知道边界情况如何处理吗？
- [ ] 我的判断与标准答案一致性 ≥ 80% 吗？

### 第四部分：质量检查清单

#### 4.1 每审查 100 个样本，自查

- [ ] **完整性**: human_verdict 都填了吗？留空率是否 < 20%？
- [ ] **一致性**: pass/fail 比例是否合理？（不应极端偏向）
- [ ] **primary_issue**: 每个 fail 样本都有问题类型吗？
- [ ] **notes**: 对不确定的样本做记录了吗？

#### 4.2 每日站会（15 分钟）

**时间**: 每天上午 10:00（建议）

**议程**:
1. 进度同步（每人 1 分钟）
2. 困难样本讨论（5 分钟）
3. 标准对齐（5 分钟）

**带上你的问题**:
- 展示 1-2 个不确定的样本
- 听听其他人的判断
- 达成共识

#### 4.3 提交前检查

完成所有样本后，最终检查：
- [ ] 文件名正确：`personX_batchY_filled.csv`
- [ ] 必填列都填了：sample_id, human_verdict, primary_issue
- [ ] 格式正确：用 Excel 打开没有乱码
- [ ] 备份：保留一份本地副本

### 第五部分：FAQ

**Q1: 如果视频无法播放怎么办？**
A: 检查文件是否存在，尝试其他播放器。如果确认损坏，在 notes 标记"video_missing"，留空 human_verdict。

**Q2: 如果 review_list 中的自动化判断和我的判断完全相反？**
A: 按你的判断填写！这正是人工审查的价值。记录理由到 notes。

**Q3: 如果一个样本有多个问题，如何选择 primary_issue？**
A: 选择最严重的那个。如果无法区分，选 `multiple_issues`。

**Q4: 留空率多少合理？**
A: 10-20% 可接受。超过 30% 说明标准不清，需要讨论。

**Q5: 我可以修改之前审查的样本吗？**
A: 可以！在提交前都可以修改。建议完成后回头检查前 50 个样本。

**Q6: 如果我比其他人快很多/慢很多？**
A: 正常。快的人会分配更多 Batch 2 样本，慢的人可以减少配额。质量优先于速度。

**Q7: 我需要审查 caption 的语法错误吗？**
A: 不需要。只关注 caption 是否准确描述视频内容，不管语法。

**Q8: 如果视频很长（> 30 秒），需要全部看完吗？**
A: 不需要。看前 10-20 秒足够判断质量。如果前面有问题，可以快进抽查。

**Q9: 什么时候该填 notes？**
A: 不确定的情况、边界样本、或你的判断与自动化差异很大时。Notes 帮助后续分析。

**Q10: 提交截止时间是？**
A: Batch 1: Week 2 Day 3 下午 5pm
    Batch 2: Week 2 Day 5 下午 5pm

**Q11: 提交到哪里？**
A: 上传到项目共享文件夹（具体路径由项目负责人提供）或发送给项目负责人。

---

## 执行计划和时间线

### Week 1：准备阶段（5 个工作日）

#### Day 1-2：全量 Stage 1+2 运行

**负责人**: 项目负责人

**任务**:
```bash
# 在 CMCC 机器上执行
cd /root/work/david_work/sana_wm_qc

# 数据根目录
DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/repair_done"

# 对每个 group 运行 Stage 1+2
for group_dir in $DATA_ROOT/final_wds-*; do
  # 跳过 driver_logs
  [ ! -d "$group_dir" ] && continue
  
  group_name=$(basename "$group_dir" | sed 's/final_wds-//')
  wds_dir="$group_dir/wds-$group_name"
  
  # 检查 wds 目录是否存在
  if [ ! -d "$wds_dir" ]; then
    echo "Warning: $wds_dir not found, skipping"
    continue
  fi
  
  echo "Processing $group_name..."
  
  python scripts/run_qc.py \
    --tar-root "$wds_dir" \
    --group "$group_name" \
    --output-dir qc_full_output/$group_name \
    --n-workers 16 \
    --read-video-frames
done

# 预期处理的 groups:
# - DL3DV-ALL-2K
# - OmniWorld-Game
# - SpatialVID-hq
# - sekai-game-drone
# - sekai-game-walking
# - sekai-real-walking-hq
```

**输出**:
- 每个 group 的 `stage1_results.jsonl`
- 每个 group 的 `stage2_results.jsonl`
- Manifests: `pass.txt`, `fail.txt`, `human_review.txt`

**预计时间**: 2 天（7 个 group 并行处理）

**检查点**:
- [ ] 所有 group 的 Stage 1+2 完成
- [ ] 统计各 group 的 pass/fail/flag 比例
- [ ] 验证输出文件格式正确

---

#### Day 3 上午：分析结果，计算配额

**负责人**: 项目负责人

**任务**: 编写并运行采样配额计算脚本

```python
# scripts/calculate_sampling_quotas.py
import json
import glob
from pathlib import Path

def calculate_quotas(stage1_files, target_total=6500):
    """基于 flag 比例计算各 group 配额"""
    
    group_stats = {}
    
    for jsonl_path in stage1_files:
        group = Path(jsonl_path).parent.name
        with open(jsonl_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]
        
        total = len(samples)
        flag_count = sum(1 for s in samples if s.get('verdict') == 'flag')
        pass_count = sum(1 for s in samples if s.get('verdict') == 'pass')
        fail_count = sum(1 for s in samples if s.get('verdict') == 'fail')
        
        group_stats[group] = {
            'total': total,
            'pass': pass_count,
            'fail': fail_count,
            'flag': flag_count,
            'flag_rate': flag_count / total if total > 0 else 0
        }
    
    # 计算配额
    total_flags = sum(s['flag'] for s in group_stats.values())
    
    quotas = {}
    for group, stats in group_stats.items():
        # 基础配额（按 flag 比例）
        base_quota = int(stats['flag'] * 0.15)  # flag 的 15%
        
        # 约束
        quota = max(500, base_quota)  # 至少 500
        quota = min(quota, int(stats['total'] * 0.10))  # 不超过总数的 10%
        
        quotas[group] = quota
    
    # 调整总数到目标
    current_total = sum(quotas.values())
    scale = target_total / current_total
    
    for group in quotas:
        quotas[group] = int(quotas[group] * scale)
    
    # 输出
    result = {
        'target_total': target_total,
        'actual_total': sum(quotas.values()),
        'groups': {}
    }
    
    for group in quotas:
        result['groups'][group] = {
            **group_stats[group],
            'quota': quotas[group],
            'sampling_rate': quotas[group] / group_stats[group]['total']
        }
    
    return result

# 运行
stage1_files = glob.glob('qc_full_output/*/stage1_results.jsonl')
quotas = calculate_quotas(stage1_files, target_total=6500)

# 保存
with open('sampling_quotas.json', 'w') as f:
    json.dump(quotas, f, indent=2)

print(f"Total quota: {quotas['actual_total']}")
for group, info in quotas['groups'].items():
    print(f"{group}: {info['quota']} samples ({info['sampling_rate']*100:.1f}% of {info['total']})")
```

**输出**: `sampling_quotas.json`

**检查点**:
- [ ] 总配额在 6,000-7,000 之间
- [ ] 每个 group 至少 500 个
- [ ] Flag 率高的 group 配额更多

---

#### Day 3 下午：生成黄金样本

**负责人**: 项目负责人

**任务**: 手动精选 15-20 个样本作为训练集

**选择标准**:
- 5 个明显 pass（完美样本）
- 5 个明显 fail（严重问题）
- 5-10 个边界样本（需要仔细判断）

**操作步骤**:
1. 从各 group 随机抽取候选样本
2. 逐个审查，选出代表性样本
3. 填写标准答案和理由
4. 打包：`golden_samples.zip`

**输出文件结构**:
```
golden_samples/
├── videos/
│   ├── sample_001.mp4
│   ├── sample_002.mp4
│   └── ...
├── golden_review_list.csv
├── golden_template.csv
└── golden_answers.csv  (标准答案 + 理由)
```

**检查点**:
- [ ] 15-20 个样本已选出
- [ ] 标准答案已填写
- [ ] 理由清晰易懂

---

#### Day 4：生成 Batch 1

**负责人**: 项目负责人

**任务**: 使用 export 脚本生成 Batch 1 审查包

```bash
# 生成 Batch 1（3,500 样本）
python scripts/export_for_review.py \
  --stage1-jsonl qc_full_output/*/stage1_results.jsonl \
  --stage2-jsonl qc_full_output/*/stage2_results.jsonl \
  --output-dir batch1_review \
  --total-samples 3500 \
  --sampling-strategy flag_driven \
  --quotas sampling_quotas.json \
  --priority-groups "sekai-real-walking,SpatialVID,OmniWorld"

# 输出文件结构
# batch1_review/
# ├── videos/  (3,500 个 mp4)
# ├── review_list.csv
# ├── decisions_template.csv
# └── sampling_report.txt
```

**分配给测试人员**:

根据 `review_list.csv`，按 sample_id 分配：

```bash
# 分割脚本
python scripts/split_batch_for_persons.py \
  --review-list batch1_review/review_list.csv \
  --template batch1_review/decisions_template.csv \
  --video-dir batch1_review/videos \
  --num-persons 3 \
  --output-dir batch1_split

# 生成 3 个压缩包
# batch1_split/person1_batch1.zip (1,200 samples)
# batch1_split/person2_batch1.zip (1,200 samples)
# batch1_split/person3_batch1.zip (1,100 samples)
```

**分发方式**:
- 上传到共享文件夹
- 通知 Person 1-3 下载

**检查点**:
- [ ] Batch 1 生成完成（3,500 样本）
- [ ] 分割成 3 个包
- [ ] 测试人员已下载

---

#### Day 5：培训和答疑

**任务**:
1. **上午**: 发送操作手册给所有测试人员
2. **下午**: 黄金样本自测
   - 测试人员自行审查 15-20 个黄金样本
   - 对比标准答案，计算一致性
   - 一致性 < 80% 需要重新学习
3. **答疑**: 项目负责人在线答疑

**检查点**:
- [ ] 所有人收到操作手册
- [ ] 所有人完成黄金样本测试
- [ ] 一致性 ≥ 80%

---

### Week 2：审查阶段（5 个工作日）

#### Day 1-3：Batch 1 审查

**参与人员**: Person 1, 2, 3

**任务**:
- Person 1: 审查 1,200 样本（400/天）
- Person 2: 审查 1,200 样本（400/天）
- Person 3: 审查 1,100 样本（370/天）

**每日节奏**:
- **10:00**: 15 分钟站会
  - 进度同步
  - 困难样本讨论
  - 标准对齐
- **工作时间**: 审查样本
- **17:00**: 提交当天进度（可选）

**质控措施**:
- 项目负责人每天随机抽查 5%（~60 个）
- 发现问题及时反馈

**检查点（Day 3 下午）**:
- [ ] Person 1-3 完成 Batch 1
- [ ] 提交 `personX_batch1_filled.csv`
- [ ] 快速检查完整性（必填列都填了）

---

#### Day 3 下午：生成 Batch 2

**负责人**: 项目
负责人

**任务**: 生成 Batch 2

```bash
# 计算剩余配额
python scripts/calculate_remaining_quotas.py \
  --original-quotas sampling_quotas.json \
  --batch1-samples batch1_review/sampling_report.txt \
  --output remaining_quotas.json

# 生成 Batch 2
python scripts/export_for_review.py \
  --stage1-jsonl qc_full_output/*/stage1_results.jsonl \
  --stage2-jsonl qc_full_output/*/stage2_results.jsonl \
  --output-dir batch2_review \
  --total-samples 3000 \
  --sampling-strategy flag_driven \
  --quotas remaining_quotas.json \
  --exclude batch1_review/review_list.csv  # 排除已在 Batch 1 的样本

# 分配
python scripts/split_batch_for_persons.py \
  --review-list batch2_review/review_list.csv \
  --template batch2_review/decisions_template.csv \
  --video-dir batch2_review/videos \
  --num-persons 2 \
  --output-dir batch2_split

# 如果 Person 1-3 有人提前完成，也可以参与 Batch 2
```

**分发**:
- 主要：Person 4, 5
- 辅助：Batch 1 完成快的人

**检查点**:
- [ ] Batch 2 生成完成
- [ ] 分发给 Person 4-5

---

#### Day 4-5：Batch 2 审查

**参与人员**: Person 4, 5（+ 可选的 Person 1-3）

**任务**:
- Person 4: 1,200-1,500 样本
- Person 5: 1,200-1,500 样本
- 快速完成者帮忙

**继续质控**:
- 每日站会
- 随机抽查

**检查点（Day 5 下午 17:00）**:
- [ ] 所有人完成审查
- [ ] 提交所有 `personX_batchY_filled.csv`
- [ ] 总审查量 ≥ 5,000

---

### Week 3：分析和决策（5 个工作日）

#### Day 1：导入和分析

**负责人**: 项目负责人

**任务 1**: 导入所有人工审查结果

```bash
# 对每个人的结果运行 import
for person in person1 person2 person3 person4 person5; do
  for batch in batch1 batch2; do
    if [ -f ${person}_${batch}_filled.csv ]; then
      python scripts/import_review_results.py \
        --review-list ${batch}_review/review_list.csv \
        --decisions ${person}_${batch}_filled.csv \
        --output-dir analysis/${person}_${batch} \
        --reviewer ${person}
    fi
  done
done
```

**输出**:
```
analysis/
├── person1_batch1/
│   ├── human_review_results.jsonl
│   └── disagreement_report.html
├── person2_batch1/
│   └── ...
└── ...
```

**任务 2**: 合并所有结果

```bash
# 合并所有 human_review_results.jsonl
cat analysis/*/human_review_results.jsonl > all_human_reviews.jsonl

# 生成汇总报告
python scripts/generate_consolidated_report.py \
  --inputs analysis/*/human_review_results.jsonl \
  --stage1 qc_full_output/*/stage1_results.jsonl \
  --output final_analysis/
```

**输出**:
```
final_analysis/
├── consolidated_report.html  (完整分析报告)
├── disagreement_summary.json  (各group的disagreement统计)
├── inter_rater_agreement.json  (测试人员间一致性，如果有重叠样本)
└── threshold_recommendations.md  (阈值优化建议)
```

**分析内容**:
1. **总体统计**:
   - 审查样本数、完成率
   - Pass/Fail 分布
   - 留空率

2. **Disagreement 分析**:
   - Auto Fail → Human Pass: X 个（可能是阈值过严）
   - Auto Pass → Human Fail: Y 个（可能是漏检）
   - 按 group 和 primary_issue 分类

3. **阈值建议**:
   - 如果某个指标的 disagreement > 20%，建议调整
   - 例如：n_jumps 阈值从 2 调整到 5

4. **测试人员一致性**:
   - 如果有重叠样本，计算 Kappa 系数
   - 识别需要额外培训的测试人员

**检查点**:
- [ ] 所有结果导入成功
- [ ] 汇总报告生成
- [ ] 阈值建议清晰

---

#### Day 2 上午：决策会议

**参与人**: 项目负责人 + 管理层

**议程**:

1. **审查汇总报告**（30 分钟）
   - 总体统计
   - 主要发现
   - Disagreement 分析

2. **阈值调整决策**（30 分钟）
   - 讨论每个建议
   - **决定**: 本次不重跑 Stage 1+2
   - **记录**: 优化建议到文档，留待下次迭代

3. **最终清单确认**（30 分钟）
   - 应用人工审查结果
   - 生成最终 pass.txt
   - 确认 Stage 3 输入

**输出**: 决策记录文档

---

#### Day 2 下午：应用人工决策

**负责人**: 项目负责人

**任务**: 使用 apply 脚本合并人工决策

```bash
# 应用人工审查结果到 Stage 1
python scripts/apply_human_review.py \
  --stage1-jsonl qc_full_output/*/stage1_results.jsonl \
  --human-review all_human_reviews.jsonl \
  --output-dir final_qc_output

# 输出
# final_qc_output/
# ├── stage1_results_merged.jsonl  (合并了人工决策)
# ├── manifests/
# │   ├── pass.txt  (最终通过，用于 Stage 3)
# │   ├── fail.txt  (最终失败)
# │   └── human_reviewed.txt  (经过人工审查的样本)
# └── summary_report.html
```

**验证**:
```bash
# 统计最终结果
echo "Final Pass samples: $(wc -l < final_qc_output/manifests/pass.txt)"
echo "Final Fail samples: $(wc -l < final_qc_output/manifests/fail.txt)"
echo "Human Reviewed: $(wc -l < final_qc_output/manifests/human_reviewed.txt)"

# 验证数据一致性
python scripts/verify_final_consistency.py \
  --merged final_qc_output/stage1_results_merged.jsonl \
  --manifests final_qc_output/manifests/
```

**检查点**:
- [ ] 人工决策成功应用
- [ ] pass.txt 生成
- [ ] 数据一致性验证通过

---

#### Day 3-5：Stage 3 执行

**负责人**: 项目负责人

**任务**: 运行 Stage 3（GPU 评估）

```bash
# 使用最终 pass.txt 作为输入
python scripts/run_stage3_cmcc.py \
  --pass-manifest final_qc_output/manifests/pass.txt \
  --tar-root /root/work/filestorage/.../repair_done \
  --output-dir stage3_output \
  --batch-size 32 \
  --n-gpus 8
```

**预计时间**: 根据 pass 样本数量，可能需要 2-3 天

**并行任务**:
- 编写项目总结报告
- 记录阈值优化建议
- 整理测试人员反馈

**检查点**:
- [ ] Stage 3 开始执行
- [ ] 项目文档完善

---

## 风险和应对措施

### 风险 1：测试人员效率差异大

**现象**: 有人 1 天完成 500 个，有人只完成 200 个

**影响**: 总样本量可能低于目标

**应对措施**:
1. **动态调整**:
   - Batch 2 根据 Batch 1 进度动态分配
   - 快的人多分配，慢的人减少

2. **降低目标**:
   - 如果整体进度慢，降低目标到 5,000 也可接受
   - 5,000 样本已经有足够统计显著性

3. **相互帮助**:
   - 快的人帮慢的人审查
   - 也可作为交叉验证

**预防**:
- Day 1-2 密切关注进度
- 早发现早调整

---

### 风险 2：标准不一致

**现象**: 同样的样本，不同人判断差异大

**影响**: 降低人工审查的可信度

**应对措施**:
1. **黄金样本培训**:
   - 必须达到 80% 一致性才开始
   - 未达标重新培训

2. **每日对齐**:
   - 站会讨论困难样本
   - 达成共识后继续

3. **抽查纠偏**:
   - 项目负责人抽查 5%
   - 发现偏差及时反馈

4. **事后分析**:
   - 如果有重叠样本，计算一致性
   - 识别需要额外培训的人

**预防**:
- 操作手册详细清晰
- 边界情况有明确指导

---

### 风险 3：Stage 1+2 运行时间超预期

**现象**: 预计 2 天，实际需要 3-4 天

**影响**: 压缩后续时间

**应对措施**:
1. **Week 1 有缓冲**:
   - Day 1-2 运行 Stage 1+2
   - Day 3-5 分析和生成 Batch 1
   - 如果 Stage 1+2 延迟到 Day 3，仍有时间

2. **先处理完成的 group**:
   - 不必等所有 group 完成
   - 先生成已完成 group 的 Batch 1
   - 边运行边采样

3. **压缩 Week 2**:
   - 最坏情况：Week 2 压缩到 4 天
   - 降低样本量到 5,000

**预防**:
- 优化 Stage 1+2 性能（增加 workers）
- 并行处理多个 group

---

### 风险 4：人工审查发现大量问题

**现象**: Disagreement > 30%，说明阈值严重偏离

**影响**: 需要决策是否重跑 Stage 1+2

**应对措施**:
1. **本次不重跑**（已决策）:
   - 人工审查结果直接应用
   - 数据质量已由人工保证

2. **记录详细建议**:
   - 哪些阈值需要调整
   - 调整幅度建议
   - 影响的样本数量

3. **下次迭代优化**:
   - 如果有新数据，应用优化后的阈值
   - 验证改进效果

**评估标准**:
- Disagreement < 20%: 自动化效果好，不需调整
- Disagreement 20-30%: 记录建议，本次不调
- Disagreement > 30%: 讨论是否重跑（权衡时间）

---

### 风险 5：测试人员中途退出

**现象**: 某人因故无法继续

**影响**: 工作量不平衡

**应对措施**:
1. **重新分配**:
   - 将该人的剩余样本分给其他人
   - 可能需要延长 1-2 天

2. **降低目标**:
   - 如果来不及重新分配，降低总样本量

3. **替补人员**:
   - 提前确定 1-2 名备选测试人员
   - 快速培训后接手

**预防**:
- 提前确认所有人的时间
- 分批分发，风险分散

---

## 成功指标

### 量化指标

1. **样本量**:
   - 目标：6,000-7,000 样本
   - 底线：5,000 样本
   - 测量：`wc -l all_human_reviews.jsonl`

2. **完成率**:
   - 目标：审查完成率 > 90%（留空率 < 10%）
   - 测量：填写 human_verdict 的样本 / 总样本

3. **一致性**:
   - 目标：测试人员间一致性 > 75%（如果有重叠样本）
   - 测量：Kappa 系数或简单一致性百分比

4. **覆盖率**:
   - 目标：每个 group 至少 500 样本
   - 测量：按 group 统计审查数量

5. **时间**:
   - 目标：Week 3 如期开始 Stage 3
   - 测量：实际开始日期

### 质量指标

6. **标准对齐**:
   - 黄金样本一致性 ≥ 80%

7. **数据有效性**:
   - 必填字段完整率 > 95%
   - Primary_issue 枚举值正确率 100%

8. **洞察价值**:
   - 识别出至少 3 个阈值优化建议
   - Disagreement 报告清晰可执行

---

## 附录

### 附录 A：脚本清单

**已有脚本**:
1. `scripts/run_qc.py` - Stage 1+2 执行
2. `scripts/export_for_review.py` - 采样和导出
3. `scripts/import_review_results.py` - 导入和验证
4. `scripts/apply_human_review.py` - 应用人工决策

**需要新增脚本**:
5. `scripts/calculate_sampling_quotas.py` - 计算配额
6. `scripts/split_batch_for_persons.py` - 分配给测试人员
7. `scripts/generate_consolidated_report.py` - 汇总报告
8. `scripts/verify_final_consistency.py` - 最终验证

### 附录 B：文件格式参考

**review_list.csv 格式**:
```csv
sample_id,group,tar_path,auto_verdict,flag_reasons,n_jumps,caption_len,black_frame_ratio,scene_cuts,caption_text,video_path
```

**decisions_template.csv 格式**:
```csv
sample_id,auto_verdict,human_verdict,video_quality,trajectory_quality,primary_issue,notes
```

**human_review_results.jsonl 格式**:
```json
{
  "sample_id": "xxx",
  "auto_verdict": "fail",
  "human_verdict": "pass",
  "video_quality": "acceptable",
  "trajectory_quality": "good",
  "primary_issue": "trajectory_minor",
  "notes": "Small jump but acceptable",
  "reviewer": "person1",
  "review_date": "2026-07-15"
}
```

### 附录 C：测试人员联系方式

*(项目负责人填写)*

| 姓名 | 联系方式 | 分配 | 备注 |
|------|---------|------|------|
| Person 1 | | Batch 1 | |
| Person 2 | | Batch 1 | |
| Person 3 | | Batch 1 | |
| Person 4 | | Batch 2 | |
| Person 5 | | Batch 2 | |

### 附录 D：重要日期

| 里程碑 | 日期 | 负责人 |
|--------|------|--------|
| Stage 1+2 完成 | Week 1 Day 2 | 项目负责人 |
| 黄金样本准备 | Week 1 Day 3 | 项目负责人 |
| Batch 1 分发 | Week 1 Day 4 | 项目负责人 |
| Batch 1 截止 | Week 2 Day 3 17:00 | 测试人员 |
| Batch 2 截止 | Week 2 Day 5 17:00 | 测试人员 |
| 决策会议 | Week 3 Day 2 10:00 | 项目负责人 + 管理层 |
| Stage 3 开始 | Week 3 Day 2 下午 | 项目负责人 |

---

## 文档版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0 | 2026-07-08 | David Wang | 初始版本 |

---

**文档结束**



