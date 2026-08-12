# 人工审查操作手册

## 概述

本手册面向人工审查人员，指导如何审查SANA-WM QC系统Stage 1+2筛选出的视频样本。

---

## 审查目标

在Stage 1+2自动筛选之后、Stage 3 GPU评估之前，人工审查1,000-3,000个边界样本，判断是否应该通过质检进入下一阶段。

**审查重点：**
- 自动筛选标记为失败但接近阈值的样本（可能是误判）
- 自动筛选标记为通过的随机样本（验证漏检）
- 存在多个问题标记的复杂样本

---

## 环境准备

### 所需工具

1. **VLC Media Player** - 视频播放
   - 下载：https://www.videolan.org/
   - 支持快捷键：空格（播放/暂停）、方向键（快进/快退）

2. **Excel / LibreOffice Calc / Google Sheets** - 填写审查结果
   - 推荐Excel或LibreOffice以保证CSV兼容性

3. **文本编辑器（可选）** - 查看review_list.csv的详细指标

### 获取审查包

技术人员会提供一个审查包目录，包含：
```
human_review_batch1/
├── review_list.csv           # 完整样本信息（只读参考）
├── decisions_template.csv    # 待填写的审查表格
├── videos/                   # 提取的视频文件
│   ├── sample_001.mp4
│   ├── sample_002.mp4
│   └── ...
└── sampling_report.txt       # 采样统计信息
```

---

## 审查流程

### Step 1: 打开文件

1. 用Excel打开 `decisions_template.csv`
2. 用文本编辑器或Excel打开 `review_list.csv`（作为参考）
3. 在VLC中打开 `videos/` 目录

### Step 2: 逐个审查样本

对于 `decisions_template.csv` 中的每一行：

#### 2.1 找到对应视频

- 根据 `sample_id` 列（例如 "DL3DV_001"）
- 在 `videos/` 目录中找到对应的 `.mp4` 文件
- 用VLC播放

#### 2.2 查看自动判断

- `auto_verdict` 列：自动系统的判断（pass/fail）
- 在 `review_list.csv` 中查看：
  - `flag_reasons`: 失败原因（如 "trajectory_jump|black_frames"）
  - `n_jumps`: 轨迹跳变次数
  - `caption_text`: 字幕文本
  - 其他指标：`black_frame_ratio`, `scene_cuts` 等

#### 2.3 观看视频（10-20秒）

**关注点：**
- **轨迹质量**：相机运动是否平滑？是否有突然跳跃？
- **视频质量**：画面是否清晰？是否有模糊、卡顿、伪影？
- **内容连贯性**：是否有突兀的场景切换？是否有黑屏？
- **字幕匹配**：字幕描述是否与视频内容一致？

#### 2.4 填写审查结果

在 `decisions_template.csv` 中填写以下列：

**必填列：**

1. **human_verdict**（必填）
   - `pass`: 该样本应该通过质检
   - `fail`: 该样本应该被淘汰
   - 留空：使用自动判断（不确定时可留空）

2. **primary_issue**（必填）
   - 从以下选项中选择主要问题：
   
   ```
   trajectory_minor_jump    - 小的轨迹跳变（可接受范围内）
   trajectory_major_jump    - 大的轨迹跳变（明显不连续）
   video_blurry             - 视频模糊
   video_artifacts          - 视频有伪影/故障
   caption_mismatch         - 字幕与内容不匹配
   caption_too_vague        - 字幕过于笼统
   black_frames             - 包含黑屏帧
   scene_cut_abrupt         - 突兀的场景切换
   multiple_issues          - 多个问题同时存在
   no_issue                 - 没有发现问题
   other                    - 其他（请在notes中说明）
   ```

**可选列：**

3. **video_quality**（可选）
   - `good`: 清晰、流畅、无伪影
   - `acceptable`: 有小问题但可用
   - `poor`: 模糊、卡顿、严重伪影

4. **trajectory_quality**（可选）
   - `good`: 平滑、合理
   - `acceptable`: 有小跳变但可接受
   - `poor`: 大跳变、不连续

5. **notes**（可选）
   - 额外说明，特别是选择 `other` 或不寻常情况时

### Step 3: 保存文件

- 定期保存 `decisions_template.csv`（建议每审查50个样本保存一次）
- 完成后另存为 `decisions_filled.csv`

---

## 判断标准

### 何时选择 PASS

即使自动系统标记为 `fail`，以下情况应该选择 `pass`：

1. **轨迹跳变很小**
   - 自动系统可能对阈值过于严格
   - 小的相机位置调整（< 0.5米）是可接受的
   - 判断：如果人眼观看感觉平滑，即使有小跳变也可通过

2. **字幕略短但足够描述**
   - 自动系统设置最小长度40字符
   - 如果字幕虽短但描述清晰（如"Person walking in a room"），可通过

3. **黑屏很少**
   - 1-2帧黑屏通常是编码问题，不影响整体质量

### 何时选择 FAIL

即使自动系统标记为 `pass`，以下情况应该选择 `fail`：

1. **视频模糊严重**
   - 自动系统目前无法检测模糊
   - 如果画面大部分时间模糊到无法识别物体，应淘汰

2. **字幕与内容明显不符**
   - 例如：字幕说"indoor bedroom"但视频是"outdoor street"

3. **场景内容不适合训练**
   - 完全静止的画面（无相机运动）
   - 纯文字屏幕/GUI界面
   - 严重的畸变或失真

### 何时留空（使用自动判断）

- 不确定时可以跳过（留空 `human_verdict`）
- 系统会自动使用 `auto_verdict` 的值
- 建议：如果观看15秒后仍无法判断，留空并继续下一个

---

## 审查技巧

### 1. 批量模式

**推荐流程：**
- 在VLC中将所有视频加入播放列表
- 逐个播放（VLC快捷键：N = 下一个，P = 上一个）
- 在Excel中同步填写

### 2. 加速播放

对于长视频（>30秒）：
- VLC快捷键：`]` 加速，`[` 减速
- 建议1.5-2倍速观看，重点关注轨迹和场景切换

### 3. 分批审查

建议分时段审查：
- 每次审查100-200个样本（约1-2小时）
- 休息10-15分钟后继续
- 避免疲劳导致判断标准飘移

### 4. 双人交叉验证（可选）

对于不确定的样本：
- 标记在notes列
- 两位审查人员独立判断
- 讨论后达成一致

---

## 常见问题

### Q1: 自动系统标记了3次跳变，但我只看到1次？

A: 可能的原因：
- 自动系统对小位移变化敏感，人眼可能忽略
- 如果你认为整体平滑，可以标记为 `pass` + `trajectory_minor_jump`

### Q2: 视频无法播放或损坏？

A: 
- 在 `review_list.csv` 中检查 `video_path` 是否为 "MISSING"
- 如果是，基于其他指标判断，或留空使用自动判断
- 记录在notes中："video corrupted, judged by metrics"

### Q3: 我对某些样本完全无法判断？

A: 
- 留空 `human_verdict`，系统会使用 `auto_verdict`
- 不强制要求100%完成率，90%以上即可

### Q4: 发现新的问题类型，不在枚举列表中？

A: 
- `primary_issue` 选择 `other`
- 在 `notes` 中详细描述问题
- 通知技术人员，可能需要更新自动检测逻辑

---

## 质量保证

### 自查清单

完成后检查：
- [ ] 至少90%的样本填写了 `human_verdict`
- [ ] 所有填写了 `human_verdict` 的行都填写了 `primary_issue`
- [ ] `human_verdict` 只包含 `pass` 或 `fail`（或留空）
- [ ] 文件另存为 `decisions_filled.csv`
- [ ] CSV文件编码为UTF-8（Excel默认可能是GBK，需检查）

### 提交前验证

技术人员会运行验证脚本：
```bash
python scripts/import_review_results.py \
  --review-list human_review_batch1/review_list.csv \
  --decisions human_review_batch1/decisions_filled.csv \
  --output-dir human_review_batch1/analysis
```

如果有错误，会返回具体行号和问题。

---

## 工作量估算

**单个样本：** 1-2分钟
- 10秒观看视频
- 5秒参考指标
- 30秒填写表格

**1000个样本：** 约20-30小时
- 建议2人并行，每人500个
- 分5-6个工作时段完成
- 总时长：2个工作日

---

## 示例

### 示例1: 自动fail → 人工pass

**review_list.csv:**
```
sample_id: DL3DV_001
auto_verdict: fail
flag_reasons: trajectory_jump
n_jumps: 3
caption_text: Person walking through a living room
```

**观察：**
- 视频播放平滑，只有2个很小的位移
- 画面清晰，内容合理

**填写 decisions_filled.csv:**
```
sample_id: DL3DV_001
auto_verdict: fail
human_verdict: pass
video_quality: good
trajectory_quality: acceptable
primary_issue: trajectory_minor_jump
notes: Small jumps but overall smooth trajectory
```

### 示例2: 自动pass → 人工fail

**review_list.csv:**
```
sample_id: RealEstate_050
auto_verdict: pass
n_jumps: 1
caption_text: Camera moving through modern apartment
```

**观察：**
- 视频严重模糊，无法识别房间细节
- 自动系统未检测到模糊

**填写 decisions_filled.csv:**
```
sample_id: RealEstate_050
auto_verdict: pass
human_verdict: fail
video_quality: poor
trajectory_quality: good
primary_issue: video_blurry
notes: Severely blurred, unusable for training
```

### 示例3: 不确定 → 留空

**review_list.csv:**
```
sample_id: Sekai_010
auto_verdict: fail
flag_reasons: multiple_issues
n_jumps: 5
scene_cuts: 2
```

**观察：**
- 有一些问题，但难以判断严重程度
- 不确定是否应该通过

**填写 decisions_filled.csv:**
```
sample_id: Sekai_010
auto_verdict: fail
human_verdict: 
video_quality: 
trajectory_quality: 
primary_issue: 
notes: Uncertain, using auto verdict
```

---

## 联系方式

**技术支持：**
- 审查过程中遇到问题，联系：[技术负责人姓名/邮箱]
- 文件格式问题、工具使用问题

**数据质量讨论：**
- 判断标准不明确
- 发现系统性问题
- 建议改进自动检测逻辑

---

## 附录：Primary Issue 完整列表

| 代码 | 中文说明 | 使用场景 |
|------|---------|---------|
| trajectory_minor_jump | 轻微轨迹跳变 | 有小的位移但整体平滑 |
| trajectory_major_jump | 严重轨迹跳变 | 明显的不连续、传送 |
| video_blurry | 视频模糊 | 画面不清晰，无法识别细节 |
| video_artifacts | 视频伪影 | 编码错误、花屏、色块 |
| caption_mismatch | 字幕不匹配 | 描述与视频内容不符 |
| caption_too_vague | 字幕过于笼统 | 如"A video"这种无效描述 |
| black_frames | 黑屏帧 | 包含大量黑色或纯色帧 |
| scene_cut_abrupt | 突兀场景切换 | 不同场景拼接在一起 |
| multiple_issues | 多个问题 | 同时存在2个以上明显问题 |
| no_issue | 无问题 | 审查后认为完全正常 |
| other | 其他 | 以上都不适用，需在notes说明 |

---

## 变更历史

| 日期 | 版本 | 变更说明 |
|------|------|---------|
| 2026-07-07 | 1.0 | 初始版本 |
