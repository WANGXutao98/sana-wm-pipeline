# 本次对话创建的分析文档清单

**对话日期**: 2026-08-14  
**主题**: SpatialVID冒烟测试 - 轨迹偏差问题深度调查

---

## 核心问题记录

### 1. PROBLEM_RECORD_TRAJECTORY_DEVIATION.md ⭐⭐⭐⭐⭐
**最重要的文档** - 供下一个对话快速了解情况

**内容**:
- 问题描述：轨迹偏差3.88-9.56x
- 已尝试的修复（两次，都失败）
- 核心发现（阈值参数完全无效）
- 下一步行动建议
- 技术细节和代码位置

---

## 详细调查文档

### 2. THRESHOLD_INEFFECTIVE_FINDING.md
**阈值测试失败分析**

**内容**:
- 三轮测试对比（2.4 → 10.0 → 100.0）
- 结论：filter_thresh和keyframe_thresh不是控制因素
- 修改记录（用于回退）
- 下一步建议：审查Phase 2逻辑

### 3. OFFICIAL_VIPE_INVESTIGATION.md
**官方sana-wm-data-clean代码对比**

**内容**:
- 官方配置与我们完全相同
- vipe_patches不涉及keyframe逻辑
- SpatialVID的精确4帧间隔无法从代码解释
- 可能的原因：版本差异、后处理、视频特征

### 4. CRITICAL_KEYFRAME_DENSITY_FINDING.md
**Keyframe密度问题发现**

**内容**:
- VIPE输出连续keyframes（indices = [0,1,2,...]）
- 参考数据是稀疏keyframes（indices = [0,4,8,...]）
- Keyframe密度差异导致BA的scale漂移
- MotionFilter机制分析

### 5. THREE_WAY_POSE_COMPARISON.md
**三方pose对比分析**

**内容**:
- 官方标注 vs VIPE标注 vs 我们的输出
- 确认应该以VIPE标注为参考（不是官方标注）
- 官方标注与VIPE标注差异很大（0.13-1.22x）
- 详细的数据表格和分析

### 6. FILTER_THRESH_ANALYSIS.md
**filter_thresh: 10.0测试分析**

**内容**:
- 第一次阈值修改测试（10.0）失败
- 可能的原因分析
- 建议尝试更大的值或修改slam/default.yaml

---

## 早期分析文档（已被后续发现推翻）

### 7. FINAL_ROOT_CAUSE_AND_SOLUTION.md
**最初的根因分析**（后被证明不完整）

**内容**:
- 识别了MotionFilter机制
- 提出增大filter_thresh的解决方案
- ⚠️ 后续测试证明此方案无效

### 8. STAGE11_FAILED_FIX_ANALYSIS.md
**第一帧归一化修复失败分析**

**内容**:
- 移除第一帧归一化只改善了9%
- 深度融合验证正确
- Scale传递验证正确
- 结论：不是主要问题

### 9. FIRST_FRAME_NORMALIZATION_DECISION.md
**第一帧归一化的决策分析**

**内容**:
- 论文App. D.3的正确理解
- 为什么本地实现添加了归一化（误读论文）
- 官方实现不做归一化
- 决定移除归一化

---

## 其他参考文档

### 10. STAGE11_INVESTIGATION_REPORT.md
**阶段11调查报告**（P0和P1阶段）

**内容**:
- 验证融合深度的物理单位
- 调查VIPE深度处理
- 初步代码对比

### 11. SANA_WM_DATA_CLEAN_ARCHITECTURE.md
**官方代码架构分析**

**内容**:
- sana-wm-data-clean的目录结构
- 关键模块说明
- 数据流程

---

## 文档使用建议

### 给下一个Claude对话

**必读**（按顺序）:
1. ⭐ `PROBLEM_RECORD_TRAJECTORY_DEVIATION.md` - 了解当前状态
2. ⭐ `THRESHOLD_INEFFECTIVE_FINDING.md` - 了解最新发现
3. `THREE_WAY_POSE_COMPARISON.md` - 了解数据对比

**选读**（如需深入）:
4. `OFFICIAL_VIPE_INVESTIGATION.md` - 官方代码对比
5. `CRITICAL_KEYFRAME_DENSITY_FINDING.md` - Keyframe问题分析

**可忽略**（已被推翻或过时）:
- `FINAL_ROOT_CAUSE_AND_SOLUTION.md` - 方案已证明无效
- 其他早期文档 - 仅作历史参考

---

## 快速上下文

**30秒版本**:
- 问题：轨迹偏大4-10x
- 原因：VIPE产生连续keyframes（32-37个）而不是稀疏（13-14个）
- 尝试：修改filter_thresh → 完全无效
- 下一步：审查VIPE Phase 2代码

**2分钟版本**:
阅读 `PROBLEM_RECORD_TRAJECTORY_DEVIATION.md` 的"核心发现"和"下一步行动"部分

---

## 代码修改状态

**当前状态**: ✅ 所有修改已回退，代码恢复到原始状态

**修改过的文件**（已回退）:
1. `third_party/vipe/configs/slam/default.yaml`
2. `third_party/vipe/configs/pipeline/vipe_sanawm.yaml`
3. `src/sana_wm_pipeline/stage02_pose/mode_default.py`（第一帧归一化已移除，此修改保留）

**验证**:
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git status
# 应该只显示第一帧归一化的修改
```
