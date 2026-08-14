# SpatialVID冒烟测试 - 轨迹偏差问题调查记录

**调查日期**: 2026-08-14  
**状态**: ❌ 未解决 - 需要深入VIPE源码分析  
**问题**: 轨迹长度比VIPE参考大3.88-9.56x

---

## 问题描述

运行SpatialVID冒烟测试后，我们的pose输出与SpatialVID参考数据对比：
- 样本1: 轨迹偏大 3.88x
- 样本2: 轨迹偏大 7.64x
- 样本3: 轨迹偏大 9.56x

根因：VIPE产生**连续keyframes**（32-37个，每1帧），而参考是**稀疏keyframes**（13-14个，每4帧）。

---

## 已尝试的修复

### 尝试1: 移除第一帧归一化 ❌

**修改**: `src/sana_wm_pipeline/stage02_pose/mode_default.py:228-231`

**结果**: 轻微改善（9%），但问题仍存在

**回退**: 已回退

---

### 尝试2: 增大filter_thresh阈值 ❌

**修改**: 
- `third_party/vipe/configs/slam/default.yaml`
  - `filter_thresh: 2.4 → 10.0 → 100.0`
  - `keyframe_thresh: 4.0 → 100.0`

**结果**: **完全无效**，keyframe数量和间隔没有任何变化

**结论**: `filter_thresh` 和 `keyframe_thresh` 不是控制keyframe选择的主要因素

**回退**: ✅ 已回退到默认值

---

## 核心发现

### 1. 官方代码与我们完全相同

对比 `sana-wm-data-clean`:
- ✅ 配置文件相同 (`vipe_patches/sanawm_pipeline.yaml`)
- ✅ VIPE调用相同 (`vipe_cli.py`)
- ✅ patches不涉及keyframe逻辑
- ❓ 但SpatialVID的keyframes是**精确4帧间隔**

### 2. 阈值参数完全无效

三轮测试证明：

| 测试 | filter_thresh | keyframe_thresh | Keyframes | 结果 |
|------|--------------|----------------|-----------|------|
| 基线 | 2.4 | 4.0 | 32-37连续 | 3.88-9.56x |
| 测试1 | 10.0 | 4.0 | 32-37连续 | 无变化 |
| 测试2 | 100.0 | 100.0 | 32-37连续 | 无变化 |

**即使阈值增大40倍，keyframe选择完全不受影响。**

### 3. VIPE必有其他机制

**可能的原因**:
1. **Phase 2强制逻辑**: VIPE的第二遍SLAM可能无条件添加所有帧
2. **版本差异**: SpatialVID使用的VIPE版本可能不同
3. **配置加载bug**: 阈值参数可能根本没有被读取
4. **环境变量覆盖**: 可能有隐藏的环境变量控制

---

## 下一步行动（待后续对话）

### 优先级1: 审查VIPE Phase 2逻辑 ⭐⭐⭐⭐⭐

**代码位置**: `third_party/vipe/vipe/slam/system.py:300-310`

```python
# SLAM Pass (2/2)
for frame_idx, frame_data_list in enumerate(...):
    self._add_keyframe(frame_idx, ..., phase=2)  # ← 没有条件判断？
```

**怀疑**: Phase 2可能无条件添加所有帧，覆盖Phase 1的MotionFilter结果。

**验证方法**:
1. 添加debug日志到 `_add_keyframe`
2. 检查phase=2时是否有条件判断
3. 对比phase 1和phase 2的keyframe列表

### 优先级2: 添加debug日志追踪 ⭐⭐⭐⭐

**目标**: 确认MotionFilter是否真的被调用和生效

**修改点**:
```python
# vipe/slam/components/motion_filter.py:138-147
def check(self, ...):
    print(f"[DEBUG] thresh={self.thresh}, dense_score={dense_motion_score}, sparse_score={sparse_motion_score}")
    if dense_motion_score > self.thresh or sparse_motion_score > self.thresh * 2:
        print(f"[DEBUG] KEYFRAME ADDED")
        return True
    else:
        print(f"[DEBUG] FRAME SKIPPED")
        return False
```

### 优先级3: 考虑替代方案 ⭐⭐⭐

**如果无法修复VIPE的keyframe选择**:

**方案A: 后处理稀疏化**
- 读取VIPE的连续keyframes
- 每4帧抽取1个
- 重新插值到全帧率

**方案B: 预处理视频降帧率**
- 输入前每4帧抽1帧
- VIPE处理稀疏视频
- 输出后插值回原帧率

**方案C: 接受偏差**
- 评估对下游训练的实际影响
- 如果模型能学习这个偏差，可能可接受

---

## 相关文档

以下文档记录了完整的调查过程：

1. **THRESHOLD_INEFFECTIVE_FINDING.md** - 阈值无效的发现和分析
2. **OFFICIAL_VIPE_INVESTIGATION.md** - 官方代码对比分析
3. **FILTER_THRESH_ANALYSIS.md** - filter_thresh测试分析
4. **CRITICAL_KEYFRAME_DENSITY_FINDING.md** - keyframe密度问题发现
5. **THREE_WAY_POSE_COMPARISON.md** - 三方pose对比（官方标注 vs VIPE标注 vs 我们）
6. **FINAL_ROOT_CAUSE_AND_SOLUTION.md** - 最初的根因分析（后被推翻）
7. **STAGE11_FAILED_FIX_ANALYSIS.md** - 第一帧归一化修复失败分析
8. **FIRST_FRAME_NORMALIZATION_DECISION.md** - 第一帧归一化的决策分析

---

## 技术细节

### 关键代码位置

**MotionFilter**: `third_party/vipe/vipe/slam/components/motion_filter.py`
- `check()` 方法决定是否添加keyframe
- 使用 `self.thresh` (来自 `filter_thresh`)

**SLAM System**: `third_party/vipe/vipe/slam/system.py`
- Phase 1 (line 270-275): 使用MotionFilter
- Phase 2 (line 300-310): 可能的问题所在

**配置文件**:
- `third_party/vipe/configs/slam/default.yaml` - SLAM默认配置
- `third_party/vipe/configs/pipeline/vipe_sanawm.yaml` - 我们的pipeline配置

### 数据对比

**SpatialVID参考数据** (`raw_samples/*.camera.npz`):
- `vipe_sparse_indices`: [0, 4, 8, 12, ...] - 精确4帧间隔
- `vipe_sparse_c2w`: 13-14个keyframes

**我们的输出** (`vipe_work_default/pose/normalized.npz`):
- `inds`: [0, 1, 2, 3, ...] - 连续
- `data`: 32-37个keyframes

---

## 当前状态

**问题**: ❌ 未解决  
**阻塞**: 需要深入VIPE源码，添加debug日志  
**估计工作量**: 2-4小时深度调试  
**优先级**: HIGH（影响pose质量和下游训练）  

**建议**: 
- 如果时间充裕，继续调查Phase 2逻辑
- 如果时间紧张，考虑替代方案（后处理稀疏化）
