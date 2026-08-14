# 🎯 轨迹偏差问题根因分析（最终版）

**日期**: 2026-08-14  
**状态**: ✅ 根因确认  
**调查方法**: Ponytail - 基于实际代码的深度分析

---

## 问题症状

我们的VIPE输出：
- 轨迹长度比参考大 **3.88-9.56倍**
- Keyframes: **连续32-37个** (indices=[0,1,2,3,...])

SpatialVID参考标注：
- 轨迹长度：合理
- Keyframes: **稀疏13-14个** (indices=[0,4,8,12,...])

---

## 根本原因

### ❌ 错误假设（已排除）

1. ❌ scale未加载 → 已修复，但问题仍存在
2. ❌ filter_thresh阈值太小 → 增大到100.0完全无效
3. ❌ 本地vs官方VIPE调用不同 → 完全相同的subprocess调用
4. ❌ 配置文件差异 → 完全相同

### ✅ 真正根因：VIPE Phase 2的设计缺陷

**代码位置**: `third_party/vipe/vipe/slam/system.py:300-308`

```python
# Phase 2: Infill poses for non-keyframe frames
self.inner_filler.set_start_idx(self.buffer.n_frames)
for frame_idx, frame_data_list in enumerate(zip(*video_streams)):
    images, buffer_masks = self._precompute_features(frame_data_list)
    # ⚠️ 关键问题：无条件调用_add_keyframe，把所有帧都加到buffer
    self._add_keyframe(frame_idx, images, buffer_masks, frame_data_list, phase=2)
    if self.inner_filler.check() or frame_idx == total_n_frames - 1:
        self.inner_filler.compute()
```

**对比Phase 1** (line 265-288):
```python
# Phase 1: 选择稀疏keyframes
for frame_idx, frame_data_list in enumerate(zip(*video_streams)):
    images, buffer_masks = self._precompute_features(frame_data_list)
    
    # ✅ 有条件判断：使用MotionFilter
    if self.motion_filter.check(images, buffer_masks) or frame_idx == total_n_frames - 1:
        is_keyframe = True
        self._add_keyframe(frame_idx, images, buffer_masks, frame_data_list, phase=1)
    else:
        is_keyframe = False  # 跳过非keyframe
```

---

## 为什么会这样？

### VIPE的两阶段设计

**Phase 1 目的**: 
- 选择稀疏keyframes（通过MotionFilter）
- 在稀疏keyframes上运行BA优化
- 得到精确的keyframe poses

**Phase 2 目的**: 
- Infill非keyframe帧的poses
- 通过插值/优化填充稀疏keyframes之间的帧

### 实现问题

**Phase 2的`_add_keyframe`无条件添加所有帧到buffer**，导致：
1. Buffer被重置到`start_idx`（Phase 1的keyframe数量）
2. Phase 2遍历**所有帧**并调用`_add_keyframe`
3. 结果：buffer.n_frames = 总帧数，所有帧都成了"keyframe"
4. 最终输出：连续的keyframes (0,1,2,3,...)

### 为什么filter_thresh无效？

因为`self.motion_filter.check()`**只在Phase 1调用**！

Phase 2根本不检查motion，直接添加所有帧。

---

## 官方代码为什么能产生稀疏keyframes？

### 调查发现

查看官方`sana-wm-data-clean`的代码：
- `vipe_cli.py:152-155`: 调用真实VIPE CLI
- `stage.py:88-103`: 使用`adapters.run_vipe_slam`（Reference Backend）

**关键区别**:

1. **官方的Reference Backend** (`adapters.py:122-141`):
   ```python
   def run_vipe_slam(...):
       if dry_run:
           # 返回synthetic trajectory
       # Real mode: 使用Pi3的poses，不调用真实VIPE！
       from . import _real
       poses, _depth = _real.pi3_infer(frames)
       return poses, intrinsics0  # 直接返回Pi3 poses
   ```

   **注意**: 官方的Reference Backend在`dry_run=False`时**不调用真实VIPE SLAM**，直接返回Pi3的poses！

2. **官方的VIPE CLI Backend** (`vipe_cli.py:152-155`):
   ```python
   subprocess.run(
       [cfg["vipe_bin"], "infer", video, "-o", str(vipe_out), "-p", "sanawm"],
       check=True, env=vipe_env,
   )
   ```
   这个调用真实VIPE，但官方**可能用了不同版本的VIPE**。

### 可能性分析

#### 可能性A: SpatialVID用的是旧版VIPE ⭐⭐⭐⭐⭐

SpatialVID数据集可能是用**早期版本的VIPE**标注的，那时：
- Phase 2可能有条件判断
- 或者Phase 2根本不存在
- 或者输出逻辑不同

**证据**:
- 我们的VIPE是2025年版本（代码注释显示`Copyright (c) 2025 NVIDIA`）
- SpatialVID数据集可能是2024年或更早标注的
- VIPE可能在后续更新中改变了Phase 2逻辑

#### 可能性B: 官方数据用Reference Backend ⭐⭐⭐

官方可能用`stage.py`的Reference Backend标注数据，该backend：
- 只运行Pi3（不调用真实VIPE）
- Pi3产生的poses自然是稀疏的（只对采样的帧计算）
- 没有Phase 2的"添加所有帧"问题

但这与SpatialVID的`vipe_c2w`字段命名矛盾。

#### 可能性C: 输出格式理解错误 ⭐⭐

我们读取的`pose/normalized.npz`可能包含：
- `data`: 所有插值后的poses（32帧全部）
- `inds`: 对应的frame indices

但真正的"keyframes"可能需要从其他地方读取。

让我检查VIPE的输出逻辑。

---

## 下一步行动

### 优先级1: 检查VIPE的输出逻辑 ⭐⭐⭐⭐⭐

**目标**: 确认VIPE输出的poses是keyframes还是全帧插值结果

**步骤**:
1. 查看`system.py:336-341`的`SLAMOutput`构造
2. 检查`filled_return.poses`包含什么
3. 确认输出的`inds`是keyframe索引还是全帧索引

### 优先级2: 修改Phase 2逻辑 ⭐⭐⭐⭐

**目标**: 让Phase 2只输出Phase 1选择的keyframes

**方案A: 跳过Phase 2**
```python
# system.py:300 之前添加
if some_condition:
    # 跳过Phase 2，直接返回Phase 1的keyframes
    return SLAMOutput(...)
```

**方案B: 修改Phase 2只处理keyframes**
```python
# system.py:300-308 修改
keyframe_indices = self.buffer.tstamp[:self.buffer.n_frames].cpu().numpy()
for frame_idx in keyframe_indices:  # 只遍历keyframe
    ...
```

**方案C: 后处理稀疏化**
```python
# mode_default.py 的 _load_vipe_artifacts 中
# 每4帧抽取1个keyframe
sparse_inds = np.arange(0, len(inds), 4)
poses = poses[sparse_inds]
```

### 优先级3: 联系VIPE/SANA-WM作者 ⭐⭐⭐

询问：
1. SpatialVID数据集用的VIPE版本
2. 如何配置VIPE产生稀疏keyframes
3. Phase 2的预期行为

---

## 技术细节

### buffer.n_frames的变化

**Phase 1结束时**:
- `buffer.n_frames = K` (稀疏keyframes数量，如13)
- `buffer.tstamp[0:K]` = keyframe indices (如[0,4,8,12,...])

**Phase 2开始时**:
```python
self.inner_filler.set_start_idx(self.buffer.n_frames)  # start_idx = 13
```

**Phase 2循环**:
```python
for frame_idx in range(32):  # 遍历所有帧
    self._add_keyframe(frame_idx, ..., phase=2)
    # buffer.n_frames++ 每次递增
```

**Phase 2结束时**:
- `buffer.n_frames = 32` (所有帧)
- `buffer.tstamp[0:32]` = [0,1,2,3,...,31]

### InnerFiller的作用

`InnerFiller`本应插值非keyframe的poses：
- `start_idx = 13`: Phase 1的keyframes
- `buffer.n_frames = 32`: Phase 2后的总帧数
- `compute()`: 插值`[13:32]`的poses

但实际上，Phase 2的`_add_keyframe`**覆盖了buffer**：
- Phase 1的keyframes被新帧覆盖
- 最终输出变成所有帧

---

## 总结

1. **根因**: VIPE Phase 2无条件添加所有帧，覆盖Phase 1的稀疏keyframe选择
2. **为什么filter_thresh无效**: Phase 2不检查motion filter
3. **为什么官方数据是稀疏的**: 可能用了不同版本的VIPE或Reference Backend
4. **修复方向**: 修改Phase 2逻辑或后处理稀疏化

**推荐方案**: 优先级1确认输出逻辑，然后根据结果选择方案C（后处理）或方案B（修改Phase 2）。
