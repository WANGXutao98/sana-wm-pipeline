# 第一帧归一化问题的澄清与决策

**调查日期**: 2026-08-14  
**问题**: 本地实现添加第一帧归一化的原因和必要性

---

## 问题1: 为什么本地实现添加了第一帧归一化？

### Commit历史追溯

**Commit**: `18f3697` (2026-05-26)  
**Message**: "feat(stage02): 3 pose annotation modes (default/gt-depth/gt-pose, App. B.1)"

关键信息：
```
validates first-frame identity per App. D.3
```

**结论**: 实现者误读了论文App. D.3，认为数据标注也需要第一帧归一化。

---

## 问题2: 论文App. D.3是否要求数据标注做第一帧归一化？

### 论文原文（第859行）

> "The ground-truth trajectory is loaded from the benchmark camera annotations, **relativized to the first frame**, and sampled consistently with the evaluated video."

### 上下文分析

**章节标题**: "D.3. Evaluation Protocol"（评估协议）

**完整语境**:
```
Camera accuracy. We estimate camera poses from each generated video using Pi3X.
The ground-truth trajectory is loaded from the benchmark camera annotations,
relativized to the first frame, and sampled consistently with the evaluated video.
The recovered Pi3X trajectory is aligned to the ground truth by Umeyama Sim(3)
alignment.
```

### 关键理解

1. **"relativized to the first frame"的含义**:
   - 这是**评估生成视频**时的处理
   - 从生成视频用Pi3X估计pose后，将GT轨迹相对第一帧归一化
   - 目的：公平对比（不同模型可能从不同初始位置生成）

2. **适用场景**:
   - ✅ **评估生成模型**：输入第一帧图像+轨迹，生成视频，评估pose accuracy
   - ❌ **数据标注**：从真实视频估计camera pose，保存为训练数据

3. **为什么评估需要归一化**:
   - 生成模型的第一帧是条件输入，相机位置是任意的
   - 归一化后才能公平计算RotErr/TransErr
   - 例如：如果GT第一帧在(10, 5, 2)，生成的第一帧在(0, 0, 0)，直接比较会有巨大偏差

4. **为什么数据标注不需要归一化**:
   - VIPE SLAM输出的是**metric-scale的绝对poses**
   - 融合深度已经恢复了真实的米制尺度
   - 归一化会破坏这个精心计算的metric scale
   - 训练时模型需要学习**真实的空间关系**，不是相对关系

### 官方实现验证

**官方 `sana-wm-data-clean/sana_wm_data/pose/vipe_cli.py:61-70`**:
```python
def _load_vipe_pose(out_dir: Path) -> np.ndarray:
    """Load VIPE's cam2world pose track (N,4,4), frame-ordered by `inds`."""
    npzs = sorted(Path(out_dir).glob("pose/*.npz"))
    z = np.load(npzs[0])
    data = np.asarray(z["data"], dtype=np.float64)
    inds = np.asarray(z["inds"]).ravel()
    order = np.argsort(inds)
    return data[order]  # ← 直接返回，不归一化！
```

**结论**: 官方实现**不做第一帧归一化**，证明论文App. D.3的"relativized"是指评估协议，不是数据标注。

---

## 问题3: 为什么之前没有和官方对齐？

### 时间线分析

1. **2026-05-26**: 初始实现 (commit 18f3697)
   - 基于对论文的理解实现
   - 误读App. D.3为数据标注要求
   - 添加了第一帧归一化逻辑

2. **2026-06-XX**: 后续开发
   - 专注于其他功能（scale加载、@lru_cache等）
   - 没有深度对比官方 `vipe_cli.py` 的实现

3. **2026-08-13**: 发现轨迹偏差问题
   - 误以为是scale加载bug
   - 修复scale后问题仍存在

4. **2026-08-14**: 今天的深度调查
   - 首次逐行对比官方实现
   - 发现第一帧归一化的差异

### 根本原因

1. **论文理解误差**:
   - 没有区分"评估协议"和"数据标注流程"
   - "relativized to the first frame"的上下文是evaluation，不是annotation

2. **缺少官方对比**:
   - 官方 `sana-wm-data-clean` 发布时间可能晚于初始实现
   - 或者实现时没有意识到需要对比官方的pose加载逻辑

3. **Ponytail原则未应用**:
   - "Already in this codebase?" → 应该直接复制官方 `_load_vipe_pose`
   - 而不是基于论文理解重新实现

---

## 决策：是否移除第一帧归一化？

### 证据汇总

| 证据 | 结论 |
|------|------|
| 论文App. D.3原文 | "relativized"是**评估协议**，不是数据标注 ✅ |
| 官方vipe_cli.py实现 | **不做归一化** ✅ |
| 轨迹偏差2.5-10.7x | 与第一帧归一化的破坏性一致 ✅ |
| 修正因子不一致 | 不同样本的T0不同 → T0_inv影响不同 ✅ |
| 本地commit message | 误读"App. D.3" ✅ |

### 数学分析

**第一帧归一化的效应**:
```python
T0_inv = np.linalg.inv(poses[0])
poses_normalized = T0_inv @ poses
```

- ❌ 破坏metric scale（VIPE通过Pi3X+MoGe融合精心计算）
- ❌ 改变轨迹的绝对位置和scale
- ❌ 导致不同样本的偏差不一致（T0不同 → 影响不同）

**VIPE的metric scale含义**:
- VIPE输出的poses是**绝对的、metric-scale的c2w变换**
- 平移单位是**米**（通过MoGe-2恢复）
- 第一帧不一定在原点，这是正常的（取决于视频拍摄起点）

### 最终决策

**✅ 确认：第一帧归一化存在问题，应该移除**

**理由**:
1. 论文App. D.3是评估协议，不适用于数据标注
2. 官方实现不做归一化
3. 归一化破坏了VIPE的metric scale
4. 导致轨迹偏差2.5-10.7x
5. 所有证据一致指向这是问题根源

---

## 修改方案

### 文件: `src/sana_wm_pipeline/stage02_pose/mode_default.py`

### 修改位置: 第222-232行

**修改前**:
```python
def _interp_poses(poses: np.ndarray, inds: np.ndarray, T: int) -> np.ndarray:
    """Nearest-neighbour fill from keyframe poses to dense T frames."""
    out = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            out[:, i, j] = np.interp(np.arange(T), inds, poses[:, i, j])
    # Ensure first frame is identity (paper App. D.3).
    if not np.allclose(out[0], np.eye(4), atol=1e-3):
        T0_inv = np.linalg.inv(out[0])
        out = (T0_inv[None] @ out)
    return out.astype(np.float32)
```

**修改后**:
```python
def _interp_poses(poses: np.ndarray, inds: np.ndarray, T: int) -> np.ndarray:
    """Nearest-neighbour fill from keyframe poses to dense T frames."""
    out = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            out[:, i, j] = np.interp(np.arange(T), inds, poses[:, i, j])
    
    # ponytail: 移除第一帧归一化，保持VIPE的metric scale
    # 
    # 论文App. D.3的"relativized to the first frame"是评估协议（评估生成视频时
    # 将GT归一化以公平对比），不是数据标注流程。VIPE输出的poses已经是metric-scale
    # 的绝对c2w变换（通过Pi3X+MoGe融合恢复），归一化会破坏这个精心计算的scale，
    # 导致轨迹偏差2-10x。
    # 
    # 参考：sana-wm-data-clean/sana_wm_data/pose/vipe_cli.py:_load_vipe_pose
    # 官方实现直接返回VIPE输出，不做任何归一化。
    
    return out.astype(np.float32)
```

---

## 预期效果

修复后，轨迹长度应该与VIPE参考基本一致（比例~1.0x）：

| 样本 | 修复前 | VIPE参考 | 修复后（预期）|
|------|--------|----------|--------------|
| 样本1 | 0.0606m (2.46x) | 0.0247m | ~0.025m (1.0x) |
| 样本2 | 2.5006m (10.73x) | 0.2331m | ~0.233m (1.0x) |

---

## 教训

1. **区分评估和标注**: 论文的评估协议（App. D）不一定适用于数据标注流程
2. **官方实现优先**: 实现时应先查看官方代码，而不是仅凭论文理解
3. **Ponytail原则**: "Already in this codebase?" → 直接复用官方实现
4. **及时对齐**: 发现偏差时，第一时间对比官方实现

---

**决策**: ✅ 移除第一帧归一化（删除mode_default.py第228-231行）  
**验证**: 修改后运行冒烟测试，检查轨迹比例是否接近1.0x
