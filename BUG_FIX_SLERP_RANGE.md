# 🐛 Bug分析：Slerp插值范围错误

**错误信息**: `Interpolation times must be within the range [0, 28], both inclusive.`

**发生位置**: `mode_default.py:179` - `R_interp = slerp(np.arange(T_full))`

---

## 根本原因

### 问题分析

**VIPE输出**:
- `pose_inds = [0, 1, 2, ..., 31]` (32帧，0-indexed)
- `pose_inds.max() = 31`

**稀疏化后**:
- `sparse_mask = (pose_inds % 4 == 0)`
- `inds_sparse = [0, 4, 8, 12, 16, 20, 24, 28]` ⚠️ **最大值是28，不是31**

**插值范围计算**:
```python
T_full = int(pose_inds.max()) + 1  # = 32
R_interp = slerp(np.arange(T_full))  # 请求插值 [0, 1, 2, ..., 31]
```

**错误**:
- Slerp初始化范围: `[0, 28]` (inds_sparse的范围)
- 插值请求范围: `[0, 31]` (T_full的范围)
- ❌ 请求的29, 30, 31超出了Slerp的范围！

### 根本原因

**稀疏化后的最后一个keyframe索引（28）小于视频的最后一帧索引（31）**

这导致：
- Slerp只能插值到28
- 但我们请求插值到31
- 29, 30, 31这3帧无法插值

---

## 修复方案

### 方案A: 确保包含最后一帧 ⭐⭐⭐⭐⭐ (推荐)

**思路**: 稀疏化时强制包含第一帧和最后一帧

```python
# 稀疏化策略：每N帧 + 强制包含首尾帧
KEYFRAME_INTERVAL = 4
sparse_mask = (pose_inds % KEYFRAME_INTERVAL == 0)

# ✅ 强制包含最后一帧（如果不在sparse_mask中）
last_idx = pose_inds[-1]
if not sparse_mask[-1]:
    sparse_mask[-1] = True

poses_sparse = poses_c2w[sparse_mask]
inds_sparse = pose_inds[sparse_mask]
```

**优点**:
- ✅ 保证插值范围完整
- ✅ 首尾帧都是keyframes（符合SLAM最佳实践）
- ✅ 轨迹长度准确（最后一帧的pose是真实计算的，不是外推的）

**效果**:
- 原本: `inds_sparse = [0, 4, 8, 12, 16, 20, 24, 28]` (8个)
- 修复后: `inds_sparse = [0, 4, 8, 12, 16, 20, 24, 28, 31]` (9个)

### 方案B: 钳位插值范围 ⭐⭐⭐

**思路**: 将插值请求钳位到Slerp的有效范围

```python
# Slerp插值旋转部分
from scipy.spatial.transform import Rotation, Slerp
R_sparse = Rotation.from_matrix(poses_sparse[:, :3, :3])
slerp = Slerp(inds_sparse, R_sparse)

# ✅ 钳位插值范围到Slerp的有效范围
interp_times = np.arange(T_full)
interp_times_clamped = np.clip(interp_times, inds_sparse.min(), inds_sparse.max())
R_interp = slerp(interp_times_clamped)
```

**缺点**:
- ❌ 最后3帧（29, 30, 31）的旋转都等于第28帧（重复值，不准确）
- ❌ 轨迹在最后几帧会"冻结"

### 方案C: 外推最后几帧 ⭐⭐

**思路**: 对超出范围的帧进行外推

```python
# 分段处理：插值 + 外推
valid_mask = interp_times <= inds_sparse.max()
R_interp = np.zeros((T_full, 3, 3), dtype=np.float32)

# 插值有效范围
R_interp[valid_mask] = slerp(interp_times[valid_mask]).as_matrix()

# 外推超出范围（简单重复最后一个keyframe）
R_interp[~valid_mask] = R_interp[valid_mask][-1]
```

**缺点**:
- ❌ 外推不准确
- ❌ 代码复杂

---

## 推荐修复代码

### 完整修复（方案A）

```python
def _load_vipe_artifacts(clip_path: Path, vipe_out: Path) -> PoseArtifact:
    """Parse VIPE's npz artifacts into PoseArtifact."""
    stem = Path(clip_path).stem
    pose_npz = vipe_out / "pose" / f"{stem}.npz"
    intr_npz = vipe_out / "intrinsics" / f"{stem}.npz"

    if not pose_npz.exists():
        raise FileNotFoundError(f"VIPE pose artifact missing: {pose_npz}")

    pose_data = np.load(pose_npz)
    poses_c2w = pose_data["data"].astype(np.float32)  # (T, 4, 4)
    pose_inds = pose_data["inds"]                      # (T,)

    if not intr_npz.exists():
        raise FileNotFoundError(f"VIPE intrinsics artifact missing: {intr_npz}")
    intr_data = np.load(intr_npz)
    intrinsics_raw = intr_data["data"].astype(np.float32)  # (T, 4) [fx,fy,cx,cy]
    intr_inds = intr_data["inds"]

    # ponytail: 稀疏化VIPE输出（每4帧1个keyframe）+ Slerp插值
    # VIPE Phase 2无条件添加所有帧 → 连续keyframes → 短基线 → scale漂移
    # 稀疏化 → 长基线 → 稳定scale（轨迹偏差从3.88-9.56x降至~1.0-1.5x）
    KEYFRAME_INTERVAL = 4
    sparse_mask = (pose_inds % KEYFRAME_INTERVAL == 0)
    
    # ✅ 修复：强制包含最后一帧（避免Slerp插值范围不足）
    # 确保插值范围覆盖所有帧 [0, T_full-1]
    if not sparse_mask[-1]:
        sparse_mask[-1] = True
    
    poses_sparse = poses_c2w[sparse_mask]
    inds_sparse = pose_inds[sparse_mask]

    T_full = int(pose_inds.max()) + 1
    
    print(f"[mode_default] Keyframe sparsification: {len(pose_inds)} → {len(inds_sparse)} frames")
    print(f"[mode_default]   Sparse indices: {inds_sparse.tolist()}")

    # Slerp插值旋转部分（避免线性插值产生非正交矩阵）
    from scipy.spatial.transform import Rotation, Slerp
    R_sparse = Rotation.from_matrix(poses_sparse[:, :3, :3])
    slerp = Slerp(inds_sparse, R_sparse)
    R_interp = slerp(np.arange(T_full))

    # 线性插值平移部分
    t_interp = np.zeros((T_full, 3), dtype=np.float32)
    for k in range(3):
        t_interp[:, k] = np.interp(np.arange(T_full), inds_sparse, poses_sparse[:, k, 3])

    # 重组4x4矩阵
    poses_c2w = np.zeros((T_full, 4, 4), dtype=np.float32)
    poses_c2w[:, :3, :3] = R_interp.as_matrix()
    poses_c2w[:, :3, 3] = t_interp
    poses_c2w[:, 3, 3] = 1.0
    intrinsics_full = _interp_intrinsics(intrinsics_raw, intr_inds, T_full)

    # ... 后续代码不变 ...
```

### 关键修改点

**位置**: `mode_default.py:169-171`

**修改前**:
```python
KEYFRAME_INTERVAL = 4
sparse_mask = (pose_inds % KEYFRAME_INTERVAL == 0)
poses_sparse = poses_c2w[sparse_mask]
inds_sparse = pose_inds[sparse_mask]
```

**修改后**:
```python
KEYFRAME_INTERVAL = 4
sparse_mask = (pose_inds % KEYFRAME_INTERVAL == 0)

# ✅ 修复：强制包含最后一帧
if not sparse_mask[-1]:
    sparse_mask[-1] = True

poses_sparse = poses_c2w[sparse_mask]
inds_sparse = pose_inds[sparse_mask]
```

---

## 验证

### 测试用例

```python
# 原始数据
pose_inds = np.array([0, 1, 2, ..., 31])  # 32帧

# 修复前
sparse_mask = (pose_inds % 4 == 0)
inds_sparse = [0, 4, 8, 12, 16, 20, 24, 28]  # ❌ max=28 < 31

# 修复后
sparse_mask = (pose_inds % 4 == 0)
if not sparse_mask[-1]:
    sparse_mask[-1] = True
inds_sparse = [0, 4, 8, 12, 16, 20, 24, 28, 31]  # ✅ max=31

# Slerp范围
slerp = Slerp([0, 4, 8, 12, 16, 20, 24, 28, 31], R_sparse)
slerp(np.arange(32))  # ✅ 成功：[0, 31]都在范围内
```

---

## 总结

**Bug根因**: 稀疏化后最后一个keyframe索引小于视频最后一帧索引，导致Slerp插值范围不足

**修复方案**: 强制包含最后一帧到稀疏keyframes中

**影响**: 
- Keyframe数量: 8个 → 9个（增加1个）
- 轨迹精度: 提升（最后一帧是真实计算的，不是外推）
- 稀疏化效果: 仍然有效（32帧 → 9个keyframes，减少72%）

**代码改动**: 仅3行（在line 169-171之间插入）
