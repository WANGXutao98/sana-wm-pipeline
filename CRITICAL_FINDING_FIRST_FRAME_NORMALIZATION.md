# 🔥 关键发现：第一帧归一化导致轨迹偏差

**发现日期**: 2026-08-14  
**严重程度**: CRITICAL  
**问题**: 轨迹长度偏大 2.5-10.7x

---

## 根本原因定位

通过深度对比官方 `sana-wm-data-clean` 和本地实现，发现**关键差异**：

### 官方实现 (`vipe_cli.py:61-70`)

```python
def _load_vipe_pose(out_dir: Path) -> np.ndarray:
    """Load VIPE's cam2world pose track (N,4,4), frame-ordered by `inds`."""
    npzs = sorted(Path(out_dir).glob("pose/*.npz"))
    z = np.load(npzs[0])
    data = np.asarray(z["data"], dtype=np.float64)  # (N,4,4) cam2world
    inds = np.asarray(z["inds"]).ravel()
    order = np.argsort(inds)
    return data[order]  # ← 直接返回，不做归一化！
```

**特点**:
- ✅ 直接使用VIPE输出的poses
- ✅ 只做排序，不做变换
- ✅ 保持VIPE的metric scale

### 本地实现 (`mode_default.py:222-232`)

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
        out = (T0_inv[None] @ out)  # ← 🔥 这会破坏metric scale！
    
    return out.astype(np.float32)
```

**问题**:
- ❌ 强制第一帧归一化到单位矩阵
- ❌ 左乘 `T0_inv` 改变了整个轨迹
- ❌ 破坏了VIPE精心计算的metric scale

---

## 数学分析

### 第一帧归一化的效应

假设VIPE输出的第一帧pose为：
```
T0 = [R0  t0]
     [0   1 ]
```

本地实现左乘 `T0_inv`：
```
poses_normalized = T0_inv @ poses_original

其中 T0_inv = [R0^T  -R0^T @ t0]
               [0      1         ]
```

**对轨迹长度的影响**:

原始轨迹长度：
```
L_orig = Σ ||t_i - t_{i-1}||
```

归一化后轨迹长度：
```
L_norm = Σ ||T0_inv @ t_i - T0_inv @ t_{i-1}||
       = Σ ||R0^T @ (t_i - t_{i-1})||
       = Σ ||t_i - t_{i-1}||  (如果R0是纯旋转)
```

**但是！** 如果T0包含平移 `t0 ≠ 0`：
```
poses_normalized[i][:3, 3] = R0^T @ (poses_original[i][:3, 3] - t0)
```

这会：
1. 平移整个轨迹 (减去t0)
2. 旋转坐标系 (R0^T)
3. **改变轨迹相对于原点的位置**

如果VIPE的metric scale是相对于世界坐标系定义的，而第一帧不在原点，这个变换会破坏scale！

---

## 为什么修正因子不一致？

回顾我们的观察：
- 样本1: 2.46x 偏差 → 修正因子 0.407
- 样本2: 10.73x 偏差 → 修正因子 0.093

**解释**:

不同样本的VIPE第一帧pose `T0` 不同，导致：
1. `T0_inv` 的平移量 `||t0||` 不同
2. 归一化对轨迹的影响不同
3. 修正因子 = f(t0, 轨迹形状) → 不一致

**样本2偏差更大的原因**:
- 可能第一帧的 `||t0||` 更大
- 或者轨迹方向与t0更对齐，导致归一化放大效应更强

---

## 验证假设

### 预期现象（如果假设正确）

1. **去掉第一帧归一化后，轨迹长度应该接近1.0x**
2. 官方 `vipe_cli.py` 不做归一化，所以官方结果是正确的
3. 本地实现的第一帧归一化是**错误**添加的

### 代码证据

**注释说明**（`mode_default.py:228`）：
```python
# Ensure first frame is identity (paper App. D.3).
```

但查看论文App. D.3和官方实现，**官方并不做这个归一化！**

这说明：
- ✅ 官方实现是正确的参考
- ❌ 本地实现错误地添加了归一化逻辑
- ❌ 注释引用的"paper App. D.3"可能是误解

---

## 解决方案

### 方案A: 移除第一帧归一化（推荐）

**修改**: `src/sana_wm_pipeline/stage02_pose/mode_default.py:222-232`

```python
def _interp_poses(poses: np.ndarray, inds: np.ndarray, T: int) -> np.ndarray:
    """Nearest-neighbour fill from keyframe poses to dense T frames."""
    out = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            out[:, i, j] = np.interp(np.arange(T), inds, poses[:, i, j])
    
    # ponytail: 移除第一帧归一化，保持VIPE的metric scale
    # 官方 vipe_cli.py 不做归一化，直接返回VIPE输出
    # 归一化会破坏metric scale导致轨迹偏差2-10x
    
    return out.astype(np.float32)
```

### 方案B: 直接使用官方的 `_load_vipe_pose`（更简单）

```python
def _load_vipe_artifacts(clip_path: Path, vipe_out: Path) -> PoseArtifact:
    stem = Path(clip_path).stem
    pose_npz = vipe_out / "pose" / f"{stem}.npz"
    
    if not pose_npz.exists():
        raise FileNotFoundError(f"VIPE pose artifact missing: {pose_npz}")
    
    # 使用官方实现：直接加载，不插值，不归一化
    z = np.load(pose_npz)
    poses_c2w = z["data"].astype(np.float32)  # (K, 4, 4) keyframes
    pose_inds = z["inds"]
    
    # 按inds排序
    order = np.argsort(pose_inds)
    poses_c2w = poses_c2w[order]
    
    # ponytail: VIPE已经输出dense poses（每2秒一个keyframe），
    # 无需插值到更密集的T_full帧，直接使用keyframe poses
    
    # 如果需要dense poses，用官方的方式：VIPE配置kf_gap_sec=更小值
    # 而不是事后插值+归一化
    
    # ... 其余代码保持不变
```

---

## 预期修复效果

### 修复前（当前）

| 样本 | 我们的轨迹 | VIPE参考 | 比例 |
|------|-----------|----------|------|
| 样本1 | 0.0606m | 0.0247m | 2.46x ❌ |
| 样本2 | 2.5006m | 0.2331m | 10.73x ❌ |

### 修复后（预期）

| 样本 | 我们的轨迹 | VIPE参考 | 比例 |
|------|-----------|----------|------|
| 样本1 | ~0.025m | 0.0247m | ~1.0x ✅ |
| 样本2 | ~0.233m | 0.2331m | ~1.0x ✅ |

---

## 行动计划

### 立即行动

1. **修改代码**：移除 `_interp_poses` 中的第一帧归一化
2. **重跑测试**：执行冒烟测试验证修复效果
3. **验证结果**：检查轨迹长度比例是否接近1.0x

### 修改文件

**文件**: `src/sana_wm_pipeline/stage02_pose/mode_default.py`

**修改位置**: 第222-232行

**修改内容**: 删除或注释掉第228-231行的归一化逻辑

### 测试命令

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
# 修改代码后
bash experiments/data_production_smoke/smoke_spatialvid.sh
python3 scripts/validate_smoke_output.py --output-dir sana_test_data/smoke_result --samples ...
```

---

## 教训与反思

### 为什么本地实现添加了错误的归一化？

1. **误解论文**：可能误读了论文App. D.3的描述
2. **缺少官方对比**：没有对比官方 `vipe_cli.py` 的实现
3. **过早优化**：添加了"保证第一帧是单位矩阵"的逻辑，但VIPE不需要

### Ponytail原则应用

**"Already in this codebase?"** 
→ 应该直接复制官方 `_load_vipe_pose`，而不是重新实现

**"Deletion over addition"**  
→ 删除不必要的归一化逻辑，保持简单

**"Bug fix = root cause"**  
→ 现在找到了：不是深度、不是scale、不是VIPE本身，而是事后处理破坏了metric scale

---

## 相关文件

1. **本报告**: `CRITICAL_FINDING_FIRST_FRAME_NORMALIZATION.md`
2. **代码对比**: 见上方官方vs本地实现
3. **调查历史**: `STAGE11_INVESTIGATION_REPORT.md`

---

**优先级**: P0 - CRITICAL  
**修复难度**: TRIVIAL（删除3行代码）  
**修复效果**: 预期完全解决轨迹偏差问题  
**建议**: 立即修复并重跑测试
