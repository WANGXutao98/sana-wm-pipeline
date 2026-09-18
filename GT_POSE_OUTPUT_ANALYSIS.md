# GT-Pose 模式输出数据来源分析报告

**分析对象**: `/mnt/afs/.../smoke_result_dl3dv/{sample}/`  
**方法**: 基于实际数据 + 代码验证  
**日期**: 2026-09-11

---

## 问题 1: 哪些是 GT，哪些是代码产出？

### 实际数据验证

```python
# 样本: DL3DV-...__images_2

# 测试 1: poses.npy
output_poses = np.load("poses.npy")              # (303, 4, 4)
gt_c2w = np.load("camera.npz")['c2w']            # (303, 4, 4)
np.allclose(output_poses, gt_c2w)                # ✅ True

# 测试 2: intrinsics.npy
output_intr = np.load("intrinsics.npy")          # (303, 1, 4)
gt_K_px = np.load("camera.npz")['K_px']          # (303, 4)
np.allclose(output_intr[:, 0, :], gt_K_px)       # ✅ True

# 测试 3: scale_per_frame.npy
camera_data.keys()  # ['c2w', 'w2c', 'K_px', ...]
'scale' in camera_data.keys()                    # ❌ False (GT 中不存在)
```

---

## 答案 1: 数据来源分类

### ✅ 直接使用 GT 的数据

#### 1. `poses.npy` → 直接使用 GT

**来源**: `camera.npz['c2w']`  
**验证**: `np.allclose(output, gt) = True`  
**代码路径**:
```python
# mode_gtpose.py:125
poses_c2w = gt_poses_full  # 直接赋值，未修改
```

**物理含义**: 
- GT camera-to-world 变换矩阵
- (303, 4, 4): 每帧一个 4×4 齐次变换矩阵
- OpenCV 坐标系

---

#### 2. `intrinsics.npy` → 直接使用 GT

**来源**: `camera.npz['K_px']`  
**验证**: `np.allclose(output[:, 0, :], gt) = True`  
**代码路径**:
```python
# mode_gtpose.py:150-157
if gt_intrinsics_path is not None and gt_intrinsics_path.exists():
    intr = _load_gt_intrinsics(gt_intrinsics_path, m)
    # _load_gt_intrinsics() 从 camera.npz 加载 K_px
```

**物理含义**:
- GT 相机内参: [fx, fy, cx, cy]
- (303, 1, 4): 每帧一组内参（本例中所有帧相同）
- fx, fy: 焦距（像素单位）
- cx, cy: 主点（像素单位）

**示例值**:
```
[862.37, 862.79, 960.0, 540.0]
```

---

### ❌ 代码计算产出的数据

#### 3. `scale_per_frame.npy` → 代码计算（Umeyama Sim(3)）

**来源**: Umeyama 算法计算  
**验证**: GT 中不存在 `scale` 字段  
**代码路径**:
```python
# mode_gtpose.py:118-122
s = recover_metric_scale(
    pred_positions,     # Pi3X 预测的 64 个相机中心
    gt_positions_sub,   # GT 子采样的 64 个相机中心
    inlier_percentile=80.0
)

# mode_gtpose.py:168
scale_per_frame = np.full(m, float(s), dtype=np.float32)
```

**计算流程**:
1. Pi3X 推理 64 帧 → `pred_positions` (64, 3)
2. GT 子采样到 64 帧 → `gt_positions_sub` (64, 3)
3. Umeyama Sim(3) 对齐 → 返回单个 scale `s` (标量)
4. 单值广播到所有帧 → `[s, s, s, ..., s]` (303 个)

**物理含义**:
- Pi3X 到 GT 的统一缩放因子
- Pi3X 预测的单位是任意的（无量纲）
- GT poses 的单位是真实世界（米）
- `scale ≈ 1.40`: 表示 Pi3X 的 "1 单位" ≈ 真实世界的 1.40 米

---

## 问题 2: 为什么所有 scale 都一样？

### 实际数据验证

```python
scale = np.load("scale_per_frame.npy")  # (303,)

# 唯一值
np.unique(scale)                        # array([1.3994777])
len(np.unique(scale))                   # 1

# 所有值完全相同
np.all(scale == scale[0])               # True
```

---

## 答案 2: Scale 单值的三个原因

### 原因 1: Umeyama Sim(3) 算法性质

**Umeyama Sim(3) 返回单个标量**:
```python
def umeyama_sim3(src, dst):
    # 输入: src (N, 3), dst (N, 3)
    # 输出: s (标量), R (3x3), t (3,)
    # ...
    return s, R, t  # ← s 是单个标量
```

**数学含义**:
- Sim(3) = Similarity(3) = Rotation + Translation + **Uniform Scale**
- "Uniform" 表示 **各向同性缩放**（xyz 三个方向相同）
- 不是 per-frame scale，是 **整个场景的全局 scale**

**对比**:
- Sim(3): 1 个 scale (s)
- Affine: 9 个参数（可以有不同的 sx, sy, sz）

---

### 原因 2: 代码实现（单值广播）

**官方代码**:
```python
# stage.py:77
scales = [s] * m  # Python list 广播
```

**本地实现**:
```python
# mode_gtpose.py:168
scale_per_frame = np.full(m, float(s), dtype=np.float32)  # numpy array 广播
```

**功能等价**:
```python
s = 1.3995
m = 303

# 官方
scales = [1.3995, 1.3995, ..., 1.3995]  # 303 个

# 本地
scale_per_frame = array([1.3995, 1.3995, ..., 1.3995])  # 303 个
```

---

### 原因 3: 物理意义（场景级别的统一 scale）

**为什么是场景级别？**

```
场景: 一个房间
  ├─ Pi3X 预测: "相机 A 距离墙 3 单位"
  ├─ Pi3X 预测: "相机 B 距离墙 5 单位"
  └─ GT 真实:    "相机 A 距离墙 4.2 米"
                 "相机 B 距离墙 7.0 米"

Scale = 真实距离 / Pi3X距离 = 4.2 / 3 = 1.4

这个 scale 对整个场景是恒定的:
  相机 B: 5 × 1.4 = 7.0 米 ✅
```

**不可能 per-frame**:
- 如果每帧 scale 不同 → 场景会"拉伸"或"压缩"
- 违反刚性场景假设
- 破坏 3D 结构一致性

---

## 总结表

| 文件 | 来源 | 是否 GT | 备注 |
|------|------|---------|------|
| `poses.npy` | camera.npz['c2w'] | ✅ GT | 完全未修改 |
| `intrinsics.npy` | camera.npz['K_px'] | ✅ GT | 修复后正确加载 |
| `scale_per_frame.npy` | Umeyama 计算 | ❌ 代码产出 | 单值广播 |
| `gt_poses.npy` | camera.npz['c2w'] | ✅ GT | 临时文件 |

---

## 关键验证点

### 验证 1: Poses 是 GT ✅
```python
output == GT: True
第一帧 pose:
[[-0.0700  0.9975 -0.0076 -0.0519]
 [ 0.9739  0.0667 -0.2170  5.5108]
 [-0.2160 -0.0226 -0.9761  2.4100]
 [ 0.0000  0.0000  0.0000  1.0000]]
```

### 验证 2: Intrinsics 是 GT ✅
```python
output == GT: True
第一帧 K_px: [862.37, 862.79, 960.0, 540.0]
```

### 验证 3: Scale 是计算值 ✅
```python
GT 中不存在 scale
所有值: 1.3995 (303 次)
Umeyama 返回单个标量 → 广播到所有帧
```

---

## 最终答案

### 问题 1: 哪些是 GT，哪些是代码产出？

**GT 数据** (2/3):
- ✅ `poses.npy`: 直接使用 camera.npz['c2w']
- ✅ `intrinsics.npy`: 直接使用 camera.npz['K_px']

**代码产出** (1/3):
- ❌ `scale_per_frame.npy`: Umeyama Sim(3) 计算

---

### 问题 2: 为什么所有 scale 都一样？

**三个原因**:
1. **算法性质**: Umeyama Sim(3) 返回单个标量（全局 uniform scale）
2. **代码实现**: `np.full(m, s)` 单值广播
3. **物理意义**: 场景级别的统一缩放因子（不是 per-frame）

**✅ 所有 scale 相同是正确的、预期的行为**

---

**分析完成！所有结论基于实际数据和代码验证！** 🎯
