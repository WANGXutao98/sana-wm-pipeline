# gt_poses.npy 文件来源分析

**问题**: 为什么结果文件夹中有 `gt_poses.npy`？它是在哪一步创建的？

---

## 📍 创建位置

**文件**: `scripts/smoke_batch_gtpose.py`  
**行号**: 89-91

```python
# 保存为临时 GT poses 文件
temp_gt_poses = sample_dir / "gt_poses.npy"
np.save(temp_gt_poses, gt_c2w)
print(f"  临时 GT poses 已保存: {temp_gt_poses.name}")
```

---

## 🔄 完整流程

### 步骤 1: 从 camera.npz 提取 GT poses

```python
# smoke_batch_gtpose.py:72-74
camera_data = np.load(camera_path)
gt_c2w = camera_data['c2w']  # (303, 4, 4)
```

**来源**: `smoke_dl3dv_test_sample/{sample_id}.camera.npz['c2w']`

---

### 步骤 2: 保存为临时文件

```python
# smoke_batch_gtpose.py:88-91
temp_gt_poses = sample_dir / "gt_poses.npy"
np.save(temp_gt_poses, gt_c2w)
```

**目的**: 
- `run_gtpose()` 的接口需要一个文件路径 (`gt_poses_path: Path`)
- 不能直接传入 numpy array
- 所以先保存为临时文件

**路径**: `smoke_result_dl3dv/{sample_id}/gt_poses.npy`

---

### 步骤 3: 传递给 run_gtpose()

```python
# smoke_batch_gtpose.py:95-101
artifact = run_gtpose(
    clip_path=video_path,
    gt_poses_path=temp_gt_poses,  # ← 传入临时文件路径
    work_dir=sample_dir,
    inlier_percentile=80.0,
    gt_intrinsics_path=camera_path,
)
```

---

### 步骤 4: run_gtpose() 内部加载

```python
# mode_gtpose.py:105-109
gt_poses_full = _load_gt_poses(gt_poses_path)
N_gt = len(gt_poses_full)
print(f"[mode_gtpose] Loaded {N_gt} GT poses from {gt_poses_path.name}")
```

**读取**: 从 `gt_poses.npy` 加载 → `gt_poses_full`

---

### 步骤 5: 直接使用 GT poses

```python
# mode_gtpose.py:125
poses_c2w = gt_poses_full  # 直接赋值，未修改
```

---

### 步骤 6: 保存最终结果

```python
# smoke_batch_gtpose.py:106-112
output_poses = sample_dir / "poses.npy"
np.save(output_poses, artifact.poses_c2w)
```

**结果**: `poses.npy` 的内容 = `gt_poses.npy` 的内容

---

## 📂 最终文件结构

```
smoke_result_dl3dv/{sample_id}/
├── gt_poses.npy          ← 步骤 2: 临时文件（从 camera.npz 提取）
├── poses.npy             ← 步骤 6: 最终输出（= gt_poses.npy）
├── intrinsics.npy        ← 步骤 6: 最终输出（GT K_px）
└── scale_per_frame.npy   ← 步骤 6: 最终输出（Umeyama 计算）
```

---

## 🎯 gt_poses.npy 的作用

### 1. **接口适配**

```python
# run_gtpose() 的接口设计
def run_gtpose(
    clip_path: Path,
    gt_poses_path: Path,  # ← 必须是文件路径，不能是 numpy array
    ...
)
```

**原因**: 
- 官方代码从 `ClipRecord` 对象读取 GT poses 路径
- 为了保持接口一致性，需要文件路径

---

### 2. **临时文件性质**

**特点**:
- ✅ 仅用于传递数据
- ✅ 内容 = camera.npz['c2w']
- ✅ 与 poses.npy 完全相同

**验证**:
```python
gt_poses = np.load("gt_poses.npy")
output_poses = np.load("poses.npy")
np.allclose(gt_poses, output_poses)  # ✅ True
```

---

## 💡 是否可以删除？

### 选项 1: 保留 (当前)

**优点**:
- ✅ 清晰显示数据流
- ✅ 可追溯中间步骤
- ✅ 便于调试

**缺点**:
- ❌ 重复存储（gt_poses.npy ≈ poses.npy）
- ❌ 占用额外磁盘空间（~20KB/样本）

---

### 选项 2: 删除临时文件

**修改方案**:
```python
# smoke_batch_gtpose.py
try:
    artifact = run_gtpose(...)
    
    # 删除临时文件
    temp_gt_poses.unlink()
    
except Exception as e:
    # 保留临时文件用于调试
    pass
```

**优点**:
- ✅ 节省磁盘空间
- ✅ 输出目录更清晰

**缺点**:
- ❌ 调试困难（无法查看原始输入）

---

### 选项 3: 修改接口支持 numpy array

**修改方案**:
```python
def run_gtpose(
    clip_path: Path,
    gt_poses: np.ndarray | Path,  # ← 支持 array 或 path
    ...
):
    if isinstance(gt_poses, Path):
        gt_poses_full = _load_gt_poses(gt_poses)
    else:
        gt_poses_full = gt_poses  # 直接使用
```

**优点**:
- ✅ 不需要临时文件
- ✅ 更灵活

**缺点**:
- ❌ 接口变更（需要测试）
- ❌ 与官方接口不一致

---

## 📋 代码执行顺序总结

```
1. 加载 camera.npz
   ├─ gt_c2w = camera_data['c2w']  # (303, 4, 4)
   └─ 来源: smoke_dl3dv_test_sample/sample.camera.npz

2. 保存临时文件
   ├─ np.save("gt_poses.npy", gt_c2w)
   └─ 位置: smoke_result_dl3dv/sample/gt_poses.npy

3. 调用 run_gtpose()
   └─ gt_poses_path = "gt_poses.npy" (Path)

4. run_gtpose() 内部
   ├─ 加载: gt_poses_full = _load_gt_poses(gt_poses_path)
   ├─ 使用: poses_c2w = gt_poses_full (直接赋值)
   └─ 返回: PoseArtifact(poses_c2w=gt_poses_full, ...)

5. 保存最终结果
   ├─ np.save("poses.npy", artifact.poses_c2w)
   ├─ np.save("intrinsics.npy", artifact.intrinsics)
   └─ np.save("scale_per_frame.npy", artifact.scale_per_frame)

结果: gt_poses.npy 保留在输出目录
```

---

## ✅ 最终答案

### gt_poses.npy 是在哪一步创建的？

**文件**: `scripts/smoke_batch_gtpose.py`  
**位置**: 第 89-91 行  
**代码**:
```python
temp_gt_poses = sample_dir / "gt_poses.npy"
np.save(temp_gt_poses, gt_c2w)
```

### 为什么需要它？

**原因**: `run_gtpose()` 接口需要文件路径，不能直接传入 numpy array

### 它的内容是什么？

**内容**: `camera.npz['c2w']` 的完整拷贝（未修改）

### 与 poses.npy 的关系？

**关系**: `poses.npy == gt_poses.npy` (完全相同)

---

**分析完成！gt_poses.npy 是临时中间文件，用于接口适配！** 🎯
