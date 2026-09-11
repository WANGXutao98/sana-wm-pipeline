# Lite Geometric Check - 使用指南

## 快速开始

### 1. 检查 JSON 文件（最简单）

```python
from sana_wm_pipeline.qc import check_pose_json

# 检查单个 pose JSON 文件
result = check_pose_json('pose_artifact_default.json')

if result.passed:
    print("✓ 所有检查通过")
else:
    print("✗ 检测失败:")
    for reason in result.failure_reasons:
        print(f"  - {reason}")
```

### 2. 检查 NumPy 数组

```python
from sana_wm_pipeline.qc import check_pose_geometry
import numpy as np

# 加载数据
poses = np.load('poses_c2w.npy')       # (N, 4, 4)
intrinsics = np.load('intrinsics.npy') # (N, 1, 4) or (N, 4)
scale = np.load('scale.npy')           # (N,)

# 执行检查
result = check_pose_geometry(
    poses_c2w=poses,
    intrinsics=intrinsics,
    scale=scale,
    image_wh=(1280, 720)
)

print(f"总体判定: {'PASS' if result.passed else 'FAIL'}")
```

### 3. 详细报告

```python
result = check_pose_json('pose_artifact_default.json')

# 分层查看结果
print(f"Level 1 (数学合法性): {'PASS' if result.level1_passed else 'FAIL'}")
print(f"  SO(3) 有效: {result.so3_valid}")
print(f"    行列式均值: {result.so3_det_mean:.8f}")
print(f"    正交性误差: {result.so3_orth_err:.3e}")
print(f"  第一帧对齐: {result.first_frame_aligned}")
print(f"    最大偏差: {result.first_frame_dev:.6f}")
print(f"  无 NaN/Inf: {result.no_nan_inf}")

print(f"\nLevel 2 (物理合理性): {'PASS' if result.level2_passed else 'FAIL'}")
print(f"  FOV 有效: {result.fov_valid}")
print(f"    水平: [{result.fov_x_min:.1f}°, {result.fov_x_max:.1f}°]")
print(f"    垂直: [{result.fov_y_min:.1f}°, {result.fov_y_max:.1f}°]")
print(f"  焦距差异: {result.focal_div_valid} (max={result.focal_div_max:.3f})")
print(f"  Scale CV: {result.scale_cv_valid} (cv={result.scale_cv:.4f})")
```

### 4. 导出 JSON 报告

```python
result = check_pose_json('pose_artifact_default.json')

# 保存为 JSON 文件
result.to_json('report.json')

# 或获取 JSON 字符串
json_str = result.to_json()
print(json_str)
```

---

## 命令行使用

```bash
# 检查单个文件并输出到终端
python -m sana_wm_pipeline.qc.lite_geometric_check pose_artifact_default.json

# 保存报告到文件
python -m sana_wm_pipeline.qc.lite_geometric_check pose_artifact_default.json report.json

# 返回值：0=通过，1=失败
echo $?
```

---

## 检查项说明

### Level 1: 数学合法性

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| SO(3) 行列式 | \|det-1\| ≤ 0.001 | 旋转矩阵行列式必须为 1 |
| SO(3) 正交性 | orth_err ≤ 0.001 | R·R^T = I |
| 第一帧对齐 | dev ≤ 0.01 | 第一帧必须是单位矩阵 |
| 无 NaN/Inf | - | 所有数值必须有限 |

### Level 2: 物理合理性

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| 水平 FOV | 25° - 120° | 相机视场角范围 |
| 垂直 FOV | 25° - 120° | 相机视场角范围 |
| 焦距差异 | ≤ 0.20 | \|fx-fy\|/((fx+fy)/2) |
| Scale CV | ≤ 2.0 | std(s)/(mean(s)+ε) |

---

## JSON 文件格式

支持的字段名（按优先级）：

```json
{
  "poses_c2w": [...],        // 或 "poses", "camera_poses", "extrinsics"
  "intrinsics": [...],       // 或 "intrinsic", "K", "camera_intrinsics"
  "scale_per_frame": [...],  // 或 "scale", "scales", "metric_scale"
  "image_wh": [1280, 720]    // 可选，可从 intrinsics 推断
}
```

**数据格式**：
- `poses_c2w`: (N, 4, 4) 相机到世界坐标系变换矩阵
- `intrinsics`: (N, 1, 4) 或 (N, 4)，[fx, fy, cx, cy]
- `scale_per_frame`: (N,) 每帧的米制尺度因子
- `image_wh`: [W, H] 图像宽高（像素）

---

## 输出结构

```python
@dataclass
class GeometryCheckResult:
    # 总体判定
    passed: bool              # 是否通过所有检查
    level1_passed: bool       # Level 1 是否通过
    level2_passed: bool       # Level 2 是否通过
    
    # Level 1 详细指标
    so3_valid: bool
    so3_det_mean: float
    so3_det_std: float
    so3_orth_err: float
    
    first_frame_aligned: bool
    first_frame_dev: float
    
    no_nan_inf: bool
    nan_inf_fields: list[str]
    
    # Level 2 详细指标
    fov_valid: bool
    fov_x_min: float
    fov_x_max: float
    fov_y_min: float
    fov_y_max: float
    
    focal_div_valid: bool
    focal_div_max: float
    
    scale_cv_valid: bool
    scale_cv: float
    
    # 失败原因
    failure_reasons: list[str]
    
    # 元信息
    num_frames: int
    image_wh: tuple[int, int]
```

---

## 常见问题

### Q1: 为什么第一帧必须是单位矩阵？
**A**: 统一坐标系原点，便于批量训练。所有样本从同一个标准起点出发。

### Q2: FOV 超出 120° 会发生什么？
**A**: 通常意味着焦距标注错误。真实相机的 FOV 很少超过 120°（超广角上限）。

### Q3: 焦距差异检查的目的是什么？
**A**: 检测常见标注错误，例如 fx/fy 与 cx/cy 的值被互换。

### Q4: Scale CV 为什么重要？
**A**: 单个视频片段的尺度应该恒定。CV 过大说明可能拼接了不同标定的片段。

### Q5: 如何处理检测失败的数据？
**A**: 
- SO(3)/第一帧/NaN → 重新运行 COLMAP 或预处理
- FOV/焦距差异 → 检查内参标注是否正确
- Scale CV → 检查是否拼接了多个片段

---

## 真实案例

### 案例 1: 正常数据
```python
result = check_pose_json('pose_artifact_default.json')
# ✓ PASS
# SO(3): det=1.00000000, orth_err=6.14e-08
# FOV: 79.9° x 50.4°
# Scale CV: 0.0129
```

### 案例 2: COLMAP 失败
```python
# SO(3) invalid: det_mean=0.857142 (expected 1.0 ± 0.001)
# → 重新运行 COLMAP 重建
```

### 案例 3: 焦距标注错误
```python
# Focal divergence too high: 0.560 (expected ≤ 0.2)
# 原因：intrinsics = [1280, 720, 800, 800]  # 错误
# 应该：intrinsics = [800, 800, 640, 360]  # 正确
```

---

## 性能

- **单样本检测**: < 10 ms（35 帧）
- **依赖**: 仅 NumPy
- **内存**: < 100 MB

---

## 与 Stage1 的区别

| 特性 | Stage1 | Lite Geometric Check |
|------|--------|---------------------|
| 输入 | Tar 文件（批量） | JSON 文件（单个） |
| 检查范围 | 完整（轨迹+caption+视觉） | 仅几何核心 |
| 执行模式 | 多进程并行 | 单进程 |
| 依赖 | PyAV, 多进程, 场景检测 | 仅 NumPy |
| 用途 | 生产数据批量 QC | 单样本快速验证 |
