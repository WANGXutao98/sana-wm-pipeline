# SANA-WM 轻量级几何质量检测模块规格说明

**文档版本**: v1.0  
**创建日期**: 2026-08-31  
**最后更新**: 2026-08-31

---

## 一、概述

### 1.1 目标
创建一个**独立的、轻量级的几何质量检测模块**，专注于相机位姿（pose）和内参（intrinsics）的数学合法性和物理合理性检测，用于快速验证单个样本的几何质量。

### 1.2 与现有模块的区别

| 特性 | 现有 Stage1 | 新模块 (Lite) |
|------|------------|--------------|
| 检查范围 | 完整（轨迹跳跃、caption、视觉） | 仅几何核心（pose + intrinsics） |
| 输入格式 | Tar 文件（批量） | 单个 JSON 文件 |
| 输出 | JSONL（PASS/FLAG/FAIL） | 结构化报告 |
| 依赖 | PyAV、多进程、场景检测 | 仅 NumPy |
| 用途 | 生产数据批量 QC | 单样本快速验证 |
| 执行模式 | 多进程并行 | 单进程串行 |

---

## 二、检测范围

### Level 1: 数学合法性（硬失败）

#### 1.1 SO(3) 旋转矩阵有效性
**数学定义**: 旋转矩阵 R ∈ SO(3) 必须满足：
- **行列式**: det(R) = 1
- **正交性**: R·R^T = I（单位矩阵）

**检测指标**:
```python
dets = np.linalg.det(R)  # 所有帧的行列式
det_mean = dets.mean()
orth_err = max(|R @ R^T - I|)  # 正交性误差
```

**判定阈值**:
- `|det_mean - 1.0| ≤ 1e-3`
- `orth_err ≤ 1e-3`

**失败原因**:
- COLMAP 重建失败产生退化矩阵
- 数值误差累积导致非正交
- 人工标注错误

**作用**: 保证旋转矩阵的数学合法性，是后续所有几何计算的基础。

---

#### 1.2 起点对齐（第一帧 = 单位矩阵）
**要求**: 第一帧位姿必须是单位矩阵 I₄

```python
poses_c2w[0] = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
```

**检测指标**:
```python
first_dev = max(|poses_c2w[0] - I|)
```

**判定阈值**:
- `first_dev ≤ 0.01`

**失败原因**:
- 预处理步骤缺失（未做 pose normalization）
- 坐标系未对齐

**作用**:
- 统一所有样本的坐标系原点
- 方便批量训练和相对位姿计算
- 模型假设起点在原点

---

#### 1.3 无 NaN/Inf 值
**要求**: 所有数值数据必须是有限实数

**检测范围**:
- `poses_c2w`: (N, 4, 4)
- `intrinsics`: (N, V, 4) 或 (N, 4)
- `scale`: (N,)

**检测方法**:
```python
np.isfinite(array).all()
```

**失败原因**:
- 计算过程中的除零错误
- 数值溢出
- 未初始化的数据

**作用**: 保证数据的数值稳定性，避免后续计算崩溃。

---

### Level 2: 物理合理性（硬失败）

#### 2.1 FOV（视场角）范围检查
**相机成像几何**:
```
θ_x = 2 * arctan(W / (2 * f_x))  # 水平视场角
θ_y = 2 * arctan(H / (2 * f_y))  # 垂直视场角
```

**判定阈值**:
- `25° ≤ θ_x ≤ 120°`
- `25° ≤ θ_y ≤ 120°`

**物理依据**:
| 相机类型 | FOV 范围 | 是否通过 |
|---------|---------|---------|
| 长焦镜头 | 20° - 40° | ✓ (边界) |
| 标准镜头 | 50° - 60° | ✓ |
| 手机摄像头 | 70° - 85° | ✓ |
| 超广角 | 100° - 120° | ✓ (边界) |
| 鱼眼镜头 | > 180° | ✗ 超出范围 |

**失败原因**:
- 焦距标注错误（如将像素尺寸误标为焦距）
- 非标准相机（鱼眼、针孔模拟）
- 单位错误（如毫米标注为像素）

**作用**: 确保相机参数符合真实物理相机的范围，过滤异常标注。

---

#### 2.2 焦距差异（fx vs fy）
**对称性检查**:
```python
focal_div = |fx - fy| / ((fx + fy) / 2)
```

**判定阈值**:
- `focal_div ≤ 0.20` (20%)

**物理依据**:
- **理想针孔相机**: fx = fy (方形像素)
- **真实传感器**: fx ≈ fy (像素略有矩形畸变，通常 < 5%)
- **异常情况**: fx 和 fy 显著不同 → 标注错误

**常见错误案例**:
```python
# 案例 1: fx/fy 与 cx/cy 标注互换
intrinsics = [1280, 720, 800, 800]  # 错误
# 应该是: [800, 800, 640, 360]
# focal_div = |1280-720|/1000 = 0.56 > 0.20 ✗

# 案例 2: 正常相机
intrinsics = [800, 810, 640, 360]
# focal_div = 10/805 = 0.012 < 0.20 ✓
```

**作用**: 检测相机内参标注的常见错误（字段顺序错误）。

---

#### 2.3 Scale 稳定性（变异系数 CV）
**Scale 的作用**:
COLMAP 重建的轨迹是**相对尺度**（up to scale），需要标定到真实米制单位：
```python
t_real = scale * t_colmap
```

**变异系数（CV）**:
```python
CV = std(scale) / (mean(scale) + ε)
```

**判定阈值**:
- `CV ≤ 2.0`

**物理含义**:
- CV = 0: scale 完全恒定（理想）
- CV < 0.5: 波动很小（正常）
- CV > 2.0: scale 变化剧烈（异常）

**失败原因**:
- 拼接了不同标定的视频片段
- COLMAP 重建不稳定
- 尺度标定算法失败

**作用**: 确保单个视频片段内的尺度一致性。

---

## 三、测试样例分析

### 3.1 样例路径
```bash
/mnt/afs/davidwang/workspace/sana_test_data/cmcc/run_20260827_125654/\
0a00f99d-9d9a-5265-9548-e97a34c1302c/vipe_work_default/pose_artifact_default.json
```

### 3.2 数据结构
```json
{
  "poses_c2w": [...],        // (35, 4, 4) - 相机位姿矩阵
  "intrinsics": [...],       // (35, 1, 4) - 相机内参 [fx, fy, cx, cy]
  "scale_per_frame": [...]   // (35,) - 每帧的尺度因子
}
```

### 3.3 数据质量评估

| 检查项 | 结果 | 数值 |
|--------|------|------|
| **Level 1** | | |
| SO(3) 行列式均值 | ✓ PASS | 1.00000000 |
| SO(3) 正交性误差 | ✓ PASS | < 1e-7 |
| 第一帧对齐 | ✓ PASS | dev = 0.000260 < 0.01 |
| 无 NaN/Inf | ✓ PASS | 所有数据有限 |
| **Level 2** | | |
| FOV 水平 | ✓ PASS | 77.3° ∈ [25°, 120°] |
| FOV 垂直 | ✓ PASS | 50.7° ∈ [25°, 120°] |
| 焦距差异 | ✓ PASS | 0.000 < 0.20 |
| Scale CV | ✓ PASS | 0.0128 < 2.0 |

**结论**: ✅ 该样例数据完全满足所有检测要求，是高质量的标注数据。

### 3.4 字段映射

| 标准字段 | JSON 字段 | 格式 |
|---------|----------|------|
| poses_c2w | `poses_c2w` | (N, 4, 4) list |
| intrinsics | `intrinsics` | (N, 1, 4) list |
| scale | `scale_per_frame` | (N,) list |
| image_wh | **缺失**，从 cx/cy 推断 | 2*(cx, cy) |

**图像尺寸推断逻辑**:
```python
# 从第一帧内参推断
cx = intrinsics[0, 0, 2]  # 640.0
cy = intrinsics[0, 0, 3]  # 360.0
image_wh = (int(cx * 2), int(cy * 2))  # (1280, 720)
# 假设主点在图像中心
```

---

## 四、模块设计

### 4.1 文件结构
```
qc/
├── __init__.py
├── lite_geometric_check.py  # 新增 - 核心模块
└── test_lite_check.py       # 新增 - 单元测试
```

### 4.2 核心数据结构

```python
@dataclass
class GeometryCheckResult:
    """几何质量检测结果"""
    
    # 总体判定
    passed: bool                      # 所有检查是否通过
    level1_passed: bool               # Level 1 是否通过
    level2_passed: bool               # Level 2 是否通过
    
    # Level 1: 数学合法性
    so3_valid: bool                   # SO(3) 是否有效
    so3_det_mean: float               # 行列式均值
    so3_det_std: float                # 行列式标准差
    so3_orth_err: float               # 正交性误差
    
    first_frame_aligned: bool         # 第一帧是否对齐
    first_frame_dev: float            # 与单位矩阵的最大偏差
    
    no_nan_inf: bool                  # 是否无 NaN/Inf
    nan_inf_fields: list[str]         # 包含 NaN/Inf 的字段
    
    # Level 2: 物理合理性
    fov_valid: bool                   # FOV 是否在范围内
    fov_x_min: float                  # 水平 FOV 最小值（度）
    fov_x_max: float                  # 水平 FOV 最大值（度）
    fov_y_min: float                  # 垂直 FOV 最小值（度）
    fov_y_max: float                  # 垂直 FOV 最大值（度）
    
    focal_div_valid: bool             # 焦距差异是否合格
    focal_div_max: float              # 最大焦距差异
    
    scale_cv_valid: bool              # Scale CV 是否合格
    scale_cv: float                   # Scale 变异系数
    
    # 失败原因（仅当 passed=False）
    failure_reasons: list[str]        # 具体失败原因列表
    
    # 元信息
    num_frames: int                   # 总帧数
    image_wh: tuple[int, int]         # 图像尺寸 (W, H)
```

### 4.3 主要函数接口

#### 4.3.1 核心检测函数
```python
def check_pose_geometry(
    poses_c2w: np.ndarray,      # (N, 4, 4) - 相机位姿
    intrinsics: np.ndarray,     # (N, V, 4) 或 (N, 4) - 内参
    scale: np.ndarray,          # (N,) - 尺度因子
    image_wh: tuple[int, int],  # (W, H) - 图像尺寸
) -> GeometryCheckResult:
    """
    执行完整的几何质量检测
    
    Args:
        poses_c2w: 相机到世界坐标系的变换矩阵
        intrinsics: 相机内参 [fx, fy, cx, cy]
        scale: 每帧的米制尺度因子
        image_wh: 图像宽度和高度（像素）
        
    Returns:
        GeometryCheckResult: 详细的检测结果
    """
```

#### 4.3.2 JSON 加载器
```python
def load_pose_from_json(json_path: Path) -> dict:
    """
    从 JSON 文件加载 pose 数据
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        {
            'poses_c2w': np.ndarray (N, 4, 4),
            'intrinsics': np.ndarray (N, V, 4),
            'scale': np.ndarray (N,),
            'image_wh': tuple[int, int],
            'missing_fields': list[str]
        }
        
    支持的字段名变体：
        - poses_c2w / poses / camera_poses / extrinsics
        - intrinsics / intrinsic / K / camera_intrinsics
        - scale_per_frame / scale / scales / metric_scale
    """
```

#### 4.3.3 便捷接口
```python
def check_pose_json(json_path: Path) -> GeometryCheckResult:
    """
    直接检查 JSON 文件
    
    Args:
        json_path: pose_artifact_default.json 路径
        
    Returns:
        GeometryCheckResult: 检测结果
    """
```

### 4.4 内部检测函数

```python
# Level 1 检测
def _check_so3(poses: np.ndarray) -> tuple[bool, float, float, float]
def _check_first_frame(poses: np.ndarray) -> tuple[bool, float]
def _check_nan_inf(poses, intrinsics, scale) -> tuple[bool, list[str]]

# Level 2 检测
def _check_fov(intrinsics: np.ndarray, image_wh: tuple) -> tuple[bool, float, float, float, float]
def _check_focal_divergence(intrinsics: np.ndarray) -> tuple[bool, float]
def _check_scale_cv(scale: np.ndarray) -> tuple[bool, float]

# 辅助函数
def _infer_image_wh(intrinsics: np.ndarray) -> tuple[int, int]
def _normalize_intrinsics_shape(intrinsics: np.ndarray) -> np.ndarray
```

---

## 五、使用方式

### 5.1 Python API

#### 示例 1：检查 JSON 文件
```python
from sana_wm_pipeline.qc.lite_geometric_check import check_pose_json

# 检查单个文件
result = check_pose_json('pose_artifact_default.json')

if result.passed:
    print("✓ 所有检查通过")
else:
    print("✗ 检测失败:")
    for reason in result.failure_reasons:
        print(f"  - {reason}")
```

#### 示例 2：检查 NumPy 数组
```python
from sana_wm_pipeline.qc.lite_geometric_check import check_pose_geometry
import numpy as np

poses = np.load('poses.npy')
intrinsics = np.load('intrinsics.npy')
scale = np.load('scale.npy')

result = check_pose_geometry(
    poses_c2w=poses,
    intrinsics=intrinsics,
    scale=scale,
    image_wh=(1280, 720)
)
```

#### 示例 3：详细报告
```python
result = check_pose_json('pose_artifact_default.json')

print(f"总体: {'PASS' if result.passed else 'FAIL'}")
print(f"\nLevel 1 (数学合法性): {'PASS' if result.level1_passed else 'FAIL'}")
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

if not result.passed:
    print(f"\n失败原因:")
    for reason in result.failure_reasons:
        print(f"  ✗ {reason}")
```

### 5.2 命令行接口（可选）

```bash
# 检查单个文件
python -m sana_wm_pipeline.qc.lite_geometric_check \
    --input pose_artifact_default.json \
    --output report.json

# 输出格式：JSON
{
  "passed": true,
  "level1_passed": true,
  "level2_passed": true,
  "so3_valid": true,
  "so3_det_mean": 1.0,
  ...
}
```

---

## 六、实现计划

### 阶段 1: 核心检测逻辑 ⏱️ 30 分钟
- [x] 样例数据分析（已完成）
- [ ] 实现 `_check_so3()`
- [ ] 实现 `_check_first_frame()`
- [ ] 实现 `_check_nan_inf()`
- [ ] 实现 `_check_fov()`
- [ ] 实现 `_check_focal_divergence()`
- [ ] 实现 `_check_scale_cv()`
- [ ] 实现 `check_pose_geometry()` 主函数
- [ ] 定义 `GeometryCheckResult` 数据类

### 阶段 2: JSON 适配器 ⏱️ 15 分钟
- [ ] 实现 `load_pose_from_json()`
- [ ] 字段名映射和容错处理
- [ ] 图像尺寸推断逻辑
- [ ] 实现 `check_pose_json()` 便捷接口

### 阶段 3: 测试验证 ⏱️ 15 分钟
- [ ] 测试真实样例（pose_artifact_default.json）
- [ ] 编写单元测试（可选）
  - [ ] 正常数据通过测试
  - [ ] SO(3) 失败测试
  - [ ] 第一帧未对齐测试
  - [ ] FOV 超范围测试
  - [ ] 焦距差异过大测试
  - [ ] Scale CV 过大测试

### 阶段 4: 文档和集成 ⏱️ 10 分钟
- [ ] 更新 `__init__.py` 导出新模块
- [ ] 编写使用示例代码
- [ ] 编写 README 章节

---

## 七、测试用例设计

### 7.1 正常数据（应该 PASS）
```python
def test_valid_sample():
    """真实样例应该通过所有检查"""
    result = check_pose_json('pose_artifact_default.json')
    assert result.passed
    assert result.level1_passed
    assert result.level2_passed
```

### 7.2 SO(3) 失败
```python
def test_so3_invalid():
    """行列式 ≠ 1 应该失败"""
    poses = np.eye(4)[None, :, :].repeat(10, axis=0)
    poses[:, :3, :3] *= 2.0  # 缩放 → det = 8
    
    result = check_pose_geometry(poses, intrinsics, scale, (1280, 720))
    assert not result.passed
    assert not result.level1_passed
    assert not result.so3_valid
    assert "SO(3)" in result.failure_reasons[0]
```

### 7.3 第一帧未对齐
```python
def test_first_frame_not_aligned():
    """第一帧 ≠ I 应该失败"""
    poses = np.eye(4)[None, :, :].repeat(10, axis=0)
    poses[0, :3, 3] = [5.0, 0.0, 0.0]  # 平移 5m
    
    result = check_pose_geometry(poses, intrinsics, scale, (1280, 720))
    assert not result.passed
    assert not result.first_frame_aligned
```

### 7.4 FOV 超范围
```python
def test_fov_out_of_range():
    """FOV > 120° 应该失败"""
    intrinsics = np.array([[[100, 100, 640, 360]]])  # 焦距过小
    # fov_x = 2*arctan(1280/(2*100)) = 147° > 120°
    
    result = check_pose_geometry(poses, intrinsics, scale, (1280, 720))
    assert not result.passed
    assert not result.level2_passed
    assert not result.fov_valid
```

### 7.5 焦距差异过大
```python
def test_focal_divergence_high():
    """fx 和 fy 相差 > 20% 应该失败"""
    intrinsics = np.array([[[800, 1000, 640, 360]]])
    # div = |800-1000| / 900 = 0.222 > 0.20
    
    result = check_pose_geometry(poses, intrinsics, scale, (1280, 720))
    assert not result.focal_div_valid
```

### 7.6 Scale CV 过大
```python
def test_scale_cv_high():
    """Scale 变化剧烈应该失败"""
    scale = np.array([0.1, 10.0, 0.1, 10.0, 0.1])  # 震荡
    
    result = check_pose_geometry(poses, intrinsics, scale, (1280, 720))
    assert not result.scale_cv_valid
```

---

## 八、检查项总结表

| 层级 | 检查项 | 数学公式 | 阈值 | 失败原因 | 作用 |
|------|--------|---------|------|---------|------|
| **Level 1** | | | | | |
| 1.1 | SO(3) 行列式 | det(R) | \|mean-1\| ≤ 1e-3 | COLMAP失败、数值误差 | 旋转矩阵合法性 |
| 1.2 | SO(3) 正交性 | max\|R·R^T - I\| | ≤ 1e-3 | 数值污染、非正交 | 旋转矩阵合法性 |
| 1.3 | 第一帧对齐 | max\|pose[0] - I\| | ≤ 0.01 | 未归一化 | 统一坐标系 |
| 1.4 | 无 NaN/Inf | isfinite(·) | all True | 除零、溢出 | 数值稳定性 |
| **Level 2** | | | | | |
| 2.1 | 水平 FOV | 2·arctan(W/2fx) | 25° - 120° | 焦距标注错误 | 相机物理合理性 |
| 2.2 | 垂直 FOV | 2·arctan(H/2fy) | 25° - 120° | 焦距标注错误 | 相机物理合理性 |
| 2.3 | 焦距差异 | \|fx-fy\|/(fx+fy)·2 | ≤ 0.20 | fx/fy 互换 | 检测标注错误 |
| 2.4 | Scale CV | std(s)/(mean(s)+ε) | ≤ 2.0 | 拼接片段、重建不稳定 | 尺度一致性 |

---

## 九、输出格式示例

### 9.1 PASS 案例
```json
{
  "passed": true,
  "level1_passed": true,
  "level2_passed": true,
  "so3_valid": true,
  "so3_det_mean": 1.0,
  "so3_det_std": 3e-08,
  "so3_orth_err": 0.0,
  "first_frame_aligned": true,
  "first_frame_dev": 0.00026,
  "no_nan_inf": true,
  "nan_inf_fields": [],
  "fov_valid": true,
  "fov_x_min": 77.3,
  "fov_x_max": 77.3,
  "fov_y_min": 50.7,
  "fov_y_max": 50.7,
  "focal_div_valid": true,
  "focal_div_max": 0.0,
  "scale_cv_valid": true,
  "scale_cv": 0.0128,
  "failure_reasons": [],
  "num_frames": 35,
  "image_wh": [1280, 720]
}
```

### 9.2 FAIL 案例
```json
{
  "passed": false,
  "level1_passed": false,
  "level2_passed": true,
  "so3_valid": false,
  "so3_det_mean": 0.857142,
  "so3_det_std": 0.12,
  "so3_orth_err": 0.234,
  "first_frame_aligned": true,
  "first_frame_dev": 0.0023,
  "no_nan_inf": true,
  "nan_inf_fields": [],
  "fov_valid": true,
  "fov_x_min": 65.2,
  "fov_x_max": 78.5,
  "fov_y_min": 42.1,
  "fov_y_max": 50.3,
  "focal_div_valid": true,
  "focal_div_max": 0.015,
  "scale_cv_valid": true,
  "scale_cv": 0.087,
  "failure_reasons": [
    "SO(3) invalid: det_mean=0.857142 (expected 1.0 ± 0.001)",
    "SO(3) invalid: orth_err=2.34e-1 (expected ≤ 0.001)"
  ],
  "num_frames": 35,
  "image_wh": [1280, 720]
}
```

---

## 十、依赖项

### 必需依赖
- `numpy` >= 1.20

### 可选依赖
- 无（完全独立模块）

---

## 十一、变更历史

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v1.0 | 2026-08-31 | 初始版本，移除轨迹跳跃检测 |

---

## 十二、待办事项

- [ ] 实现核心检测模块
- [ ] 实现 JSON 加载器
- [ ] 测试真实样例
- [ ] 编写单元测试（可选）
- [ ] 更新 `__init__.py`
- [ ] 编写使用文档

---

**文档维护者**: Claude  
**审阅状态**: 待审阅
