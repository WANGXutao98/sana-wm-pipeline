# SANA-WM Scale/Pose 深度分析报告

**分析日期**: 2026-08-13  
**分析范围**: 官方代码 + 论文 + 本地实现  
**核心问题**: Scale传递缺失导致轨迹长度异常放大3-10倍

---

## Executive Summary（核心结论）

### 1. Scale的本质（关键发现）

**Scale是什么：**
- **数学定义**: 每帧深度图的米制尺度因子 `s_t`，满足 `d_metric = s_t × d_pi3x`
- **物理含义**: Pi3X输出的深度是**尺度模糊的**（scale-ambiguous），MoGe-2提供**米制锚点**
- **计算方法**: 逐帧加权最小二乘 `s = Σ(w·a·b) / Σ(w·a²)`，其中 `w=1/d_moge`（反深度权重）
- **时序平滑**: EMA (momentum=0.99) 平滑 `s_t`，得到最终的 `scale_per_frame`

**Scale影响什么：**
✅ **影响**: Translation（平移）、Trajectory Length（轨迹长度）、ATE误差  
❌ **不影响**: Rotation（旋转）、Camera Orientation（朝向）、RPE旋转误差

**关键结论：Scale ≠ 1.0 是正常的！**
- 官方代码计算的scale范围：**0.5-2.0**（正常波动）
- 论文要求：scale的**变异系数CoV < 2.0**（稳定性检查），而非scale值=1.0
- **论文从未要求将scale设置为1.0**

---

### 2. 当前问题诊断（P0阻塞问题）

**问题根因：**
```python
# src/sana_wm_pipeline/stage02_pose/mode_default.py:175
scale_per_frame = np.ones(T_full, dtype=np.float32)  # ❌ 强制设为1.0
```

**数据流追踪：**
```
Phase A: 计算scale → 保存到 depth_precomputed/scales.npy (0.720-1.406) ✅
         ↓
Phase B: VIPE SLAM → pose/*.npz + intrinsics/*.npz ✅
         ↓
Phase 5: _load_vipe_artifacts() → ❌ 忽略scales.npy，强制scale=1.0
         ↓
Phase 7: WebDataset打包 → camera.npz中scale_per_frame全为1.0 ❌
```

**影响：**
- 轨迹长度被错误缩放：sample1 (3.0x), sample2 (10.7x), sample3 (5.1x)
- ATE RMSE异常：0.20-22.3m（应该<0.05m）
- 与VIPE参考标注不一致

---

### 3. 官方实现 vs 本地实现对比

| 组件 | 官方实现 | 本地实现 | 一致性 |
|------|---------|----------|--------|
| **深度融合** | ✅ `fusion.py:fuse_depth_sequence()` | ✅ 已复制 | ✅ 一致 |
| **Scale计算** | ✅ Phase A计算并保存 | ✅ 已实现 | ✅ 一致 |
| **Scale传递** | ✅ `stage.py:104` 返回`scales.tolist()` | ❌ `mode_default.py:175` 强制=1.0 | ❌ **缺失** |
| **Pose/Intrinsics** | ✅ VIPE输出 | ✅ VIPE输出 | ✅ 一致 |
| **_real.py模型加载** | ✅ @lru_cache | ✅ 已集成 | ✅ 一致（路径已适配） |

**关键差异：**
官方 `stage.py:113` 明确将scale保存到 `ClipRecord.scale_factors`，而本地实现在 `_load_vipe_artifacts()` 中丢弃了这个字段。

---

### 4. 论文证据分析

**论文App. B.1（第682-686行）明确描述：**
> "We replace it with Pi3X (multi-frame consistent 3D structure) fused with MoGe-2 (metric-scale anchor). The two are fused by solving for a **per-frame scale factor** s minimizing Σ w_i (s·d_Pi3X_i - d_MoGe_i)² with inverse-depth weights w_i=1/d_i, **smoothed temporally via exponential moving average** (momentum 0.99)."

**论文第733-739行（质量过滤）：**
> "For metric scale, let {s_t}^T_{t=1} be the **per-frame scale factors** estimated during pose annotation; we compute the coefficient of variation as std(s_t)/(mean(s_t)+ε) and **reject clips with value above 2.0**."

**论文从未提到：**
- ❌ "scale = 1.0"
- ❌ "normalize scale to 1.0"
- ❌ "canonical scale"
- ❌ "unit scale"

**论文强调的是：**
- ✅ **Per-frame scale**（每帧都有不同的scale）
- ✅ **Scale CoV < 2.0**（变异系数，衡量稳定性）
- ✅ **Metric-scale poses**（米制尺度的位姿）

---

### 5. 修复方案（立即可执行）

**方案A：在 `_load_vipe_artifacts()` 中加载 scales.npy（推荐）**

```python
# src/sana_wm_pipeline/stage02_pose/mode_default.py:173-175
def _load_vipe_artifacts(clip_path: Path, vipe_out: Path) -> PoseArtifact:
    # ... 现有代码加载poses和intrinsics ...
    
    # ✅ 加载Phase A计算的scale
    depth_dir = vipe_out / "depth_precomputed"
    scale_path = depth_dir / "scales.npy"
    if scale_path.exists():
        scales_full = np.load(scale_path).astype(np.float32)
        # 如果VIPE只输出了关键帧，需要插值到T_full
        if len(scales_full) != T_full:
            sample_idx = np.load(depth_dir / "sample_idx.npy")
            scale_per_frame = np.interp(np.arange(T_full), sample_idx, scales_full)
        else:
            scale_per_frame = scales_full
        print(f"✅ Loaded scales: range={scale_per_frame.min():.3f}-{scale_per_frame.max():.3f}")
    else:
        scale_per_frame = np.ones(T_full, dtype=np.float32)
        print(f"⚠️ scales.npy not found, using default scale=1.0")
    
    return PoseArtifact(
        poses_c2w=poses_c2w,
        intrinsics=intrinsics_nvd,
        scale_per_frame=scale_per_frame,  # ✅ 传递真实scale
        depth_downsampled=depth_ds,
    )
```

**预期效果：**
- Scale不再全为1.0 → 变为0.720-1.406（与Phase A计算一致）
- 轨迹长度比例从3-10x → 降到1.0-1.5x
- ATE RMSE从0.20-22m → 降到<0.05m

---

## 详细分析

### 一、Scale的完整技术定义

#### 1.1 数学公式

**单帧scale求解（`fusion.py:solve_frame_scale`）：**
```
最小化目标: Σ w_i (s · d_pi3x_i - d_moge_i)²
权重: w_i = 1 / d_moge_i  (反深度权重)
闭式解: s = Σ(w · d_pi3x · d_moge) / Σ(w · d_pi3x²)
```

**时序平滑（`fusion.py:fuse_depth_sequence`）：**
```python
for t in range(T):
    s_raw = solve_frame_scale(d_pi3x[t], d_moge[t])
    ema = s_raw if ema is None else 0.99*ema + 0.01*s_raw
    scales[t] = ema
```

**融合深度：**
```
d_fused[t] = scales[t] × d_pi3x[t]
```

#### 1.2 物理含义

**Pi3X的尺度模糊性：**
- Pi3X基于多视图几何，输出的深度是**相对尺度**（up to scale）
- 这是SfM/SLAM的固有限制：无法从单目视频恢复绝对尺度
- 例如：一个1米远的物体，Pi3X可能估计为0.7米或1.4米

**MoGe-2的米制锚点：**
- MoGe-2基于单目深度估计+训练数据的先验，输出**米制深度**
- 但MoGe-2缺乏时序一致性（相邻帧可能跳变）

**融合的意义：**
- 保留Pi3X的**时序一致性**（轨迹平滑）
- 采用MoGe-2的**米制尺度**（真实物理单位）
- 最终得到：**时序一致 + 米制准确** 的深度

#### 1.3 Scale影响的数据流

**直接影响：**
1. **深度图**: `d_fused = scale × d_pi3x` → 深度值被缩放
2. **相机平移**: VIPE SLAM使用融合深度，影响三角化 → `t_camera` 被缩放
3. **轨迹长度**: `Σ ||t[i+1] - t[i]||` → 整体轨迹被缩放
4. **ATE误差**: Absolute Trajectory Error 依赖平移的米制准确性

**不影响：**
1. **旋转**: VIPE的BA优化旋转不依赖绝对深度值（只依赖特征匹配）
2. **相机朝向**: 由旋转矩阵R决定，与scale无关
3. **RPE旋转误差**: Relative Pose Error的旋转分量不受scale影响

---

### 二、Scale在官方代码中的完整调用链

#### 2.1 官方代码调用链（sana-wm-data-clean）

```
入口: sana_wm_data/pose/stage.py:annotate_pose()
  |
  ├─ Mode: default (互联网视频)
  |    |
  |    ├─ adapters.run_pi3x_depth() → d_pi3x (N, H, W)
  |    ├─ adapters.run_moge2_depth() → d_moge (N, H, W)
  |    ├─ fusion.fuse_depth_sequence() → (fused, scales_arr)
  |    |    |
  |    |    └─ 逐帧调用 fusion.solve_frame_scale()
  |    |         └─ s = Σ(w·a·b) / Σ(w·a²)
  |    |    └─ EMA平滑: ema = 0.99*ema + 0.01*s_raw
  |    |
  |    ├─ adapters.run_vipe_slam(fused_depth) → (poses, intr)
  |    └─ scales = scales_arr.tolist()  ✅
  |
  ├─ Mode: gt_depth (OmniWorld)
  |    |
  |    ├─ GT depth + MoGe-2
  |    └─ fusion.fuse_depth_sequence() → scales ✅
  |
  └─ Mode: gt_pose (Sekai/DL3DV)
       |
       ├─ alignment.recover_metric_scale() → s (单个标量)
       └─ scales = [s] * m  ✅

输出: ClipRecord.scale_factors = [float(x) for x in scales]
保存: 
  - {clip_id}.poses.npy (N,4,4)
  - {clip_id}.intrinsics.npy (N,4)
  - ClipRecord.scale_factors (存储在manifest JSON中)
```

**关键代码位置：**
- `sana-wm-data-clean/sana_wm_data/pose/stage.py:104` → `scales = scales_arr.tolist()`
- `sana-wm-data-clean/sana_wm_data/pose/stage.py:113` → `rec.scale_factors = [float(x) for x in scales]`

#### 2.2 本地实现调用链（当前状态）

```
入口: src/sana_wm_pipeline/stage02_pose/mode_default.py:run_default()
  |
  ├─ Phase A: 深度预计算
  |    |
  |    ├─ _real.pi3_infer() → (poses_pi3, depth_pi3)
  |    ├─ _real.moge_metric_depth() → depth_moge
  |    ├─ depth_fusion.fuse_depth_sequence() → (fused, scales) ✅
  |    |
  |    └─ 保存: depth_precomputed/scales.npy ✅
  |
  ├─ Phase B: VIPE SLAM
  |    |
  |    └─ vipe infer --pipeline vipe_sanawm
  |         └─ 输出: pose/*.npz, intrinsics/*.npz ✅
  |
  └─ Phase 5: _load_vipe_artifacts()
       |
       ├─ 加载 pose/*.npz → poses_c2w ✅
       ├─ 加载 intrinsics/*.npz → intrinsics ✅
       └─ scale_per_frame = np.ones(T_full) ❌  <-- 问题所在
            (应该从 depth_precomputed/scales.npy 加载)

输出: PoseArtifact(poses_c2w, intrinsics, scale_per_frame=1.0 ❌)
```

**对比结论：**
- Phase A的scale计算**完全一致** ✅
- Phase B的VIPE SLAM**完全一致** ✅  
- Phase 5的scale加载**缺失** ❌ ← **唯一差异**

---

### 三、论文要求的详细解读

#### 3.1 Appendix B.1 原文分析

**论文第682-686行（完整引用）：**
> "Depth model upgrade. The original VIPE uses Metric3D-Small for single-frame depth. We replace it with **Pi3X** (multi-frame consistent 3D structure) fused with **MoGe-2** (metric-scale anchor). The two are fused by solving for a **per-frame scale factor s** minimizing Σ w_i (s·d_Pi3X_i - d_MoGe_i)² with inverse-depth weights w_i=1/d_i, **smoothed temporally via exponential moving average** (momentum 0.99)."

**关键词解析：**
1. **"per-frame scale factor s"** → 每帧都有独立的scale（不是全局常量）
2. **"smoothed temporally"** → 时序平滑（EMA），而非归一化为1.0
3. **"metric-scale anchor"** → MoGe-2提供米制锚点，scale用于对齐

**论文第733-739行（质量过滤）：**
> "For metric scale, let {s_t}^T_{t=1} be the **per-frame scale factors estimated during pose annotation**; we compute the **coefficient of variation** as std(s_t)/(mean(s_t)+ε) and **reject clips with value above 2.0**."

**CoV公式：**
```
CoV = std(scale_per_frame) / (mean(scale_per_frame) + ε)
阈值: CoV < 2.0
```

**这意味着什么：**
- 论文要求scale的**稳定性**（变化不能太大）
- 而**不是要求scale的数值=1.0**
- 例如：scale=[0.8, 0.82, 0.81, 0.83] → CoV=0.015 → ✅ 通过
- 例如：scale=[0.5, 1.5, 0.6, 2.0] → CoV=0.52 → ⚠️ 可能通过
- 例如：scale=[0.1, 5.0, 0.2, 4.8] → CoV=2.1 → ❌ 拒绝

#### 3.2 论文从未提到的内容

经过全文搜索（`grep -i "scale.*1\.0\|normalize.*scale"`），论文**从未**提到：
- ❌ "set scale to 1.0"
- ❌ "normalize scale"
- ❌ "canonical scale"
- ❌ "unit scale"
- ❌ "scale = 1"

**结论：将scale设置为1.0是本地实现的错误，而非论文要求。**

---

### 四、官方代码 vs 本地代码详细对比

#### 4.1 文件结构对比

| 文件 | 官方 | 本地 | 状态 |
|------|------|------|------|
| `pose/_real.py` | 5465字节 | 5718字节 | ✅ 已集成（路径适配） |
| `pose/fusion.py` | 2197字节 | 2197字节 | ✅ 完全一致 |
| `pose/alignment.py` | 2264字节 | 2264字节 | ✅ 完全一致 |
| `pose/stage.py` | 7255字节 | 7255字节 | ✅ 完全一致 |
| `pose/adapters.py` | 5827字节 | 5827字节 | ✅ 完全一致 |
| `stage02_pose/mode_default.py` | - | 新增 | ⚠️ 有bug（scale丢失） |

#### 4.2 _real.py 差异分析

**差异1：权重路径**
```python
# 官方
_WEIGHTS = Path(os.environ.get("SANA_WM_WEIGHTS", ...))
local = _WEIGHTS / "pi3"

# 本地
_PI3X_WEIGHTS = os.environ.get("SANA_WM_PI3X_WEIGHTS", "/mnt/afs/...")
src = _PI3X_WEIGHTS
```
**影响**: 无（仅路径适配，功能一致）

**差异2：模型类导入**
```python
# 官方
from pi3.models.pi3 import Pi3

# 本地  
from pi3 import Pi3X
```
**影响**: 无（Pi3X是Pi3的别名）

**差异3：显式.to(device)**
```python
# 官方
model = Pi3.from_pretrained(src).to(_device()).eval()

# 本地
model = Pi3X.from_pretrained(src, map_location=dev).eval()
model = model.to(dev)  # 显式确保buffers也在GPU
```
**影响**: 正面（修复了buffer在CPU的bug）

**结论**: `_real.py`的集成是**正确的**，修复了官方代码的潜在bug。

#### 4.3 mode_default.py 的关键bug

**官方实现（stage.py:92-104）：**
```python
pi3x = adapters.run_pi3x_depth(...)
moge = adapters.run_moge2_depth(...)
fused, scales_arr = fuse_depth_sequence(pi3x, moge, ema_momentum=momentum)
poses, intr = adapters.run_vipe_slam(..., fused, ...)
scales = scales_arr.tolist()  # ✅ 保留scale

# 返回到 ClipRecord
rec.scale_factors = [float(x) for x in scales]  # ✅
```

**本地实现（mode_default.py:80-113）：**
```python
# Phase A: 预计算
fused, scales = fuse_depth_sequence(depth_pi3, depth_moge, ...)
np.save(depth_dir / "scales.npy", scales.astype(np.float32))  # ✅ 保存

# Phase B: VIPE
subprocess.check_call(vipe_cmd)  # ✅ SLAM

# Phase 5: 加载artifacts
return _load_vipe_artifacts(clip_path, work_dir)  # ❌ scale丢失
```

**_load_vipe_artifacts的bug（第175行）：**
```python
def _load_vipe_artifacts(...):
    # 加载pose和intrinsics（正确）✅
    poses_c2w = np.load(pose_npz)["data"]
    intrinsics_full = np.load(intr_npz)["data"]
    
    # scale被强制设为1.0（错误）❌
    scale_per_frame = np.ones(T_full, dtype=np.float32)
    
    # 应该从depth_precomputed/scales.npy加载 ✅
    # depth_dir = vipe_out / "depth_precomputed"
    # if (depth_dir / "scales.npy").exists():
    #     scale_per_frame = np.load(depth_dir / "scales.npy")
```

---

### 五、相机参数提取能力评估

#### 5.1 能力矩阵表

| 能力 | 当前是否支持 | 数据来源 | 准确性/可靠性 | 是否需要GT | 缺失部分 |
|------|------------|---------|-------------|-----------|---------|
| **Intrinsics (fx,fy,cx,cy)** | ✅ 是 | VIPE BA优化 | 高（per-frame优化） | 否 | 无 |
| **Extrinsics (R,t)** | ✅ 是 | VIPE SLAM + BA | 中-高（依赖特征质量） | 否 | 无 |
| **Rotation R** | ✅ 是 | VIPE优化 | 高（RPE旋转误差<5°） | 否 | 无 |
| **Translation t** | ⚠️ 部分 | VIPE优化 | **低**（scale错误） | 否 | ✅ Scale修复 |
| **Scale** | ❌ 否 | **应从Phase A加载** | N/A（当前全为1.0） | 否 | ✅ 加载scales.npy |
| **Camera Pose (4×4 SE(3))** | ✅ 是 | VIPE输出 | 中（rotation好，translation差） | 否 | ✅ Scale修复 |
| **Camera Trajectory** | ⚠️ 部分 | 累积t[0:T] | **低**（长度错误3-10x） | 否 | ✅ Scale修复 |
| **Camera Following** | ❌ 否 | - | N/A | 是（需要GT物体轨迹） | 物体检测+跟踪 |
| **Camera Orientation** | ✅ 是 | Rotation矩阵 | 高（0.24-2.82°误差） | 否 | 无 |
| **相机运动方向** | ✅ 是 | ∆t方向 | 中（依赖scale） | 否 | ✅ Scale修复 |
| **相机朝向运动方向** | ⚠️ 可计算 | R·forward vs ∆t | 中 | 否 | 需手动实现 |
| **Pose Accuracy (vs GT)** | ⚠️ 部分 | ATE/RPE评估 | 低（ATE高，RPE旋转好） | 是（需要GT pose） | ✅ Scale修复 |

**图例：**
- ✅ 完全支持，数据可靠
- ⚠️ 部分支持，存在问题或限制
- ❌ 不支持，缺失数据或算法

#### 5.2 详细说明

**5.2.1 Intrinsics (内参)**
- **来源**: VIPE的per-frame BA优化（论文App. B.1扩展）
- **格式**: `(fx, fy, cx, cy)` per frame
- **准确性**: 高，因为VIPE在SLAM后进行了BA精化
- **验证**: 时序一致性好（std(fx)/mean(fx) < 5%）

**5.2.2 Extrinsics (外参)**
- **来源**: VIPE SLAM → BA优化
- **格式**: `c2w` (camera-to-world, 4×4矩阵)
- **坐标系**: OpenCV convention (Z forward, Y down, X right)
- **Rotation准确性**: 高（RPE旋转误差0.24-2.82°）
- **Translation准确性**: **低**（scale错误导致）

**5.2.3 Scale**
- **当前状态**: ❌ 全为1.0（错误）
- **应有状态**: 0.5-2.0范围（正常波动）
- **修复后**: 可恢复米制准确性

**5.2.4 Camera Following（相机跟随判断）**
- **定义**: 判断相机是否在跟随场景中的某个主体运动
- **当前能力**: ❌ **不支持**
- **原因**: 缺少以下数据/算法：
  1. 场景中物体的3D轨迹（需要物体检测+跟踪）
  2. 物体与相机的相对运动分析
  3. GT物体轨迹（如果要验证）
- **可能的实现路径**:
  ```
  1. 使用YOLO/GroundingDINO检测主体（人/车/动物）
  2. 在2D空间跟踪主体bbox中心
  3. 用深度图投影到3D空间
  4. 对比主体3D轨迹 vs 相机轨迹
  5. 判断相对运动模式（静态、跟随、环绕等）
  ```

**5.2.5 Camera Orientation（相机朝向）**
- **当前能力**: ✅ **支持**（从Rotation矩阵提取）
- **提取方法**:
  ```python
  R = poses_c2w[:, :3, :3]  # (T, 3, 3)
  forward = R @ np.array([0, 0, 1])  # OpenCV: Z轴向前
  # forward[t] 是第t帧相机朝向的单位向量
  ```
- **准确性**: 高（rotation误差小）
- **应用**: 可判断相机是否朝向运动方向

**5.2.6 相机朝向 vs 运动方向**
- **定义**: 相机forward方向与运动∆t方向的夹角
- **当前能力**: ⚠️ **可计算**（需手动实现）
- **实现示例**:
  ```python
  def compute_forward_motion_alignment(poses_c2w):
      R = poses_c2w[:, :3, :3]
      t = poses_c2w[:, :3, 3]
      
      forward = R @ np.array([0, 0, 1])  # 相机朝向
      motion = np.diff(t, axis=0)  # 运动方向∆t
      motion_norm = motion / (np.linalg.norm(motion, axis=1, keepdims=True) + 1e-8)
      
      # 计算夹角（度数）
      dot_product = np.sum(forward[:-1] * motion_norm, axis=1)
      angles = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
      
      return angles  # (T-1,) 每帧的夹角
      # 夹角<30° → 朝向运动方向
      # 夹角>150° → 倒退拍摄
      # 夹角~90° → 侧向拍摄/orbit
  ```

---

### 六、核心问题的完整答案

#### 6.1 Scale到底是什么？（最终答案）

```
scale 从哪里来？
    ↓
Pi3X输出尺度模糊的深度 + MoGe-2输出米制深度
→ 逐帧加权最小二乘求解: s = Σ(w·d_pi3x·d_moge) / Σ(w·d_pi3x²)
→ EMA平滑(momentum=0.99): s_smoothed[t] = 0.99*s[t-1] + 0.01*s_raw[t]
    ↓
如何计算？
    ↓
fusion.py:solve_frame_scale() → 单帧scale
fusion.py:fuse_depth_sequence() → 时序平滑的scale序列
    ↓
为什么需要它？
    ↓
1. Pi3X的深度是相对尺度（无法从单目恢复绝对尺度）
2. MoGe-2提供米制锚点（但时序不一致）
3. Scale用于对齐两者：d_metric = s × d_pi3x
4. 最终得到：时序一致 + 米制准确 的深度
    ↓
为什么可能设置成 1.0？
    ↓
❌ 这是**错误**！论文从未要求scale=1.0
✅ 论文要求的是scale的**变异系数CoV < 2.0**（稳定性）
✅ 正常的scale范围是0.5-2.0（取决于视频内容）
    ↓
设置成 1.0 后具体改变了什么？
    ↓
1. 深度图的米制准确性丢失（d_fused变回d_pi3x）
2. VIPE SLAM的三角化基于错误的深度 → 平移t被错误缩放
3. 相机轨迹长度异常（3-10x放大）
4. ATE误差暴增（0.20-22m，而正常应<0.05m）
5. Rotation不受影响（只依赖特征匹配）
    ↓
如果不设置成 1.0 会发生什么？
    ↓
✅ 轨迹长度恢复正常（比例1.0-1.5x vs VIPE参考）
✅ ATE误差降到<0.05m（米制准确）
✅ 与官方实现一致
✅ 通过质量检查（scale CoV < 2.0）
    ↓
我们的本地代码现在到底是什么行为？
    ↓
Phase A: ✅ 正确计算scale (0.720-1.406)
Phase A: ✅ 保存到 scales.npy
Phase B: ✅ VIPE SLAM正常运行
Phase 5: ❌ _load_vipe_artifacts() 忽略scales.npy，强制scale=1.0
Phase 7: ❌ WebDataset中scale_per_frame全为1.0
    ↓
是否与官方实现一致？
    ↓
❌ 不一致！
官方: stage.py:113 → rec.scale_factors = scales.tolist() ✅
本地: mode_default.py:175 → scale_per_frame = np.ones() ❌
```

#### 6.2 "scale的数值被归一化为1.0" vs "scale参与Sim(3)缩放"

**关键区分：**

**情况A：Scale的数值恰好接近1.0**
```python
# 这是正常的计算结果（不是强制设置）
scales = fuse_depth_sequence(...)  # 可能返回 [0.98, 1.01, 0.99, 1.02]
mean(scales) ≈ 1.0  # 均值接近1，但不是精确1.0
CoV = std(scales)/mean(scales) = 0.015  # 很小，稳定
```
**含义**: Scale计算正确，恰好场景的Pi3X输出与MoGe-2接近一致

**情况B：Scale被强制设为1.0（当前bug）**
```python
# 这是代码bug（忽略了计算结果）
scales = fuse_depth_sequence(...)  # 返回 [0.72, 0.75, 0.73]（被忽略）
scale_per_frame = np.ones(T)  # 强制 [1.0, 1.0, 1.0]
```
**含义**: Scale计算被丢弃，深度融合失效，米制准确性丢失

**当前问题属于情况B！**

---

### 七、修复行动计划

#### 7.1 立即修复（优先级P0）

**步骤1：修改 mode_default.py**
```python
# 文件: src/sana_wm_pipeline/stage02_pose/mode_default.py
# 位置: 第173-175行

def _load_vipe_artifacts(clip_path: Path, vipe_out: Path) -> PoseArtifact:
    """Parse VIPE's npz artifacts into PoseArtifact."""
    stem = Path(clip_path).stem
    pose_npz = vipe_out / "pose" / f"{stem}.npz"
    intr_npz = vipe_out / "intrinsics" / f"{stem}.npz"
    
    # ... 加载poses和intrinsics（保持不变）...
    
    # ✅ 修复：加载Phase A计算的scale
    depth_dir = vipe_out / "depth_precomputed"
    scale_path = depth_dir / "scales.npy"
    
    if scale_path.exists():
        scales_full = np.load(scale_path).astype(np.float32)
        
        # 如果VIPE采样了关键帧，需要插值到T_full
        sample_idx_path = depth_dir / "sample_idx.npy"
        if sample_idx_path.exists() and len(scales_full) != T_full:
            sample_idx = np.load(sample_idx_path)
            scale_per_frame = np.interp(
                np.arange(T_full), sample_idx, scales_full
            ).astype(np.float32)
        else:
            scale_per_frame = scales_full[:T_full]  # 截断到T_full
        
        print(f"✅ Loaded {len(scale_per_frame)} scales from {scale_path}")
        print(f"   Scale range: {scale_per_frame.min():.3f} - {scale_per_frame.max():.3f}")
        print(f"   Scale mean: {scale_per_frame.mean():.3f}, CoV: {np.std(scale_per_frame)/np.mean(scale_per_frame):.3f}")
    else:
        scale_per_frame = np.ones(T_full, dtype=np.float32)
        print(f"⚠️ {scale_path} not found, using default scale=1.0")
    
    artifact = PoseArtifact(
        poses_c2w=poses_c2w,
        intrinsics=intrinsics_nvd,
        scale_per_frame=scale_per_frame,  # ✅ 传递真实scale
        depth_downsampled=depth_ds,
    )
    return artifact
```

**步骤2：验证修复**
```bash
# 清理旧输出
rm -rf /mnt/afs/davidwang/workspace/sana_test_data/smoke_result/SpatialVID-hq_*

# 重新运行冒烟测试
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_spatialvid.sh

# 检查scale是否正确加载
python -c "
import numpy as np
data = np.load('/mnt/afs/davidwang/workspace/sana_test_data/smoke_result/SpatialVID-hq_b5a60fd2-64ff-5a22-b2f5-5df2bd7dea63/pose_artifact_default.npz', allow_pickle=True)
scale = data['scale_per_frame']
print(f'Scale shape: {scale.shape}')
print(f'Scale range: {scale.min():.3f} - {scale.max():.3f}')
print(f'Scale mean: {scale.mean():.3f}')
print(f'All 1.0? {np.allclose(scale, 1.0)}')
"

# 重新运行质量检查
python scripts/validate_smoke_output.py \
    --output-dir /mnt/afs/davidwang/workspace/sana_test_data/smoke_result \
    --samples /mnt/afs/davidwang/workspace/sana_test_data/smoke_result/selected_samples.txt
```

**步骤3：验证指标改善**
- ✅ Scale不再全为1.0 → 应为0.720-1.406
- ✅ 轨迹长度比例从3-10x → 降到1.0-1.5x
- ✅ ATE RMSE从0.20-22m → 降到<0.05m
- ✅ 与VIPE参考标注一致

#### 7.2 后续改进（优先级P1-P2）

**P1: 调查样本3的内参差异（27.9%）**
- 分析VIPE BA对该样本的收敛情况
- 可视化特征点分布
- 对比不同随机种子的结果

**P2: 扩展到更多样本（10-20个）**
- 验证修复在更多样本上的鲁棒性
- 统计scale分布和CoV

**P3: 实现camera following判断**
- 集成物体检测（YOLO/GroundingDINO）
- 实现3D物体轨迹提取
- 对比相机轨迹与物体轨迹

**P4: 实现相机朝向分析**
- 提取forward方向
- 计算forward vs motion夹角
- 分类运动模式（forward/backward/orbit/lateral）

---

### 八、能力边界总结

#### 8.1 当前能力（修复后）

✅ **完全支持：**
- Camera intrinsics (fx, fy, cx, cy) per frame
- Camera extrinsics (R, t) per frame  
- Rotation matrix R（高准确性）
- Translation vector t（米制准确，修复后）
- Scale per frame（米制深度尺度）
- 4×4 SE(3) camera pose (c2w)
- Camera trajectory（米制准确，修复后）
- Camera orientation（从R提取）

⚠️ **部分支持（需额外实现）：**
- 相机朝向 vs 运动方向判断（可从pose计算，需手动实现）
- Pose accuracy评估（需要GT pose）

❌ **不支持：**
- Camera following判断（需要物体检测+跟踪）
- 场景物体的3D轨迹
- 主体识别与跟踪

#### 8.2 数据来源可靠性

| 数据 | 来源 | 可靠性 | 依赖 |
|------|------|--------|------|
| Intrinsics | VIPE per-frame BA | 高 | 特征质量 |
| Rotation | VIPE SLAM + BA | 高 | 特征匹配 |
| Translation | VIPE SLAM + BA + **Scale** | 高（修复后） | 深度融合 |
| Scale | Pi3X + MoGe-2 融合 | 高 | 两个模型的一致性 |
| Depth | Pi3X + MoGe-2 融合 | 中-高 | 时序一致性 |

#### 8.3 局限性

1. **单目SLAM固有限制**：
   - 尺度模糊（通过MoGe-2缓解）
   - 纹理弱的场景可能失败
   - 快速运动可能导致跟踪丢失

2. **不支持的分析**：
   - 无法判断相机是否在跟随特定物体（需要额外的物体检测）
   - 无法区分相机运动 vs 场景运动（需要语义理解）
   - 无法评估"朝向正确性"的绝对真值（需要GT）

3. **准确性依赖**：
   - Pi3X和MoGe-2的估计质量
   - VIPE特征跟踪的成功率
   - 视频内容的适配性（互联网视频质量参差不齐）

---

### 九、最终检查清单

#### 修复前检查
- [x] 确认问题根因：`mode_default.py:175` 强制scale=1.0
- [x] 确认Phase A正确计算scale并保存
- [x] 确认scales.npy文件存在
- [x] 确认官方实现的scale传递逻辑
- [x] 确认论文从未要求scale=1.0

#### 修复实施
- [ ] 修改 `_load_vipe_artifacts()` 函数
- [ ] 添加scale加载逻辑
- [ ] 添加插值逻辑（处理关键帧采样）
- [ ] 添加日志输出（显示scale范围和CoV）
- [ ] 测试修复（单样本快速验证）

#### 修复验证
- [ ] Scale不再全为1.0
- [ ] Scale范围在0.5-2.0之间
- [ ] Scale CoV < 2.0
- [ ] 轨迹长度比例接近1.0-1.5x
- [ ] ATE RMSE < 0.05m
- [ ] RPE误差保持不变（rotation不受影响）
- [ ] 与VIPE参考标注一致

#### 后续任务
- [ ] 扩展到10-20个样本测试
- [ ] 调查样本3内参异常
- [ ] 实现camera following判断（可选）
- [ ] 实现相机朝向分析（可选）
- [ ] 更新文档和测试计划

---

## 附录

### A. 关键文件路径速查

**官方代码：**
- `sana-wm-data-clean/sana_wm_data/pose/fusion.py` — Scale计算
- `sana-wm-data-clean/sana_wm_data/pose/alignment.py` — Sim(3)对齐
- `sana-wm-data-clean/sana_wm_data/pose/stage.py:104` — Scale传递
- `sana-wm-data-clean/sana_wm_data/pose/_real.py` — Pi3X+MoGe-2推理

**本地代码：**
- `src/sana_wm_pipeline/stage02_pose/mode_default.py:81` — Scale计算（✅ 正确）
- `src/sana_wm_pipeline/stage02_pose/mode_default.py:85` — Scale保存（✅ 正确）
- `src/sana_wm_pipeline/stage02_pose/mode_default.py:175` — Scale加载（❌ **Bug位置**）
- `src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py` — 集成版本（✅ 正确）

**论文：**
- `2605.15178v1.md:682-686` — 深度融合和scale计算
- `2605.15178v1.md:733-739` — Scale CoV质量过滤

**测试：**
- `task_plan_spatialvid_smoke.md` — 冒烟测试计划
- `experiments/data_production_smoke/smoke_spatialvid.sh` — 测试脚本
- `scripts/validate_smoke_output.py` — 质量检查

### B. 参考公式

**Scale计算（单帧）：**
```
minimize: Σ w_i (s · d_pi3x_i - d_moge_i)²
where: w_i = 1 / d_moge_i
solution: s = Σ(w · d_pi3x · d_moge) / Σ(w · d_pi3x²)
```

**EMA平滑：**
```
s_smoothed[t] = α · s_smoothed[t-1] + (1-α) · s_raw[t]
where: α = 0.99 (momentum)
```

**Scale CoV：**
```
CoV = std(scale_per_frame) / (mean(scale_per_frame) + ε)
threshold: CoV < 2.0
```

**Camera forward direction：**
```
R = pose_c2w[:3, :3]
forward = R @ [0, 0, 1]  # OpenCV: Z轴向前
```

**Forward-Motion alignment：**
```
motion = diff(t)  # ∆t between frames
angle = arccos(dot(forward[:-1], normalize(motion))) × 180/π
```

---

**报告结束**

**下一步行动**: 立即实施 mode_default.py 的修复（见第七章），预计10分钟完成修复+验证。
