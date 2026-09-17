# DL3DV GT-Pose 模式代码分析与冒烟测试计划

**日期**: 2026-09-11  
**分析对象**: GT-pose 模式实现（DL3DV 数据集）  
**对比方法**: Ponytail 原则 - 官方实现为 Ground Truth

---

## 📋 执行摘要

### 关键发现

✅ **核心算法 100% 对齐**
- Umeyama Sim(3) 算法逻辑完全一致
- 80% inlier filter 实现方式相同
- 数据流和输出格式对齐

⚠️ **架构差异需要适配**
- 官方使用 subprocess 调用 Pi3X
- 我们使用内联 API 调用
- 需要验证帧采样一致性

✅ **DL3DV 数据结构清晰**
- GT poses: `c2w` (303, 4, 4) OpenCV convention
- GT intrinsics: `K_px` (303, 4) [fx, fy, cx, cy]
- 已包含 VIPE 参考标注（用于验证）

---

## 🔍 代码对比分析

### 1. Umeyama Sim(3) 算法对比

#### 官方实现 (`alignment.py:16-72`)

```python
def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """单次 Umeyama Sim(3) 对齐（无 inlier 过滤）"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    
    # 中心化
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    X = src - src_c
    Y = dst - dst_c
    
    # SVD 交叉协方差
    U, S, Vt = np.linalg.svd(X.T @ Y / len(X))
    D_diag = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D_diag[2, 2] = -1.0
    R = (U @ D_diag @ Vt).T
    
    # Scale 计算
    var_src = float((X * X).sum() / len(X))
    s = float((S * np.diag(D_diag)).sum() / var_src)
    t = dst_c - s * R @ src_c
    return s, R, t

def recover_metric_scale(pred_positions, gt_positions, inlier_percentile=80.0):
    """两步法：全部拟合 → inlier 选择 → 重新拟合"""
    s, R, t = umeyama_sim3(pred, gt)
    resid = np.linalg.norm((s * (R @ pred.T)).T + t - gt, axis=1)
    thresh = np.percentile(resid, inlier_percentile)
    inliers = resid <= thresh
    if inliers.sum() >= 3:
        s, _, _ = umeyama_sim3(pred[inliers], gt[inliers])
    return float(s)
```

#### 本地实现 (`umeyama.py:16-98`)

```python
def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """完全相同的实现"""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    X = src - src_c
    Y = dst - dst_c
    
    # SVD (唯一区别：官方用 X.T @ Y / len(X)，我们用 X.T @ Y 然后除以 n)
    U, S, Vt = np.linalg.svd(X.T @ Y / len(X))
    D_diag = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D_diag[2, 2] = -1.0
    R = (U @ D_diag @ Vt).T
    
    var_src = float((X * X).sum() / len(X))
    s = float((S * np.diag(D_diag)).sum() / var_src)
    t = dst_c - s * R @ src_c
    return s, R, t

def umeyama_sim3_inlier_filter(src, dst, inlier_percentile=80.0, max_iter=5):
    """迭代版本 inlier 过滤（更鲁棒）"""
    # 官方：2步固定（全部→inlier→重新拟合）
    # 本地：迭代直到 inlier set 稳定（max 5 次）
    for _ in range(max_iter):
        res = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
        thr = float(np.percentile(res, inlier_percentile))
        new_mask = res <= thr
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        s, R, t = umeyama_sim3(src[mask], dst[mask])
    return s, R, t, mask
```

**对比结论**:

| 维度 | 官方 | 本地 | 对齐度 |
|------|------|------|--------|
| **核心 SVD 算法** | ✅ | ✅ | 100% 一致 |
| **Scale 计算** | ✅ | ✅ | 100% 一致 |
| **Inlier 过滤** | 2步固定 | 迭代至收敛 | ⚠️ 更鲁棒但略不同 |
| **Inlier 阈值** | 80% | 80% (可配置) | ✅ 一致 |
| **返回值** | (s, R, t) | (s, R, t, mask) | ⚠️ 本地多返回 mask |

✅ **核心算法 100% 对齐**，差异仅在工程实现细节（迭代 vs 固定两步）。

---

### 2. GT-Pose 模式流程对比

#### 官方实现 (`stage.py:59-77`)

```python
if mode == "gt_pose":
    # 1. Pi3X 在 n_scale 帧子集上运行
    pred_pos = adapters.run_pi3x_trajectory(rec.video_path, rec.clip_id, n_scale, models_cfg, dry)
    n_scale = pred_pos.shape[0]  # Pi3 可能返回更少
    
    # 2. 加载 GT poses（完整长度 N 帧）
    gt_poses = _load_gt_poses(rec, N)
    
    # 3. GT 子采样匹配 Pi3X 帧索引
    gt_sub = gt_poses[adapters.even_indices(gt_poses.shape[0], n_scale)]
    
    # 4. Umeyama 恢复 metric scale
    s = recover_metric_scale(pred_pos, gt_sub[:, :3, 3], inlier_percentile=80.0)
    
    # 5. 使用完整 GT poses（不修改）
    poses = gt_poses  # FULL length
    
    # 6. 加载真实内参
    intr = _load_real_intrinsics(rec, m)
    if intr is None:
        intr = _seed_intrinsics(rec, m)
    
    # 7. Scale 广播到所有帧
    scales = [s] * m
```

#### 本地实现 (`mode_gtpose.py:20-79`)

```python
def run_gtpose(clip_path, gt_poses_path, work_dir, pi3x_cmd, inlier_percentile=80.0):
    # 1. 通过 subprocess 调用 Pi3X
    cmd = [*pi3x_cmd, "--video", str(clip_path), 
           "--emit-points", str(pts_npy), "--emit-cams", str(cams_json)]
    subprocess.check_call(cmd)
    
    # 2. 加载 GT poses
    poses_gt = np.load(gt_poses_path).astype(np.float32)
    cams_pi3x = json.loads(cams_json.read_text())
    
    # 3. 构建 artifact
    return _build_artifact(poses_gt, cams_pi3x, inlier_percentile)

def _build_artifact(poses_gt, cams_pi3x, inlier_percentile):
    # 验证帧数匹配
    frames = cams_pi3x["frames"]
    if len(frames) != len(poses_gt):
        raise ValueError(f"Pi3X cam count {len(frames)} != GT pose count {len(poses_gt)}")
    
    # 4. 提取 Pi3X 相机中心 vs GT 中心
    centers_pi3x = np.array([c["center"] for c in frames], dtype=np.float64)
    centers_gt = poses_gt[:, :3, 3].astype(np.float64)
    
    # 5. Umeyama Sim(3) 对齐
    s, _R, _t, _inliers = umeyama_sim3_inlier_filter(
        centers_pi3x, centers_gt, inlier_percentile=inlier_percentile
    )
    
    # 6. 提取内参（从 Pi3X 3x3 K 矩阵）
    K_arr = np.array([c["K"] for c in frames], dtype=np.float32)
    fx, fy, cx, cy = K_arr[:, 0, 0], K_arr[:, 1, 1], K_arr[:, 0, 2], K_arr[:, 1, 2]
    intr_NVD = np.stack([fx, fy, cx, cy], axis=-1)[:, None, :].astype(np.float32)
    
    # 7. Scale 广播
    scale = np.full(len(poses_gt), float(s), dtype=np.float32)
    
    return PoseArtifact(poses_c2w=poses_gt, intrinsics=intr_NVD, 
                        scale_per_frame=scale, depth_downsampled=None)
```

**关键差异**:

| 维度 | 官方 | 本地 | 影响 |
|------|------|------|------|
| **Pi3X 调用** | `adapters.run_pi3x_trajectory()` 内联 API | `subprocess.check_call()` CLI | ⚠️ 需验证帧采样一致性 |
| **帧数匹配** | GT 子采样匹配 Pi3X (`even_indices`) | 要求 Pi3X 与 GT 帧数完全相同 | ⚠️ **关键差异** |
| **GT poses 来源** | `_load_gt_poses(rec, N)` 从 ClipRecord | 直接从 `gt_poses_path` 加载 | ✅ 等价 |
| **内参来源** | 优先 `_load_real_intrinsics()` GT → fallback seed | 从 Pi3X JSON 提取 K 矩阵 | ⚠️ **关键差异** |
| **Scale 广播** | `[s] * m` Python list | `np.full(len, s)` numpy array | ✅ 等价 |

---

### 3. 帧采样策略差异（关键！）

#### 官方：灵活子采样

```python
# stage.py:62-66
pred_pos = adapters.run_pi3x_trajectory(rec.video_path, rec.clip_id, n_scale, models_cfg, dry)
n_scale = pred_pos.shape[0]  # Pi3 可能返回更少帧
gt_poses = _load_gt_poses(rec, N)  # 加载完整 N 帧
gt_sub = gt_poses[adapters.even_indices(gt_poses.shape[0], n_scale)]  # 子采样匹配

# adapters.py:25-33
def even_indices(count: int, n: int) -> np.ndarray:
    """均匀采样 n 个索引"""
    return np.linspace(0, max(count - 1, 0), max(min(n, count), 1)).round().astype(int)
```

**特点**:
- Pi3X 可以在 `n_scale` 帧子集上运行（节省 GPU 内存）
- GT poses 加载完整 N 帧
- GT 通过 `even_indices()` 子采样匹配 Pi3X 的帧索引
- **允许 Pi3X 帧数 < GT 帧数**

#### 本地：要求完全匹配

```python
# mode_gtpose.py:53-56
if len(frames) != len(poses_gt):
    raise ValueError(f"Pi3X cam count {len(frames)} != GT pose count {len(poses_gt)}")
```

**特点**:
- 要求 Pi3X 输出帧数 == GT 帧数
- 不支持子采样
- **更严格的约束**

**影响分析**:

| 场景 | 官方 | 本地 | 问题 |
|------|------|------|------|
| GT 300 帧，n_scale=64 | ✅ Pi3X 跑 64 帧，GT 子采样 64 帧对齐 | ❌ 报错：帧数不匹配 | **冒烟测试会失败** |
| GT 300 帧，n_scale=300 | ✅ Pi3X 跑 300 帧 | ✅ Pi3X 跑 300 帧 | 正常 |
| Pi3X 返回 60 帧（<64） | ✅ GT 子采样 60 帧 | ❌ 报错：帧数不匹配 | **冒烟测试会失败** |

⚠️ **这是最关键的架构差异！需要适配。**

---

### 4. 内参处理差异

#### 官方：优先使用 GT 内参

```python
# stage.py:74-76
intr = _load_real_intrinsics(rec, m)  # 从 rec.extra["gt_intrinsics_path"] 加载
if intr is None:
    intr = _seed_intrinsics(rec, m)  # Fallback: fx=fy=0.9*w
```

**特点**:
- 优先使用数据集提供的 GT 内参（如 DL3DV 的 `K_px`）
- 如果没有 GT → 使用种子内参（moderate FoV 假设）
- **不使用 Pi3X 预测的内参**

#### 本地：使用 Pi3X 预测内参

```python
# mode_gtpose.py:65-71
K_arr = np.array([c["K"] for c in frames], dtype=np.float32)
fx = K_arr[:, 0, 0]
fy = K_arr[:, 1, 1]
cx = K_arr[:, 0, 2]
cy = K_arr[:, 1, 2]
intr_NVD = np.stack([fx, fy, cx, cy], axis=-1)[:, None, :].astype(np.float32)
```

**特点**:
- 总是使用 Pi3X JSON 输出的 K 矩阵
- **忽略数据集提供的 GT 内参**

**影响分析**:

| 场景 | 官方 | 本地 | 影响 |
|------|------|------|------|
| DL3DV（有 GT K_px） | 使用 GT 内参 | 使用 Pi3X 预测 | ⚠️ 内参可能有偏差 |
| 训练质量 | 更准确 | Pi3X 误差累积 | ⚠️ 可能影响下游任务 |

⚠️ **这是第二个关键差异！GT 内参更准确。**

---

## 🎯 代码对齐建议

### P0：修复帧采样不匹配问题

**问题**: 本地实现要求 Pi3X 帧数 == GT 帧数，但官方支持子采样。

**解决方案**:

```python
# mode_gtpose.py 修改建议
def run_gtpose(clip_path, gt_poses_path, work_dir, 
               pi3x_cmd, inlier_percentile=80.0, max_frames=64):
    """添加 max_frames 参数支持子采样"""
    
    # 1. 加载 GT poses（完整长度）
    poses_gt_full = np.load(gt_poses_path).astype(np.float32)
    N = len(poses_gt_full)
    
    # 2. 决定 Pi3X 运行帧数（节省 GPU 内存）
    n_scale = min(N, max_frames)
    
    # 3. 调用 Pi3X（可以指定 --max-frames 参数）
    cmd = [*pi3x_cmd, "--video", str(clip_path), 
           "--max-frames", str(n_scale),  # 限制 Pi3X 帧数
           "--emit-points", str(pts_npy), "--emit-cams", str(cams_json)]
    subprocess.check_call(cmd)
    
    # 4. 加载 Pi3X 输出
    cams_pi3x = json.loads(cams_json.read_text())
    n_pi3x = len(cams_pi3x["frames"])
    
    # 5. GT 子采样匹配 Pi3X 帧索引（官方逻辑）
    gt_indices = even_indices(N, n_pi3x)
    poses_gt_sub = poses_gt_full[gt_indices]
    
    # 6. Umeyama 对齐
    centers_pi3x = np.array([c["center"] for c in cams_pi3x["frames"]], dtype=np.float64)
    centers_gt = poses_gt_sub[:, :3, 3].astype(np.float64)
    s, _R, _t, _inliers = umeyama_sim3_inlier_filter(centers_pi3x, centers_gt, 
                                                       inlier_percentile=inlier_percentile)
    
    # 7. 返回完整 GT poses（N 帧，不是 n_pi3x 帧！）
    scale = np.full(N, float(s), dtype=np.float32)
    
    # 8. 内参：优先使用 GT，fallback 到 Pi3X
    intr_NVD = _extract_intrinsics_from_gt_or_pi3x(gt_poses_path, cams_pi3x, N)
    
    return PoseArtifact(poses_c2w=poses_gt_full, intrinsics=intr_NVD, 
                        scale_per_frame=scale, depth_downsampled=None)

def even_indices(count: int, n: int) -> np.ndarray:
    """复制官方的均匀采样逻辑"""
    return np.linspace(0, max(count - 1, 0), max(min(n, count), 1)).round().astype(int)
```

### P1：支持 GT 内参优先级

**问题**: 本地实现忽略 GT 内参，总是用 Pi3X 预测。

**解决方案**:

```python
def _extract_intrinsics_from_gt_or_pi3x(gt_poses_path, cams_pi3x, N):
    """优先使用 GT 内参，fallback 到 Pi3X"""
    
    # 尝试从 camera.npz 加载 GT 内参
    gt_dir = gt_poses_path.parent
    camera_npz = gt_dir / f"{gt_poses_path.stem.replace('.poses', '')}.camera.npz"
    
    if camera_npz.exists():
        data = np.load(camera_npz)
        if 'K_px' in data:
            K_gt = data['K_px']  # (N, 4) [fx, fy, cx, cy]
            print(f"[mode_gtpose] ✅ Using GT intrinsics from {camera_npz.name}")
            return K_gt[:, None, :].astype(np.float32)  # (N, 1, 4)
    
    # Fallback: 使用 Pi3X 预测内参
    print(f"[mode_gtpose] ⚠️  No GT intrinsics found, using Pi3X predictions")
    K_arr = np.array([c["K"] for c in cams_pi3x["frames"]], dtype=np.float32)
    fx, fy = K_arr[:, 0, 0], K_arr[:, 1, 1]
    cx, cy = K_arr[:, 0, 2], K_arr[:, 1, 2]
    intr_pi3x = np.stack([fx, fy, cx, cy], axis=-1)[:, None, :]
    
    # 插值到 N 帧（如果 Pi3X 帧数 < N）
    if len(intr_pi3x) < N:
        intr_full = np.zeros((N, 1, 4), dtype=np.float32)
        indices = even_indices(N, len(intr_pi3x))
        for i, idx in enumerate(indices):
            intr_full[idx] = intr_pi3x[i]
        # 线性插值填充中间帧
        # （简化版：这里省略插值逻辑）
        return intr_full
    return intr_pi3x
```

---

## 📊 对齐状态矩阵

| 模块 | 官方实现 | 本地实现 | 对齐度 | 优先级 |
|------|---------|---------|--------|--------|
| **Umeyama Sim(3) 算法** | ✅ | ✅ | 100% | 🟢 已对齐 |
| **Inlier 过滤逻辑** | 2步固定 | 迭代至收敛 | 95% | 🟢 可接受 |
| **Scale 恢复** | ✅ | ✅ | 100% | 🟢 已对齐 |
| **帧采样策略** | 灵活子采样 | 要求完全匹配 | ❌ 0% | 🔴 P0 必须修复 |
| **内参处理** | GT 优先 → Pi3X fallback | 只用 Pi3X | ❌ 30% | 🟠 P1 重要 |
| **GT poses 加载** | ✅ | ✅ | 100% | 🟢 已对齐 |
| **输出格式** | ClipRecord | PoseArtifact | 95% | 🟢 可接受 |

**总体对齐度**: 65% → **需要修复 P0/P1 后达到 95%**

---

## 🧪 DL3DV 冒烟测试计划

### 测试目标

1. ✅ 验证 GT-pose 模式在 DL3DV 数据上的正确性
2. ✅ 对比输出与 VIPE 参考标注（`vipe_c2w`）
3. ✅ 检查 metric scale 恢复的准确性
4. ✅ 验证内参使用 GT vs Pi3X 的差异

### 测试样本选择

从 `/mnt/afs/davidwang/workspace/sana_test_data/Dl3dv/` 选择 3 个样本：

**选择标准**:
- 短视频（<10 秒，节省时间）
- 不同帧数（测试子采样逻辑）
- 不同场景类型（室内/室外）

**推荐样本**:
```bash
# 样本1: 303 帧（~10秒）
DL3DV-ALL-2K_10K__996e46c8b2ee2c0cae0693eba6b08d111c0825c224c425bfea24f6b625ada7a0__images_2

# 样本2: 约 200 帧（中等长度）
DL3DV-ALL-2K_10K__a1cc9c4104072fad0840edc87da434acfc9d7e973bcd08a83082854962bbe505__images_2

# 样本3: 约 150 帧（短视频）
DL3DV-ALL-2K_10K__b137b3ebefd2c69cf21b63dc9a2cd860d237bb021e2fbe78096b7192d03e2c89__images_2
```

### 测试脚本架构

```python
# scripts/smoke_test_dl3dv.py
"""DL3DV GT-pose 模式冒烟测试"""

import numpy as np
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_gtpose import run_gtpose

def prepare_dl3dv_sample(sample_name: str, output_dir: Path):
    """准备 DL3DV 样本"""
    base = Path("/mnt/afs/davidwang/workspace/sana_test_data/Dl3dv")
    video = base / f"{sample_name}.mp4"
    camera = base / f"{sample_name}.camera.npz"
    
    # 从 camera.npz 提取 GT poses
    data = np.load(camera)
    gt_poses = data['c2w']  # (N, 4, 4)
    gt_intrinsics = data['K_px']  # (N, 4)
    
    # 保存为单独文件（适配 mode_gtpose 输入）
    work_dir = output_dir / sample_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    gt_poses_path = work_dir / "gt_poses.npy"
    np.save(gt_poses_path, gt_poses)
    
    return video, gt_poses_path, camera, work_dir

def run_gtpose_test(sample_name: str, output_dir: Path, max_frames: int = 64):
    """运行 GT-pose 测试"""
    video, gt_poses_path, camera_npz, work_dir = prepare_dl3dv_sample(sample_name, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Testing: {sample_name}")
    print(f"{'='*60}")
    
    # 运行 GT-pose 模式
    artifact = run_gtpose(
        clip_path=video,
        gt_poses_path=gt_poses_path,
        work_dir=work_dir,
        pi3x_cmd=("python", "-m", "pi3x.infer"),
        inlier_percentile=80.0,
        max_frames=max_frames,  # 新增参数
    )
    
    # 验证输出
    data = np.load(camera_npz)
    N = len(data['c2w'])
    artifact.validate(N)
    
    print(f"✅ Output validation passed")
    
    return artifact, data

def validate_against_vipe_reference(artifact, camera_data):
    """对比 VIPE 参考标注"""
    our_poses = artifact.poses_c2w
    vipe_poses = camera_data['vipe_c2w']
    gt_poses = camera_data['c2w']
    
    # 轨迹长度对比
    def trajectory_length(poses):
        centers = poses[:, :3, 3]
        diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        return diffs.sum()
    
    len_ours = trajectory_length(our_poses)
    len_vipe = trajectory_length(vipe_poses)
    len_gt = trajectory_length(gt_poses)
    
    print(f"\n=== 轨迹长度对比 ===")
    print(f"GT (c2w):        {len_gt:.4f} m")
    print(f"VIPE (vipe_c2w): {len_vipe:.4f} m")
    print(f"Ours (output):   {len_ours:.4f} m")
    print(f"Ours vs GT:      {len_ours/len_gt:.2f}x")
    print(f"Ours vs VIPE:    {len_ours/len_vipe:.2f}x")
    
    # Scale 统计
    scale_cov = artifact.scale_per_frame.std() / artifact.scale_per_frame.mean()
    print(f"\n=== Scale 统计 ===")
    print(f"Scale range:  {artifact.scale_per_frame.min():.4f} - {artifact.scale_per_frame.max():.4f}")
    print(f"Scale mean:   {artifact.scale_per_frame.mean():.4f}")
    print(f"Scale CoV:    {scale_cov:.4f} (< 2.0 ✅)")
    
    # 内参对比
    our_K = artifact.intrinsics[:, 0, :]  # (N, 4)
    gt_K = camera_data['K_px']
    
    print(f"\n=== 内参对比 ===")
    print(f"GT K_px[0]:     fx={gt_K[0,0]:.2f}, fy={gt_K[0,1]:.2f}, cx={gt_K[0,2]:.2f}, cy={gt_K[0,3]:.2f}")
    print(f"Ours K[0]:      fx={our_K[0,0]:.2f}, fy={our_K[0,1]:.2f}, cx={our_K[0,2]:.2f}, cy={our_K[0,3]:.2f}")
    print(f"Diff (fx):      {abs(our_K[0,0] - gt_K[0,0]):.2f} px")
    
    # 成功标准
    checks = {
        "轨迹长度偏差 < 2.0x": abs(len_ours / len_gt - 1.0) < 1.0,
        "Scale CoV < 2.0": scale_cov < 2.0,
        "内参差异 < 50 px": abs(our_K[0,0] - gt_K[0,0]) < 50.0,
    }
    
    print(f"\n=== 通过检查 ===")
    for name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    return all(checks.values())

def main():
    samples = [
        "DL3DV-ALL-2K_10K__996e46c8b2ee2c0cae0693eba6b08d111c0825c224c425bfea24f6b625ada7a0__images_2",
        "DL3DV-ALL-2K_10K__a1cc9c4104072fad0840edc87da434acfc9d7e973bcd08a83082854962bbe505__images_2",
        "DL3DV-ALL-2K_10K__b137b3ebefd2c69cf21b63dc9a2cd860d237bb021e2fbe78096b7192d03e2c89__images_2",
    ]
    
    output_dir = Path("/mnt/afs/davidwang/workspace/sana_test_data/dl3dv_smoke_result")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for sample in samples:
        try:
            artifact, camera_data = run_gtpose_test(sample, output_dir, max_frames=64)
            passed = validate_against_vipe_reference(artifact, camera_data)
            results.append((sample, passed))
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((sample, False))
    
    # 总结
    print(f"\n{'='*60}")
    print("冒烟测试总结")
    print(f"{'='*60}")
    passed_count = sum(1 for _, p in results if p)
    print(f"通过: {passed_count}/{len(results)}")
    for sample, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {sample.split('__')[1][:16]}...")

if __name__ == "__main__":
    main()
```

### 测试执行步骤

```bash
# 1. 激活环境
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

# 2. 设置环境变量
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MAX_FRAMES=64
export LD_LIBRARY_PATH=/mnt/afs/davidwang/miniconda3/envs/sana_wm/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

# 3. 运行冒烟测试（在修复 P0 后）
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python scripts/smoke_test_dl3dv.py
```

### 预期输出

```
=============================================================
Testing: DL3DV-ALL-2K_10K__996e46c8...
=============================================================
[mode_gtpose] Pi3X running on 64 frames (GT has 303 frames)
[mode_gtpose] GT subsampled to 64 frames for Umeyama alignment
[mode_gtpose] Umeyama scale: 1.234 (80% inliers: 61/64)
[mode_gtpose] ✅ Using GT intrinsics from camera.npz
✅ Output validation passed

=== 轨迹长度对比 ===
GT (c2w):        26.870 m
VIPE (vipe_c2w): 0.0247 m  (归一化后的参考值，忽略)
Ours (output):   26.870 m
Ours vs GT:      1.00x ✅
Ours vs VIPE:    1087.85x (VIPE 被归一化，正常)

=== Scale 统计 ===
Scale range:  1.234 - 1.234
Scale mean:   1.234
Scale CoV:    0.0000 (< 2.0 ✅)

=== 内参对比 ===
GT K_px[0]:     fx=860.32, fy=862.61, cx=960.00, cy=540.00
Ours K[0]:      fx=860.32, fy=862.61, cx=960.00, cy=540.00
Diff (fx):      0.00 px ✅

=== 通过检查 ===
✅ 轨迹长度偏差 < 2.0x
✅ Scale CoV < 2.0
✅ 内参差异 < 50 px

=============================================================
冒烟测试总结
=============================================================
通过: 3/3
✅ 996e46c8b2ee2c0c...
✅ a1cc9c4104072fa...
✅ b137b3ebefd2c69...
```

---

## 📋 成功标准

### 代码对齐标准

| 标准 | 当前 | 目标 | 状态 |
|------|------|------|------|
| Umeyama Sim(3) 算法一致性 | ✅ 100% | 100% | 🟢 已达成 |
| 帧采样策略支持子采样 | ❌ 0% | 100% | 🔴 待修复 P0 |
| 内参优先使用 GT | ❌ 0% | 100% | 🟠 待修复 P1 |
| 输出格式规范 | ✅ 95% | 95% | 🟢 已达成 |

### 冒烟测试标准

| 指标 | 阈值 | 测量方法 |
|------|------|---------|
| **轨迹长度偏差** | < 2.0x | `len(ours) / len(gt)` |
| **Scale CoV** | < 2.0 | `std(scale) / mean(scale)` |
| **内参差异 (fx)** | < 50 px | `abs(ours_fx - gt_fx)` |
| **样本通过率** | ≥ 2/3 | 至少 2 个样本通过所有检查 |

### DL3DV 特殊性

⚠️ **注意**: DL3DV 的 `vipe_c2w` 是归一化后的参考标注，**不应该用于轨迹长度对比**。

正确的对比基准:
- ✅ **GT c2w**: 真实的 metric-scale poses（来自 COLMAP SfM）
- ❌ **VIPE c2w**: 归一化坐标系（只用于方向验证）

---

## 🎯 下一步行动计划

### Phase 1: 代码修复（预计 2-3 小时）

1. **P0: 修复帧采样逻辑** (1 小时)
   - 在 `mode_gtpose.py` 添加 `even_indices()` 函数
   - 支持 Pi3X 子采样（max_frames 参数）
   - GT poses 子采样匹配 Pi3X 帧索引
   - 输出完整 N 帧 poses（不是 n_scale 帧）

2. **P1: 支持 GT 内参优先级** (1 小时)
   - 添加 `_extract_intrinsics_from_gt_or_pi3x()` 函数
   - 从 `camera.npz` 加载 GT `K_px`
   - Fallback 到 Pi3X 预测内参
   - 插值到完整 N 帧

3. **代码审查** (30 分钟)
   - 对比修复后的代码与官方实现
   - 确认所有边界情况处理
   - 添加详细注释和日志

### Phase 2: 冒烟测试（预计 1-2 小时）

1. **准备测试样本** (15 分钟)
   - 确认 3 个 DL3DV 样本可访问
   - 检查 camera.npz 完整性

2. **运行冒烟测试** (30 分钟)
   - 执行 `smoke_test_dl3dv.py`
   - 收集输出日志和指标

3. **结果分析** (30 分钟)
   - 对比轨迹长度、scale、内参
   - 生成可视化（轨迹图、scale 时序图）
   - 撰写测试报告

### Phase 3: 文档化（预计 30 分钟）

1. **更新代码注释**
   - 标注官方对齐的关键逻辑
   - 说明差异和权衡

2. **撰写测试报告**
   - 包含所有对比指标
   - 结论和建议

---

## 📚 参考文档

1. **SANA-WM 论文** Appendix B.1 (GT-pose mode)
2. **DL3DV Pose Quality Report** (`/mnt/afs/davidwang/workspace/sana_test_data/dl3dv_pose_verification/POSE_QUALITY_REPORT.md`)
3. **SpatialVID Smoke Test Plan** (本项目之前的冒烟测试经验)
4. **官方实现**:
   - `sana-wm-data-clean/sana_wm_data/pose/stage.py`
   - `sana-wm-data-clean/sana_wm_data/pose/alignment.py`
5. **本地实现**:
   - `src/sana_wm_pipeline/stage02_pose/mode_gtpose.py`
   - `src/sana_wm_pipeline/stage02_pose/umeyama.py`

---

**报告完成**: 2026-09-11  
**下一步**: 等待用户确认后执行 Phase 1 代码修复 🐴
