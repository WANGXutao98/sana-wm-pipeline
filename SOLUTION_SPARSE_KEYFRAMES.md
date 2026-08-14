# ✅ 轨迹偏差问题的解决方案

**日期**: 2026-08-14  
**根因**: VIPE Phase 2输出所有帧的插值poses，而不是稀疏keyframes  
**解决方案**: 后处理稀疏化 + 验证脚本修正

---

## 根因总结

### VIPE的输出逻辑（基于实际代码）

**保存函数**: `third_party/vipe/vipe/utils/io.py:146-164`

```python
def save_pose_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False):
    pose_list = cached_final_stream.get_stream_attribute(FrameAttribute.POSE)
    
    # 关键：只保存非None的poses
    pose_list = [
        (frame_idx, pose_data.matrix().cpu().numpy())
        for frame_idx, pose_data in enumerate(pose_list)
        if pose_data is not None
    ]
    
    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        np.savez(path, data=pose_data, inds=pose_inds)
```

**关键发现**:
- `cached_final_stream` 来自 `system.py:336` 的 `filled_return.poses`
- `filled_return` 来自 `InnerFiller.get_result()` 
- **InnerFiller插值了所有帧的poses**（Phase 2的作用）
- 所以 `inds = [0, 1, 2, 3, ..., 31]`（所有帧）

### 为什么官方SpatialVID有稀疏keyframes？

**两种可能性**:

#### 可能性A: 官方用了不同的输出逻辑 ⭐⭐⭐⭐⭐

官方可能修改了VIPE的输出逻辑，只保存Phase 1的keyframes：

```python
# 假设的官方修改
keyframe_indices = buffer.tstamp[:start_idx]  # 只取Phase 1的keyframes
pose_list = [(idx, poses[idx]) for idx in keyframe_indices]
```

#### 可能性B: SpatialVID标注时VIPE版本不同 ⭐⭐⭐⭐

VIPE的Phase 2逻辑可能在后续版本中发生了变化。

---

## 解决方案

### 方案A: 后处理稀疏化（推荐）⭐⭐⭐⭐⭐

**优点**:
- 不修改VIPE源码，维护简单
- 可以灵活调整稀疏间隔
- 不影响其他使用VIPE的流程

**实现**: 修改 `mode_default.py` 的 `_load_vipe_artifacts`

```python
def _load_vipe_artifacts(clip_path: Path, vipe_out: Path) -> PoseArtifact:
    """Parse VIPE's npz artifacts into PoseArtifact."""
    stem = Path(clip_path).stem
    pose_npz = vipe_out / "pose" / f"{stem}.npz"
    intr_npz = vipe_out / "intrinsics" / f"{stem}.npz"

    if not pose_npz.exists():
        raise FileNotFoundError(f"VIPE pose not found: {pose_npz}")

    # 加载VIPE输出（所有帧的插值poses）
    pose_data_full = np.load(pose_npz)
    poses_full = pose_data_full["data"].astype(np.float64)  # (T_full, 4, 4)
    inds_full = pose_data_full["inds"].astype(int)          # (T_full,)
    
    # ✅ 关键修复：稀疏化keyframes（每4帧取1个）
    KEYFRAME_INTERVAL = 4
    sparse_mask = (inds_full % KEYFRAME_INTERVAL == 0)
    inds_sparse = inds_full[sparse_mask]
    poses_sparse = poses_full[sparse_mask]
    
    print(f"[mode_default] Sparsified keyframes: {len(poses_full)} -> {len(poses_sparse)} (interval={KEYFRAME_INTERVAL})")
    
    # 加载intrinsics（使用稀疏keyframes）
    intr_data_full = np.load(intr_npz)
    intr_full = intr_data_full["data"].astype(np.float64)  # (T_full, 4)
    intr_sparse = intr_full[sparse_mask]
    
    # 插值回全帧率（用于下游任务）
    T_full = len(inds_full)
    poses_interp = _interpolate_poses(inds_sparse, poses_sparse, T_full)
    intr_interp = _interpolate_intrinsics(inds_sparse, intr_sparse, T_full)
    
    # 加载scales（如果存在）
    depth_dir = vipe_out / "depth_precomputed"
    scale_path = depth_dir / "scales.npy"
    if scale_path.exists():
        scales_full = np.load(scale_path).astype(np.float32)
        sample_idx_path = depth_dir / "sample_idx.npy"
        if sample_idx_path.exists() and len(scales_full) < T_full:
            sample_idx = np.load(sample_idx_path).astype(int)
            scale_per_frame = np.interp(np.arange(T_full), sample_idx, scales_full)
        else:
            scale_per_frame = scales_full[:T_full]
    else:
        scale_per_frame = np.ones(T_full, dtype=np.float32)
    
    return PoseArtifact(
        c2w=poses_interp,           # (T_full, 4, 4) 插值到全帧
        intrinsics=intr_interp,     # (T_full, 1, 4) 插值到全帧
        scale_per_frame=scale_per_frame,  # (T_full,)
    )


def _interpolate_poses(keyframe_inds: np.ndarray, keyframe_poses: np.ndarray, T_full: int) -> np.ndarray:
    """插值稀疏keyframe poses到全帧率
    
    Args:
        keyframe_inds: (K,) keyframe索引 [0, 4, 8, ...]
        keyframe_poses: (K, 4, 4) keyframe poses
        T_full: 总帧数
    
    Returns:
        (T_full, 4, 4) 插值后的poses
    """
    from scipy.spatial.transform import Rotation, Slerp
    
    # 提取rotation和translation
    rotations = Rotation.from_matrix(keyframe_poses[:, :3, :3])
    translations = keyframe_poses[:, :3, 3]
    
    # Slerp插值rotation
    slerp = Slerp(keyframe_inds, rotations)
    full_inds = np.arange(T_full)
    interp_rotations = slerp(full_inds)
    
    # 线性插值translation
    interp_translations = np.stack([
        np.interp(full_inds, keyframe_inds, translations[:, i])
        for i in range(3)
    ], axis=1)
    
    # 组合成4x4矩阵
    poses_interp = np.tile(np.eye(4), (T_full, 1, 1))
    poses_interp[:, :3, :3] = interp_rotations.as_matrix()
    poses_interp[:, :3, 3] = interp_translations
    
    return poses_interp


def _interpolate_intrinsics(keyframe_inds: np.ndarray, keyframe_intr: np.ndarray, T_full: int) -> np.ndarray:
    """插值稀疏keyframe intrinsics到全帧率
    
    Args:
        keyframe_inds: (K,) keyframe索引
        keyframe_intr: (K, 4) keyframe intrinsics [fx, fy, cx, cy]
        T_full: 总帧数
    
    Returns:
        (T_full, 1, 4) 插值后的intrinsics
    """
    full_inds = np.arange(T_full)
    intr_interp = np.stack([
        np.interp(full_inds, keyframe_inds, keyframe_intr[:, i])
        for i in range(4)
    ], axis=1)
    
    return intr_interp[:, None, :]  # (T_full, 1, 4)
```

**安装scipy**（如果需要）:
```bash
conda activate sana_wm
pip install scipy
```

---

### 方案B: 修改VIPE输出逻辑 ⭐⭐⭐

**优点**:
- 输出的就是稀疏keyframes，不需要后处理
- 更符合SLAM的语义

**缺点**:
- 需要修改VIPE源码
- 可能影响其他使用VIPE的项目

**实现**: 修改 `system.py:336-341`

```python
# system.py 末尾（第336行附近）

# 原代码：
# return SLAMOutput(
#     trajectory=filled_return.poses.inv(),
#     intrinsics=original_intrinsics,
#     rig=SE3(self.buffer.rig.clone()),
#     slam_map=slam_map,
# )

# 修改为：只返回Phase 1的keyframes
keyframe_count = self.inner_filler.start_idx
keyframe_indices = self.buffer.tstamp[:keyframe_count].cpu().numpy()

# 提取keyframe poses
keyframe_poses = SE3(self.buffer.poses[:keyframe_count])

# 提取keyframe intrinsics
if hasattr(self.buffer, "intrinsics_pf"):
    keyframe_intrinsics = self.buffer.intrinsics_pf[:keyframe_count]
    keyframe_intrinsics = torch.stack([resizers[0].recover_intrinsics(k) for k in keyframe_intrinsics])
else:
    keyframe_intrinsics = original_intrinsics

return SLAMOutput(
    trajectory=keyframe_poses.inv(),
    intrinsics=keyframe_intrinsics[None],  # (1, K, 4)
    rig=SE3(self.buffer.rig.clone()),
    slam_map=slam_map,
    keyframe_indices=keyframe_indices,  # 新增字段
)
```

---

### 方案C: 跳过Phase 2 ⭐⭐

**优点**:
- 最简单，直接返回Phase 1的结果
- 不需要插值逻辑

**缺点**:
- 输出的poses数量很少（13-14个）
- 需要下游任务自行插值

**实现**: 修改 `system.py:299` 添加早期返回

```python
# system.py:299 添加
if os.environ.get("SANA_WM_SPARSE_KEYFRAMES_ONLY", "0") == "1":
    # 跳过Phase 2，直接返回Phase 1的keyframes
    slam_map = self.buffer.extract_slam_map(filter_thresh=self.config.map_filter_thresh)
    slam_map.backend_graph = self.backend.last_graph
    
    keyframe_count = self.buffer.n_frames
    original_intrinsics = torch.stack([
        resizer.recover_intrinsics(self.buffer.intrinsics[v]) 
        for v, resizer in enumerate(resizers)
    ])
    
    return SLAMOutput(
        trajectory=SE3(self.buffer.poses[:keyframe_count]).inv(),
        intrinsics=original_intrinsics,
        rig=SE3(self.buffer.rig.clone()),
        slam_map=slam_map,
    )

# 否则继续Phase 2...
self.inner_filler.set_start_idx(self.buffer.n_frames)
```

使用时设置环境变量：
```bash
export SANA_WM_SPARSE_KEYFRAMES_ONLY=1
```

---

## 推荐方案：方案A（后处理稀疏化）

### 理由

1. **不修改VIPE源码** → 维护简单，不影响其他项目
2. **灵活性高** → 可以调整稀疏间隔（4帧、8帧等）
3. **完整输出** → 插值后输出全帧率poses，适配下游任务
4. **可验证** → 可以对比稀疏keyframes和插值结果

### 实施步骤

1. 修改 `mode_default.py` 添加稀疏化逻辑
2. 安装 `scipy`（用于Slerp旋转插值）
3. 测试3个样本
4. 验证轨迹长度是否改善到~1.0x

---

## 预期效果

### 修复前（当前）

- Keyframes: 32-37个（连续）
- 轨迹偏差: 3.88-9.56x
- 原因: 短基线 → BA scale漂移

### 修复后（稀疏化）

- Keyframes: 8-10个（每4帧）
- 轨迹偏差: ~1.0-1.5x（预期）
- 原因: 长基线 → BA scale稳定

---

## 下一步

1. 实施方案A
2. 测试3个样本
3. 如果轨迹仍有偏差，调整稀疏间隔（试试每8帧）
4. 扩展到更多样本验证鲁棒性
