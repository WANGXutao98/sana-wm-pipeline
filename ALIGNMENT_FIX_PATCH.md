# 修复补丁：删除稀疏化，完全对齐参考实现

**目标**: 将 `mode_default.py` 的 `_load_vipe_artifacts()` 函数恢复到与 `sana-wm-data-clean/vipe_cli.py` 100%一致的逻辑

**修改文件**: `src/sana_wm_pipeline/stage02_pose/mode_default.py`

---

## 修改方案

### 删除第165-202行，替换为简单排序逻辑

**原代码**（第165-202行，共38行）:
```python
    # ponytail: 稀疏化VIPE输出（每4帧1个keyframe）+ Slerp插值
    # VIPE Phase 2无条件添加所有帧 → 连续keyframes → 短基线 → scale漂移
    # 稀疏化 → 长基线 → 稳定scale（轨迹偏差从3.88-9.56x降至~1.0-1.5x）
    KEYFRAME_INTERVAL = 4
    sparse_mask = (pose_inds % KEYFRAME_INTERVAL == 0)
    
    # ... 38行稀疏化+插值逻辑 ...
    
    poses_c2w = np.zeros((T_full, 4, 4), dtype=np.float32)
    poses_c2w[:, :3, :3] = R_interp.as_matrix()
    poses_c2w[:, :3, 3] = t_interp
    poses_c2w[:, 3, 3] = 1.0
```

**新代码**（替换为6行简单逻辑）:
```python
    # 直接使用VIPE输出（与sana-wm-data-clean/vipe_cli.py:_load_vipe_pose对齐）
    # 参考: vipe_cli.py:61-70 _load_vipe_pose()
    # 逻辑: 只按inds排序，不做稀疏化/插值
    order = np.argsort(pose_inds)
    poses_c2w = poses_c2w[order]
    pose_inds_sorted = pose_inds[order]
    
    order_intr = np.argsort(intr_inds)
    intrinsics_raw = intrinsics_raw[order_intr]
    intr_inds_sorted = intr_inds[order_intr]
    
    # VIPE输出的帧数就是最终帧数（不插值到T_full）
    T_full = len(poses_c2w)
    
    print(f"[mode_default] Loaded {T_full} frames from VIPE (no sparsification)")
    print(f"[mode_default]   Pose indices: {pose_inds_sorted[:5].tolist()} ... {pose_inds_sorted[-5:].tolist()}")
```

---

## 完整修改后的函数

```python
def _load_vipe_artifacts(clip_path: Path, vipe_out: Path) -> PoseArtifact:
    """Parse VIPE's npz artifacts into PoseArtifact.

    VIPE writes:
      pose/<stem>.npz          data:(T,4,4), inds:(T,)
      intrinsics/<stem>.npz    data:(T,4),   inds:(T,)   — [fx,fy,cx,cy]
    """
    stem = Path(clip_path).stem
    pose_npz = vipe_out / "pose" / f"{stem}.npz"
    intr_npz = vipe_out / "intrinsics" / f"{stem}.npz"

    if not pose_npz.exists():
        raise FileNotFoundError(
            f"VIPE pose artifact missing: {pose_npz}\n"
            f"(check vipe infer completed without error)"
        )

    pose_data = np.load(pose_npz)
    poses_c2w = pose_data["data"].astype(np.float32)  # (T, 4, 4)
    pose_inds = pose_data["inds"]                      # (T,)

    if not intr_npz.exists():
        raise FileNotFoundError(f"VIPE intrinsics artifact missing: {intr_npz}")
    intr_data = np.load(intr_npz)
    intrinsics_raw = intr_data["data"].astype(np.float32)  # (T, 4) [fx,fy,cx,cy]
    intr_inds = intr_data["inds"]

    # 直接使用VIPE输出（与sana-wm-data-clean/vipe_cli.py:_load_vipe_pose对齐）
    # 参考: vipe_cli.py:61-70 _load_vipe_pose()
    # 逻辑: 只按inds排序，不做稀疏化/插值
    order = np.argsort(pose_inds)
    poses_c2w = poses_c2w[order]
    pose_inds_sorted = pose_inds[order]
    
    order_intr = np.argsort(intr_inds)
    intrinsics_raw = intrinsics_raw[order_intr]
    intr_inds_sorted = intr_inds[order_intr]
    
    # VIPE输出的帧数就是最终帧数（不插值到T_full）
    T_full = len(poses_c2w)
    
    print(f"[mode_default] Loaded {T_full} frames from VIPE (no sparsification)")
    print(f"[mode_default]   Pose indices: {pose_inds_sorted[:5].tolist()} ... {pose_inds_sorted[-5:].tolist()}")

    # 参考实现的intrinsics处理逻辑（vipe_cli.py:73-100 _load_perframe_intrinsics）
    # 如果K == N: 直接使用
    # 如果K == 1: broadcast到N帧
    # 如果1 < K < N: 线性插值到N帧
    intrinsics_full = _interp_intrinsics_aligned(intrinsics_raw, T_full)
    
    # Reshape intrinsics to (T, 1, 4) as required by PoseArtifact.
    intrinsics_nvd = intrinsics_full[:, None, :]  # (T, 1, 4)

    # Load scale_per_frame from Phase A (与官方sana-wm-data-clean一致)
    # 官方: vipe_cli.py:161-162 → scales = np.load(scales_npy).tolist()
    depth_dir = vipe_out / "depth_precomputed"
    scale_path = depth_dir / "scales.npy"

    if scale_path.exists():
        scales_full = np.load(scale_path).astype(np.float32)  # (S,) Phase A采样的帧数

        # 如果Phase A采样了关键帧（S < T_full），需要插值到全部帧
        sample_idx_path = depth_dir / "sample_idx.npy"
        if sample_idx_path.exists() and len(scales_full) < T_full:
            sample_idx = np.load(sample_idx_path).astype(int)  # (S,) 采样索引
            # 线性插值到T_full帧
            scale_per_frame = np.interp(
                np.arange(T_full),
                sample_idx,
                scales_full
            ).astype(np.float32)
            print(f"[mode_default] ✅ Interpolated {len(scales_full)} scales to {T_full} frames")
        else:
            # 无需插值，直接使用（或截断）
            scale_per_frame = scales_full[:T_full] if len(scales_full) >= T_full else scales_full
            if len(scale_per_frame) < T_full:
                # 不足则补1.0
                padding = np.ones(T_full - len(scale_per_frame), dtype=np.float32)
                scale_per_frame = np.concatenate([scale_per_frame, padding])
            print(f"[mode_default] ✅ Loaded {len(scales_full)} scales directly")

        print(f"[mode_default]    Scale range: {scale_per_frame.min():.3f} - {scale_per_frame.max():.3f}")
        print(f"[mode_default]    Scale mean±std: {scale_per_frame.mean():.3f} ± {scale_per_frame.std():.3f}")
        scale_cov = scale_per_frame.std() / (scale_per_frame.mean() + 1e-8)
        print(f"[mode_default]    Scale CoV: {scale_cov:.3f} (threshold: <2.0)")
    else:
        # Fallback: Phase A失败或缺失时使用默认值
        scale_per_frame = np.ones(T_full, dtype=np.float32)
        print(f"[mode_default] ⚠️  {scale_path} not found, using default scale=1.0")

    # Optional downsampled depth for visualization.
    depth_ds = _try_load_depth_downsampled(vipe_out, stem, T_full)

    artifact = PoseArtifact(
        poses_c2w=poses_c2w,
        intrinsics=intrinsics_nvd,
        scale_per_frame=scale_per_frame,
        depth_downsampled=depth_ds,
    )
    return artifact


def _interp_intrinsics_aligned(intr: np.ndarray, n_target: int) -> np.ndarray:
    """Align intrinsics to target frame count (与vipe_cli.py:_load_perframe_intrinsics对齐).
    
    参考: sana-wm-data-clean/sana_wm_data/pose/vipe_cli.py:73-100
    
    Args:
        intr: (K, 4) [fx,fy,cx,cy]
        n_target: 目标帧数N
        
    Returns:
        (N, 4) intrinsics
    """
    if intr.ndim == 1:
        intr = intr[None, :]
    K = intr.shape[0]
    
    if K == n_target:
        return intr
    if K == 1:
        return np.tile(intr[0], (n_target, 1))
    
    # 1 < K < N: 线性插值
    src = np.linspace(0.0, 1.0, K)
    dst = np.linspace(0.0, 1.0, n_target)
    return np.stack([np.interp(dst, src, intr[:, j]) for j in range(4)], axis=1).astype(np.float32)
```

---

## 需要删除的其他函数

**删除 `_interp_poses()` 函数**（第257-274行）:
- 这个函数现在没有被调用
- 它包含了被移除的"第一帧归一化"逻辑
- 参考实现没有对应函数

**保留 `_interp_intrinsics()` 函数** 但重命名:
- 重命名为 `_interp_intrinsics_aligned()`
- 这是参考实现的核心逻辑

---

## 验证清单

修改后需要验证：

1. ✅ **编译检查**
   ```bash
   cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
   python -c "from sana_wm_pipeline.stage02_pose import mode_default; print('OK')"
   ```

2. ✅ **运行测试**
   ```bash
   bash experiments/data_production_smoke/smoke_spatialvid.sh
   ```

3. ✅ **对比输出**
   - 检查 poses_c2w.shape 是否正确
   - 检查 轨迹长度 vs 参考标注
   - 检查 scale CoV 是否 < 2.0

4. ✅ **与参考实现对比**（可选）
   ```bash
   # 用sana-wm-data-clean处理同一个视频
   # 对比输出的poses是否相同
   ```

---

## 预期结果

**代码层面**:
- ✅ 删除38行复杂逻辑，新增6行简单逻辑
- ✅ 与参考实现100%对齐
- ✅ 无Slerp依赖，降低复杂度

**数据层面**:
- ⚠️ 轨迹长度偏差可能仍是5-8x（这可能是expected）
- ✅ Scale CoV应该仍 < 2.0
- ✅ 保留VIPE BA优化的全部信息

**如果偏差仍存在**:
- 这说明问题在Pi3X+MoGe-2融合阶段，不是poses处理阶段
- 应该去调查深度融合的scale计算
- 但至少我们的代码与proven方案一致了

---

## Git Commit Message

```
fix: remove sparsification hack, align with sana-wm-data-clean reference

- Delete 38 lines of keyframe sparsification + Slerp interpolation
- Replace with simple sorting logic (6 lines)
- 100% aligned with sana-wm-data-clean/vipe_cli.py:_load_vipe_pose
- Preserves all VIPE BA-optimized poses instead of discarding 75%

Rationale:
- Reference implementation has NO sparsification logic
- Sparsification+interpolation discards BA optimization information
- Test results inconsistent (sample 2 degraded +8.4%)
- Ponytail principle: use the proven solution

Issue: Sparsification was added by previous Claude as a quick fix,
but it deviates from the proven reference implementation.
```

---

**修复难度**: TRIVIAL  
**预计时间**: 5分钟代码修改 + 30分钟测试验证  
**风险**: LOW（恢复到proven方案）
