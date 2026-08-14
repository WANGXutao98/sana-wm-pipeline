#!/usr/bin/env python3
"""分析10组SpatialVID冒烟测试结果的综合评估脚本"""

import json
import tarfile
from pathlib import Path
import numpy as np

# ponytail: 最简分析，只提取关键指标

SMOKE_DIR = Path("/mnt/afs/davidwang/workspace/sana_test_data/smoke_result")
SAMPLES_FILE = SMOKE_DIR / "selected_samples.txt"

# 读取样本列表
samples = []
with open(SAMPLES_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            samples.append((parts[0], int(parts[1])))

print(f"总样本数: {len(samples)}")
print()

results = []

for sample_id, expected_frames in samples:
    sample_dir = SMOKE_DIR / sample_id
    tar_path = sample_dir / f"{sample_id}.tar"
    json_path = sample_dir / "pose_artifact_default.json"

    if not tar_path.exists():
        print(f"⚠️  {sample_id}: tar不存在")
        continue

    # 解压tar到临时目录
    extract_dir = sample_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_dir)

    # 读取poses/intrinsics/scale
    poses_path = extract_dir / f"{sample_id}.poses_c2w.npy"
    intr_path = extract_dir / f"{sample_id}.intrinsics.npy"
    scale_path = extract_dir / f"{sample_id}.scale.npy"  # 注意：tar内是scale.npy不是scale_per_frame.npy

    if not poses_path.exists():
        print(f"⚠️  {sample_id}: poses不存在")
        continue

    poses = np.load(poses_path)  # (T,4,4)
    intr = np.load(intr_path)    # (T,1,4)
    scale = np.load(scale_path)  # (T,)

    T = poses.shape[0]

    # 计算轨迹长度
    traj = poses[:, :3, 3]  # (T,3)
    dists = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    traj_len = float(dists.sum())

    # 旋转正交性
    R = poses[:, :3, :3]
    RtR = R @ R.transpose(0, 2, 1)
    I = np.eye(3)[None]
    ortho_err = np.abs(RtR - I).max()

    # Scale CoV
    scale_cov = float(scale.std() / scale.mean()) if scale.mean() > 0 else 0

    # 读取参考标注（如果有）
    camera_npz = SMOKE_DIR / "raw_samples" / f"{sample_id}.camera.npz"
    ref_traj_len = None
    ratio = None

    if camera_npz.exists():
        cam = np.load(camera_npz)
        if "vipe_c2w" in cam:
            ref_c2w = cam["vipe_c2w"]
            ref_traj = ref_c2w[:, :3, 3]
            ref_dists = np.linalg.norm(np.diff(ref_traj, axis=0), axis=1)
            ref_traj_len = float(ref_dists.sum())
            ratio = traj_len / ref_traj_len if ref_traj_len > 0 else None

    results.append({
        "sample_id": sample_id,
        "expected_frames": expected_frames,
        "actual_frames": T,
        "traj_len": traj_len,
        "ref_traj_len": ref_traj_len,
        "ratio": ratio,
        "ortho_err": ortho_err,
        "scale_cov": scale_cov,
        "scale_range": (float(scale.min()), float(scale.max())),
        "fx_mean": float(intr[:, 0, 0].mean()),
    })

    print(f"✅ {sample_id}")
    print(f"   帧数: {T} (预期{expected_frames})")
    print(f"   轨迹: {traj_len:.3f}m")
    if ref_traj_len:
        print(f"   参考: {ref_traj_len:.3f}m (比例: {ratio:.2f}x)")
    print(f"   Scale CoV: {scale_cov:.4f}")
    print(f"   正交误差: {ortho_err:.2e}")
    print()

# 汇总统计
print("=" * 60)
print("汇总统计")
print("=" * 60)

ratios = [r["ratio"] for r in results if r["ratio"] is not None]
scale_covs = [r["scale_cov"] for r in results]
ortho_errs = [r["ortho_err"] for r in results]

print(f"总样本数: {len(results)}")
print(f"有参考标注: {len(ratios)}")
print()

if ratios:
    print(f"轨迹偏差比例:")
    print(f"  范围: {min(ratios):.2f}x - {max(ratios):.2f}x")
    print(f"  中位数: {np.median(ratios):.2f}x")
    print(f"  均值±标准差: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
    print()

print(f"Scale CoV:")
print(f"  范围: {min(scale_covs):.4f} - {max(scale_covs):.4f}")
print(f"  中位数: {np.median(scale_covs):.4f}")
print(f"  < 2.0阈值: {sum(1 for c in scale_covs if c < 2.0)}/{len(scale_covs)}")
print()

print(f"旋转正交性:")
print(f"  最大误差: {max(ortho_errs):.2e}")
print(f"  中位数: {np.median(ortho_errs):.2e}")
print()

# 保存详细结果
import json
with open(SMOKE_DIR / "analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"详细结果已保存到: {SMOKE_DIR / 'analysis_results.json'}")
