#!/usr/bin/env python3
"""诊断VIPE内参变化导致的深度缩放问题"""

import numpy as np
from pathlib import Path
import json

samples = [
    ("样本1", "SpatialVID-hq_b5a60fd2-64ff-5a22-b2f5-5df2bd7dea63"),
    ("样本2", "SpatialVID-hq_a884fb06-ac39-5950-a2b4-288bf4d93efe"),
]

print("=" * 100)
print("VIPE 内参变化 vs 深度缩放 诊断")
print("=" * 100)

for name, sid in samples:
    print(f"\n{name} ({sid[:8]}...)")

    # 读取输出内参
    artifact = Path(f"/mnt/afs/davidwang/workspace/sana_test_data/smoke_result/{sid}/pose_artifact_default.json")
    with open(artifact) as f:
        data = json.load(f)

    intrinsics = np.array(data["intrinsics"])  # (N, 4) [fx, fy, cx, cy]
    fx = intrinsics[:, 0]

    print(f"  内参fx范围: {fx.min():.1f} - {fx.max():.1f}")
    print(f"  内参fx变化: {(fx.max() - fx.min()) / fx.mean() * 100:.2f}%")

    # 计算VIPE可能应用的焦距缩放
    fx_ratio = fx[-1] / fx[0]  # 最后帧 / 第一帧
    print(f"  焦距比例 (last/first): {fx_ratio:.4f}")

    # 理论上的深度缩放效应
    # 如果VIPE用 disp *= fx_old / fx_new，那么depth *= fx_new / fx_old
    depth_scale_effect = 1.0 / fx_ratio
    print(f"  深度缩放效应: {depth_scale_effect:.4f}x")

    # 读取实际轨迹比例
    poses = np.array(data["poses_c2w"])
    traj_len = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()

    ref_cam = np.load(f"/mnt/afs/davidwang/workspace/sana_test_data/smoke_result/raw_samples/{sid}.camera.npz")
    vipe_c2w = ref_cam["vipe_c2w"]
    ref_traj = np.linalg.norm(np.diff(vipe_c2w[:, :3, 3], axis=0), axis=1).sum()

    actual_ratio = traj_len / ref_traj
    print(f"  实际轨迹比例: {actual_ratio:.2f}x")

    # 如果焦距变化是原因，修正后应该接近1.0
    corrected_ratio = actual_ratio * fx_ratio
    print(f"  💡 修正后比例: {corrected_ratio:.2f}x (如果焦距缩放是原因)")

print("\n" + "=" * 100)
print("分析结论:")
print("如果'修正后比例'接近1.0，说明问题是VIPE的焦距缩放逻辑")
print("如果'修正后比例'仍然偏大，说明还有其他问题")
print("=" * 100)
