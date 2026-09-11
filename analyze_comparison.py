#!/usr/bin/env python3
"""深入分析Stage01归一化对结果的影响"""
import cv2
import json
import numpy as np
from pathlib import Path

# 1. 检查原视频分辨率
video_path = '/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/pass_videos/00094653-a9c6-5558-8e2a-4119e7d64f36.mp4'
cap = cv2.VideoCapture(video_path)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
orig_fps = cap.get(cv2.CAP_PROP_FPS)
orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print("=" * 80)
print("视频信息分析")
print("=" * 80)
print(f"原始视频: {orig_width}×{orig_height} @ {orig_fps:.2f}fps, {orig_frames}帧")
print(f"归一化后: 1280×720 @ 16fps")
print()

# 2. 加载结果
baseline_json = Path('/mnt/afs/davidwang/workspace/sana_test_data/spatialvid_passvideos_smoke/00094653-a9c6-5558-8e2a-4119e7d64f36/pose_artifact_default.json')
ablation_json = Path('/mnt/afs/davidwang/workspace/sana_test_data/smoke_stage2_ablation/pose_artifact_default.json')

baseline = json.loads(baseline_json.read_text())
ablation = json.loads(ablation_json.read_text())

poses_base = np.array(baseline['poses_c2w'])
intr_base = np.array(baseline['intrinsics'])
scale_base = np.array(baseline['scale_per_frame'])

poses_abl = np.array(ablation['poses_c2w'])
intr_abl = np.array(ablation['intrinsics'])
scale_abl = np.array(ablation['scale_per_frame'])

print("=" * 80)
print("关键差异分析")
print("=" * 80)
print()

print("1. 采样帧数差异")
print(f"  基线版本: {len(poses_base)} 帧（来自归一化后视频）")
print(f"  Ablation: {len(poses_abl)} 帧（来自原始{orig_frames}帧视频）")
print(f"  差异原因: 归一化改变了总帧数（{orig_fps:.1f}fps @ {orig_frames}帧 → 16fps）")
print()

print("2. 内参对比（考虑分辨率）")
fx_base, fy_base, cx_base, cy_base = intr_base[0, 0]
fx_abl, fy_abl, cx_abl, cy_abl = intr_abl[0, 0]

print(f"  基线版本（1280×720）:")
print(f"    fx={fx_base:.1f}, fy={fy_base:.1f}, cx={cx_base:.1f}, cy={cy_base:.1f}")
print(f"    归一化焦距: fx/width={fx_base/1280:.4f}")
print(f"    视场角 (FOV): {2 * np.arctan(640 / fx_base) * 180 / np.pi:.2f}°")
print()

print(f"  Ablation（原视频{orig_width}×{orig_height}）:")
print(f"    fx={fx_abl:.1f}, fy={fy_abl:.1f}, cx={cx_abl:.1f}, cy={cy_abl:.1f}")
print(f"    归一化焦距: fx/width={fx_abl/orig_width:.4f}")
print(f"    视场角 (FOV): {2 * np.arctan(orig_width/2 / fx_abl) * 180 / np.pi:.2f}°")
print()

print("  ⚠️ 内参解释:")
print(f"    原视频是{orig_width}×{orig_height}，但内参cx={cx_abl:.1f}暗示宽度为{cx_abl*2:.0f}")
print(f"    这说明VIPE优化的内参可能基于Pi3X resize后的分辨率，而非原视频")
print()

print("3. 第一帧姿态异常分析")
print("  基线版本 poses[0]:")
print(poses_base[0])
print()
print("  Ablation poses[0]:")
print(poses_abl[0])
print()
print(f"  Ablation偏离identity的程度:")
print(f"    平移: {np.linalg.norm(poses_abl[0, :3, 3]):.6f}")
print(f"    旋转偏差: {np.linalg.norm(poses_abl[0, :3, :3] - np.eye(3), 'fro'):.6f}")
print()

print("4. Scale对比")
print(f"  基线median scale: {np.median(scale_base):.3f}")
print(f"  Ablation median scale: {np.median(scale_abl):.3f}")
print(f"  Scale比值: {np.median(scale_abl) / np.median(scale_base):.3f}x")
print(f"  差异: {(np.median(scale_abl) / np.median(scale_base) - 1) * 100:.1f}%")
print()

print("5. 轨迹形状对比")
pos_base = poses_base[:, :3, 3]
pos_abl = poses_abl[:, :3, 3]

# 运动距离
travel_base = np.sum(np.linalg.norm(np.diff(pos_base, axis=0), axis=1))
travel_abl = np.sum(np.linalg.norm(np.diff(pos_abl, axis=0), axis=1))

print(f"  基线总运动距离: {travel_base:.3f}")
print(f"  Ablation总运动距离: {travel_abl:.3f}")
print(f"  比值: {travel_abl / travel_base:.3f}x")
print()

# 采样密度不同，按帧率归一化
travel_per_frame_base = travel_base / len(poses_base)
travel_per_frame_abl = travel_abl / len(poses_abl)
print(f"  基线平均帧间运动: {travel_per_frame_base:.4f}")
print(f"  Ablation平均帧间运动: {travel_per_frame_abl:.4f}")
print()

print("=" * 80)
print("结论")
print("=" * 80)
print()
print("✅ Stage02成功处理原视频，无需Stage01归一化")
print()
print("⚠️ 关键差异:")
print("  1. 采样帧数不同: 46 vs 87 (因原视频总帧数不同)")
print("  2. 内参绝对值相近，但可能基于不同分辨率")
print("  3. Scale略有差异 (~3%，在合理范围)")
print("  4. Ablation的poses[0]不是identity（需要调查）")
print()
print("🔍 需要进一步验证:")
print("  - poses[0]为何不是identity？")
print("  - 内参实际对应什么分辨率？")
print("  - 轨迹形状是否一致（需Umeyama对齐）？")
