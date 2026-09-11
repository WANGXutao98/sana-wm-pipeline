#!/usr/bin/env python3
"""验证脚本 - 测试官方对齐实现"""
import sys
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")

VIDEO = "/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001/00d77a61-531a-58f4-acf7-da49c23af0ca.mp4"

print("=" * 60)
print("方法 1: 使用 DOVER 官方脚本")
print("=" * 60)

import subprocess
result = subprocess.run([
    "python", "evaluate_one_video.py",
    "-v", VIDEO,
    "-f"
], capture_output=True, text=True, cwd="/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")

# 提取分数
for line in result.stdout.split('\n'):
    if "Normalized fused overall score" in line:
        official_score = float(line.split(':')[-1].strip())
        print(f"官方分数: {official_score:.4f}")
        break

print("\n" + "=" * 60)
print("方法 2: 我们的对齐实现")
print("=" * 60)

# 运行我们的实现
from stage3_batch_official import load_models, compute_dover_score_official
import decord
import numpy as np

device = "cuda"
dover_model, unimatch_model, dover_opt = load_models(device)

decord.bridge.set_bridge('torch')
vr = decord.VideoReader(VIDEO, ctx=decord.cpu(0))
frames = vr[:].numpy()

dover_result = compute_dover_score_official(frames, dover_model, dover_opt, device)

print(f"对齐实现分数: {dover_result['fused']:.4f}")
print(f"  TQE: {dover_result['tqe']:.4f}")
print(f"  AQE: {dover_result['aqe']:.4f}")

print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)

diff = abs(dover_result['fused'] - official_score)
print(f"官方分数:   {official_score:.4f}")
print(f"对齐分数:   {dover_result['fused']:.4f}")
print(f"差异:       {diff:.4f}")
print(f"对齐状态:   {'✅ 完全一致 (<0.01)' if diff < 0.01 else '❌ 存在偏差'}")
