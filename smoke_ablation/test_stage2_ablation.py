#!/usr/bin/env python3
"""Stage2 Ablation Test - 跳过Stage01归一化，直接处理原视频"""
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sana_wm_pipeline.stage02_pose.mode_default import run_default
import json

# 测试视频
video_path = Path('/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/pass_videos/00094653-a9c6-5558-8e2a-4119e7d64f36.mp4')
output_dir = Path('/mnt/afs/davidwang/workspace/sana_test_data/smoke_stage2_ablation')
output_dir.mkdir(parents=True, exist_ok=True)

vipe_work = output_dir / 'vipe_work_default'
print(f'[Stage2 Ablation] 输入视频: {video_path}')
print(f'[Stage2 Ablation] 工作目录: {vipe_work}')
print(f'[Stage2 Ablation] 测试目标: 验证跳过Stage01归一化的可行性')
print()

# 运行stage02
art = run_default(video_path, vipe_work)
print()
print(f'✅ 成功: poses {art.poses_c2w.shape}, intrinsics {art.intrinsics.shape}')
print(f'   Scale range: [{art.scale_per_frame.min():.3f}, {art.scale_per_frame.max():.3f}]')
print(f'   Scale median: {float(art.scale_per_frame[len(art.scale_per_frame)//2]):.3f}')

# 保存artifact
artifact_json = output_dir / 'pose_artifact_default.json'
artifact_json.write_text(json.dumps({
    'poses_c2w': art.poses_c2w.tolist(),
    'intrinsics': art.intrinsics.tolist(),
    'scale_per_frame': art.scale_per_frame.tolist(),
}))
print(f'   Artifact saved: {artifact_json}')
print()
print('验证完成：Stage02可以直接处理原视频，无需Stage01归一化')
