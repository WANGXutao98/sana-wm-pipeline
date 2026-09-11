#!/usr/bin/env python3
"""验证 DOVER 修复 - 对比错误实现 vs 正确实现"""
import sys, torch, numpy as np
from pathlib import Path

VIDEO = "/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001/00d77a61-531a-58f4-acf7-da49c23af0ca.mp4"

# 加载视频
import decord
decord.bridge.set_bridge('torch')
vr = decord.VideoReader(VIDEO, ctx=decord.cpu(0))
frames = vr[:].numpy()
print(f"视频: {frames.shape}")

# 加载 DOVER
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
import yaml
from dover import DOVER

with open("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml") as f:
    opt = yaml.safe_load(f)

device = "cuda"
model = DOVER(**opt["model"]["args"]).to(device).eval()
model.load_state_dict(torch.load(
    "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth",
    map_location=device, weights_only=False
))

# 参数
IMAGENET_MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
IMAGENET_STD = torch.FloatTensor([58.395, 57.12, 57.375])

def fuse_results(tqe, aqe):
    x = (tqe - 0.1107) / 0.07355 * 0.6104 + (aqe + 0.08285) / 0.03774 * 0.3896
    return 1 / (1 + np.exp(-x))

print("\n" + "="*60)
print("错误实现 (当前批量任务)")
print("="*60)

chunk = frames[:32]
# 错误: 简单 /255
t_wrong = torch.from_numpy(chunk).float() / 255.0
t_wrong = t_wrong.permute(3, 0, 1, 2).unsqueeze(0).to(device)

views = {"technical": t_wrong, "aesthetic": t_wrong}
with torch.no_grad():
    results = model(views)

tqe_wrong = results[0].mean().item()
aqe_wrong = results[1].mean().item()
avg_wrong = (tqe_wrong + aqe_wrong) / 2

print(f"预处理: 简单 /255")
print(f"TQE (原始): {tqe_wrong:.4f}")
print(f"AQE (原始): {aqe_wrong:.4f}")
print(f"简单平均:   {avg_wrong:.4f}  ← 当前批量任务输出")
print(f"官方融合:   {fuse_results(tqe_wrong, aqe_wrong):.4f}")

del t_wrong, views, results
torch.cuda.empty_cache()

print("\n" + "="*60)
print("正确实现 (修复版)")
print("="*60)

# 正确: ImageNet 归一化
t_correct = torch.from_numpy(chunk).float().to(device)
t_correct = t_correct.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
t_correct = t_correct.permute(0, 2, 3, 4, 1)  # (1, T, H, W, C)

mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
std = IMAGENET_STD.view(1, 3, 1, 1).to(device)
t_correct = (t_correct - mean) / std
t_correct = t_correct.permute(0, 4, 1, 2, 3)  # (1, C, T, H, W)

views = {"technical": t_correct, "aesthetic": t_correct}
with torch.no_grad():
    results = model(views)

tqe_correct = results[0].mean().item()
aqe_correct = results[1].mean().item()
fused_correct = fuse_results(tqe_correct, aqe_correct)

print(f"预处理: ImageNet 归一化 (x - mean) / std")
print(f"TQE (原始): {tqe_correct:.4f}")
print(f"AQE (原始): {aqe_correct:.4f}")
print(f"官方融合:   {fused_correct:.4f}  ← 修复后输出")

print("\n" + "="*60)
print("对比总结")
print("="*60)
print(f"错误实现分数: {avg_wrong:.4f}  (错误的简单平均)")
print(f"正确实现分数: {fused_correct:.4f}  (ImageNet归一化 + 官方融合)")
print(f"差异:         {fused_correct - avg_wrong:+.4f}")
print(f"\n阈值判定 (DOVER ∈ [0.35, 1.0]):")
print(f"  错误实现: {'FAIL' if avg_wrong < 0.35 else 'PASS'}")
print(f"  正确实现: {'FAIL' if fused_correct < 0.35 else 'PASS'}")
