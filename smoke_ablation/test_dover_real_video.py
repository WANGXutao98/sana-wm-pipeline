#!/usr/bin/env python3
import sys, torch, numpy as np, subprocess
from pathlib import Path

# Load video
import decord
decord.bridge.set_bridge('torch')
vr = decord.VideoReader("/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001/00094653-a9c6-5558-8e2a-4119e7d64f36.mp4", ctx=decord.cpu(0))
frames = vr[:].numpy()  # (87, H, W, 3)
print(f"Video loaded: {frames.shape}")

# Load DOVER
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
import yaml
from dover import DOVER

device = "cuda"
with open("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml") as f:
    opt = yaml.safe_load(f)
model = DOVER(**opt["model"]["args"]).to(device).eval()
model.load_state_dict(torch.load(
    "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth",
    map_location=device, weights_only=False
))

def check_mem():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 
                       capture_output=True, text=True)
    return int(r.stdout.strip())

print(f"Model loaded: {check_mem()} MB")

# Test 80 frames (5s @ 16fps)
print(f"\n=== Testing 80 frames (论文方案) ===")
chunk = frames[:80]
try:
    t = torch.from_numpy(chunk).float().to(device) / 255.0
    t = t.permute(3, 0, 1, 2).unsqueeze(0)
    print(f"Tensor on GPU: {check_mem()} MB")
    
    views = {"technical": t, "aesthetic": t}
    print(f"Views created: {check_mem()} MB")
    
    with torch.no_grad():
        output = model(views)
    print(f"After inference: {check_mem()} MB")
    print("✅ Success")
except torch.cuda.OutOfMemoryError as e:
    print(f"❌ OOM: {str(e)[:200]}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)[:200]}")
