#!/usr/bin/env python3
import sys, torch, numpy as np
from pathlib import Path

# Load DOVER
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
import yaml
from dover import DOVER

device = "cuda"
with open("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml") as f:
    opt = yaml.safe_load(f)
model = DOVER(**opt["model"]["args"])
model.load_state_dict(torch.load(
    "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth",
    map_location=device, weights_only=False
))
model = model.to(device).eval()

print("Model loaded, checking memory...")
import subprocess
result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 
                       capture_output=True, text=True)
print(f"After model load: {result.stdout.strip()} MB")

# Test different chunk sizes
for chunk_size in [32, 64, 80, 96]:
    frames = np.random.randint(0, 255, (chunk_size, 224, 224, 3), dtype=np.uint8)
    print(f"\nTesting {chunk_size} frames...")
    
    try:
        t = torch.from_numpy(frames).float().to(device) / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0)
        
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 
                               capture_output=True, text=True)
        print(f"  After tensor load: {result.stdout.strip()} MB")
        
        views = {"technical": t, "aesthetic": t}
        with torch.no_grad():
            output = model(views)
        
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 
                               capture_output=True, text=True)
        print(f"  After inference: {result.stdout.strip()} MB")
        print(f"  ✅ Success")
        
        del t, views, output
        torch.cuda.empty_cache()
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ OOM: {e}")
        break
