#!/usr/bin/env python3
"""Synthesize a varying-intrinsics (zoom) clip with EXACT per-frame GT intrinsics
from a real GT-calibrated clip.

Take a clip with real camera parallax + known base intrinsics, and apply a smooth
per-frame center-crop-zoom ramp. Because a center-crop of box (x0,y0,cw,ch) resized
back to (W,H) maps intrinsics exactly as
    fx' = fx*(W/cw),  fy' = fy*(H/ch),  cx' = (cx-x0)*(W/cw),  cy' = (cy-y0)*(H/ch),
the per-frame GT intrinsics are known to machine precision. Real texture+parallax
(so SLAM works) + exactly-known focal ramp (so per-frame-intrinsics BA can be scored).

    python3 make_zoom_clip.py <src_clip_dir> <out_dir> [zmax]

src_clip_dir: has video.mp4 (+ optional poses.npy) and intrinsics.npy ((N,4)=fx,fy,cx,cy
or (3,3)). out_dir gets video.mp4 + gt_intrinsics.npy ((N,4)) + poses.npy (copied).
"""
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

src, out = sys.argv[1], sys.argv[2]
zmax = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8
os.makedirs(out, exist_ok=True)
FF = os.path.join(os.environ.get("SANA_WM_ROOT", "."), "bin", "ffmpeg")
if not os.path.exists(FF):
    FF = "ffmpeg"

# read frames
cap = cv2.VideoCapture(os.path.join(src, "video.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
N = len(frames)
H, W = frames[0].shape[:2]
assert N > 1, "need >1 frame"

# base intrinsics
K = np.load(os.path.join(src, "intrinsics.npy"))
if K.ndim == 3:           # (N,3,3)
    fx, fy, cx, cy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
elif K.ndim == 2 and K.shape[1] == 4:   # (N,4) fx,fy,cx,cy
    fx, fy, cx, cy = K[0]
elif K.shape == (3, 3):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
else:
    raise SystemExit(f"unexpected intrinsics shape {K.shape}")
fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
print(f"src: {N} frames {W}x{H} fps~{fps:.1f}; base fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}", flush=True)

fdir = os.path.join(out, "_frames")
os.makedirs(fdir, exist_ok=True)
gt = np.zeros((N, 4), dtype=np.float32)
for i, f in enumerate(frames):
    z = 1.0 + (zmax - 1.0) * (i / (N - 1))      # linear focal ramp 1.0 -> zmax
    cw, ch = int(round(W / z)), int(round(H / z))
    x0 = (W - cw) // 2                          # crop centered on image center
    y0 = (H - ch) // 2
    crop = f[y0:y0 + ch, x0:x0 + cw]
    z_img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(fdir, f"{i:05d}.png"), z_img)
    sx, sy = W / cw, H / ch                      # EXACT scale from the actual crop box
    gt[i] = [fx * sx, fy * sy, (cx - x0) * sx, (cy - y0) * sy]

np.save(os.path.join(out, "gt_intrinsics.npy"), gt)
if os.path.exists(os.path.join(src, "poses.npy")):
    shutil.copy(os.path.join(src, "poses.npy"), os.path.join(out, "poses.npy"))

# assemble mp4 (same fps, high quality so SLAM features survive)
mp4 = os.path.join(out, "video.mp4")
subprocess.run([FF, "-hide_banner", "-y", "-framerate", str(int(round(fps))),
                "-i", os.path.join(fdir, "%05d.png"),
                "-pix_fmt", "yuv420p", "-crf", "16", "-c:v", "libx264", mp4], check=True)
shutil.rmtree(fdir)
print(f"ZOOM_CLIP_DONE {mp4}", flush=True)
print(f"GT focal ramp: f[0]={gt[0,0]:.1f} -> f[-1]={gt[-1,0]:.1f}  ({zmax:.2f}x)", flush=True)
print("gt_intrinsics.npy shape", gt.shape, flush=True)
