#!/usr/bin/env python3
"""
生成 Method A vs B 可视化对比：
  1. side_by_side.mp4  — RGB | Depth-A | Depth-B 三列视频
  2. trajectory.png    — 已由 evaluate.py 生成的俯视轨迹图
  3. depth_sample/     — 每隔50帧抽样深度帧 PNG（肉眼预览）
"""
import zipfile
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath

BASE = Path("experiments/vipe_comparison/results")
OUT  = BASE / "viz"
OUT.mkdir(exist_ok=True)
SAMPLE_DIR = OUT / "depth_sample"
SAMPLE_DIR.mkdir(exist_ok=True)

def read_exr_depth(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    data = zf.read(name)
    tmp = Path("/tmp/_tmp_depth.exr")
    tmp.write_bytes(data)
    f = OpenEXR.InputFile(str(tmp))
    dw = f.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    ch = f.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(ch, dtype=np.float32).reshape(h, w)
    return depth

def colorize_depth(d: np.ndarray) -> np.ndarray:
    valid = d[d > 0]
    if len(valid) == 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    d_norm = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    colored = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored[d <= 0] = 0
    return colored

print("[render] Opening depth zips...")
za = zipfile.ZipFile(BASE / "method_A/depth/video.zip")
zb = zipfile.ZipFile(BASE / "method_B/depth/video.zip")
names = sorted(za.namelist())
N = len(names)

# --- 1. side_by_side.mp4 ---
cap = cv2.VideoCapture(str(BASE / "method_A/rgb/video.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 10
w0  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h0  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_w, out_h = w0 * 3, h0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(str(OUT / "side_by_side.mp4"), fourcc, fps, (out_w, out_h))

LABEL_A = "A: unidepth-l"
LABEL_B = "B: MoGe-2"
FONT = cv2.FONT_HERSHEY_SIMPLEX

print(f"[render] Writing side_by_side.mp4 ({N} frames @ {w0}x{h0})...")
for i, name in enumerate(names):
    ret, rgb = cap.read()
    if not ret:
        break
    da = colorize_depth(read_exr_depth(za, name))
    db = colorize_depth(read_exr_depth(zb, name))

    # labels
    cv2.putText(rgb, f"RGB #{i:04d}", (8, 24), FONT, 0.7, (255,255,255), 2)
    cv2.putText(da,  LABEL_A,         (8, 24), FONT, 0.7, (255,255,255), 2)
    cv2.putText(db,  LABEL_B,         (8, 24), FONT, 0.7, (255,255,255), 2)

    row = np.concatenate([rgb, da, db], axis=1)
    vw.write(row)

    # sample frames every 50
    if i % 50 == 0:
        cv2.imwrite(str(SAMPLE_DIR / f"frame_{i:04d}.png"), row)
        print(f"  [sample] frame {i:04d} saved")

vw.release()
cap.release()
print(f"[render] side_by_side.mp4 written → {OUT}/side_by_side.mp4")

# --- 2. depth diff video ---
cap_a = cv2.VideoCapture(str(BASE / "method_A/rgb/video.mp4"))
out_diff = cv2.VideoWriter(str(OUT / "depth_diff.mp4"), fourcc, fps, (w0, h0))

print("[render] Writing depth_diff.mp4...")
za2 = zipfile.ZipFile(BASE / "method_A/depth/video.zip")
zb2 = zipfile.ZipFile(BASE / "method_B/depth/video.zip")
for name in names:
    da = read_exr_depth(za2, name)
    db = read_exr_depth(zb2, name)
    valid = (da > 0) & (db > 0)
    diff = np.zeros_like(da)
    if valid.any():
        diff[valid] = np.abs(da[valid] - db[valid])
    hi = np.percentile(diff[valid], 95) if valid.any() else 1.0
    diff_norm = np.clip(diff / (hi + 1e-6), 0, 1)
    diff_vis = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    diff_vis[~valid] = 0
    cv2.putText(diff_vis, "|Depth_A - Depth_B|", (8, 24), FONT, 0.7, (255,255,255), 2)
    out_diff.write(diff_vis)

out_diff.release()
cap_a.release()
print(f"[render] depth_diff.mp4 written → {OUT}/depth_diff.mp4")

print("\n[done] Output files:")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        print(f"  {f}  ({f.stat().st_size//1024} KB)")
