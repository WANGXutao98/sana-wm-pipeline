# GT-depth 模式修复 + OmniWorld Smoke Test 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `mode_gtdepth.py`（当前使用 VIPE 不存在的 flags），并在 OmniWorld 单样本上端到端验证 GT-depth 管线，产出 WebDataset shard + pose 评估结果。

**Architecture:** GT-depth 模式与 Default 模式架构完全一致（两阶段 VIPE），唯一区别是用 OmniWorld GT depth 替代 Pi3X 推理。新 `mode_gtdepth.py` 复用 `mode_default._load_vipe_artifacts` 和 `depth_fusion.fuse_metric_scale`，加入 MoGe-2 metric anchor。准备脚本 `prepare_omniworld.py` 将 OmniWorld clip 转换为管线标准格式（video.mp4 + gt_depth.npy + gt_poses.npy + gt_intrinsics.npy）。

**Tech Stack:** Python 3.10, conda env `sana_wm`, MoGe-2, VIPE (`vipe infer --pipeline vipe_cached_depth`), evo (ATE 评估)

---

## 背景：GT-depth 模式技术原理（论文 App. B.1）

### 三种模式对比

| 模式 | 目标数据集 | 深度来源 | SLAM 输入 |
|---|---|---|---|
| **GT-pose** | DL3DV, Sekai-Game | Pi3X（仅用于 Umeyama 尺度对齐） | 不跑 SLAM，直接用 GT 位姿 |
| **Default** | SpatialVID-HQ, MiraData | Pi3X（几何形状）+ MoGe-2（metric anchor）→ EMA 融合 | `vipe infer --pipeline vipe_cached_depth` |
| **GT-depth** | OmniWorld | GT depth（完美几何）+ MoGe-2（metric anchor）→ EMA 融合 | `vipe infer --pipeline vipe_cached_depth` |

### GT-depth 具体流程

```
OmniWorld GT depth (d_gt: T,H,W) — 合成场景完美深度，任意绝对单位
    ↓
MoGe-2 逐帧推理 → d_moge (T,H,W) — metric anchor（绝对米制）
    ↓ EMA scale fusion（论文 App. B.1，depth_fusion.fuse_metric_scale）
  scale_t = EMA(Σ(w·d_gt·d_moge)/Σ(w·d_gt²))   w=1/d_gt
  fused_depth = d_gt × scale_t                  [metric-scaled GT shape]
    ↓ 写入 _depth_cache.npz {depths:(T,H,W), scale_history:(T,)}
  SANA_WM_CACHED_DEPTH_PATH = _depth_cache.npz
    ↓
  vipe infer clip.mp4 --output work_dir --pipeline vipe_cached_depth
  └─ CachedDepthModel 按帧号查表注入 SLAM BA
  └─ GeoCalib 估计内参（optimize_intrinsics=True）
    ↓
  poses_c2w (T,4,4) + intrinsics (T,1,4)  [VIPE artifacts]
```

### 当前 mode_gtdepth.py 的 Bug

```python
# ❌ 当前实现（VIPE 不支持这些 flags）
cmd = [
    *vipe_cmd,
    "--video", str(clip_path),
    "--depth-backend", "gt_depth",      # VIPE 没有这个 flag
    "--gt-depth-path", str(gt_depth_path),  # VIPE 没有这个 flag
    "--emit-moge2", str(moge_npy),      # VIPE 没有这个 flag
    "--per-frame-intrinsics",
    "--out", str(pose_json),
]
```

测试之所以 pass 是因为 `subprocess.check_call` 被 monkeypatch 了，真实运行必然报错。

### 输入数据格式要求

`run_gtdepth(clip_path, gt_depth_path, work_dir)` 需要：
- `clip_path`: 视频文件（.mp4，16fps，1280×720，与 gt_depth 帧数对齐）
- `gt_depth_path`: `(T, H, W)` float32 npy 文件，深度值 > 0（任意单位，MoGe-2 恢复 metric）
- `SANA_WM_MOGE2_WEIGHTS` 环境变量（Pi3X 不需要，因为用 GT 深度代替）

---

## 文件变更清单

| 操作 | 文件 |
|---|---|
| **完全重写** | `src/sana_wm_pipeline/stage02_pose/mode_gtdepth.py` |
| 更新测试 | `tests/test_pose_modes.py` — `test_gtdepth_mode_recovers_scale` |
| 新建 | `experiments/data_production_smoke/prepare_omniworld.py` |
| 新建 | `experiments/data_production_smoke/run_e2e_gtdepth.sh` |

`configs/filter_thresholds.yaml` 已有 `OmniWorld` 条目（vmaf_motion/unimatch_flow/dover/vlm 全部配置），**无需修改**。

---

## Task 1: 重写 mode_gtdepth.py（核心修复）

**Files:**
- Modify (完全重写): `src/sana_wm_pipeline/stage02_pose/mode_gtdepth.py`

- [ ] **Step 1.1: 写失败测试（先确认旧测试反映正确行为）**

在 `tests/test_pose_modes.py` 中，找到 `test_gtdepth_mode_recovers_scale` 并替换为新的测试，使其 monkeypatch `_precompute_gt_depth_cache` 而非 `subprocess.check_call`（因为新实现的 `subprocess.check_call` 接收 VIPE 格式 args，输出 npz 而非 pose.json）：

```python
def test_gtdepth_mode_recovers_scale(monkeypatch, tmp_path: Path):
    """GT-depth mode uses two-phase approach: precompute cache + VIPE.
    scale_per_frame comes from EMA fusion of GT depth vs MoGe-2.
    """
    T_frames = T
    poses = _eye_poses(T_frames)
    intr_raw = np.tile(
        np.array([700.0, 700.0, 640.0, 360.0], dtype=np.float32), (T_frames, 1)
    )
    gt_depth = tmp_path / "gt_depth.npy"
    np.save(gt_depth, np.full((T_frames, 90, 160), 1.0, dtype=np.float32))

    def fake_precompute(clip_path, gt_depth_path, cache_path, scale_path,
                        moge2_weights, device="cuda"):
        # Simulate: GT depth=1.0, MoGe-2=2.0 → scale=2.0
        np.savez_compressed(
            str(cache_path),
            depths=np.full((T_frames, 90, 160), 2.0, dtype=np.float32),
            scale_history=np.full(T_frames, 2.0, dtype=np.float32),
        )
        np.save(str(scale_path), np.full(T_frames, 2.0, dtype=np.float32))

    def fake_vipe(cmd, **kw):
        cmd = list(cmd)
        out_idx = cmd.index("--output") + 1
        work_dir = Path(cmd[out_idx])
        stem = "clip"
        pose_dir = work_dir / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pose_dir / f"{stem}.npz",
                 data=poses, inds=np.arange(T_frames))
        intr_dir = work_dir / "intrinsics"
        intr_dir.mkdir(parents=True, exist_ok=True)
        np.savez(intr_dir / f"{stem}.npz",
                 data=intr_raw, inds=np.arange(T_frames))

    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_gtdepth._precompute_gt_depth_cache",
        fake_precompute,
    )
    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_gtdepth.subprocess.check_call",
        fake_vipe,
    )
    monkeypatch.setenv("SANA_WM_MOGE2_WEIGHTS", "/fake/moge2")

    art = mode_gtdepth.run_gtdepth(Path("clip.mp4"), gt_depth, tmp_path)
    art.validate(T)
    assert isinstance(art, PoseArtifact)
    assert art.intrinsics.shape == (T, 1, 4)
    assert art.scale_per_frame.shape == (T,)
    # scale_history=2.0 → PoseArtifact.scale_per_frame ≈ 2.0
    assert art.scale_per_frame.mean() == pytest.approx(2.0, rel=1e-3)
```

- [ ] **Step 1.2: 运行失败测试确认 FAIL**

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_pose_modes.py::test_gtdepth_mode_recovers_scale -v
```

期望：FAIL（因为旧实现的 monkeypatch 路径不对）。

- [ ] **Step 1.3: 完全重写 mode_gtdepth.py**

用以下内容替换 `src/sana_wm_pipeline/stage02_pose/mode_gtdepth.py`：

```python
"""GT-depth pose-annotation mode (paper App. B.1).

Targets: OmniWorld (synthetic, has perfectly-known depth).
Pipeline:
  1. Load GT depth (T,H,W) from .npy file (any unit, may be relative).
  2. MoGe-2 per-frame inference → metric anchor d_moge.
  3. EMA scale fusion: scale_t = EMA(d_moge / d_gt) → fused = d_gt * scale_t.
  4. Write fused depths to _depth_cache.npz (same format as mode_default).
  5. VIPE infer --pipeline vipe_cached_depth (CachedDepthModel injects BA).
  6. Load VIPE artifacts; attach EMA scale to PoseArtifact.scale_per_frame.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common import PoseArtifact
from .depth_fusion import fuse_metric_scale
from .mode_default import VIPE_CMD, _load_vipe_artifacts

VIPE_PIPELINE = "vipe_cached_depth"
SAMPLE_GRID = 32


def _precompute_gt_depth_cache(
    clip_path: Path,
    gt_depth_path: Path,
    cache_path: Path,
    scale_path: Path,
    moge2_weights: str,
    device: str = "cuda",
) -> None:
    """Load GT depth, run MoGe-2 metric anchor, EMA-fuse → write cache + scale.

    Writes:
      cache_path.npz — {depths:(T,H,W) float32, scale_history:(T,) float32}
      scale_path.npy — (T,) float32, same as scale_history (survives cache deletion)
    """
    import cv2
    import torch

    # ── 1. Load GT depth ─────────────────────────────────────────────────────
    d_gt = np.load(gt_depth_path).astype(np.float32)  # (T, H, W)
    if d_gt.ndim != 3:
        raise ValueError(f"GT depth must be (T,H,W), got {d_gt.shape}")
    T_gt, H, W = d_gt.shape

    # ── 2. Read video frames for MoGe-2 ──────────────────────────────────────
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    T_vid = len(frames)
    T = min(T_gt, T_vid)
    frames_np = np.stack(frames[:T], axis=0).astype(np.float32) / 255.0
    d_gt = d_gt[:T]
    frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)  # (T,3,H,W)

    # ── 3. MoGe-2 per-frame inference ────────────────────────────────────────
    from moge.model.v2 import MoGeModel  # type: ignore

    moge2_path = Path(moge2_weights)
    moge2_ckpt = moge2_path / "model.pt" if moge2_path.is_dir() else moge2_path
    moge2_model = MoGeModel.from_pretrained(str(moge2_ckpt)).to(device).eval()
    fov_x = math.degrees(2 * math.atan(W / (2 * 525.0)))
    d_moge_list = []
    with torch.no_grad():
        for i in range(T):
            out = moge2_model.infer(frames_t[i : i + 1], fov_x=fov_x)
            d_moge_list.append(out["depth"].squeeze(0).cpu().numpy())
    d_moge = np.stack(d_moge_list, axis=0)  # (T, H, W)
    del moge2_model
    torch.cuda.empty_cache()

    # ── 4. EMA scale fusion (论文 App. B.1) ──────────────────────────────────
    ys = np.linspace(0, H - 1, SAMPLE_GRID).astype(int)
    xs = np.linspace(0, W - 1, SAMPLE_GRID).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d_gt_pts = d_gt[:, yy, xx].reshape(T, -1).astype(np.float64)
    d_moge_pts = d_moge[:, yy, xx].reshape(T, -1).astype(np.float64)
    scale_history = fuse_metric_scale(d_gt_pts, d_moge_pts, momentum=0.99).astype(np.float32)

    # Forward-fill NaN (degenerate frames carry last valid scale)
    last_valid = 1.0
    for i in range(len(scale_history)):
        if np.isnan(scale_history[i]):
            scale_history[i] = last_valid
        else:
            last_valid = float(scale_history[i])

    depths_fused = (d_gt * scale_history[:, None, None]).astype(np.float32)

    # ── 5. Write cache ────────────────────────────────────────────────────────
    np.savez_compressed(str(cache_path), depths=depths_fused, scale_history=scale_history)
    np.save(str(scale_path), scale_history)  # survives cache deletion in finally block


def run_gtdepth(
    clip_path: Path,
    gt_depth_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = VIPE_CMD,
    pipeline: str = VIPE_PIPELINE,
) -> PoseArtifact:
    """Two-phase VIPE: precompute GT-depth+MoGe-2 cache, then cached SLAM.

    Phase A: MoGe-2 provides metric anchor; fused = GT depth * EMA_scale.
    Phase B: VIPE vipe_cached_depth (CachedDepthModel injects BA).
    scale_per_frame is set to the EMA scale recovered in Phase A.

    Requires env var SANA_WM_MOGE2_WEIGHTS (Pi3X not needed).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "")
    if not moge2_weights:
        raise RuntimeError("SANA_WM_MOGE2_WEIGHTS must be set")

    cache_path = work_dir / "_depth_cache.npz"
    scale_path = work_dir / "_scale_history.npy"

    _precompute_gt_depth_cache(
        clip_path, gt_depth_path, cache_path, scale_path,
        moge2_weights=moge2_weights,
    )

    os.environ["SANA_WM_CACHED_DEPTH_PATH"] = str(cache_path)
    try:
        cmd = [
            *vipe_cmd,
            str(clip_path),
            "--output", str(work_dir),
            "--pipeline", pipeline,
        ]
        subprocess.check_call(cmd)
    finally:
        os.environ.pop("SANA_WM_CACHED_DEPTH_PATH", None)
        cache_path.unlink(missing_ok=True)

    artifact = _load_vipe_artifacts(clip_path, work_dir)

    # Override scale_per_frame with EMA-recovered GT-depth scale
    if scale_path.exists():
        scale_hist = np.load(scale_path)
        scale_path.unlink(missing_ok=True)
        T_full = artifact.poses_c2w.shape[0]
        if len(scale_hist) != T_full:
            # Resample if VIPE dropped/interpolated frames
            scale_hist = np.interp(
                np.linspace(0, len(scale_hist) - 1, T_full),
                np.arange(len(scale_hist)),
                scale_hist,
            ).astype(np.float32)
        artifact.scale_per_frame = scale_hist.astype(np.float32)

    return artifact
```

- [ ] **Step 1.4: 运行新测试确认 PASS**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_pose_modes.py -v
```

期望：4 个 pose mode 测试全部 PASS（default / gtdepth / gtpose / validate）。

- [ ] **Step 1.5: 运行全量测试确认不回归**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/ -q --tb=short 2>&1 | tail -10
```

期望：141 passed（新测试替换了旧的，总数不变）。

- [ ] **Step 1.6: Commit**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add src/sana_wm_pipeline/stage02_pose/mode_gtdepth.py tests/test_pose_modes.py
git commit -m "fix: rewrite mode_gtdepth to use vipe_cached_depth pipeline (GT depth replaces Pi3X)"
```

---

## Task 2: 创建 prepare_omniworld.py

**Files:**
- Create: `experiments/data_production_smoke/prepare_omniworld.py`

OmniWorld 是合成数据集，每个 clip 包含 RGB 帧 + 深度图 + 相机参数。本脚本将一个 OmniWorld clip 转换为管线标准格式。

- [ ] **Step 2.1: 调查 OmniWorld 数据格式**

在用户下载样本后，运行以下命令查看结构：

```bash
find /mnt/afs/davidwang/workspace/data/omniworld_smoke -type f | head -30
ls -lh /mnt/afs/davidwang/workspace/data/omniworld_smoke/
```

OmniWorld 预期结构（根据 InternRobotics 数据集惯例）：
```
<clip_id>/
  rgb/000000.png, 000001.png, ...    # uint8 RGB
  depth/000000.exr 或 depth.npy     # float32 depth（米或归一化）
  camera.json                        # fx, fy, cx, cy, c2w poses
```

或可能为 HDF5/Zarr:
```
<clip_id>.h5  →  keys: rgb (T,H,W,3), depth (T,H,W), camera_K (3,3), poses_c2w (T,4,4)
```

- [ ] **Step 2.2: 创建 prepare_omniworld.py（支持文件夹和 HDF5 两种格式）**

创建 `experiments/data_production_smoke/prepare_omniworld.py`：

```python
#!/usr/bin/env python3
"""Convert an OmniWorld clip to sana_wm_pipeline standard format.

Outputs (written into <clip_dir>/):
  video.mp4         — RGB video at orig fps (from rgb/ frames)
  gt_depth.npy      — (T, H, W) float32, depth in meters (or raw units)
  gt_poses.npy      — (T, 4, 4) float32, c2w poses
  gt_intrinsics.npy — (4,) float32, [fx, fy, cx, cy]
  orig_fps.txt      — original frame rate

Usage:
  python prepare_omniworld.py <clip_dir>        # folder format
  python prepare_omniworld.py <clip.h5>         # HDF5 format
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _save_video_from_frames(rgb_dir: Path, out_mp4: Path, fps: float) -> None:
    """Use ffmpeg to pack sorted RGB frames into MP4."""
    frames = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")
    # ffmpeg glob pattern
    first = frames[0]
    pattern = str(first.parent / f"%06d{first.suffix}")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    # Try static-ffmpeg if system ffmpeg missing
    try:
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL)


def _load_depth_folder(depth_dir: Path) -> np.ndarray:
    """Load depth frames from folder: EXR or 16-bit PNG with /1000 scale."""
    depth_files = sorted(depth_dir.glob("*.exr")) or sorted(depth_dir.glob("*.png"))
    if not depth_files:
        raise FileNotFoundError(f"No depth files in {depth_dir}")
    frames = []
    for f in depth_files:
        if f.suffix == ".exr":
            try:
                import OpenEXR, Imath  # type: ignore
                exr_file = OpenEXR.InputFile(str(f))
                header = exr_file.header()
                dw = header["dataWindow"]
                W = dw.max.x - dw.min.x + 1
                H = dw.max.y - dw.min.y + 1
                raw = exr_file.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
                depth_frame = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
            except ImportError:
                # Try imageio as fallback
                import imageio.v3 as iio
                depth_frame = iio.imread(str(f), plugin="EXR-FI").squeeze()
        else:
            import cv2
            d = cv2.imread(str(f), cv2.IMREAD_ANYDEPTH)
            depth_frame = d.astype(np.float32) / 1000.0  # mm → m
        frames.append(depth_frame)
    return np.stack(frames, axis=0).astype(np.float32)


def prepare_folder(clip_dir: Path) -> None:
    """Process a clip in folder format: rgb/ + depth/ + camera.json."""
    clip_dir = Path(clip_dir)

    # ── FPS ──────────────────────────────────────────────────────────────────
    fps = 30.0  # OmniWorld default; override if metadata available
    meta_candidates = ["metadata.json", "camera.json", "meta.json", "params.json"]
    meta = {}
    for c in meta_candidates:
        p = clip_dir / c
        if p.exists():
            meta = json.loads(p.read_text())
            break
    if "fps" in meta:
        fps = float(meta["fps"])
    elif "frame_rate" in meta:
        fps = float(meta["frame_rate"])

    # ── RGB → video.mp4 ──────────────────────────────────────────────────────
    rgb_dir = clip_dir / "rgb"
    if not rgb_dir.exists():
        rgb_dir = clip_dir / "color"
    if not rgb_dir.exists():
        rgb_dir = clip_dir / "images"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found under {clip_dir}")

    out_mp4 = clip_dir / "video.mp4"
    if not out_mp4.exists():
        print(f"  Creating video.mp4 from {rgb_dir} ...")
        _save_video_from_frames(rgb_dir, out_mp4, fps)
    T = len(sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg")))

    # ── GT depth ─────────────────────────────────────────────────────────────
    gt_depth_path = clip_dir / "gt_depth.npy"
    if not gt_depth_path.exists():
        depth_dir = clip_dir / "depth"
        depth_npy = clip_dir / "depth.npy"
        depth_npz = clip_dir / "depth.npz"
        if depth_dir.exists():
            print(f"  Loading depth from {depth_dir} ...")
            d = _load_depth_folder(depth_dir)
        elif depth_npy.exists():
            d = np.load(str(depth_npy)).astype(np.float32)
        elif depth_npz.exists():
            d = np.load(str(depth_npz))["depth"].astype(np.float32)
        else:
            raise FileNotFoundError(f"No depth found under {clip_dir}")
        np.save(str(gt_depth_path), d[:T])
        print(f"  gt_depth.npy saved: shape={d[:T].shape}")
    else:
        print(f"  gt_depth.npy already exists: {gt_depth_path}")

    # ── Camera poses & intrinsics ─────────────────────────────────────────────
    gt_poses_path = clip_dir / "gt_poses.npy"
    gt_intrinsics_path = clip_dir / "gt_intrinsics.npy"
    if not gt_poses_path.exists():
        # Try reading from metadata JSON
        if not meta:
            raise FileNotFoundError(
                f"Camera metadata not found. Expected one of: {meta_candidates}"
            )
        # Common OmniWorld formats:
        # 1) meta["frames"][i]["transform_matrix"] (NeRF convention)
        # 2) meta["poses_c2w"] (T,4,4) array
        # 3) meta["extrinsics"][i] (c2w 4x4)
        if "frames" in meta:
            poses = np.array(
                [f["transform_matrix"] for f in meta["frames"]], dtype=np.float32
            )
        elif "poses_c2w" in meta:
            poses = np.array(meta["poses_c2w"], dtype=np.float32)
        elif "extrinsics" in meta:
            poses = np.array(meta["extrinsics"], dtype=np.float32)
        else:
            raise KeyError(
                f"Cannot find poses in metadata keys: {list(meta.keys())}"
            )
        np.save(str(gt_poses_path), poses[:T])
        print(f"  gt_poses.npy saved: shape={poses[:T].shape}")

        # Intrinsics
        if "fl_x" in meta:
            intr = np.array([meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]], dtype=np.float32)
        elif "intrinsics" in meta:
            K = np.array(meta["intrinsics"])
            intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
        elif "camera_K" in meta:
            K = np.array(meta["camera_K"])
            intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
        else:
            # Fallback: estimate from image size
            import cv2
            frame0 = sorted((rgb_dir).glob("*.png"))[0]
            img = cv2.imread(str(frame0))
            h, w = img.shape[:2]
            f = max(h, w)
            intr = np.array([f, f, w / 2, h / 2], dtype=np.float32)
            print(f"  [WARN] Intrinsics not found, using fallback: fx=fy={f}")
        np.save(str(gt_intrinsics_path), intr)
        print(f"  gt_intrinsics.npy saved: {intr}")
    else:
        print(f"  gt_poses.npy + gt_intrinsics.npy already exist")

    # ── FPS file ─────────────────────────────────────────────────────────────
    (clip_dir / "orig_fps.txt").write_text(f"{fps:.1f}")
    print(f"  orig_fps.txt: {fps:.1f}")
    print(f"[DONE] {clip_dir.name}: T={T} frames, fps={fps}")


def prepare_hdf5(h5_path: Path) -> None:
    """Process a clip in HDF5 format."""
    import h5py  # type: ignore

    out_dir = h5_path.parent / h5_path.stem
    out_dir.mkdir(exist_ok=True)
    with h5py.File(str(h5_path), "r") as f:
        print(f"  HDF5 keys: {list(f.keys())}")
        # RGB
        rgb_key = next((k for k in f.keys() if "rgb" in k.lower() or "color" in k.lower()), None)
        if rgb_key is None:
            raise KeyError(f"No RGB key found in {h5_path}")
        rgb = f[rgb_key][:]  # (T, H, W, 3)

        # Depth
        depth_key = next((k for k in f.keys() if "depth" in k.lower()), None)
        if depth_key is None:
            raise KeyError(f"No depth key found in {h5_path}")
        depth = f[depth_key][:].astype(np.float32)  # (T, H, W)

        # Poses
        pose_key = next(
            (k for k in f.keys() if "pose" in k.lower() or "extrinsic" in k.lower()), None
        )
        poses = f[pose_key][:].astype(np.float32) if pose_key else None

        # Intrinsics
        intr_key = next(
            (k for k in f.keys() if "intrinsic" in k.lower() or "camera_k" in k.lower()), None
        )
        intr_raw = f[intr_key][:] if intr_key else None
        fps = float(f.attrs.get("fps", 30.0))

    T = len(rgb)
    # Save RGB as PNG sequence, then convert to mp4
    rgb_dir = out_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    import cv2
    for i, frame in enumerate(rgb):
        cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out_mp4 = out_dir / "video.mp4"
    if not out_mp4.exists():
        _save_video_from_frames(rgb_dir, out_mp4, fps)

    np.save(str(out_dir / "gt_depth.npy"), depth)
    if poses is not None:
        np.save(str(out_dir / "gt_poses.npy"), poses)
    if intr_raw is not None:
        K = np.array(intr_raw)
        if K.shape == (3, 3):
            intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
        elif K.ndim == 1 and len(K) == 4:
            intr = K.astype(np.float32)
        else:
            intr = np.array([K.ravel()[0], K.ravel()[4], K.ravel()[2], K.ravel()[5]], dtype=np.float32)
        np.save(str(out_dir / "gt_intrinsics.npy"), intr)
    (out_dir / "orig_fps.txt").write_text(f"{fps:.1f}")
    print(f"[DONE] {h5_path.name} → {out_dir}: T={T} frames")


def main():
    parser = argparse.ArgumentParser(description="Prepare OmniWorld clip for sana_wm_pipeline.")
    parser.add_argument("clip_path", type=Path, help="Clip directory or .h5 file")
    args = parser.parse_args()

    p = args.clip_path
    if p.suffix in (".h5", ".hdf5"):
        prepare_hdf5(p)
    elif p.is_dir():
        prepare_folder(p)
    else:
        print(f"ERROR: {p} is not a directory or .h5 file", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.3: 语法检查**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
python -c "import ast; ast.parse(open('experiments/data_production_smoke/prepare_omniworld.py').read()); print('Syntax OK')"
```

期望：`Syntax OK`

- [ ] **Step 2.4: Commit**

```bash
git add experiments/data_production_smoke/prepare_omniworld.py
git commit -m "feat: add prepare_omniworld.py for OmniWorld GT-depth smoke test"
```

---

## Task 3: 创建 run_e2e_gtdepth.sh

**Files:**
- Create: `experiments/data_production_smoke/run_e2e_gtdepth.sh`

- [ ] **Step 3.1: 创建端到端脚本**

创建 `experiments/data_production_smoke/run_e2e_gtdepth.sh`：

```bash
#!/usr/bin/env bash
# End-to-end smoke test: OmniWorld GT-depth mode (Stage 01→02_gtdepth→05→06)
# Stage 04 (filter) is skipped: smoke env lacks UniMatch/DOVER deps.
#
# Usage: bash experiments/data_production_smoke/run_e2e_gtdepth.sh [DATA_DIR]
# Default DATA_DIR: /mnt/afs/davidwang/workspace/data/omniworld_smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 环境 ──────────────────────────────────────────────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
# GT-depth mode 只需要 MoGe-2，不需要 Pi3X
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

DATA_DIR="${1:-/mnt/afs/davidwang/workspace/data/omniworld_smoke}"
OUT_DIR="/mnt/afs/davidwang/workspace/data/omniworld_smoke_shards_gtdepth"
mkdir -p "${OUT_DIR}"

cd "${PROJECT_ROOT}"

# ── 找到所有 clip 目录 ────────────────────────────────────────────────────────
mapfile -t SCENES < <(
  ls -d "${DATA_DIR}"/*/  2>/dev/null | grep -v "shards" || true
)

if [ "${#SCENES[@]}" -eq 0 ]; then
  echo "ERROR: No clip directories found under ${DATA_DIR}"
  echo "  Run: python experiments/data_production_smoke/prepare_omniworld.py <clip_dir>"
  exit 1
fi
echo "Found ${#SCENES[@]} clips in ${DATA_DIR}"

for SCENE_DIR in "${SCENES[@]}"; do
  SCENE_DIR="${SCENE_DIR%/}"
  SCENE_ID="$(basename "${SCENE_DIR}")"
  echo ""
  echo "===== Clip: ${SCENE_ID} ====="

  WORK_DIR="${OUT_DIR}/work/${SCENE_ID}"
  mkdir -p "${WORK_DIR}"

  # ── Step 0: prepare (若 video.mp4 / gt_depth.npy 不存在则运行 prepare) ────
  if [ ! -f "${SCENE_DIR}/video.mp4" ] || [ ! -f "${SCENE_DIR}/gt_depth.npy" ]; then
    echo "  [Step 0] Preparing OmniWorld clip..."
    python "${SCRIPT_DIR}/prepare_omniworld.py" "${SCENE_DIR}"
  else
    echo "  [Step 0] video.mp4 + gt_depth.npy already exist, skipping."
  fi

  GT_DEPTH="${SCENE_DIR}/gt_depth.npy"
  GT_POSES="${SCENE_DIR}/gt_poses.npy"
  if [ ! -f "${GT_DEPTH}" ]; then
    echo "ERROR: gt_depth.npy not found at ${GT_DEPTH}"
    exit 1
  fi

  # ── Step 1: Normalize (→ 1280×720 @ 16fps) ───────────────────────────────
  NORM_VIDEO="${WORK_DIR}/normalized.mp4"
  if [ ! -f "${NORM_VIDEO}" ]; then
    echo "  [Step 1] Normalizing to 1280×720 @ 16fps..."
    python - <<PYEOF
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
from pathlib import Path
info = normalize_video(Path("${SCENE_DIR}/video.mp4"), Path("${NORM_VIDEO}"))
print(f"  Normalized: {info.n_frames} frames @ {info.fps} fps  ({info.width}x{info.height})")
PYEOF
  else
    echo "  [Step 1] normalized.mp4 already exists, skipping."
  fi

  # ── Step 1b: 同步 GT depth 到归一化帧数 ──────────────────────────────────
  # normalize_video 重采样到 16fps；gt_depth 需要同步降采样
  NORM_DEPTH="${WORK_DIR}/gt_depth_normalized.npy"
  if [ ! -f "${NORM_DEPTH}" ]; then
    echo "  [Step 1b] Resampling GT depth to match normalized frame count..."
    python - <<PYEOF
import numpy as np
from pathlib import Path
import cv2

# 获取归一化后帧数
cap = cv2.VideoCapture("${NORM_VIDEO}")
T_norm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# 加载原始 GT depth
d_orig = np.load("${GT_DEPTH}")  # (T_orig, H, W)
T_orig = len(d_orig)

# 读取原始 fps
orig_fps_file = Path("${SCENE_DIR}/orig_fps.txt")
orig_fps = float(orig_fps_file.read_text()) if orig_fps_file.exists() else 30.0

# 按时间戳采样（16fps 对应原始 fps 的哪些帧）
target_fps = 16.0
t_norm = np.arange(T_norm) / target_fps
t_orig = np.arange(T_orig) / orig_fps
# 对每个 T_norm 时间戳找最近的原始帧
indices = np.round(np.interp(t_norm, t_orig, np.arange(T_orig))).astype(int)
indices = np.clip(indices, 0, T_orig - 1)
d_resampled = d_orig[indices]
np.save("${NORM_DEPTH}", d_resampled)
print(f"  GT depth resampled: {T_orig} → {T_norm} frames (orig_fps={orig_fps}, target=16fps)")
PYEOF
  else
    echo "  [Step 1b] gt_depth_normalized.npy already exists, skipping."
  fi

  # ── Step 2: GT-depth pose estimation ─────────────────────────────────────
  POSE_DIR="${WORK_DIR}/pose_work"
  POSE_ARTIFACT="${WORK_DIR}/pose_artifact.npz"
  if [ ! -f "${POSE_ARTIFACT}" ]; then
    echo "  [Step 2] GT-depth pose estimation (MoGe-2 metric anchor + VIPE SLAM)..."
    python - <<PYEOF
from sana_wm_pipeline.stage02_pose import mode_gtdepth
from pathlib import Path
import numpy as np
art = mode_gtdepth.run_gtdepth(
    Path("${NORM_VIDEO}"),
    Path("${NORM_DEPTH}"),
    Path("${POSE_DIR}"),
)
np.savez_compressed(
    "${POSE_ARTIFACT}",
    poses_c2w=art.poses_c2w,
    intrinsics=art.intrinsics,
    scale_per_frame=art.scale_per_frame,
)
T = art.poses_c2w.shape[0]
s_mean = art.scale_per_frame.mean()
print(f"  Pose artifact saved: T={T} frames, scale_mean={s_mean:.4f}  -> ${POSE_ARTIFACT}")
PYEOF
  else
    echo "  [Step 2] pose_artifact.npz already exists, skipping."
  fi

  # ── Step 3 (Stage 04): Filter — SKIPPED ──────────────────────────────────
  echo "  [Step 3/Stage04] Filter skipped (no UniMatch/DOVER in smoke env)."

  # ── Step 4 (Stage 05): Caption — stub fallback ───────────────────────────
  CAPTION_FILE="${WORK_DIR}/caption.txt"
  if [ ! -f "${CAPTION_FILE}" ]; then
    echo "  [Step 4/Stage05] Caption stub..."
    python - <<PYEOF
from sana_wm_pipeline.stage05_caption.qwen35_vl_runner import CAPTION_FALLBACK
from pathlib import Path
Path("${CAPTION_FILE}").write_text(CAPTION_FALLBACK, encoding="utf-8")
print(f"  Caption: {CAPTION_FALLBACK}")
PYEOF
  fi

  # ── Step 5 (Stage 06): Pack WebDataset shard ─────────────────────────────
  echo "  [Step 5/Stage06] Packing WebDataset shard (strict_frames=False)..."
  python - <<PYEOF
import numpy as np
from pathlib import Path
from sana_wm_pipeline.stage06_pack.schema import Sample
from sana_wm_pipeline.stage06_pack.webdataset_writer import ShardWriter

data       = np.load("${POSE_ARTIFACT}")
poses_c2w  = data["poses_c2w"].astype(np.float32)
intrinsics = data["intrinsics"].astype(np.float32)
scale      = data["scale_per_frame"].astype(np.float32)
caption    = Path("${CAPTION_FILE}").read_text(encoding="utf-8").strip()

sample = Sample(
    sample_id="${SCENE_ID}",
    video_path="${NORM_VIDEO}",
    poses_c2w=poses_c2w,
    intrinsics_NVD=intrinsics,
    scale_per_frame=scale,
    caption=caption,
    meta={
        "source": "OmniWorld",
        "pose_mode": "gtdepth",
        "scene_id": "${SCENE_ID}",
    },
)
with ShardWriter("${OUT_DIR}", samples_per_shard=100, strict_frames=False) as w:
    w.write(sample)
print(f"  Shard written to ${OUT_DIR}")
PYEOF

  echo "  [DONE] ${SCENE_ID}"
done

echo ""
echo "===== All clips complete ====="
echo "Shards at: ${OUT_DIR}"
```

- [ ] **Step 3.2: 语法检查 + chmod**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
bash -n experiments/data_production_smoke/run_e2e_gtdepth.sh && echo "Syntax OK"
chmod +x experiments/data_production_smoke/run_e2e_gtdepth.sh
```

期望：`Syntax OK`

- [ ] **Step 3.3: 更新 configs/filter_thresholds.yaml（已有 OmniWorld，添加 smoke 用的宽松版）**

`filter_thresholds.yaml` 已有 `OmniWorld` 完整配置（unimatch_flow/dover 等都设了阈值），smoke test 中这些 filter 会被跳过（Stage 04 被 skip），无需修改。**跳过此步。**

- [ ] **Step 3.4: Commit**

```bash
git add experiments/data_production_smoke/run_e2e_gtdepth.sh
git commit -m "feat: add run_e2e_gtdepth.sh for OmniWorld GT-depth smoke test"
```

---

## Task 4: Pose 评估脚本扩展（支持 OmniWorld GT poses）

**Files:**
- Modify: `experiments/data_production_smoke/verify_and_eval.py`（小扩展）

`verify_and_eval.py` 已支持从 shard 提取 poses 并与 `gt_poses.npy` 对比。OmniWorld 与 DL3DV 目录结构相同（都有 `gt_poses.npy`），只需确认 `--scenes-dir` 指向正确路径。

- [ ] **Step 4.1: 确认 verify_and_eval.py 的兼容性**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python experiments/data_production_smoke/verify_and_eval.py --help
```

期望：输出 `--mode {schema,pose-eval}` 选项。

如果 pose-eval 模式依赖 `gt_poses.npy`（OmniWorld 有），则兼容。检查：

```bash
grep -n "gt_poses" experiments/data_production_smoke/verify_and_eval.py | head -10
```

如果不兼容（查找逻辑写死 DL3DV 路径），则在 Step 4.2 修复。

- [ ] **Step 4.2: 若需要修复路径查找逻辑**

在 `verify_and_eval.py` 的 pose-eval 部分，确保 scene 目录查找逻辑支持 OmniWorld 的平坦结构：

```python
# 当前可能的路径查找逻辑
scene_dir = scenes_dir / sample_id  # 直接子目录，OmniWorld 兼容
gt_poses_path = scene_dir / "gt_poses.npy"
```

若需修改，将写死的 `/1K/` 子路径移除，改为通用查找：
```python
for candidate in [scenes_dir / sample_id, scenes_dir / "1K" / sample_id]:
    if candidate.exists():
        scene_dir = candidate
        break
```

- [ ] **Step 4.3: Commit（若有修改）**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add experiments/data_production_smoke/verify_and_eval.py
git commit -m "fix: verify_and_eval supports both DL3DV (1K/) and OmniWorld (flat) layouts"
```

---

## Task 5: （用户手动）下载 OmniWorld 样本

**这是唯一需要用户手动操作的步骤。**

### 数据集页面访问

```
https://modelscope.cn/datasets/InternRobotics/OmniWorld/files
```

### 随机样本选取原则（论文对齐）

- 选包含**连续相机运动**（非静态）的 clip
- 时长 ≥ 5 秒（≥ 150 帧 @ 30fps）
- 室内/室外均可（OmniWorld 是合成场景）

### 下载命令

**方案 A：用 modelscope CLI（推荐）**

```bash
pip install modelscope -q

# 下载单个 clip（替换 <CLIP_ID> 为在网页上选取的样本 ID）
modelscope download \
  --dataset InternRobotics/OmniWorld \
  --local_dir /mnt/afs/davidwang/workspace/data/omniworld_smoke \
  --include "<CLIP_ID>/*"
```

**方案 B：下载整个数据集的一个分片**

```bash
modelscope download \
  --dataset InternRobotics/OmniWorld \
  --local_dir /mnt/afs/davidwang/workspace/data/omniworld_smoke \
  --include "data/train/shard_00000/*"
```

**方案 C：Python API**

```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load(
    'InternRobotics/OmniWorld',
    split='train',
    subset_name='default',
)
# 取第一个样本
sample = ds[0]
print(sample.keys())  # 查看数据格式
```

### 下载后确认

```bash
find /mnt/afs/davidwang/workspace/data/omniworld_smoke -type f | head -20
ls -lh /mnt/afs/davidwang/workspace/data/omniworld_smoke/
```

---

## Task 6: 端到端运行与验证（用户下载后执行）

**Files:** 仅运行，无代码修改

- [ ] **Step 6.1: 运行 prepare_omniworld.py**

```bash
conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 替换 <CLIP_DIR> 为下载的 clip 目录路径
python experiments/data_production_smoke/prepare_omniworld.py \
  /mnt/afs/davidwang/workspace/data/omniworld_smoke/<CLIP_DIR>
```

期望输出：
```
Creating video.mp4 from .../rgb ...
gt_depth.npy saved: shape=(T, H, W)
gt_poses.npy saved: shape=(T, 4, 4)
gt_intrinsics.npy saved: [fx fy cx cy]
orig_fps.txt: 30.0
[DONE] <CLIP_ID>: T=XXX frames, fps=30.0
```

- [ ] **Step 6.2: 运行 GT-depth 端到端**

```bash
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface

nohup bash experiments/data_production_smoke/run_e2e_gtdepth.sh \
  /mnt/afs/davidwang/workspace/data/omniworld_smoke \
  > /mnt/afs/davidwang/workspace/data/omniworld_smoke_shards_gtdepth/run_gtdepth.log 2>&1 &
echo "PID: $!"
tail -f /mnt/afs/davidwang/workspace/data/omniworld_smoke_shards_gtdepth/run_gtdepth.log
```

预期耗时：MoGe-2 推理 ~1-3min（按帧数），VIPE SLAM ~5-15min，共 ~10-20min（单 clip）。

- [ ] **Step 6.3: Schema 校验**

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir /mnt/afs/davidwang/workspace/data/omniworld_smoke_shards_gtdepth
```

期望：`N/N shards valid`

- [ ] **Step 6.4: Pose 评估（若 OmniWorld 有 GT poses）**

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir /mnt/afs/davidwang/workspace/data/omniworld_smoke_shards_gtdepth \
  --scenes-dir /mnt/afs/davidwang/workspace/data/omniworld_smoke \
  --out-dir /mnt/afs/davidwang/workspace/data/omniworld_smoke_shards_gtdepth/eval_output
```

期望：ATE RMSE 输出（GT-depth 模式应比 Default 模式低，接近 GT-pose 数量级）。

---

## 期望结果与验证标准

| 检查项 | 期望 |
|---|---|
| `mode_gtdepth.py` 重写后 | 141 pytest passed（不回归） |
| `prepare_omniworld.py` 语法 | 无错误 |
| `run_e2e_gtdepth.sh` | GT-depth shard schema PASS |
| `scale_per_frame` 均值 | 接近 1.0（OmniWorld GT 深度已是绝对 metric，MoGe-2 scale≈1） |
| ATE RMSE（GT-depth） | 显著低于 Default 模式（0.127m），预期 <0.05m |

---

## 注意事项

1. **帧数对齐**：OmniWorld GT depth 帧数必须与归一化后视频帧数一致。`run_e2e_gtdepth.sh` 的 Step 1b 处理了降采样。
2. **内存**：MoGe-2 逐帧推理（无 Pi3X），GPU 内存需求低于 Default 模式（约 4-8GB）。
3. **VIPE 配置**：使用与 Default 模式相同的 `vipe_cached_depth.yaml`，`CachedDepthModel` 从 `SANA_WM_CACHED_DEPTH_PATH` 读取 GT+MoGe-2 融合深度。
4. **OmniWorld 格式**：`prepare_omniworld.py` 支持 folder（rgb/ + depth/ + camera.json）和 HDF5 两种格式。下载后先检查格式，若不匹配请告知，可快速扩展脚本。
5. **Pi3X 不需要**：GT-depth 模式不调用 Pi3X，`SANA_WM_PI3X_WEIGHTS` 无需设置。
