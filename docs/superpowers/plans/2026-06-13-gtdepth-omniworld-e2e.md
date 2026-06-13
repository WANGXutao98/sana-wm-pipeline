# GT-depth 模式修复 + OmniWorld 端到端测试实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `mode_gtdepth.py`（当前调用不存在的 VIPE `gt_depth` 后端），改为复用已有 `cached` 后端；并新建 OmniWorld 数据准备脚本，在单个 OmniWorld-Game 样本上完成 GT-depth 模式端到端验证。

**Architecture:** 
GT-depth 模式的正确实现：GT 深度图格式化为 `cached` npz → 注入已有 `vipe_cached_depth` SLAM 管线 → MoGe-2 单独推理获取度量锚点 → `fuse_metric_scale(d_gt, d_moge)` 恢复 `scale_per_frame`。无需新增 VIPE 后端。

**Tech Stack:** Python 3.10, conda env `sana_wm`, VIPE (`vipe_cached_depth` pipeline), MoGe-2, ModelScope CLI, ffmpeg, numpy

---

## 前置状态（2026-06-13 已确认）

| 状态 | 内容 |
|---|---|
| ✅ 已有 | `mode_default.py` 完整实现，`_load_vipe_artifacts` / `_interp_poses` 工作正常 |
| ✅ 已有 | `CachedDepthModel` 在 VIPE 已注册（读取 `SANA_WM_CACHED_DEPTH_PATH` 环境变量） |
| ✅ 已有 | `vipe_cached_depth.yaml` pipeline config |
| ✅ 已有 | `fuse_metric_scale` 在 `depth_fusion.py` |
| ✅ 已有 | `tests/test_pose_modes.py::test_gtdepth_mode_recovers_scale` 测试存在 |
| ❌ **broken** | `mode_gtdepth.py` 调用不存在的 `--depth-backend gt_depth` VIPE flag |
| ❌ **broken** | 测试 mock 写旧格式 `pose.json`，但新实现用 `pose/<stem>.npz` |
| ❌ 缺失 | `prepare_omniworld.py`：16-bit PNG depth → `gt_depth.npy` |
| ❌ 缺失 | `run_e2e_gtdepth.sh`：OmniWorld 端到端脚本 |

## 关键技术说明

**GT-depth 模式工作原理**（论文 App. B.1）：
```
OmniWorld GT depth (T,H,W) float32 metres
  ├─ 格式化为 cached npz (depths key) → VIPE CachedDepthModel 注入 BA
  │   └─ VIPE SLAM 输出: poses_c2w (T,4,4), intrinsics (T,1,4)
  └─ MoGe-2 逐帧推理 → d_moge (T,H,W) 度量深度锚点
       └─ fuse_metric_scale(d_gt, d_moge, momentum=0.99)
            └─ scale_per_frame (T,) — 用于 WebDataset shard 标注
```

**与 Default 模式的差别**：Default = Pi3X+MoGe-2 融合替代预测深度；GT-depth = GT 深度直接替代，MoGe-2 仅作度量锚。

**OmniWorld 深度格式**（16-bit PNG）：
- 值单位：毫米（uint16 / 1000.0 = 米）
- 文件路径：`<scene>/depth/frame_XXXXXX.png`
- 相机文件：`<scene>/camera/split_0.json`（含 intrinsics + extrinsics）

---

## 文件结构

| 操作 | 文件 | 说明 |
|---|---|---|
| **Rewrite** | `src/sana_wm_pipeline/stage02_pose/mode_gtdepth.py` | 改为 cached 后端 |
| **Update** | `tests/test_pose_modes.py` | 更新 mock 适配新实现 |
| **Create** | `experiments/data_production_smoke/prepare_omniworld.py` | OmniWorld → pipeline 格式 |
| **Create** | `experiments/data_production_smoke/run_e2e_gtdepth.sh` | GT-depth E2E 脚本 |

---

## Task 1: 修复 mode_gtdepth.py（改用 cached 后端）

**Files:**
- Rewrite: `src/sana_wm_pipeline/stage02_pose/mode_gtdepth.py`

- [ ] **Step 1.1: 完整重写 mode_gtdepth.py**

将 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage02_pose/mode_gtdepth.py` 替换为：

```python
"""GT-depth pose-annotation mode (paper App. B.1).

Targets: OmniWorld (synthetic, perfectly-known depth maps).

Pipeline:
  1. Format GT depth (.npy, T×H×W float32 metres) as CachedDepthModel npz.
  2. Run MoGe-2 per-frame to get metric depth anchor.
  3. VIPE SLAM with vipe_cached_depth pipeline (GT depth injected into BA).
  4. fuse_metric_scale(d_gt_grid, d_moge_grid) → per-frame metric scale s_t.
  5. Return PoseArtifact.

No new VIPE backend required — reuses existing `cached` depth backend.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common import PoseArtifact
from .depth_fusion import fuse_metric_scale
from .mode_default import _load_vipe_artifacts

VIPE_CMD: Sequence[str] = ("vipe", "infer")
VIPE_PIPELINE = "vipe_cached_depth"
SAMPLE_GRID = 32


def _run_moge2(
    clip_path: Path,
    moge_out: Path,
    moge2_weights: str,
    fov_x_deg: float = 60.0,
    device: str = "cuda",
) -> np.ndarray:
    """Run MoGe-2 on every frame; return (T, H, W) float32 metric depth."""
    import cv2
    import torch
    from moge.model.v2 import MoGeModel  # type: ignore

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {clip_path}")

    moge2_path = Path(moge2_weights)
    ckpt = moge2_path / "model.pt" if moge2_path.is_dir() else moge2_path
    model = MoGeModel.from_pretrained(str(ckpt)).to(device).eval()

    H, W = frames[0].shape[:2]
    depths = np.zeros((len(frames), H, W), dtype=np.float32)
    with torch.no_grad():
        for i, frame in enumerate(frames):
            ft = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model.infer(ft, fov_x=fov_x_deg)
            depths[i] = out["depth"].squeeze(0).cpu().numpy()
    del model

    np.save(str(moge_out), depths)
    return depths


def run_gtdepth(
    clip_path: Path,
    gt_depth_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = VIPE_CMD,
    pipeline: str = VIPE_PIPELINE,
) -> PoseArtifact:
    """GT-depth annotation: inject OmniWorld GT depth into VIPE BA.

    Args:
        clip_path: normalized video (.mp4), T frames.
        gt_depth_path: (T, H, W) float32 numpy file, depth in metres.
        work_dir: scratch directory; VIPE writes pose/ and intrinsics/ here.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "")
    if not moge2_weights:
        raise RuntimeError("SANA_WM_MOGE2_WEIGHTS must be set")

    # Phase 1: format GT depth as CachedDepthModel npz
    d_gt = np.load(str(gt_depth_path)).astype(np.float32)  # (T, H, W)
    cache_path = work_dir / "_gt_depth_cache.npz"
    np.savez_compressed(str(cache_path), depths=d_gt)

    # Phase 2: run MoGe-2 for metric scale anchor (skip if cached)
    moge_npy = work_dir / "_moge2_depth.npy"
    if moge_npy.exists():
        d_moge = np.load(str(moge_npy)).astype(np.float32)
    else:
        d_moge = _run_moge2(clip_path, moge_npy, moge2_weights)

    # Phase 3: VIPE SLAM with GT depth injected via CachedDepthModel
    os.environ["SANA_WM_CACHED_DEPTH_PATH"] = str(cache_path)
    try:
        cmd = [*vipe_cmd, str(clip_path), "--output", str(work_dir), "--pipeline", pipeline]
        subprocess.check_call(cmd)
    finally:
        os.environ.pop("SANA_WM_CACHED_DEPTH_PATH", None)
        cache_path.unlink(missing_ok=True)

    # Phase 4: load VIPE pose + intrinsics artifacts (same format as default mode)
    artifact = _load_vipe_artifacts(clip_path, work_dir)
    T = len(artifact.poses_c2w)

    # Phase 5: per-frame metric scale via grid-sampled GT vs MoGe-2 depths
    H_d, W_d = d_gt.shape[1], d_gt.shape[2]
    ys = np.linspace(0, H_d - 1, SAMPLE_GRID).astype(int)
    xs = np.linspace(0, W_d - 1, SAMPLE_GRID).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d_gt_grid = d_gt[:T, yy, xx].reshape(T, -1)    # (T, SAMPLE_GRID²) float32
    d_moge_grid = d_moge[:T, yy, xx].reshape(T, -1)  # (T, SAMPLE_GRID²) float32
    scale = fuse_metric_scale(d_gt_grid, d_moge_grid, momentum=0.99).astype(np.float32)

    return PoseArtifact(
        poses_c2w=artifact.poses_c2w,
        intrinsics=artifact.intrinsics,
        scale_per_frame=scale,
        depth_downsampled=artifact.depth_downsampled,
    )
```

- [ ] **Step 1.2: 语法检查**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -c "from sana_wm_pipeline.stage02_pose.mode_gtdepth import run_gtdepth; print('import OK')"
```

期望：`import OK`

---

## Task 2: 更新 test_gtdepth_mode_recovers_scale

**Files:**
- Modify: `tests/test_pose_modes.py` 中的 `test_gtdepth_mode_recovers_scale` 函数

新实现的 `run_gtdepth` 调用链：
1. `np.savez_compressed(cache_path, depths=d_gt)` — 不需要 mock
2. `_run_moge2(...)` → 需要 monkeypatch
3. `subprocess.check_call` → VIPE，写 `pose/<stem>.npz` + `intrinsics/<stem>.npz`
4. `_load_vipe_artifacts(clip_path, work_dir)` → 读上面两个文件

- [ ] **Step 2.1: 读当前 test_pose_modes.py 中 test_gtdepth_mode_recovers_scale**

当前函数在第 59-86 行。需要整体替换。

- [ ] **Step 2.2: 替换 test_gtdepth_mode_recovers_scale 函数**

将 `tests/test_pose_modes.py` 中 `test_gtdepth_mode_recovers_scale` 函数替换为：

```python
def test_gtdepth_mode_recovers_scale(monkeypatch, tmp_path: Path):
    """mode_gtdepth: GT depth=1, MoGe-2 depth=2 → scale_per_frame ≈ 2.0."""
    gt_depth = tmp_path / "gt_depth.npy"
    # GT depth array (T, H, W) — constant 1.0 metres
    np.save(gt_depth, np.full((T, 90, 160), 1.0, dtype=np.float32))

    def fake_moge2(clip_path, moge_out, moge2_weights, fov_x_deg=60.0, device="cuda"):
        # MoGe-2 returns 2.0 m everywhere → fuse_metric_scale gives s=2.0
        depths = np.full((T, 90, 160), 2.0, dtype=np.float32)
        np.save(str(moge_out), depths)
        return depths

    def fake_vipe(cmd: Sequence[str], **kw):
        # VIPE writes pose/<stem>.npz + intrinsics/<stem>.npz
        cmd = list(cmd)
        out_idx = cmd.index("--output") + 1
        work_dir = Path(cmd[out_idx])
        stem = Path(cmd[1]).stem  # clip_path is cmd[1] after "vipe infer"

        pose_dir = work_dir / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pose_dir / f"{stem}.npz",
                 data=_eye_poses(), inds=np.arange(T))

        intr_dir = work_dir / "intrinsics"
        intr_dir.mkdir(parents=True, exist_ok=True)
        intr_raw = np.tile(
            np.array([700.0, 700.0, 640.0, 360.0], dtype=np.float32), (T, 1)
        )
        np.savez(intr_dir / f"{stem}.npz",
                 data=intr_raw, inds=np.arange(T))

    monkeypatch.setenv("SANA_WM_MOGE2_WEIGHTS", "/fake/moge2")
    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_gtdepth._run_moge2",
        fake_moge2,
    )
    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_gtdepth.subprocess.check_call",
        fake_vipe,
    )

    art = mode_gtdepth.run_gtdepth(Path("clip.mp4"), gt_depth, tmp_path)
    art.validate(T)
    # fuse_metric_scale(d_gt=1, d_moge=2) = 2.0 (closed form, EMA stays at 2.0)
    assert art.scale_per_frame.mean() == pytest.approx(2.0, rel=1e-3)
    assert art.intrinsics.shape == (T, 1, 4)
```

- [ ] **Step 2.3: 运行 gtdepth 测试**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
python -m pytest tests/test_pose_modes.py::test_gtdepth_mode_recovers_scale -v
```

期望：`PASSED`

- [ ] **Step 2.4: 运行全部 pose mode 测试**

```bash
python -m pytest tests/test_pose_modes.py -v
```

期望：4/4 PASSED

- [ ] **Step 2.5: 运行全量测试**

```bash
python -m pytest -q
```

期望：141 passed (原 141 全过)

- [ ] **Step 2.6: Commit**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add src/sana_wm_pipeline/stage02_pose/mode_gtdepth.py tests/test_pose_modes.py
git commit -m "fix(stage02): rewrite mode_gtdepth to use cached depth backend + fix test mock"
```

---

## Task 3: 编写 prepare_omniworld.py

**Files:**
- Create: `experiments/data_production_smoke/prepare_omniworld.py`

OmniWorld 场景结构：
```
<scene_id>/
  color/frame_000000.png ...   # RGB, uint8
  depth/frame_000000.png ...   # uint16 depth, 值/1000 = 米
  camera/split_0.json          # intrinsics (fx,fy,cx,cy) + extrinsics per frame
  fps.txt                      # 帧率
```

- [ ] **Step 3.1: 编写 prepare_omniworld.py**

创建 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/data_production_smoke/prepare_omniworld.py`：

```python
#!/usr/bin/env python3
"""Convert a single OmniWorld-Game scene to sana_wm_pipeline input format.

Usage:
  python prepare_omniworld.py \
    --scene-dir /mnt/afs/davidwang/data/omniworld/OmniWorld-Game/b04f88d1f85a \
    --out-dir   /mnt/afs/davidwang/workspace/data/omniworld_smoke/b04f88d1f85a \
    [--depth-scale 1000.0]  # uint16 / depth_scale = metres; default 1000 (mm→m)
    [--fps 30.0]            # override if fps.txt missing

Outputs:
  video.mp4         — RGB frames as H264 video (ffmpeg)
  gt_depth.npy      — (T, H, W) float32, metres
  gt_poses.npy      — (T, 4, 4) float32, camera-to-world
  gt_intrinsics.npy — (4,) float32 [fx, fy, cx, cy] (first frame)
  orig_fps.txt      — frame rate string
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def load_camera_json(camera_dir: Path) -> dict:
    """Load camera/split_0.json (or first available split_*.json)."""
    candidates = sorted(camera_dir.glob("split_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No split_*.json found in {camera_dir}")
    return json.loads(candidates[0].read_text())


def parse_poses_and_intrinsics(cam_data: dict, T: int):
    """Extract (T, 4, 4) c2w poses and (4,) intrinsics from camera JSON.

    OmniWorld camera JSON format (inferred from paper + ICLR 2026 release):
      {
        "fx": 960.0, "fy": 960.0, "cx": 640.0, "cy": 360.0,
        "frames": [
          {"file_path": "color/frame_000000.png",
           "transform_matrix": [[r00,r01,r02,tx],[r10,...],[r20,...],[0,0,0,1]]},
          ...
        ]
      }
    If intrinsics are per-frame they may be nested inside each frame dict.
    """
    # Intrinsics — try top-level first, then per-frame
    fx = cam_data.get("fx") or cam_data["frames"][0].get("fx")
    fy = cam_data.get("fy") or cam_data["frames"][0].get("fy")
    cx = cam_data.get("cx") or cam_data["frames"][0].get("cx")
    cy = cam_data.get("cy") or cam_data["frames"][0].get("cy")
    if any(v is None for v in [fx, fy, cx, cy]):
        # Fallback: check "intrinsics" sub-key
        intr = cam_data.get("intrinsics") or cam_data["frames"][0].get("intrinsics", {})
        fx = intr.get("fx", fx)
        fy = intr.get("fy", fy)
        cx = intr.get("cx", cx)
        cy = intr.get("cy", cy)
    if any(v is None for v in [fx, fy, cx, cy]):
        raise ValueError(f"Cannot find fx/fy/cx/cy in camera JSON. Keys: {list(cam_data.keys())}")
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    # Poses — (T, 4, 4) c2w
    frames = cam_data["frames"][:T]
    if len(frames) < T:
        raise ValueError(f"Camera JSON has {len(frames)} frames, expected {T}")
    poses = np.stack(
        [np.array(f["transform_matrix"], dtype=np.float32) for f in frames],
        axis=0,
    )  # (T, 4, 4)
    return poses, intrinsics


def main():
    p = argparse.ArgumentParser(description="Prepare OmniWorld scene for sana_wm_pipeline.")
    p.add_argument("--scene-dir", required=True, type=Path,
                   help="OmniWorld scene directory (contains color/, depth/, camera/)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for pipeline artifacts")
    p.add_argument("--depth-scale", type=float, default=1000.0,
                   help="Divide uint16 depth values by this to get metres (default 1000)")
    p.add_argument("--fps", type=float, default=None,
                   help="Override FPS (reads fps.txt if omitted)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Truncate to first N frames (for smoke tests)")
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- FPS ---
    fps_txt = scene_dir / "fps.txt"
    if args.fps is not None:
        fps = args.fps
    elif fps_txt.exists():
        fps = float(fps_txt.read_text().strip())
    else:
        fps = 30.0
        print(f"[WARN] fps.txt not found; assuming {fps} fps")
    (out_dir / "orig_fps.txt").write_text(str(fps))
    print(f"FPS: {fps}")

    # --- RGB frames ---
    color_dir = scene_dir / "color"
    rgb_files = sorted(color_dir.glob("frame_*.png"))
    if not rgb_files:
        rgb_files = sorted(color_dir.glob("*.png"))
    if not rgb_files:
        print(f"[ERROR] No PNG frames found in {color_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_frames:
        rgb_files = rgb_files[:args.max_frames]
    T = len(rgb_files)
    print(f"Found {T} RGB frames")

    # Write video.mp4 via ffmpeg (lossless-ish H264)
    video_path = out_dir / "video.mp4"
    if not video_path.exists():
        first_frame = cv2.imread(str(rgb_files[0]))
        H, W = first_frame.shape[:2]
        # Write a temp file list for ffmpeg concat
        list_file = out_dir / "_frame_list.txt"
        list_file.write_text("\n".join(f"file '{f}'" for f in rgb_files) + "\n")
        subprocess.check_call([
            "ffmpeg", "-y", "-r", str(fps),
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(video_path),
        ])
        list_file.unlink()
        print(f"Video: {video_path}  ({W}×{H}, {T} frames, {fps}fps)")
    else:
        print(f"Video already exists: {video_path}")

    # --- Depth frames ---
    depth_dir = scene_dir / "depth"
    depth_files = sorted(depth_dir.glob("frame_*.png"))
    if not depth_files:
        depth_files = sorted(depth_dir.glob("*.png"))
    if len(depth_files) < T:
        print(f"[WARN] Only {len(depth_files)} depth frames for {T} RGB frames; truncating")
        T = min(T, len(depth_files))
        depth_files = depth_files[:T]

    gt_depth_path = out_dir / "gt_depth.npy"
    if not gt_depth_path.exists():
        sample_d = cv2.imread(str(depth_files[0]), cv2.IMREAD_ANYDEPTH)
        DH, DW = sample_d.shape
        depths = np.zeros((T, DH, DW), dtype=np.float32)
        for i, df in enumerate(depth_files[:T]):
            d16 = cv2.imread(str(df), cv2.IMREAD_ANYDEPTH)
            if d16 is None:
                raise RuntimeError(f"Cannot read depth: {df}")
            depths[i] = d16.astype(np.float32) / args.depth_scale  # → metres
        np.save(str(gt_depth_path), depths)
        print(f"GT depth: {gt_depth_path}  shape={depths.shape}  "
              f"range=[{depths.min():.2f}, {depths.max():.2f}]m")
    else:
        depths = np.load(str(gt_depth_path))
        print(f"GT depth already exists: {gt_depth_path}  shape={depths.shape}")

    # --- Camera (poses + intrinsics) ---
    camera_dir = scene_dir / "camera"
    if not camera_dir.exists():
        print(f"[WARN] No camera/ dir found — skipping pose/intrinsics output")
    else:
        cam_data = load_camera_json(camera_dir)
        poses, intrinsics = parse_poses_and_intrinsics(cam_data, T)
        np.save(str(out_dir / "gt_poses.npy"), poses)
        np.save(str(out_dir / "gt_intrinsics.npy"), intrinsics)
        print(f"GT poses: {out_dir/'gt_poses.npy'}  shape={poses.shape}")
        print(f"GT intrinsics: {intrinsics}  (fx,fy,cx,cy)")

    print(f"\nDone. Output in: {out_dir}")
    print(f"  video.mp4       — {T} frames @ {fps}fps")
    print(f"  gt_depth.npy    — (T={T}, H, W) float32 metres")
    if (out_dir / "gt_poses.npy").exists():
        print(f"  gt_poses.npy    — (T={T}, 4, 4) c2w")
        print(f"  gt_intrinsics.npy — [fx, fy, cx, cy]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: 验证脚本语法**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -c "import ast; ast.parse(open('experiments/data_production_smoke/prepare_omniworld.py').read()); print('syntax OK')"
```

期望：`syntax OK`

- [ ] **Step 3.3: Commit**

```bash
git add experiments/data_production_smoke/prepare_omniworld.py
git commit -m "feat(data): add prepare_omniworld.py for GT-depth mode data prep"
```

---

## Task 4: 编写 run_e2e_gtdepth.sh

**Files:**
- Create: `experiments/data_production_smoke/run_e2e_gtdepth.sh`

- [ ] **Step 4.1: 编写脚本**

创建 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/data_production_smoke/run_e2e_gtdepth.sh`：

```bash
#!/usr/bin/env bash
# GT-depth 模式端到端：OmniWorld 单场景 → WebDataset shard
#
# 用法：
#   bash experiments/data_production_smoke/run_e2e_gtdepth.sh \
#     /mnt/afs/davidwang/data/omniworld/OmniWorld-Game/<scene_id>
#
# 前置：
#   - conda env sana_wm 已激活（含 MoGe-2 + VIPE）
#   - SANA_WM_MOGE2_WEIGHTS 已设置
#   - 场景目录含 color/ depth/ camera/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 环境 ──────────────────────────────────────────────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export DISABLE_XFORMERS=1

SCENE_DIR="${1:?Usage: $0 <omniworld_scene_dir>}"
SCENE_ID="$(basename "${SCENE_DIR}")"
WORK_BASE="/mnt/afs/davidwang/workspace/data/omniworld_smoke"
PREP_DIR="${WORK_BASE}/${SCENE_ID}"
SHARDS_DIR="${WORK_BASE}/shards_gtdepth"
VIPE_WORK="${WORK_BASE}/${SCENE_ID}/vipe_work"

mkdir -p "${PREP_DIR}" "${SHARDS_DIR}" "${VIPE_WORK}"

cd "${PROJECT_ROOT}"

# ── Stage 0: 准备 OmniWorld 场景 ──────────────────────────────────────────────
echo "=== Stage 0: prepare OmniWorld scene (${SCENE_ID}) ==="
python experiments/data_production_smoke/prepare_omniworld.py \
  --scene-dir "${SCENE_DIR}" \
  --out-dir   "${PREP_DIR}"

VIDEO="${PREP_DIR}/video.mp4"
GT_DEPTH="${PREP_DIR}/gt_depth.npy"

# ── Stage 1: normalize（统一分辨率/帧率，sana_wm pipeline 标准）────────────────
echo "=== Stage 1: normalize video ==="
NORM_VIDEO="${PREP_DIR}/normalized.mp4"
if [ ! -f "${NORM_VIDEO}" ]; then
  python -c "
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_clip
normalize_clip(Path('${VIDEO}'), Path('${NORM_VIDEO}'))
print('Normalized:', '${NORM_VIDEO}')
"
fi

# ── Stage 2: GT-depth VIPE ───────────────────────────────────────────────────
echo "=== Stage 2: GT-depth VIPE SLAM ==="
python -c "
import numpy as np
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_gtdepth import run_gtdepth

clip = Path('${NORM_VIDEO}')
gt_depth = Path('${GT_DEPTH}')
work_dir = Path('${VIPE_WORK}')

print(f'GT depth shape: {np.load(str(gt_depth)).shape}')
artifact = run_gtdepth(clip, gt_depth, work_dir)
print(f'Poses: {artifact.poses_c2w.shape}')
print(f'Intrinsics: {artifact.intrinsics.shape}')
print(f'Scale mean: {artifact.scale_per_frame.mean():.4f}')

import pathlib, json
out = work_dir / 'pose_artifact.json'
out.write_text(json.dumps({
    'poses_c2w': artifact.poses_c2w.tolist(),
    'intrinsics': artifact.intrinsics.tolist(),
    'scale_per_frame': artifact.scale_per_frame.tolist(),
}))
print(f'Pose artifact saved: {out}')
"

# ── Stage 5: caption（stub）────────────────────────────────────────────────
echo "=== Stage 5: stub caption ==="
CAPTION_FILE="${PREP_DIR}/caption.txt"
echo "Indoor synthetic scene from OmniWorld-Game with known depth." > "${CAPTION_FILE}"

# ── Stage 6: pack → WebDataset shard ────────────────────────────────────────
echo "=== Stage 6: pack WebDataset shard ==="
python -c "
import json, numpy as np, tarfile, io
from pathlib import Path

scene_id  = '${SCENE_ID}'
norm_video = Path('${NORM_VIDEO}')
vipe_work  = Path('${VIPE_WORK}')
caption    = '${CAPTION_FILE}'
shards_dir = Path('${SHARDS_DIR}')

# Load artifact
art = json.loads((vipe_work / 'pose_artifact.json').read_text())
poses_c2w      = np.array(art['poses_c2w'],      dtype=np.float32)
intrinsics_nvd = np.array(art['intrinsics'],      dtype=np.float32)
scale_pf       = np.array(art['scale_per_frame'], dtype=np.float32)

# Load GT depth (downsampled 4× for shard)
gt_depth = np.load('${GT_DEPTH}').astype(np.float32)
T = len(poses_c2w)
gt_depth_ds = gt_depth[:T, ::4, ::4]

# Write shard-000001.tar
shard = shards_dir / 'shard-000001.tar'
with tarfile.open(shard, 'w') as tf:
    def add_npy(name, arr):
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        ti = tarfile.TarInfo(f'{scene_id}/{name}')
        ti.size = len(buf.getvalue())
        tf.addfile(ti, buf)

    # video.mp4
    vbytes = norm_video.read_bytes()
    ti = tarfile.TarInfo(f'{scene_id}/video.mp4')
    ti.size = len(vbytes)
    tf.addfile(ti, io.BytesIO(vbytes))

    add_npy('poses_c2w.npy',       poses_c2w)
    add_npy('intrinsics_NVD.npy',  intrinsics_nvd)
    add_npy('scale_per_frame.npy', scale_pf)
    add_npy('depth_downsampled.npy', gt_depth_ds)

    # caption
    cbytes = open(caption, 'rb').read()
    ti = tarfile.TarInfo(f'{scene_id}/caption.txt')
    ti.size = len(cbytes)
    tf.addfile(ti, io.BytesIO(cbytes))

    # meta
    meta = {'scene_id': scene_id, 'T': T, 'mode': 'gt_depth', 'dataset': 'OmniWorld'}
    mbytes = json.dumps(meta).encode()
    ti = tarfile.TarInfo(f'{scene_id}/meta.json')
    ti.size = len(mbytes)
    tf.addfile(ti, io.BytesIO(mbytes))

print(f'Shard written: {shard}  ({shard.stat().st_size/1e6:.1f} MB)')
print('Contents:', [m.name for m in tarfile.open(shard).getmembers()])
"

# ── 验证 schema ───────────────────────────────────────────────────────────────
echo "=== Schema check ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir "${SHARDS_DIR}"

echo ""
echo "=== GT-depth E2E 完成 ==="
echo "  Shard: ${SHARDS_DIR}/shard-000001.tar"
echo "  场景:  ${SCENE_ID}"
```

- [ ] **Step 4.2: 添加执行权限并验证语法**

```bash
chmod +x /mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/data_production_smoke/run_e2e_gtdepth.sh
bash -n /mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/data_production_smoke/run_e2e_gtdepth.sh
echo "Shell syntax OK"
```

期望：`Shell syntax OK`

- [ ] **Step 4.3: Commit**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add experiments/data_production_smoke/run_e2e_gtdepth.sh
git commit -m "feat(data): add run_e2e_gtdepth.sh for OmniWorld GT-depth E2E"
```

---

## Task 5: 【用户手动】OmniWorld 数据下载

> ⚠️ 此 Task 需要您手动执行。以下命令已全部准备完毕，直接运行即可。

**Step 5.1: 安装 ModelScope SDK（如未安装）**

```bash
conda activate sana_wm
pip install modelscope
```

**Step 5.2: 下载单个 OmniWorld-Game 场景**

```bash
# 下载场景 b04f88d1f85a（OmniWorld-Game 中的已知示例场景）
# 约 ~2-5 GB（含 color/ depth/ camera/ 等）

python -c "
from modelscope.hub.file_download import model_file_download
from modelscope.hub.snapshot_download import snapshot_download

# 下载整个场景目录
snapshot_download(
    'InternRobotics/OmniWorld',
    repo_type='dataset',
    revision='master',
    allow_patterns=['OmniWorld-Game/b04f88d1f85a/*'],
    local_dir='/mnt/afs/davidwang/data/omniworld',
    ignore_file_pattern=None,
)
print('Download complete')
"
```

如果 snapshot_download 不支持 allow_patterns，备用命令：

```bash
# 备用：使用 git lfs + sparse-checkout（慢但稳）
cd /mnt/afs/davidwang/data
git clone https://www.modelscope.cn/datasets/InternRobotics/OmniWorld.git omniworld_repo
cd omniworld_repo
git sparse-checkout set OmniWorld-Game/b04f88d1f85a
git lfs pull
```

**Step 5.3: 验证下载完整性**

```bash
ls /mnt/afs/davidwang/data/omniworld/OmniWorld-Game/b04f88d1f85a/
# 期望输出: color/  depth/  camera/  fps.txt  split_info.json

ls /mnt/afs/davidwang/data/omniworld/OmniWorld-Game/b04f88d1f85a/depth/ | head -5
# 期望: frame_000000.png frame_000001.png ...

# 验证深度图格式（应为 16-bit uint16）
python -c "
import cv2, numpy as np
d = cv2.imread('/mnt/afs/davidwang/data/omniworld/OmniWorld-Game/b04f88d1f85a/depth/frame_000000.png', cv2.IMREAD_ANYDEPTH)
print(f'dtype={d.dtype} shape={d.shape} max={d.max()} (÷1000={d.max()/1000:.2f}m)')
"
# 期望: dtype=uint16, max 值约 5000-30000（对应 5-30 米）
```

**若场景 ID `b04f88d1f85a` 不存在**，先运行以下命令查看可用场景：

```bash
# 获取数据集文件列表（只下 metadata）
python -c "
from modelscope.hub.api import HubApi
api = HubApi()
files = api.get_dataset_file_list('InternRobotics/OmniWorld', revision='master')
# 打印 OmniWorld-Game 下的场景目录名（前20个）
game_scenes = [f['Name'] for f in files if 'OmniWorld-Game/' in f.get('Path','')][:20]
print(game_scenes)
"
```

---

## Task 6: 运行 GT-depth 端到端测试（数据下载后执行）

> 前置：Task 5（数据下载）已完成

- [ ] **Step 6.1: 依赖检查**

```bash
conda activate sana_wm
python -c "
import cv2, numpy as np, torch
from moge.model.v2 import MoGeModel
import vipe
print('All dependencies OK')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  MoGe-2 weights: /mnt/afs/davidwang/models/moge2/model.pt exists:', end=' ')
from pathlib import Path; print(Path('/mnt/afs/davidwang/models/moge2/model.pt').exists())
"
```

期望：`All dependencies OK` + GPU 信息 + `True`

- [ ] **Step 6.2: 运行 E2E（约 15-30 分钟：MoGe-2 ~5min + VIPE SLAM ~10-20min）**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
SCENE_ID=b04f88d1f85a  # 替换为实际下载的场景 ID

nohup bash experiments/data_production_smoke/run_e2e_gtdepth.sh \
  /mnt/afs/davidwang/data/omniworld/OmniWorld-Game/${SCENE_ID} \
  > /mnt/afs/davidwang/workspace/data/omniworld_smoke/run_gtdepth_${SCENE_ID}.log 2>&1 &

echo "PID: $!  —  监控: tail -f /mnt/afs/davidwang/workspace/data/omniworld_smoke/run_gtdepth_${SCENE_ID}.log"
```

- [ ] **Step 6.3: 验证结果**

```bash
# Schema check
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir /mnt/afs/davidwang/workspace/data/omniworld_smoke/shards_gtdepth

# 检查 pose artifact（scale_per_frame 应约为 1.0，因为 MoGe-2 对合成场景精度高）
python -c "
import json, numpy as np
from pathlib import Path
art = json.loads(Path('/mnt/afs/davidwang/workspace/data/omniworld_smoke/b04f88d1f85a/vipe_work/pose_artifact.json').read_text())
poses = np.array(art['poses_c2w'])
scale = np.array(art['scale_per_frame'])
print(f'T={len(poses)} frames')
print(f'scale_per_frame: mean={scale.mean():.4f} std={scale.std():.4f}')
print(f'poses[0]: {poses[0]}')
"
```

期望：`T > 0`；`scale_per_frame` 接近 1.0（GT depth 与 MoGe-2 尺度接近）；`poses[0]` 接近单位矩阵。

- [ ] **Step 6.4: Commit 结果日志**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add docs/operation_logs/  # 如果有新的运行日志
git commit -m "exp(data): OmniWorld GT-depth E2E smoke PASSED (scene=${SCENE_ID})"
```

---

## 验证标准

| 检查项 | 期望 |
|---|---|
| `python -m pytest tests/test_pose_modes.py -v` | 4/4 PASSED |
| `python -m pytest -q` | 141 passed |
| E2E schema check | shard-000001.tar 包含 6 个必需文件 |
| `scale_per_frame` 均值 | 0.8~1.5（GT depth ≈ MoGe-2 尺度，合成数据精度高） |
| 无 FileNotFoundError / RuntimeError | VIPE 正常退出 |

---

## 注意事项

1. **VIPE 环境变量顺序**：`SANA_WM_CACHED_DEPTH_PATH` 必须在 `subprocess.check_call` 前设置，在 finally 块清理。新实现已处理。
2. **MoGe-2 fov_x 估计**：使用默认 60°（合成数据内参已知，但 VIPE 会用 GeoCalib 估计，MoGe-2 仅作度量锚，fov 估计误差影响不大）。
3. **深度图单位**：OmniWorld 使用 uint16 / 1000.0 = metres。若实际数据不符，调整 `--depth-scale` 参数。
4. **帧数对齐**：`prepare_omniworld.py` 中 RGB 帧数 = 深度帧数 = poses 数目，三者严格对齐。
5. **首帧归一化**：`_load_vipe_artifacts` 中 `_interp_poses` 已确保 `poses[0] ≈ I₄`（论文 App. D.3）。

---

## 相关文档索引

| 文档 | 路径 |
|---|---|
| GT-pose vs Default 对比 | `docs/operation_logs/2026-06-13-gtpose-vs-default-mode-experiment.md` |
| DL3DV E2E 实施记录 | `docs/operation_logs/2026-06-12-dl3dv-e2e-implementation.md` |
| 数据集下载指南 | `docs/DATASETS.md` |
| mode_default 参考实现 | `src/sana_wm_pipeline/stage02_pose/mode_default.py` |
