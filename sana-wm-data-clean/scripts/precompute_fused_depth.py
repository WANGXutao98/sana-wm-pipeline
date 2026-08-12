#!/usr/bin/env python3
"""Precompute Pi3X + MoGe-2 fused metric depth for a clip (torch-2.5 env).

Runs in the env where Pi3/MoGe already work and writes one ``NNNNNN.npy`` metric
depth map per clip frame into <out_dir>, which VIPE's pi3xmoge backend then loads
(decoupled from VIPE's torch 2.7). Pi3 is run on an evenly-sampled subset (memory)
and each full-clip frame is assigned its nearest sampled fused depth.

    SANA_WM_WEIGHTS=... PYTHONPATH=<repo>:<Pi3repo> \
      python3 scripts/precompute_fused_depth.py <video.mp4> <out_dir> [max_frames]
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_wm_data.pose import _real
from sana_wm_data.pose.adapters import read_frames
from sana_wm_data.pose.fusion import fuse_depth_sequence


def main():
    video, out_dir = sys.argv[1], sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("SANA_WM_MAX_FRAMES", "64"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # total frame count (decord), and the sampled subset Pi3 runs on
    import decord
    n_total = len(decord.VideoReader(str(video)))
    frames = read_frames(video, max_frames)                  # (S,H,W,3) evenly sampled
    s = frames.shape[0]
    sample_idx = np.linspace(0, n_total - 1, s).round().astype(int)

    _poses, pi3_depth = _real.pi3_infer(frames)              # (S,h,w)
    moge_depth = _real.moge_metric_depth(frames, ref_hw=pi3_depth.shape[1:])  # (S,h,w)
    fused, scales = fuse_depth_sequence(pi3_depth, np.abs(moge_depth), ema_momentum=0.99)

    # Per-frame signatures so VIPE's backend can match an incoming keyframe RGB to its
    # fused depth without a frame index. RGB 16x16 (768-d) not gray 8x8 (64-d): color +
    # 4x spatial resolution make near-duplicate frames separable, so the nearest-signature
    # match can't hand a wrong-frame depth into BA. The matcher in pi3x_moge.py uses the
    # identical signature; keep the two in lock-step if either changes.
    import cv2
    sig = np.stack([
        cv2.resize(f, (16, 16)).astype(np.float32).ravel()
        for f in frames
    ])  # (S, 768) RGB
    np.save(out / "fused.npy", fused.astype(np.float32))   # (S, h, w)
    np.save(out / "sig.npy", sig)                          # (S, 64)
    np.save(out / "sample_idx.npy", sample_idx)
    # per-frame metric scale s_t (S,): the pose stage's gt_depth/default modes use
    # this for the scale-CoV camera filter. The real-VIPE backend (pose/vipe_cli.py)
    # loads it from here instead of recomputing the fusion.
    np.save(out / "scales.npy", np.asarray(scales, dtype=np.float32))
    print(f"PRECOMPUTE_DONE fused{fused.shape} ({s} sampled of {n_total}), scale~{float(np.median(scales)):.3f} -> {out}")


if __name__ == "__main__":
    main()
