"""VIPE patch — replace the default depth back-end with our Pi3X + MoGe-2 fusion.

Paper App. B.1 protocol (per clip):
  1. Pi3X infers (T,H,W) long-sequence-consistent depth + 3D structure.
  2. MoGe-2 infers (T,H,W) per-frame metric depth.
  3. SLAM tracks (track_id -> (frame, (u, v))) are read from VIPE's front-end.
  4. fuse_metric_scale (App. B.1 closed-form + EMA momentum 0.99) returns
     per-frame scale `s_t`.
  5. The patched back-end returns `s_t * d_pi3x` as the "metric depth" fed
     into the rest of VIPE (BA + global pose graph).

This file is copied into `third_party/vipe/vipe/backends/` by the setup
script — it cannot live inside the upstream tree without our edits.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from sana_wm_pipeline.stage02_pose.depth_fusion import fuse_metric_scale


GRID_SAMPLE_SIZE = 16     # per-frame fallback grid when tracks are sparse.
MIN_TRACKS_PER_FRAME = 16


class Pi3XMoGe2Depth:
    """VIPE-compatible depth back-end wrapper around Pi3X + MoGe-2."""

    def __init__(self, pi3x_model, moge2_model, device: str = "cuda",
                 ema_momentum: float = 0.99):
        self.pi3x = pi3x_model
        self.moge2 = moge2_model
        self.device = device
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def __call__(self, frames_thwc: np.ndarray,
                 slam_tracks: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute metric depth + per-frame scale.

        Args:
          frames_thwc: (T,H,W,3) uint8 RGB.
          slam_tracks: list of dicts with keys {track_id, frame_id, uv}.

        Returns:
          metric_depth: (T,H,W) float32, == Pi3X depth * per-frame scale.
          scale_per_frame: (T,) float32.
        """
        d_pi3x = self.pi3x.infer_video(frames_thwc).astype(np.float32)
        d_moge = self.moge2.infer_video(frames_thwc).astype(np.float32)
        T, H, W = d_pi3x.shape
        if d_moge.shape != (T, H, W):
            raise ValueError(
                f"Pi3X/MoGe-2 shape mismatch: {d_pi3x.shape} vs {d_moge.shape}"
            )

        pi3x_pts, moge_pts = self._gather_paired_depths(d_pi3x, d_moge, slam_tracks)
        scale = fuse_metric_scale(pi3x_pts, moge_pts, momentum=self.ema_momentum)
        metric = d_pi3x * scale[:, None, None]
        return metric.astype(np.float32), scale.astype(np.float32)

    @staticmethod
    def _gather_paired_depths(
        d_pi3x: np.ndarray, d_moge: np.ndarray, slam_tracks: List[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (T, K) arrays of paired Pi3X / MoGe-2 depths.

        Pads with NaN so `fuse_metric_scale`'s validity filter excludes them.
        """
        T, H, W = d_pi3x.shape
        # group tracks by frame
        by_frame: dict[int, List[Tuple[int, int]]] = {}
        for tr in slam_tracks:
            t = int(tr["frame_id"])
            u, v = tr["uv"]
            by_frame.setdefault(t, []).append((int(round(v)), int(round(u))))

        pi3x_rows: List[np.ndarray] = []
        moge_rows: List[np.ndarray] = []
        for t in range(T):
            pts = by_frame.get(t, [])
            if len(pts) >= MIN_TRACKS_PER_FRAME:
                ys = np.array([p[0] for p in pts], dtype=int).clip(0, H - 1)
                xs = np.array([p[1] for p in pts], dtype=int).clip(0, W - 1)
                pi3x_rows.append(d_pi3x[t, ys, xs])
                moge_rows.append(d_moge[t, ys, xs])
            else:
                ys = np.linspace(0, H - 1, GRID_SAMPLE_SIZE).astype(int)
                xs = np.linspace(0, W - 1, GRID_SAMPLE_SIZE).astype(int)
                yy, xx = np.meshgrid(ys, xs, indexing="ij")
                pi3x_rows.append(d_pi3x[t, yy, xx].flatten())
                moge_rows.append(d_moge[t, yy, xx].flatten())
        max_n = max(len(r) for r in pi3x_rows)
        pad = lambda a: np.pad(a, (0, max_n - len(a)), constant_values=np.nan)
        return (
            np.stack([pad(r) for r in pi3x_rows]).astype(np.float32),
            np.stack([pad(r) for r in moge_rows]).astype(np.float32),
        )
