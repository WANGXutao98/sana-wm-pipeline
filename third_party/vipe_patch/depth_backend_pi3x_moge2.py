"""VIPE-compatible depth backend: Pi3X (consistent) + MoGe-2 (metric scale).

Paper App. B.1 protocol:
  1. Pi3X infers (T,H,W) long-sequence-consistent relative depth.
  2. MoGe-2 infers (T,H,W) per-frame metric depth.
  3. EMA-momentum-0.99 closed-form scale fusion (App. B.1) gives per-frame
     scale s_t that converts Pi3X depth to metric.

Integration: drop into vipe/priors/depth/ and register in vipe/priors/depth/__init__.py
as make_depth_model("pi3x_moge2") -> Pi3XMoGe2DepthModel().

Install prerequisites:
  pip install git+https://github.com/yyfz/Pi3.git   # Pi3X code
  hf download yyfz233/Pi3X --local-dir <weights>    # Pi3X weights
  pip install git+https://github.com/microsoft/MoGe.git  # MoGe code
  hf download Ruicheng/moge-2-vitl-normal --local-dir <weights>  # MoGe-2 weights
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


def _fuse_ema(d_consistent: np.ndarray, d_metric: np.ndarray,
              momentum: float = 0.99) -> np.ndarray:
    """Closed-form EMA scale fusion (App. B.1).

    s_t = Σ(w · a · b) / Σ(w · a²),  w = 1/a  (inverse-depth weighting)
    where a = d_consistent[t], b = d_metric[t].
    """
    T = d_consistent.shape[0]
    scale = np.ones(T, dtype=np.float32)
    ema_s = None
    for t in range(T):
        a = d_consistent[t].flatten().astype(np.float64)
        b = d_metric[t].flatten().astype(np.float64)
        valid = (a > 1e-6) & (b > 1e-6) & np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 16:
            s_t = ema_s if ema_s is not None else 1.0
        else:
            av, bv = a[valid], b[valid]
            w = 1.0 / np.clip(av, 1e-6, None)
            s_t = float(np.sum(w * av * bv) / np.sum(w * av * av))
        ema_s = s_t if ema_s is None else momentum * ema_s + (1 - momentum) * s_t
        scale[t] = float(ema_s)
    return scale


class Pi3XMoGe2DepthModel(DepthEstimationModel):
    """Pi3X + MoGe-2 fused depth for SANA-WM (App. B.1).

    Environment variables:
      SANA_WM_PI3X_WEIGHTS  — path to yyfz233/Pi3X local_dir (required)
      SANA_WM_MOGE2_WEIGHTS — path to Ruicheng/moge-2-vitl-normal local_dir (required)
    """

    def __init__(self, device: str = "cuda", ema_momentum: float = 0.99) -> None:
        super().__init__()
        self.device = device
        self.ema_momentum = ema_momentum
        self._pi3x: Optional[object] = None
        self._moge2: Optional[object] = None
        self._video_buffer: list[np.ndarray] = []

    def _lazy_load(self) -> None:
        if self._pi3x is not None:
            return

        pi3x_weights = os.environ.get("SANA_WM_PI3X_WEIGHTS")
        moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS")

        if pi3x_weights is None or moge2_weights is None:
            raise RuntimeError(
                "Set SANA_WM_PI3X_WEIGHTS and SANA_WM_MOGE2_WEIGHTS env vars "
                "to the local weight directories before using Pi3XMoGe2DepthModel."
            )

        try:
            from pi3 import Pi3  # type: ignore
            self._pi3x = Pi3.from_pretrained(pi3x_weights).to(self.device).eval()
        except ImportError as e:
            raise RuntimeError(
                "Pi3X (pip install git+https://github.com/yyfz/Pi3.git) not found."
            ) from e

        try:
            from moge.model import MoGeModel  # type: ignore
            self._moge2 = MoGeModel.from_pretrained(moge2_weights).to(self.device).eval()
        except ImportError as e:
            raise RuntimeError(
                "MoGe (pip install git+https://github.com/microsoft/MoGe.git) not found."
            ) from e

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """Process a single frame (VIPE calls this per-frame during SLAM).

        Pi3X requires the full video context for sequence-consistent depth.
        We buffer all frames and run Pi3X in batch on the first call after the
        video is complete (video_frame_list is not None).

        For per-frame SLAM keyframe calls (rgb only), we fall back to MoGe-2
        metric depth directly (Pi3X sequence context not available yet).
        """
        self._lazy_load()

        # Video-sequence mode (post-SLAM depth alignment): process full clip.
        if src.video_frame_list is not None:
            return self._estimate_video(src)

        # Per-frame mode (SLAM keyframe): use MoGe-2 only.
        return self._estimate_single(src)

    @torch.no_grad()
    def _estimate_single(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb = unpack_optional(src.rgb).to(self.device)
        if rgb.dim() == 3:
            rgb = rgb[None]
        # MoGe-2 inference
        inp = rgb.permute(0, 3, 1, 2)  # (1,3,H,W)
        fov_x = None
        if src.intrinsics is not None:
            fx = src.intrinsics[0].item()
            w = rgb.shape[2]
            import math
            fov_x = math.degrees(2 * math.atan(w / (2 * fx)))
        out = self._moge2.infer(inp, fov_x=fov_x)  # type: ignore[union-attr]
        depth = out["depth"].squeeze(0)  # (H,W)
        return DepthEstimationResult(metric_depth=depth)

    @torch.no_grad()
    def _estimate_video(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """Run Pi3X (full sequence) + MoGe-2 (per-frame) and fuse."""
        frames = src.video_frame_list  # list of (H,W,3) float32 [0,1]
        assert frames is not None
        T = len(frames)

        # Pi3X: batch (T, H, W, 3) -> consistent depth (T, H, W)
        frames_np = np.stack(frames, axis=0)  # (T,H,W,3)
        frames_t = torch.from_numpy(frames_np).to(self.device).permute(0, 3, 1, 2)  # (T,3,H,W)
        pi3x_out = self._pi3x.infer(frames_t)  # type: ignore[union-attr]
        d_pi3x = pi3x_out["depth"].cpu().numpy()  # (T,H,W)

        # MoGe-2: per-frame metric depth
        fov_x = None
        if src.intrinsics is not None:
            fx = src.intrinsics[0].item()
            w = frames_np.shape[2]
            import math
            fov_x = math.degrees(2 * math.atan(w / (2 * fx)))

        d_moge_list = []
        for i in range(T):
            f = frames_t[i:i+1]
            out = self._moge2.infer(f, fov_x=fov_x)  # type: ignore[union-attr]
            d_moge_list.append(out["depth"].squeeze(0).cpu().numpy())
        d_moge = np.stack(d_moge_list, axis=0)  # (T,H,W)

        # EMA scale fusion
        scale = _fuse_ema(d_pi3x, d_moge, self.ema_momentum)  # (T,)
        metric_depth = d_pi3x * scale[:, None, None]

        return DepthEstimationResult(
            metric_depth=torch.from_numpy(metric_depth.astype(np.float32)).to(self.device)
        )
