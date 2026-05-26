"""Inference / rendering wrapper for fitted FCGS scenes.

The renderer is injected — call sites supply an FCGS renderer object that
has `.render(scene_params, c2w, intrinsics)` returning an (H,W,3) uint8 image
and a tile-empty ratio.  In tests we stub the renderer entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class RenderResult:
    frames: np.ndarray            # (N, H, W, 3) uint8
    splat_counts: np.ndarray      # (N_sampled,) int
    tile_empty: np.ndarray        # (N_sampled,) float
    near_black_fraction: float


def render_trajectory(
    gs_params_path,
    poses_c2w: np.ndarray,
    intrinsics_fxfycxcy: np.ndarray,
    image_wh: Tuple[int, int],
    renderer_fn: Callable,
    sample_every: int = 10,
    near_black_pixel_thr: int = 8,
    near_black_frame_thr: float = 0.95,
) -> RenderResult:
    """Render poses with `renderer_fn(params, c2w_t, intr_t, wh) -> (img, n_splats, tile_empty)`.

    Samples coverage stats every `sample_every` frames per paper App. B.2.
    """
    W, H = image_wh
    n = len(poses_c2w)
    frames = np.zeros((n, H, W, 3), dtype=np.uint8)
    splats: List[int] = []
    tile_empty: List[float] = []
    sampled = set(range(0, n, sample_every))
    near_black_count = 0
    for t in range(n):
        img, ns, te = renderer_fn(
            gs_params_path, poses_c2w[t], intrinsics_fxfycxcy[t], image_wh,
        )
        frames[t] = img
        if t in sampled:
            splats.append(int(ns))
            tile_empty.append(float(te))
        if (img.mean(axis=-1) < near_black_pixel_thr).mean() > near_black_frame_thr:
            near_black_count += 1
    return RenderResult(
        frames=frames,
        splat_counts=np.array(splats, dtype=int),
        tile_empty=np.array(tile_empty, dtype=np.float32),
        near_black_fraction=near_black_count / max(n, 1),
    )
