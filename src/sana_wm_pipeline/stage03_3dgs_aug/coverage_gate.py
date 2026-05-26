"""3DGS coverage gate (paper App. B.2).

Two-tier rejection:
  1. Frame-level — every 10th frame is sampled; ≥70% of sampled frames must
     project enough splats (>= MIN_SPLATS).
  2. Tile-level   — 32×32 tile grid; a view is rejected if >65% of tiles
     are empty.

Both predicates are returned in a single helper so the caller can decide
whether to apply them per-frame, per-view, or as the joint gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


MIN_SPLATS: int = 1000
FRAME_PASS_FRACTION: float = 0.70
TILE_EMPTY_MAX: float = 0.65
NEAR_BLACK_FRAME_MAX: float = 0.30   # post-render: ≤30 % near-empty frames


@dataclass
class CoverageReport:
    passed: bool
    frame_pass_fraction: float
    tile_empty_max: float
    reason: str


def passes_coverage(
    splat_counts: np.ndarray,
    tile_empty_ratio: np.ndarray,
    min_splats: int = MIN_SPLATS,
    frame_pass_fraction: float = FRAME_PASS_FRACTION,
    tile_empty_max: float = TILE_EMPTY_MAX,
) -> Tuple[bool, str]:
    """Return (accept, reason).  Paper App. B.2."""
    splat_counts = np.asarray(splat_counts)
    tile_empty_ratio = np.asarray(tile_empty_ratio, dtype=np.float32)
    if splat_counts.size == 0:
        return False, "no_sampled_frames"
    enough = float((splat_counts >= min_splats).mean())
    if enough < frame_pass_fraction:
        return False, f"frame_pass={enough:.2%} < {frame_pass_fraction:.0%}"
    if (tile_empty_ratio > tile_empty_max).any():
        return False, f"tile_empty={tile_empty_ratio.max():.2%} > {tile_empty_max:.0%}"
    return True, "ok"


def passes_post_render(near_black_fraction: float,
                       max_fraction: float = NEAR_BLACK_FRAME_MAX) -> Tuple[bool, str]:
    """Reject clips with too many near-empty frames after FCGS rendering."""
    if near_black_fraction > max_fraction:
        return False, f"near_black_frac={near_black_fraction:.2%} > {max_fraction:.0%}"
    return True, "ok"
