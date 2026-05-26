"""Shared dataclass for stage-02 pose-annotation outputs (paper App. B.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PoseArtifact:
    """One clip's per-frame camera state after Stage-02.

    Field shapes (T = camera_frames, paper §5.1 = 961):
      poses_c2w        : (T, 4, 4)  float32  — first frame anchored to I_4
      intrinsics       : (T, 1, 4)  float32  — (fx, fy, cx, cy) per App. B.1
      scale_per_frame  : (T,)       float32  — metric scale for QC
      depth_downsampled: optional (T, H/4, W/4) float32 for visualisation
    """

    poses_c2w: np.ndarray
    intrinsics: np.ndarray
    scale_per_frame: np.ndarray
    depth_downsampled: Optional[np.ndarray] = None

    def validate(self, T_expected: int) -> None:
        if self.poses_c2w.shape != (T_expected, 4, 4):
            raise AssertionError(
                f"poses_c2w shape {self.poses_c2w.shape} != ({T_expected}, 4, 4)"
            )
        if self.intrinsics.shape != (T_expected, 1, 4):
            raise AssertionError(
                f"intrinsics shape {self.intrinsics.shape} != ({T_expected}, 1, 4)"
            )
        if self.scale_per_frame.shape != (T_expected,):
            raise AssertionError(
                f"scale_per_frame shape {self.scale_per_frame.shape} != ({T_expected},)"
            )
        # First frame anchored to identity (paper App. D.3 benchmark convention).
        if not np.allclose(self.poses_c2w[0], np.eye(4), atol=1e-3):
            raise AssertionError("poses_c2w[0] is not the identity")
