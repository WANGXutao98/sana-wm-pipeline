"""WebDataset sample schema for SANA-WM training corpus.

Each sample group inside a .tar shard contains:
  {id}.mp4              — 1280x720, 16fps, 961 frames (camera frames)
  {id}.poses_c2w.npy    — (961, 4, 4) float32, frame 0 anchored as identity
  {id}.intrinsics.npy   — (961, 1, 4) float32 (paper App. B.1 N,V,D tensor)
  {id}.scale.npy        — (961,) float32 per-frame metric scale (paper App. B.1)
  {id}.caption.txt      — UTF-8 scene-static caption (no camera-action verbs)
  {id}.meta.json        — source, pose_mode, qc_scores, license, original_url
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

CAMERA_FRAMES = 961                    # paper App. D.1
EXPECTED_WH: tuple[int, int] = (1280, 720)
INTRINSICS_VIEWS = 1
INTRINSICS_DIM = 4

@dataclass(frozen=True)
class Sample:
    """One training sample. Validates shapes against paper-fixed constants."""
    sample_id: str
    video_path: str                    # path to existing .mp4 on disk
    poses_c2w: np.ndarray              # (961, 4, 4) float32
    intrinsics_NVD: np.ndarray         # (961, 1, 4) float32
    scale_per_frame: np.ndarray        # (961,) float32
    caption: str
    meta: dict

    def validate(self) -> None:
        T = CAMERA_FRAMES
        if self.poses_c2w.shape != (T, 4, 4):
            raise ValueError(f"poses_c2w shape {self.poses_c2w.shape} != ({T},4,4)")
        if self.poses_c2w.dtype != np.float32:
            raise ValueError(f"poses_c2w dtype {self.poses_c2w.dtype} != float32")
        if self.intrinsics_NVD.shape != (T, INTRINSICS_VIEWS, INTRINSICS_DIM):
            raise ValueError(
                f"intrinsics_NVD shape {self.intrinsics_NVD.shape} "
                f"!= ({T},{INTRINSICS_VIEWS},{INTRINSICS_DIM})"
            )
        if self.intrinsics_NVD.dtype != np.float32:
            raise ValueError(f"intrinsics_NVD dtype {self.intrinsics_NVD.dtype} != float32")
        if self.scale_per_frame.shape != (T,):
            raise ValueError(f"scale_per_frame shape {self.scale_per_frame.shape} != ({T},)")
        if self.scale_per_frame.dtype != np.float32:
            raise ValueError(f"scale_per_frame dtype != float32")
        if not self.caption.strip():
            raise ValueError("caption must be non-empty")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        # First-frame canonicalization per benchmark convention (App. D.3):
        # "ground-truth trajectory is loaded ..., relativized to the first frame"
        if not np.allclose(self.poses_c2w[0], np.eye(4, dtype=np.float32), atol=1e-3):
            raise ValueError("poses_c2w[0] must be identity (first-frame anchor)")
