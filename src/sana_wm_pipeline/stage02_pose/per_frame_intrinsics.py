"""Per-frame camera intrinsics container per paper App. B.1.

Paper App. B.1: "...treat (fx, fy, cx, cy) as independent variables per frame,
stored as an (N, V, D) tensor (frames × views × intrinsics dimension)."

For our pipeline V=1 (single view per frame), D=4 (fx, fy, cx, cy).
The N×V×D layout is preserved so future multi-view extensions are drop-in.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

INTRINSICS_DIM = 4   # (fx, fy, cx, cy)
DEFAULT_VIEWS = 1    # our pipeline is monocular

@dataclass(frozen=True)
class PerFrameIntrinsics:
    """(N, V, D) tensor of per-frame camera intrinsics.

    N = number of frames
    V = number of views (we use 1 for monocular)
    D = INTRINSICS_DIM = 4: layout is (fx, fy, cx, cy)
    """
    tensor: np.ndarray   # shape (N, V, D), dtype float32

    def __post_init__(self):
        t = self.tensor
        if t.ndim != 3 or t.shape[2] != INTRINSICS_DIM:
            raise ValueError(
                f"PerFrameIntrinsics tensor must be (N, V, {INTRINSICS_DIM}); got {t.shape}"
            )
        if t.dtype != np.float32:
            raise ValueError(f"tensor dtype must be float32; got {t.dtype}")

    @classmethod
    def from_flat(cls, fx: np.ndarray, fy: np.ndarray,
                  cx: np.ndarray, cy: np.ndarray,
                  views: int = DEFAULT_VIEWS) -> "PerFrameIntrinsics":
        """Construct from 4 length-N arrays (single-view; broadcast V if >1)."""
        arrs = [np.asarray(a, dtype=np.float32) for a in (fx, fy, cx, cy)]
        N = arrs[0].shape[0]
        for a in arrs:
            if a.shape != (N,):
                raise ValueError(f"all of fx/fy/cx/cy must have shape (N,); got {a.shape}")
        # stack to (N, 4), broadcast across V views
        flat = np.stack(arrs, axis=-1)               # (N, 4)
        tensor = np.broadcast_to(flat[:, None, :], (N, views, INTRINSICS_DIM)).astype(np.float32).copy()
        return cls(tensor=tensor)

    @property
    def n_frames(self) -> int:
        return self.tensor.shape[0]

    @property
    def n_views(self) -> int:
        return self.tensor.shape[1]

    @property
    def fx(self) -> np.ndarray:
        """(N, V) float32"""
        return self.tensor[..., 0]

    @property
    def fy(self) -> np.ndarray:
        return self.tensor[..., 1]

    @property
    def cx(self) -> np.ndarray:
        return self.tensor[..., 2]

    @property
    def cy(self) -> np.ndarray:
        return self.tensor[..., 3]

    def to_K(self) -> np.ndarray:
        """Build (N, V, 3, 3) intrinsic matrices."""
        N, V = self.n_frames, self.n_views
        K = np.zeros((N, V, 3, 3), dtype=np.float32)
        K[..., 0, 0] = self.fx
        K[..., 1, 1] = self.fy
        K[..., 0, 2] = self.cx
        K[..., 1, 2] = self.cy
        K[..., 2, 2] = 1.0
        return K
