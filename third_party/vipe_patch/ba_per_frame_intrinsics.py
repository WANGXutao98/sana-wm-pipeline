"""VIPE patch — lift (fx, fy, cx, cy) to per-frame BA variables.

Paper App. B.1 says VIPE's original single-shared-intrinsics assumption is
replaced by storing intrinsics as the **(N, V, D)** tensor used elsewhere in
the codebase: N frames, V views (1 for monocular here), D=4 (fx, fy, cx, cy).

This module provides `PerFrameIntrinsicsParam`, a `torch.nn.Module` that owns
the (N, 1, 4) parameter and exposes per-frame K matrices.  VIPE's BA loop
imports it instead of the original single-tensor intrinsics holder.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PerFrameIntrinsicsParam(nn.Module):
    """Owns the per-frame (N, 1, 4) intrinsics tensor as a BA variable."""

    def __init__(self, init_fxfycxcy: torch.Tensor):
        """init_fxfycxcy: (N, 4) initial guess."""
        super().__init__()
        if init_fxfycxcy.ndim != 2 or init_fxfycxcy.shape[1] != 4:
            raise ValueError(
                f"init_fxfycxcy must be (N, 4); got {tuple(init_fxfycxcy.shape)}"
            )
        # Lift to (N, 1, 4) — V=1 monocular path.
        self.intr = nn.Parameter(init_fxfycxcy[:, None, :].clone().contiguous())

    @property
    def n_frames(self) -> int:
        return self.intr.shape[0]

    def K(self, t: int) -> torch.Tensor:
        """3x3 K matrix for frame `t`."""
        fx, fy, cx, cy = self.intr[t, 0].unbind(-1)
        K = torch.zeros(3, 3, device=self.intr.device, dtype=self.intr.dtype)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K[2, 2] = 1.0
        return K

    def to_NVD(self) -> torch.Tensor:
        """Detach and return the (N, 1, 4) tensor for serialization."""
        return self.intr.detach().contiguous()
