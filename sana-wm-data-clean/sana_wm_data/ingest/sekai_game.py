"""Sekai-Game pose-convention helpers.

The release's ``extrinsic`` array is already the authoritative camera-to-world
trajectory consumed by this project.  In particular, its first camera rotation
is normally close to ``diag(1, -1, -1)``.  That is part of the source convention,
not a signal that another camera-axis conversion is required.

An older ingestion path incorrectly post-multiplied every pose by
``diag(1, -1, -1, 1)``.  Because that matrix is self-inverse, blindly applying it
again toggles between the correct and incorrect conventions.  Keep the raw pose
unchanged here and only apply the temporal resampling indices.
"""

from __future__ import annotations

import numpy as np


def select_sekai_game_c2w(extrinsic: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return temporally selected Sekai-Game c2w poses without axis conversion."""
    c2w = np.asarray(extrinsic)
    sel = np.asarray(indices)

    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"expected Sekai-Game c2w [T,4,4], got {c2w.shape}")
    if sel.ndim != 1 or not np.issubdtype(sel.dtype, np.integer):
        raise ValueError(f"expected 1-D integer frame indices, got {sel.shape} {sel.dtype}")
    if sel.size and (int(sel.min()) < 0 or int(sel.max()) >= c2w.shape[0]):
        raise IndexError(f"Sekai-Game frame indices outside [0,{c2w.shape[0]})")

    # Deliberately no ``@ diag(1,-1,-1,1)`` here.  See the module docstring.
    return c2w[sel].copy()
