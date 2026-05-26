"""DiFix3D refinement of FCGS renders (paper App. B.2).

Paper App. B.2 fixes these inference hyper-parameters verbatim:
  num_steps = 1
  prompt    = "remove degradation"
  timestep  = 199
  guidance  = 0.0

We freeze them in DIFIX3D_PARAMS and refuse to load a non-matching config.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# Paper App. B.2 — DO NOT change without re-reading the paper.
DIFIX3D_PARAMS = {
    "num_steps": 1,
    "prompt": "remove degradation",
    "timestep": 199,
    "guidance": 0.0,
}


def assert_difix3d_params_match_paper(params: dict) -> None:
    """Raise if `params` deviate from the paper App. B.2 spec."""
    for k, v in DIFIX3D_PARAMS.items():
        if params.get(k) != v:
            raise AssertionError(
                f"DiFix3D param {k}={params.get(k)!r} != paper-fixed {v!r}"
            )


def refine_clip(
    frames_rgb: np.ndarray,
    difix3d_pipeline: Optional[Callable] = None,
) -> np.ndarray:
    """Run DiFix3D frame-by-frame.

    `difix3d_pipeline` must be a callable that accepts the keyword args in
    DIFIX3D_PARAMS and returns an RGB uint8 array of the same shape.
    """
    if difix3d_pipeline is None:
        raise NotImplementedError(
            "difix3d_pipeline must be injected — wire it in at deploy time"
        )
    out = np.empty_like(frames_rgb)
    for t in range(len(frames_rgb)):
        out[t] = difix3d_pipeline(image=frames_rgb[t], **DIFIX3D_PARAMS)
    return out
