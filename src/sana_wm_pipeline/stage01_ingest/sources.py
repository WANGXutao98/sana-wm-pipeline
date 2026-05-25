"""Typed loader for ``configs/sources.yaml``.

Source counts trace to paper Table 1 (arXiv:2605.15178v1, §4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

PoseMode = Literal["default", "gt_pose", "gt_depth"]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    citation: str
    type: str
    pose_mode: PoseMode
    nominal_duration_s: int
    real_or_synthetic: Literal["real", "synthetic"]
    target_clips: int
    repo_id: str | None = None
    repo: str | None = None
    subset: str | None = None


def load_sources(cfg_path: Path) -> dict[str, SourceSpec]:
    """Load and validate the sources manifest.

    Raises ``AssertionError`` if the per-source clip counts do not sum to
    ``totals.total_clips`` (paper Table 1 sum = 212,975).
    """
    raw = yaml.safe_load(Path(cfg_path).read_text())
    out: dict[str, SourceSpec] = {}
    for name, body in raw["sources"].items():
        out[name] = SourceSpec(name=name, **body)
    total = sum(s.target_clips for s in out.values())
    expected = raw["totals"]["total_clips"]
    assert total == expected, f"clip counts {total} != {expected}"
    return out
