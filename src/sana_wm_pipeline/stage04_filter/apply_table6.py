"""Apply paper Table-6 per-source acceptance rules.

A clip is accepted iff every applicable filter passes.  "Applicable" means the
threshold for that source is not `null` (paper's "—").  Missing scores in the
input dict are treated as failures when the rule is applicable, so callers
must either compute every metric the source needs or explicitly skip the
filter by setting the YAML entry to null.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import yaml


_METRIC_TO_RULE = (
    ("vmaf_motion",      "vmaf_motion"),
    ("unimatch_flow",    "unimatch_flow"),
    ("dover",            "dover"),
    ("color_saturation", "color_saturation"),
    ("vlm_entity_count", "vlm_entity"),
    ("vlm_quality",      "vlm_quality"),
)


def load_thresholds(path: str | Path) -> dict:
    """Read filter_thresholds.yaml from disk."""
    with open(path) as f:
        return yaml.safe_load(f)


def _in_range(value, rng) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    lo, hi = rng
    return lo <= value <= hi


def evaluate(source: str, scores: Dict[str, float], cfg: dict) -> Dict[str, object]:
    """Return a structured pass/fail report:
      {"accepted": bool, "reasons": [str, ...]}
    """
    if source not in cfg["per_source"]:
        raise KeyError(f"unknown source '{source}' (known: {list(cfg['per_source'])})")
    rules = cfg["per_source"][source]
    reasons = []

    for k_score, k_rule in _METRIC_TO_RULE:
        rng = rules.get(k_rule)
        if rng is None:
            continue
        v = scores.get(k_score)
        if not _in_range(v, rng):
            reasons.append(f"{k_score}={v!r} not in {rng}")

    sc_max = rules.get("scene_cuts_max")
    if sc_max is not None:
        v = scores.get("scene_cuts", 0)
        if v is None or v > sc_max:
            reasons.append(f"scene_cuts={v!r} > {sc_max}")

    return {"accepted": len(reasons) == 0, "reasons": reasons}


def accept(source: str, scores: Dict[str, float], cfg: dict) -> bool:
    return bool(evaluate(source, scores, cfg)["accepted"])
