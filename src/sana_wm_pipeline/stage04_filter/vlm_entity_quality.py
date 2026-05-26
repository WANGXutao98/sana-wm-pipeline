"""Stage-04b VLM entity & quality annotation (paper §4 + App. B.3).

Uses the Qwen3.5-VL family ([102] in paper).  Falls back to the most recent
publicly released sibling (Qwen2.5-VL) when Qwen3.5-VL weights are not yet
downloadable on this machine — the prompt is unchanged so downstream Table-6
thresholds remain comparable.

The model itself is injected as a callable so unit tests don't need GPU.
"""
from __future__ import annotations

import json
from typing import Callable, List

import numpy as np


ENTITY_QUALITY_PROMPT = """You are a video annotator. Below are 8 evenly-sampled keyframes from a clip.

Output STRICT JSON with exactly these fields and types:
  "people":   <integer count of distinct persons visible across the clip>
  "vehicles": <integer count of distinct vehicles>
  "animals":  <integer count of distinct animals>
  "quality":  <float in [0.0, 2.0] — 0 unusable, 1 acceptable, 2 excellent>
  "too_dark": <bool>
  "blurry":   <bool>

Respond with the JSON object ONLY, no prose, no markdown fences."""


def _sample_keyframes(frames_rgb: np.ndarray, n: int = 8) -> List[np.ndarray]:
    if len(frames_rgb) == 0:
        return []
    idx = np.linspace(0, len(frames_rgb) - 1, n).astype(int)
    return [frames_rgb[i] for i in idx]


def _safe_json_parse(raw: str) -> dict:
    try:
        a = raw.find("{")
        b = raw.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("no JSON object braces found")
        return json.loads(raw[a:b + 1])
    except (ValueError, json.JSONDecodeError):
        return {}


def annotate(
    frames_rgb: np.ndarray,
    vlm_call: Callable[[str, List[np.ndarray]], str],
) -> dict:
    """Call the injected `vlm_call(prompt, keyframes)` and parse its JSON.

    On any failure we emit sentinel values (-1 / False) so the Table-6
    apply layer can treat the clip as failing the VLM rule rather than
    silently passing it.
    """
    keyframes = _sample_keyframes(frames_rgb)
    if not keyframes:
        return _sentinel()
    raw = vlm_call(ENTITY_QUALITY_PROMPT, keyframes)
    parsed = _safe_json_parse(raw or "")
    if not parsed:
        return _sentinel()
    return _coerce(parsed)


def _sentinel() -> dict:
    return {
        "people": -1, "vehicles": -1, "animals": -1,
        "quality": -1.0, "too_dark": False, "blurry": False,
    }


def _coerce(d: dict) -> dict:
    def _int(v, default=-1):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _float(v, default=-1.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _bool(v):
        return bool(v) if isinstance(v, (bool, int, float)) else False

    return {
        "people":   _int(d.get("people")),
        "vehicles": _int(d.get("vehicles")),
        "animals":  _int(d.get("animals")),
        "quality":  _float(d.get("quality")),
        "too_dark": _bool(d.get("too_dark")),
        "blurry":   _bool(d.get("blurry")),
    }


def to_table6_scores(vlm_result: dict) -> dict:
    """Project the VLM dict into the score keys expected by apply_table6.

    Paper Table 6 uses two VLM-derived metrics per source:
      - vlm_entity_count = people + vehicles + animals   (cap-based filter)
      - vlm_quality      = quality                       (mid-range filter)
    """
    pe = max(0, vlm_result.get("people", 0))
    ve = max(0, vlm_result.get("vehicles", 0))
    an = max(0, vlm_result.get("animals", 0))
    return {
        "vlm_entity_count": pe + ve + an,
        "vlm_quality": vlm_result.get("quality", -1.0),
    }
