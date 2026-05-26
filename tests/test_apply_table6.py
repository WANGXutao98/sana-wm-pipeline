"""Tests for stage04_filter.apply_table6 — verbatim paper Table-6 rules."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from sana_wm_pipeline.stage04_filter.apply_table6 import (
    accept,
    evaluate,
    load_thresholds,
)
from sana_wm_pipeline.stage04_filter.vlm_entity_quality import (
    annotate,
    to_table6_scores,
)


CFG_PATH = Path(__file__).parent.parent / "configs" / "filter_thresholds.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_thresholds(CFG_PATH)


# ---- Table 6 verbatim values are in the YAML -------------------------------
def test_yaml_has_all_six_sources(cfg):
    expected = {"OmniWorld", "Sekai_Game", "Sekai_Walking",
                "MiraData", "DL3DV_GS", "SpatialVID"}
    assert set(cfg["per_source"].keys()) == expected


def test_dl3dv_gs_vmaf_lower_bound_is_6(cfg):
    # The only source whose vmaf_motion lower bound differs from 0.5.
    assert cfg["per_source"]["DL3DV_GS"]["vmaf_motion"] == [6, 50]


def test_camera_uniform_filters_match_paper(cfg):
    cam = cfg["camera"]
    assert cam["fov_deg"] == [25, 120]
    assert cam["focal_div_max"] == 0.20
    assert cam["scale_cv_max"] == 2.0


# ---- accept() acceptance behaviour ----------------------------------------
def _ok_scores():
    """Scores that satisfy every paper threshold."""
    return dict(
        vmaf_motion=10.0,
        unimatch_flow=10.0,
        dover=0.55,
        color_saturation=50.0,
        scene_cuts=0,
        vlm_entity_count=3,
        vlm_quality=1.0,
    )


def test_omniworld_accept_baseline(cfg):
    assert accept("OmniWorld", _ok_scores(), cfg)


def test_omniworld_rejects_dover_below_035(cfg):
    s = _ok_scores(); s["dover"] = 0.30
    out = evaluate("OmniWorld", s, cfg)
    assert not out["accepted"]
    assert any("dover" in r for r in out["reasons"])


def test_miradata_rejects_scene_cuts_2(cfg):
    s = _ok_scores(); s["scene_cuts"] = 2
    out = evaluate("MiraData", s, cfg)
    assert not out["accepted"]
    assert any("scene_cuts" in r for r in out["reasons"])


def test_miradata_ignores_vlm_fields(cfg):
    # MiraData has vlm_entity / vlm_quality = null  ⇒ those scores never reject.
    s = _ok_scores(); s["vlm_entity_count"] = 999; s["vlm_quality"] = -5.0
    assert accept("MiraData", s, cfg)


def test_dl3dv_gs_rejects_vmaf_below_6(cfg):
    s = _ok_scores(); s["vmaf_motion"] = 1.0
    out = evaluate("DL3DV_GS", s, cfg)
    assert not out["accepted"]
    assert any("vmaf_motion" in r for r in out["reasons"])


def test_omniworld_ignores_saturation(cfg):
    # OmniWorld has color_saturation = null  ⇒ value range is irrelevant.
    s = _ok_scores(); s["color_saturation"] = 9999
    assert accept("OmniWorld", s, cfg)


def test_applicable_missing_score_rejects(cfg):
    # SpatialVID requires dover; omitting it must NOT silently pass.
    s = _ok_scores(); del s["dover"]
    out = evaluate("SpatialVID", s, cfg)
    assert not out["accepted"]
    assert any("dover" in r for r in out["reasons"])


def test_nan_score_rejects(cfg):
    s = _ok_scores(); s["dover"] = float("nan")
    assert not accept("OmniWorld", s, cfg)


def test_unknown_source_raises(cfg):
    with pytest.raises(KeyError):
        evaluate("NotASource", _ok_scores(), cfg)


# ---- VLM helpers ----------------------------------------------------------
def test_annotate_parses_valid_json():
    import numpy as np
    frames = np.zeros((20, 4, 4, 3), dtype=np.uint8)

    def fake_vlm(prompt, imgs):
        return ('{"people": 2, "vehicles": 0, "animals": 1, '
                '"quality": 1.2, "too_dark": false, "blurry": false}')

    out = annotate(frames, fake_vlm)
    assert out["people"] == 2
    assert out["vehicles"] == 0
    assert out["animals"] == 1
    assert out["quality"] == pytest.approx(1.2)


def test_annotate_sentinel_on_garbage():
    import numpy as np
    frames = np.zeros((20, 4, 4, 3), dtype=np.uint8)
    out = annotate(frames, lambda p, i: "definitely not json")
    assert out == {
        "people": -1, "vehicles": -1, "animals": -1,
        "quality": -1.0, "too_dark": False, "blurry": False,
    }


def test_to_table6_scores_sums_entities():
    proj = to_table6_scores({"people": 3, "vehicles": 1, "animals": 2,
                             "quality": 1.3})
    assert proj["vlm_entity_count"] == 6
    assert proj["vlm_quality"] == pytest.approx(1.3)
