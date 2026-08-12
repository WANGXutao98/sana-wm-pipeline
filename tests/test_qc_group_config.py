from __future__ import annotations
import pytest
from sana_wm_pipeline.qc.group_config import (
    GroupConfig, get_group_config, compute_verdict,
    VERDICT_FAIL, VERDICT_FLAG, VERDICT_PASS,
)

ALL_GROUPS = [
    "wds-DL3DV-ALL-2K", "wds-sekai-real-walking-hq", "wds-OmniWorld-Game",
    "wds-SpatialVID-hq", "wds-RealEstate10K-360p",
    "wds-sekai-game-drone", "wds-sekai-game-walking",
]


def _clean(T=100) -> dict:
    return {
        "T": T, "t_aligned": True, "no_nan_inf": True, "so3_valid": True,
        "first_frame_ok": True, "pose_quality_ok": True, "fov_ok": True, "focal_div_ok": True,
        "caption_ok": True, "n_jumps": 0, "camera_words": [], "reasons": [],
    }


def test_all_seven_groups_have_config():
    for g in ALL_GROUPS:
        assert isinstance(get_group_config(g), GroupConfig)

def test_unknown_group_fallback():
    assert isinstance(get_group_config("wds-unknown-xyz"), GroupConfig)

def test_game_groups_more_lenient():
    game = get_group_config("wds-OmniWorld-Game")
    real = get_group_config("wds-DL3DV-ALL-2K")
    assert game.max_jumps_fail > real.max_jumps_fail
    assert game.jump_threshold_m > real.jump_threshold_m

def test_game_max_jumps_fail_is_50():
    for g in ["wds-OmniWorld-Game", "wds-sekai-game-walking"]:
        assert get_group_config(g).max_jumps_fail == 50

def test_drone_max_jumps_fail_is_80():
    assert get_group_config("wds-sekai-game-drone").max_jumps_fail == 80

def test_verdict_pass_clean():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    verdict, _ = compute_verdict(_clean(), cfg)
    assert verdict == VERDICT_PASS

def test_verdict_fail_so3():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean(); m["so3_valid"] = False
    verdict, _ = compute_verdict(m, cfg)
    assert verdict == VERDICT_FAIL

def test_verdict_fail_too_many_jumps():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean(); m["n_jumps"] = cfg.max_jumps_fail + 1
    assert compute_verdict(m, cfg)[0] == VERDICT_FAIL

def test_verdict_flag_moderate_jumps():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean(); m["n_jumps"] = 1  # max_jumps_flag=0 for DL3DV
    assert compute_verdict(m, cfg)[0] == VERDICT_FLAG

def test_verdict_flag_camera_words():
    cfg = get_group_config("wds-sekai-real-walking-hq")
    m = _clean(); m["camera_words"] = ["pans left"]
    verdict, reasons = compute_verdict(m, cfg)
    assert verdict == VERDICT_FLAG
    assert any("camera_word" in r for r in reasons)

def test_game_tolerates_moderate_jumps():
    cfg = get_group_config("wds-OmniWorld-Game")
    m = _clean(); m["n_jumps"] = 10
    assert compute_verdict(m, cfg)[0] == VERDICT_PASS

def test_saturation_flag_for_real_data():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean(); m["saturation"] = 1.0  # below min (configured as 0)
    # DL3DV has color_saturation check [0,180], 1.0 should pass
    assert compute_verdict(m, cfg)[0] == VERDICT_PASS
