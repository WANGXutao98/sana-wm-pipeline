"""Tests for the sources manifest (paper Table 1, §4).

Source counts must sum to 212,975 clips.
"""
from pathlib import Path

from sana_wm_pipeline.stage01_ingest.sources import load_sources

CFG = Path(__file__).parent.parent / "configs" / "sources.yaml"


def test_load_sources_total_matches_paper():
    specs = load_sources(CFG)
    total = sum(s.target_clips for s in specs.values())
    assert total == 212975, f"sum of target_clips = {total} != 212975"


def test_load_sources_has_7_sources():
    specs = load_sources(CFG)
    assert len(specs) == 7


def test_pose_modes_match_paper_assignment():
    specs = load_sources(CFG)
    assert specs["spatialvid_hq"].pose_mode == "default"
    assert specs["dl3dv_real"].pose_mode == "gt_pose"
    assert specs["dl3dv_gs_refined"].pose_mode == "gt_pose"
    assert specs["omniworld"].pose_mode == "gt_depth"
    assert specs["sekai_game"].pose_mode == "gt_pose"
    assert specs["sekai_walking_hq"].pose_mode == "default"
    assert specs["miradata"].pose_mode == "default"
