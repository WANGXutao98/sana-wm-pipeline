# tests/test_qc_stage2.py
from __future__ import annotations
import io, json, tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.qc.stage2_deep import (
    count_video_frames_av, check_black_frame_ratio,
    check_trajectory_frozen, deep_check_sample, run_stage2,
)

TESTDATA = Path(__file__).parent.parent / "testdata"
SAMPLE_ID = "OmniWorld-Game_1f79eb96f021__splits_013-015"
GROUP = "wds-OmniWorld-Game"
EXPECTED_T = 782  # confirmed from testdata analysis


@pytest.fixture(scope="session")
def tiny_tar(tmp_path_factory):
    p = tmp_path_factory.mktemp("s2") / "shard.tar"
    with tarfile.open(p, "w") as tf:
        for ext in [".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"]:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return p


@pytest.fixture(scope="session")
def video_bytes():
    return (TESTDATA / (SAMPLE_ID + ".mp4")).read_bytes()


def test_count_frames_av(video_bytes):
    T = count_video_frames_av(video_bytes)
    assert T == EXPECTED_T

def test_black_frame_ratio_normal_video(video_bytes):
    ratio = check_black_frame_ratio(video_bytes)
    assert 0.0 <= ratio <= 0.3  # normal video should be mostly non-black

def test_trajectory_not_frozen():
    p = np.tile(np.eye(4, dtype=np.float32), (100, 1, 1))
    p[:, 0, 3] = np.arange(100, dtype=np.float32) * 0.1
    frozen, ratio = check_trajectory_frozen(p)
    assert not frozen and ratio < 0.01

def test_trajectory_frozen_detected():
    p = np.tile(np.eye(4, dtype=np.float32), (100, 1, 1))
    p[10:90, 0, 3] = 5.0  # 80% frames have same position
    frozen, ratio = check_trajectory_frozen(p)
    assert frozen and ratio > 0.7

def test_deep_check_structure(tiny_tar):
    r = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert "sample_id" in r and "stage2" in r
    for k in ("video_T", "video_T_matches_npy", "traj_frozen", "frozen_ratio",
               "black_frame_ratio", "scene_cuts", "reasons"):
        assert k in r["stage2"], f"missing key: {k}"

def test_deep_check_video_T(tiny_tar):
    r = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert r["stage2"]["video_T"] == EXPECTED_T
    assert r["stage2"]["video_T_matches_npy"] is True

def test_deep_check_not_frozen(tiny_tar):
    r = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert r["stage2"]["traj_frozen"] is False

def test_run_stage2_processes_flag(tiny_tar, tmp_path):
    s1 = tmp_path / "s1.jsonl"
    s1.write_text(json.dumps({
        "sample_id": SAMPLE_ID, "group": GROUP, "tar_path": str(tiny_tar),
        "verdict": "flag", "flag_reasons": ["n_jumps=5"], "metrics": {"T": EXPECTED_T},
    }) + "\n")
    out = tmp_path / "s2.jsonl"
    count = run_stage2(s1, out, sample_frac=0.0, n_workers=1)
    assert count == 1
    rec = json.loads(out.read_text().splitlines()[0])
    assert "stage2" in rec

def test_run_stage2_skips_fail(tiny_tar, tmp_path):
    s1 = tmp_path / "s1.jsonl"
    s1.write_text(json.dumps({
        "sample_id": SAMPLE_ID, "group": GROUP, "tar_path": str(tiny_tar),
        "verdict": "fail", "flag_reasons": ["so3"], "metrics": {},
    }) + "\n")
    out = tmp_path / "s2.jsonl"
    count = run_stage2(s1, out, sample_frac=0.0, n_workers=1)
    assert count == 0
