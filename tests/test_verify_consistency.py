"""Tests for scripts/verify_consistency.py."""
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
import numpy as np
import pytest

# Import via path injection mirroring the script itself
ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_consistency.py"
sys.path.insert(0, str(ROOT / "scripts"))
import verify_consistency as VC                       # type: ignore

from sana_wm_pipeline.stage06_pack.schema import (
    Sample, CAMERA_FRAMES, EXPECTED_WH, INTRINSICS_VIEWS, INTRINSICS_DIM,
)
from sana_wm_pipeline.stage06_pack.webdataset_writer import ShardWriter


def _good_sample(tmp_path: Path, sample_id: str = "good_001", caption: str | None = None) -> Sample:
    video = tmp_path / f"{sample_id}.mp4"
    video.write_bytes(b"\x00fake")
    poses = np.tile(np.eye(4, dtype=np.float32), (CAMERA_FRAMES, 1, 1))
    intrinsics = np.tile([[[700.0, 700.0, 640.0, 360.0]]], (CAMERA_FRAMES, 1, 1)).astype(np.float32)
    scale = np.ones(CAMERA_FRAMES, dtype=np.float32)
    return Sample(
        sample_id=sample_id,
        video_path=str(video),
        poses_c2w=poses,
        intrinsics_NVD=intrinsics,
        scale_per_frame=scale,
        caption=caption or "a wooden table and ceramic mugs under afternoon light",
        meta={"source": "test"},
    )


def test_forbidden_verb_detection_pan():
    has, hits = VC.caption_has_forbidden_verbs("the camera pans left across the room")
    assert has and "pan" in hits


def test_forbidden_verb_detection_walking():
    has, hits = VC.caption_has_forbidden_verbs("first-person view walking through a forest")
    assert has and ("walking" in hits or "camera-phrase" in hits)


def test_forbidden_verb_clean_caption():
    has, hits = VC.caption_has_forbidden_verbs("a serene alpine lake at dawn with snow-dusted peaks")
    assert not has, hits


def test_forbidden_verb_does_not_false_positive_on_substring():
    # 'transit' should NOT match 'tilt' or 'pan' etc.
    has, hits = VC.caption_has_forbidden_verbs("a busy rail transit station with concrete platforms")
    assert not has, hits


def test_verify_sample_good(tmp_path):
    sample = _good_sample(tmp_path, "g_001")
    with ShardWriter(tmp_path / "shards", samples_per_shard=10) as w:
        w.write(sample)
    shard = tmp_path / "shards" / "shard-000000.tar"
    n_ok, n_fail, errs = VC.verify_shard(shard)
    assert n_ok == 1 and n_fail == 0, errs


def test_verify_sample_bad_caption_fails(tmp_path):
    sample = _good_sample(tmp_path, "bad_cap_001",
                          caption="the camera dolly-zooms toward the table")
    with ShardWriter(tmp_path / "shards", samples_per_shard=10) as w:
        w.write(sample)
    shard = tmp_path / "shards" / "shard-000000.tar"
    n_ok, n_fail, errs = VC.verify_shard(shard)
    assert n_fail == 1
    assert any("camera-action" in e for e in errs)


def test_verify_sample_bad_fov_fails(tmp_path):
    sample = _good_sample(tmp_path, "bad_fov_001")
    # Mutate intrinsics to give absurdly wide FOV (fx too small)
    bad_intr = sample.intrinsics_NVD.copy()
    bad_intr[..., 0] = 100.0     # fx=100 → fov_x ≈ 161° > 120
    sample2 = Sample(**{**sample.__dict__, "intrinsics_NVD": bad_intr})
    with ShardWriter(tmp_path / "shards", samples_per_shard=10) as w:
        w.write(sample2)
    shard = tmp_path / "shards" / "shard-000000.tar"
    n_ok, n_fail, errs = VC.verify_shard(shard)
    assert n_fail == 1
    assert any("fov_x" in e for e in errs)


def test_cli_exit_code_pass(tmp_path):
    sample = _good_sample(tmp_path, "cli_001")
    with ShardWriter(tmp_path / "shards", samples_per_shard=10) as w:
        w.write(sample)
    shard = tmp_path / "shards" / "shard-000000.tar"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(shard)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_exit_code_fail(tmp_path):
    sample = _good_sample(tmp_path, "cli_bad_001",
                          caption="rotating around the centerpiece")
    with ShardWriter(tmp_path / "shards", samples_per_shard=10) as w:
        w.write(sample)
    shard = tmp_path / "shards" / "shard-000000.tar"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(shard)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "FAIL" in r.stdout
