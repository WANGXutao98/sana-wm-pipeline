"""Tests for Stage-01 normalize module (1280x720 @ 16fps).

Paper §5.1, App. D.1: all videos resampled to 720p / 16fps before annotation.
"""
from pathlib import Path

from sana_wm_pipeline.stage01_ingest.normalize import normalize_video, probe

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_video_5s.mp4"


def test_probe_reads_video_metadata():
    info = probe(FIXTURE)
    assert info.width == 640
    assert info.height == 480
    # ffmpeg lavfi testsrc emits exactly 24fps
    assert abs(info.fps - 24.0) < 1e-3
    assert info.n_frames > 0


def test_normalize_outputs_1280x720_16fps(tmp_path):
    dst = tmp_path / "out.mp4"
    out_info = normalize_video(FIXTURE, dst)
    assert dst.exists()
    probed = probe(dst)
    assert probed.width == 1280
    assert probed.height == 720
    assert abs(probed.fps - 16.0) < 1e-3
    # Returned info should match what probe sees on disk
    assert out_info.width == probed.width
    assert out_info.height == probed.height
    assert abs(out_info.fps - probed.fps) < 1e-3
