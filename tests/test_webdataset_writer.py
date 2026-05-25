"""Tests for WebDataset shard writer."""
import io
import json
import tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.stage06_pack.schema import (
    Sample, CAMERA_FRAMES, EXPECTED_WH, INTRINSICS_VIEWS, INTRINSICS_DIM,
)
from sana_wm_pipeline.stage06_pack.webdataset_writer import ShardWriter


def _make_valid_sample(tmp_path: Path, sample_id: str = "clip_0") -> Sample:
    video = tmp_path / f"{sample_id}.mp4"
    video.write_bytes(b"\x00fake video bytes")  # writer only reads as bytes
    poses = np.tile(np.eye(4, dtype=np.float32), (CAMERA_FRAMES, 1, 1))
    intrinsics = np.tile([[[700.0, 700.0, 640.0, 360.0]]],
                         (CAMERA_FRAMES, 1, 1)).astype(np.float32)
    scale = np.ones(CAMERA_FRAMES, dtype=np.float32)
    return Sample(
        sample_id=sample_id,
        video_path=str(video),
        poses_c2w=poses,
        intrinsics_NVD=intrinsics,
        scale_per_frame=scale,
        caption="a wooden table under afternoon light",
        meta={"source": "test", "pose_mode": "default"},
    )


def test_paper_constants():
    assert CAMERA_FRAMES == 961
    assert EXPECTED_WH == (1280, 720)
    assert INTRINSICS_VIEWS == 1
    assert INTRINSICS_DIM == 4


def test_valid_sample_validates():
    sample = _make_valid_sample(Path("/tmp"))
    sample.validate()   # no raise


def test_wrong_poses_shape_raises():
    sample = _make_valid_sample(Path("/tmp"))
    bad = np.zeros((100, 4, 4), dtype=np.float32)
    sample2 = Sample(**{**sample.__dict__, "poses_c2w": bad})
    with pytest.raises(ValueError, match="poses_c2w shape"):
        sample2.validate()


def test_poses_must_be_float32():
    sample = _make_valid_sample(Path("/tmp"))
    bad = np.tile(np.eye(4, dtype=np.float64), (CAMERA_FRAMES, 1, 1))
    sample2 = Sample(**{**sample.__dict__, "poses_c2w": bad})
    with pytest.raises(ValueError, match="float32"):
        sample2.validate()


def test_intrinsics_NVD_shape_enforced():
    sample = _make_valid_sample(Path("/tmp"))
    bad = np.zeros((CAMERA_FRAMES, 2, 4), dtype=np.float32)  # V=2 not 1
    sample2 = Sample(**{**sample.__dict__, "intrinsics_NVD": bad})
    with pytest.raises(ValueError, match="intrinsics_NVD shape"):
        sample2.validate()


def test_first_frame_must_be_identity():
    sample = _make_valid_sample(Path("/tmp"))
    bad = sample.poses_c2w.copy()
    bad[0, 0, 3] = 5.0   # translate first frame
    sample2 = Sample(**{**sample.__dict__, "poses_c2w": bad})
    with pytest.raises(ValueError, match="identity"):
        sample2.validate()


def test_empty_caption_raises():
    sample = _make_valid_sample(Path("/tmp"))
    sample2 = Sample(**{**sample.__dict__, "caption": "   "})
    with pytest.raises(ValueError, match="caption"):
        sample2.validate()


def test_shard_writer_writes_six_files_per_sample(tmp_path):
    sample = _make_valid_sample(tmp_path, "abc_001")
    with ShardWriter(tmp_path / "shards", samples_per_shard=10) as w:
        w.write(sample)
    shard = tmp_path / "shards" / "shard-000000.tar"
    assert shard.exists()
    with tarfile.open(shard) as t:
        names = set(t.getnames())
    expected = {f"abc_001.{ext}" for ext in
                ("mp4", "poses_c2w.npy", "intrinsics.npy", "scale.npy",
                 "caption.txt", "meta.json")}
    assert expected <= names


def test_shard_rotates_at_capacity(tmp_path):
    out = tmp_path / "shards"
    with ShardWriter(out, samples_per_shard=2) as w:
        for i in range(5):
            s = _make_valid_sample(tmp_path, f"clip_{i:03d}")
            w.write(s)
    shards = sorted(out.glob("shard-*.tar"))
    # 5 samples / 2 per shard = 3 shards (2+2+1)
    assert len(shards) == 3
    sizes = []
    for sh in shards:
        with tarfile.open(sh) as t:
            ids = {n.rsplit(".", 1)[0] for n in t.getnames()
                   if n.endswith(".meta.json")}
            sizes.append(len(ids))
    assert sizes == [2, 2, 1]


def test_round_trip_npy_payload(tmp_path):
    sample = _make_valid_sample(tmp_path, "rt_001")
    with ShardWriter(tmp_path / "shards", samples_per_shard=10) as w:
        w.write(sample)
    shard = tmp_path / "shards" / "shard-000000.tar"
    with tarfile.open(shard) as t:
        poses_bytes = t.extractfile("rt_001.poses_c2w.npy").read()
        scale_bytes = t.extractfile("rt_001.scale.npy").read()
    poses = np.load(io.BytesIO(poses_bytes))
    scale = np.load(io.BytesIO(scale_bytes))
    np.testing.assert_array_equal(poses, sample.poses_c2w)
    np.testing.assert_array_equal(scale, sample.scale_per_frame)


def test_meta_json_round_trip(tmp_path):
    sample = _make_valid_sample(tmp_path, "meta_001")
    with ShardWriter(tmp_path / "shards", samples_per_shard=10) as w:
        w.write(sample)
    shard = tmp_path / "shards" / "shard-000000.tar"
    with tarfile.open(shard) as t:
        meta_bytes = t.extractfile("meta_001.meta.json").read()
    meta = json.loads(meta_bytes)
    assert meta == sample.meta
