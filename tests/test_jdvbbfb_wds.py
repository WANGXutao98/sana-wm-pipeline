import io
import json
import tarfile

import numpy as np
import pytest

from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import (
    CameraGT,
    SampleRef,
    iter_tar_samples,
    load_camera_gt,
    read_index,
    write_scene_dir,
)


def _synth_camera_npz(T=5) -> bytes:
    """Build an in-memory per_frame_camera_npz_v1 like the real dataset."""
    c2w = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    for t in range(T):
        c2w[t, 0, 3] = float(t)
    buf = io.BytesIO()
    np.savez(
        buf,
        c2w=c2w,
        w2c=np.linalg.inv(c2w).astype(np.float32),
        K_px=np.tile(np.array([500, 500, 320, 240], np.float32), (T, 1)),
        frame_indices=np.arange(T, dtype=np.int32),
        width=np.int32(1920), height=np.int32(1080), fps=np.float32(30.0),
        pose_convention=np.array("opencv_c2w"),
        vipe_c2w=c2w.copy(),
    )
    return buf.getvalue()


# ── Task 2: load_camera_gt ──────────────────────────────────────────────────

def test_load_camera_gt_basic():
    gt = load_camera_gt(_synth_camera_npz(T=5))
    assert isinstance(gt, CameraGT)
    assert gt.c2w.shape == (5, 4, 4) and gt.c2w.dtype == np.float32
    assert gt.k_px.shape == (5, 4)
    assert gt.fps == pytest.approx(30.0)
    assert gt.width == 1920 and gt.height == 1080
    assert gt.vipe_c2w is not None and gt.vipe_c2w.shape == (5, 4, 4)


def test_load_camera_gt_missing_vipe_is_none():
    buf = io.BytesIO()
    np.savez(buf,
             c2w=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
             K_px=np.zeros((3, 4), np.float32),
             width=np.int32(640), height=np.int32(360), fps=np.float32(24.0))
    gt = load_camera_gt(buf.getvalue())
    assert gt.vipe_c2w is None
    assert gt.fps == pytest.approx(24.0)


# ── Task 3: read_index ──────────────────────────────────────────────────────

def test_read_index_parses_records(tmp_path):
    rec = {
        "sample_id": "DL3DV-ALL-2K/6K__abc__images_2",
        "key": "DL3DV-ALL-2K_6K__abc__images_2",
        "shard": "shards/DL3DV-ALL-2K-000000.tar",
        "video_member": "DL3DV-ALL-2K_6K__abc__images_2.mp4",
        "camera_member": "DL3DV-ALL-2K_6K__abc__images_2.camera.npz",
        "manifest": {"video": {"fps": 30.0, "num_frames": 300},
                     "prompt": {"text": "a calm indoor lounge"}},
    }
    p = tmp_path / "index.jsonl"
    p.write_text(json.dumps(rec) + "\n" + json.dumps({**rec, "key": "k2",
                 "shard": "shards/DL3DV-ALL-2K-000001.tar"}) + "\n")

    refs = read_index(p)
    assert len(refs) == 2
    r0 = refs[0]
    assert isinstance(r0, SampleRef)
    assert r0.key == "DL3DV-ALL-2K_6K__abc__images_2"
    assert r0.shard == "shards/DL3DV-ALL-2K-000000.tar"
    assert r0.video_member.endswith(".mp4")
    assert r0.camera_member.endswith(".camera.npz")
    assert r0.caption == "a calm indoor lounge"
    assert r0.fps == pytest.approx(30.0)


def test_read_index_caption_fallback(tmp_path):
    rec = {"key": "k", "shard": "s.tar",
           "video_member": "k.mp4", "camera_member": "k.camera.npz",
           "manifest": {"video": {}}}
    p = tmp_path / "index.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    refs = read_index(p)
    assert refs[0].caption == ""


# ── Task 4: iter_tar_samples ────────────────────────────────────────────────

def _make_shard(tmp_path, keys):
    shard = tmp_path / "shard.tar"
    with tarfile.open(shard, "w") as tf:
        for k in keys:
            for ext, payload in [(".mp4", b"FAKEVIDEO" + k.encode()),
                                 (".camera.npz", b"FAKENPZ" + k.encode())]:
                data = payload
                ti = tarfile.TarInfo(f"{k}{ext}"); ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return shard


def test_iter_tar_samples_pairs_mp4_and_npz(tmp_path):
    shard = _make_shard(tmp_path, ["sampleA", "sampleB"])
    got = list(iter_tar_samples(open(shard, "rb"), limit=None))
    assert [k for k, _, _ in got] == ["sampleA", "sampleB"]
    k0, mp4_0, npz_0 = got[0]
    assert mp4_0 == b"FAKEVIDEOsampleA"
    assert npz_0 == b"FAKENPZsampleA"


def test_iter_tar_samples_respects_limit(tmp_path):
    shard = _make_shard(tmp_path, ["a", "b", "c"])
    got = list(iter_tar_samples(open(shard, "rb"), limit=2))
    assert len(got) == 2


# ── Task 5: write_scene_dir ─────────────────────────────────────────────────

def test_write_scene_dir_layout(tmp_path):
    cam = _synth_camera_npz(T=4)
    scene = write_scene_dir(
        out_base=tmp_path,
        scene_id="DL3DV-ALL-2K_6K__abc__images_2",
        mp4_bytes=b"FAKEVIDEO",
        camera_npz_bytes=cam,
        caption="a calm indoor lounge",
    )
    assert (scene / "video.mp4").read_bytes() == b"FAKEVIDEO"
    assert (scene / "caption.txt").read_text() == "a calm indoor lounge"
    assert (scene / "orig_fps.txt").read_text().strip() == "30.0"
    gt = np.load(scene / "gt_poses.npy")
    assert gt.shape == (4, 4, 4) and gt.dtype == np.float32
    intr = np.load(scene / "gt_intrinsics.npy")
    assert intr.shape == (4, 4)
    assert (scene / "vipe_ref_poses.npy").exists()


def test_write_scene_dir_empty_caption_gets_stub(tmp_path):
    buf = io.BytesIO()
    np.savez(buf, c2w=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
             K_px=np.zeros((2, 4), np.float32), fps=np.float32(24.0))
    scene = write_scene_dir(tmp_path, "scene_x", b"v", buf.getvalue(), caption="")
    txt = (scene / "caption.txt").read_text()
    assert txt.strip()
