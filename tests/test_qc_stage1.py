# tests/test_qc_stage1.py
from __future__ import annotations
import io, json, tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.qc.stage1_fast import scan_tar, run_stage1

TESTDATA = Path(__file__).parent.parent / "testdata"
SAMPLE_ID = "OmniWorld-Game_1f79eb96f021__splits_013-015"
GROUP = "wds-OmniWorld-Game"


@pytest.fixture(scope="session")
def tiny_tar(tmp_path_factory):
    p = tmp_path_factory.mktemp("s1") / "shard-000000.tar"
    exts = [".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"]
    with tarfile.open(p, "w") as tf:
        for ext in exts:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return p


@pytest.fixture(scope="session")
def corrupt_tar(tmp_path_factory):
    p = tmp_path_factory.mktemp("s1c") / "shard-000000.tar"
    with tarfile.open(p, "w") as tf:
        for ext in [".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".caption.txt", ".meta.json"]:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return p  # missing .scale.npy → fail


def test_scan_tar_returns_list(tiny_tar):
    results = scan_tar(tiny_tar, GROUP)
    assert isinstance(results, list) and len(results) == 1

def test_scan_tar_has_keys(tiny_tar):
    r = scan_tar(tiny_tar, GROUP)[0]
    for k in ("sample_id", "group", "tar_path", "verdict", "flag_reasons", "metrics"):
        assert k in r

def test_scan_tar_sample_id(tiny_tar):
    assert scan_tar(tiny_tar, GROUP)[0]["sample_id"] == SAMPLE_ID

def test_scan_tar_testdata_passes(tiny_tar):
    r = scan_tar(tiny_tar, GROUP)[0]
    assert r["metrics"]["so3_valid"] and r["metrics"]["first_frame_ok"]
    assert r["verdict"] in ("pass", "flag")  # OmniWorld may flag for jumps

def test_scan_tar_missing_file_fails(corrupt_tar):
    r = scan_tar(corrupt_tar, GROUP)[0]
    assert r["verdict"] == "fail"
    assert any("missing" in x.lower() for x in r["flag_reasons"])

def test_run_stage1_writes_jsonl(tiny_tar, tmp_path):
    out = tmp_path / "s1.jsonl"
    count = run_stage1([tiny_tar], GROUP, out, n_workers=1)
    assert count == 1 and out.exists()
    assert json.loads(out.read_text().splitlines()[0])["sample_id"] == SAMPLE_ID

def test_run_stage1_two_tars(tiny_tar, tmp_path):
    out = tmp_path / "s1.jsonl"
    count = run_stage1([tiny_tar, tiny_tar], GROUP, out, n_workers=1)
    assert count == 2 and len(out.read_text().strip().splitlines()) == 2
