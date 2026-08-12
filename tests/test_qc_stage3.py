from __future__ import annotations
import io, json, tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.qc.stage3_gpu import process_sample_stage3, run_stage3

TESTDATA = Path(__file__).parent.parent / "testdata"
SAMPLE_ID = "OmniWorld-Game_1f79eb96f021__splits_013-015"
GROUP = "wds-OmniWorld-Game"


@pytest.fixture(scope="session")
def tiny_tar(tmp_path_factory):
    p = tmp_path_factory.mktemp("s3") / "shard.tar"
    with tarfile.open(p, "w") as tf:
        for ext in [".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"]:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return p


def _mock_flow_fn(img_a, img_b):
    H, W = img_a.shape[:2]
    return np.ones((H, W, 2), dtype=np.float32) * 10.0  # constant flow = 10px


def _mock_dover_fn(frames_rgb):
    return 0.7  # reasonable quality score


def _mock_vlm_call(prompt, keyframes):
    return json.dumps({
        "people": 2, "vehicles": 0, "animals": 0,
        "quality": 1.0, "too_dark": False, "blurry": False,
    })


@pytest.fixture
def table6_cfg():
    from sana_wm_pipeline.stage04_filter.apply_table6 import load_thresholds
    cfg_path = Path(__file__).parent.parent / "configs" / "filter_thresholds.yaml"
    return load_thresholds(cfg_path)


def test_process_sample_stage3_structure(tiny_tar, table6_cfg):
    result = process_sample_stage3(
        SAMPLE_ID, tiny_tar, GROUP,
        flow_fn=_mock_flow_fn, dover_fn=_mock_dover_fn, vlm_call=_mock_vlm_call,
        table6_cfg=table6_cfg, has_camera_words=False,
    )
    assert "sample_id" in result and "stage3" in result
    s3 = result["stage3"]
    for k in ("unimatch_flow", "dover", "vlm_entity_count", "vlm_quality", "table6_accepted", "reasons"):
        assert k in s3, f"missing key: {k}"


def test_process_sample_stage3_mock_values(tiny_tar, table6_cfg):
    result = process_sample_stage3(
        SAMPLE_ID, tiny_tar, GROUP,
        flow_fn=_mock_flow_fn, dover_fn=_mock_dover_fn, vlm_call=_mock_vlm_call,
        table6_cfg=table6_cfg, has_camera_words=False,
    )
    s3 = result["stage3"]
    assert s3["dover"] == pytest.approx(0.7, abs=0.01)
    assert s3["unimatch_flow"] > 0


def test_process_sample_caption_rewrite_when_camera_words(tiny_tar, table6_cfg):
    def vlm_with_rewrite(prompt, kf):
        r = json.loads(_mock_vlm_call(prompt, kf))
        if "Caption:" in prompt:
            r["caption_revised"] = "A city street scene."
        return json.dumps(r)
    result = process_sample_stage3(
        SAMPLE_ID, tiny_tar, GROUP,
        flow_fn=_mock_flow_fn, dover_fn=_mock_dover_fn, vlm_call=vlm_with_rewrite,
        table6_cfg=table6_cfg, has_camera_words=True,
    )
    assert result["stage3"]["caption_revised"] is not None
    assert result["stage3"]["caption_revised"] == "A city street scene."


def test_run_stage3_end_to_end(tiny_tar, table6_cfg, tmp_path):
    s1 = tmp_path / "s1.jsonl"
    s1.write_text(json.dumps({
        "sample_id": SAMPLE_ID, "group": GROUP, "tar_path": str(tiny_tar),
        "verdict": "pass", "flag_reasons": [], "metrics": {"camera_words": []},
    }) + "\n")
    out = tmp_path / "s3.jsonl"
    cap_out = tmp_path / "caption_overrides.jsonl"
    count = run_stage3(
        s1, out, cap_out,
        flow_fn=_mock_flow_fn, dover_fn=_mock_dover_fn, vlm_call=_mock_vlm_call,
        table6_cfg=table6_cfg,
    )
    assert count == 1 and out.exists()
    rec = json.loads(out.read_text().splitlines()[0])
    assert "stage3" in rec
