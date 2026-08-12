from __future__ import annotations
import json
from pathlib import Path
import pytest
from sana_wm_pipeline.qc.report import merge_results, write_manifests, write_html_report


def _w(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


@pytest.fixture
def base_records():
    base = {"group": "wds-OmniWorld-Game", "tar_path": "/f.tar", "flag_reasons": [],
            "metrics": {"T": 100, "n_jumps": 0, "caption_len": 200}}
    return [
        {**base, "sample_id": "p1", "verdict": "pass"},
        {**base, "sample_id": "p2", "verdict": "pass"},
        {**base, "sample_id": "fl1", "verdict": "flag", "flag_reasons": ["n_jumps=5"]},
        {**base, "sample_id": "fa1", "verdict": "fail", "flag_reasons": ["so3"]},
    ]


def test_merge_stage1_only(tmp_path, base_records):
    s1 = tmp_path / "s1.jsonl"; _w(s1, base_records)
    results = merge_results(s1)
    assert len(results) == 4

def test_merge_stage3_upgrades_to_fail(tmp_path, base_records):
    s1 = tmp_path / "s1.jsonl"; _w(s1, base_records)
    s3 = tmp_path / "s3.jsonl"
    _w(s3, [{**base_records[0], "stage3": {"table6_accepted": False, "reasons": ["dover=0.1"], "unimatch_flow": 5.0, "dover": 0.1, "vlm_entity_count": 2, "vlm_quality": 0.8, "caption_revised": None}}])
    results = merge_results(s1, stage3_jsonl=s3)
    p1 = next(r for r in results if r["sample_id"] == "p1")
    assert p1["verdict"] == "fail"

def test_write_manifests_counts(tmp_path, base_records):
    write_manifests(base_records, tmp_path)
    assert len((tmp_path / "manifests" / "pass.txt").read_text().splitlines()) == 2
    assert len((tmp_path / "manifests" / "reject.txt").read_text().splitlines()) == 1
    assert len((tmp_path / "manifests" / "human_review.txt").read_text().splitlines()) == 1

def test_html_report_created(tmp_path, base_records):
    write_html_report(base_records, tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "<html" in html.lower() and "OmniWorld" in html
