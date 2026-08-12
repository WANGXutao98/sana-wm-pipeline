# tests/test_apply_human_review.py
import json
from pathlib import Path
import pytest
from scripts.apply_human_review import merge_decisions, generate_manifests, generate_summary_report

def test_merge_decisions_override():
    stage1_results = [
        {"sample_id": "A", "verdict": "fail", "n_jumps": 3},
        {"sample_id": "B", "verdict": "pass", "n_jumps": 1},
    ]
    human_review = [
        {"sample_id": "A", "auto_verdict": "fail", "human_verdict": "pass",
         "video_quality": "good", "primary_issue": "trajectory_minor", "notes": "OK"},
    ]

    merged = merge_decisions(stage1_results, human_review)

    assert len(merged) == 2
    # A的verdict被人工覆盖
    assert merged[0]["sample_id"] == "A"
    assert merged[0]["verdict"] == "pass"
    assert merged[0]["human_reviewed"] is True
    assert merged[0]["human_feedback"]["primary_issue"] == "trajectory_minor"

    # B保持自动决策
    assert merged[1]["sample_id"] == "B"
    assert merged[1]["verdict"] == "pass"
    assert merged[1]["human_reviewed"] is False

def test_merge_decisions_no_human_review():
    stage1_results = [
        {"sample_id": "A", "verdict": "fail", "n_jumps": 3},
    ]
    human_review = []

    merged = merge_decisions(stage1_results, human_review)

    assert len(merged) == 1
    assert merged[0]["verdict"] == "fail"
    assert merged[0]["human_reviewed"] is False

def test_generate_manifests(tmp_path):
    merged_results = [
        {"sample_id": "A", "verdict": "pass", "human_reviewed": True},
        {"sample_id": "B", "verdict": "fail", "human_reviewed": False},
        {"sample_id": "C", "verdict": "pass", "human_reviewed": False},
    ]

    manifests_dir = tmp_path / "manifests"
    generate_manifests(merged_results, manifests_dir)

    # 检查pass.txt
    pass_file = manifests_dir / "pass.txt"
    assert pass_file.exists()
    pass_ids = pass_file.read_text().strip().split("\n")
    assert set(pass_ids) == {"A", "C"}

    # 检查fail.txt
    fail_file = manifests_dir / "fail.txt"
    assert fail_file.exists()
    fail_ids = fail_file.read_text().strip().split("\n")
    assert fail_ids == ["B"]

    # 检查human_reviewed.txt
    human_file = manifests_dir / "human_reviewed.txt"
    assert human_file.exists()
    human_ids = human_file.read_text().strip().split("\n")
    assert human_ids == ["A"]

def test_generate_summary_report(tmp_path):
    merged_results = [
        {"sample_id": "A", "verdict": "pass", "human_reviewed": True,
         "human_feedback": {"auto_verdict": "fail"}},
        {"sample_id": "B", "verdict": "fail", "human_reviewed": True,
         "human_feedback": {"auto_verdict": "pass"}},
        {"sample_id": "C", "verdict": "pass", "human_reviewed": False},
        {"sample_id": "D", "verdict": "fail", "human_reviewed": False},
    ]

    output_path = tmp_path / "summary.html"
    generate_summary_report(merged_results, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Final Statistics" in content
    assert "Human Review Impact" in content
