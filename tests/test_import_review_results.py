# tests/test_import_review_results.py
import pandas as pd
import pytest
import json
from pathlib import Path
from scripts.import_review_results import validate_decisions, generate_jsonl, generate_disagreement_report

def test_validate_decisions_valid():
    review_list = pd.DataFrame([
        {"sample_id": "A", "auto_verdict": "fail"},
        {"sample_id": "B", "auto_verdict": "pass"}
    ])
    decisions = pd.DataFrame([
        {"sample_id": "A", "auto_verdict": "fail", "human_verdict": "pass", "primary_issue": "trajectory_minor"},
        {"sample_id": "B", "auto_verdict": "pass", "human_verdict": "pass", "primary_issue": "no_issue"}
    ])

    errors = validate_decisions(decisions, review_list)
    assert len(errors) == 0

def test_validate_decisions_missing_verdict():
    review_list = pd.DataFrame([{"sample_id": "A", "auto_verdict": "fail"}])
    decisions = pd.DataFrame([{"sample_id": "A", "auto_verdict": "fail", "human_verdict": "", "primary_issue": ""}])

    errors = validate_decisions(decisions, review_list)
    assert len(errors) == 0  # 空verdict是允许的

def test_validate_decisions_invalid_verdict():
    review_list = pd.DataFrame([{"sample_id": "A", "auto_verdict": "fail"}])
    decisions = pd.DataFrame([{"sample_id": "A", "auto_verdict": "fail", "human_verdict": "maybe", "primary_issue": "other"}])

    errors = validate_decisions(decisions, review_list)
    assert len(errors) == 1
    assert "invalid verdict" in errors[0].lower()

def test_validate_decisions_unknown_sample():
    review_list = pd.DataFrame([{"sample_id": "A", "auto_verdict": "fail"}])
    decisions = pd.DataFrame([{"sample_id": "Z", "auto_verdict": "fail", "human_verdict": "pass", "primary_issue": "other"}])

    errors = validate_decisions(decisions, review_list)
    assert len(errors) == 1
    assert "not in review_list" in errors[0]

def test_generate_jsonl(tmp_path):
    review_df = pd.DataFrame([
        {"sample_id": "A", "auto_verdict": "fail"},
        {"sample_id": "B", "auto_verdict": "pass"},
    ])
    decisions_df = pd.DataFrame([
        {"sample_id": "A", "auto_verdict": "fail", "human_verdict": "pass",
         "video_quality": "good", "trajectory_quality": "acceptable",
         "primary_issue": "trajectory_minor", "notes": "Small jump"},
        {"sample_id": "B", "auto_verdict": "pass", "human_verdict": "",
         "video_quality": "", "trajectory_quality": "",
         "primary_issue": "", "notes": ""},
    ])

    output_path = tmp_path / "results.jsonl"
    generate_jsonl(review_df, decisions_df, output_path, "batch1")

    assert output_path.exists()
    with open(output_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2
    assert lines[0]["sample_id"] == "A"
    assert lines[0]["human_verdict"] == "pass"
    assert lines[0]["reviewer"] == "batch1"

    # B没有填写human_verdict，应该使用auto_verdict
    assert lines[1]["sample_id"] == "B"
    assert lines[1]["human_verdict"] == "pass"

def test_generate_disagreement_report(tmp_path):
    review_df = pd.DataFrame([
        {"sample_id": "A", "auto_verdict": "fail", "n_jumps": 3},
        {"sample_id": "B", "auto_verdict": "pass", "n_jumps": 1},
        {"sample_id": "C", "auto_verdict": "fail", "n_jumps": 10},
    ])
    decisions_df = pd.DataFrame([
        {"sample_id": "A", "auto_verdict": "fail", "human_verdict": "pass", "primary_issue": "trajectory_minor"},
        {"sample_id": "B", "auto_verdict": "pass", "human_verdict": "pass", "primary_issue": "no_issue"},
        {"sample_id": "C", "auto_verdict": "fail", "human_verdict": "fail", "primary_issue": "trajectory_major"},
    ])

    output_path = tmp_path / "report.html"
    generate_disagreement_report(review_df, decisions_df, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Overall Statistics" in content
    assert "Agreement Analysis" in content
    assert "Disagreement" in content
