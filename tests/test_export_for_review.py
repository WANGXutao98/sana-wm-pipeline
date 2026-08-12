# tests/test_export_for_review.py
import json
from pathlib import Path
import pytest
import tarfile
import pandas as pd
from scripts.export_for_review import load_and_merge_results, balanced_sampling, extract_videos, generate_review_list, generate_template

def test_load_stage1_only(tmp_path):
    s1 = tmp_path / "stage1.jsonl"
    s1.write_text(json.dumps({"sample_id": "A", "verdict": "pass", "n_jumps": 1}) + "\n")

    results = load_and_merge_results([s1], [])
    assert len(results) == 1
    assert results[0]["sample_id"] == "A"
    assert results[0]["verdict"] == "pass"

def test_merge_stage1_and_stage2(tmp_path):
    s1 = tmp_path / "stage1.jsonl"
    s2 = tmp_path / "stage2.jsonl"
    s1.write_text(json.dumps({"sample_id": "A", "verdict": "pass", "n_jumps": 1}) + "\n")
    s2.write_text(json.dumps({"sample_id": "A", "black_frame_ratio": 0.02, "scene_cuts": 1}) + "\n")

    results = load_and_merge_results([s1], [s2])
    assert len(results) == 1
    assert results[0]["sample_id"] == "A"
    assert results[0]["black_frame_ratio"] == 0.02
    assert results[0]["scene_cuts"] == 1

def test_balanced_sampling_fail_near_threshold():
    results = [
        {"sample_id": "A", "verdict": "fail", "n_jumps": 3, "flag_reasons": "trajectory_jump"},
        {"sample_id": "B", "verdict": "fail", "n_jumps": 10, "flag_reasons": "trajectory_jump"},
        {"sample_id": "C", "verdict": "pass", "n_jumps": 1, "flag_reasons": ""},
    ]
    config = {
        "total_samples": 1,
        "buckets": {"fail_near_threshold": 1, "multiple_reasons": 0, "pass_random": 0, "fail_random": 0}
    }

    sampled = balanced_sampling(results, config)
    assert len(sampled) == 1
    assert sampled[0]["sample_id"] == "A"  # n_jumps=3更接近阈值

def test_balanced_sampling_multiple_reasons():
    results = [
        {"sample_id": "A", "verdict": "fail", "flag_reasons": "trajectory_jump"},
        {"sample_id": "B", "verdict": "fail", "flag_reasons": "trajectory_jump|black_frames"},
        {"sample_id": "C", "verdict": "fail", "flag_reasons": "trajectory_jump|black_frames|caption_short"},
    ]
    config = {
        "total_samples": 1,
        "buckets": {"fail_near_threshold": 0, "multiple_reasons": 1, "pass_random": 0, "fail_random": 0}
    }

    sampled = balanced_sampling(results, config)
    assert len(sampled) == 1
    assert sampled[0]["sample_id"] in ["B", "C"]  # 多原因样本

def test_extract_videos(tmp_path):
    # 创建测试tar文件
    tar_path = tmp_path / "test.tar"
    video_content = b"fake_mp4_content"

    with tarfile.open(tar_path, "w") as tf:
        import io
        video_file = io.BytesIO(video_content)
        info = tarfile.TarInfo("sample_A.mp4")
        info.size = len(video_content)
        tf.addfile(info, video_file)

    samples = [{"sample_id": "sample_A", "tar_path": str(tar_path)}]
    output_dir = tmp_path / "videos"

    extract_videos(samples, output_dir)

    assert (output_dir / "sample_A.mp4").exists()
    assert (output_dir / "sample_A.mp4").read_bytes() == video_content

def test_generate_review_list(tmp_path):
    samples = [{
        "sample_id": "A",
        "group": "wds-DL3DV-ALL-2K",
        "tar_path": "/data/shard-001.tar",
        "verdict": "fail",
        "flag_reasons": "trajectory_jump",
        "n_jumps": 3,
        "caption_len": 45,
        "black_frame_ratio": 0.01,
        "scene_cuts": 0,
        "caption_text": "Person walking",
    }]
    output_path = tmp_path / "review_list.csv"

    # 创建视频目录和文件以测试存在的情况
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "A.mp4").write_bytes(b"fake_video")

    generate_review_list(samples, output_path, video_dir)

    df = pd.read_csv(output_path)
    assert len(df) == 1
    assert df.iloc[0]["sample_id"] == "A"
    assert df.iloc[0]["auto_verdict"] == "fail"
    assert df.iloc[0]["video_path"] == "videos/A.mp4"

def test_generate_template(tmp_path):
    samples = [{"sample_id": "A", "verdict": "fail"}]
    output_path = tmp_path / "template.csv"

    generate_template(samples, output_path)

    df = pd.read_csv(output_path)
    assert len(df) == 1
    assert df.iloc[0]["sample_id"] == "A"
    assert df.iloc[0]["auto_verdict"] == "fail"
    assert pd.isna(df.iloc[0]["human_verdict"])
