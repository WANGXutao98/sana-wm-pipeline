#!/usr/bin/env python3
# scripts/export_for_review.py
from __future__ import annotations
import json, glob, sys, random, tarfile, argparse
from pathlib import Path
from typing import Any
import pandas as pd

def load_and_merge_results(
    stage1_paths: list[Path],
    stage2_paths: list[Path]
) -> list[dict[str, Any]]:
    """加载Stage 1结果，可选地合并Stage 2数据。"""
    # 加载所有stage1文件
    stage1_data = {}
    for path in stage1_paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    stage1_data[rec["sample_id"]] = rec

    # 合并stage2数据（如果提供）
    if stage2_paths:
        for path in stage2_paths:
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        sid = rec["sample_id"]
                        if sid in stage1_data:
                            stage1_data[sid].update({
                                "black_frame_ratio": rec.get("black_frame_ratio"),
                                "scene_cuts": rec.get("scene_cuts"),
                                "video_T": rec.get("video_T"),
                                "traj_frozen": rec.get("traj_frozen")
                            })

    return list(stage1_data.values())

def balanced_sampling(
    results: list[dict[str, Any]],
    config: dict[str, Any]
) -> list[dict[str, Any]]:
    """根据配置的bucket策略采样。"""
    buckets = config["buckets"]
    total = config["total_samples"]
    sampled = []

    # Bucket 1: fail_near_threshold - 接近阈值的失败样本
    if buckets.get("fail_near_threshold", 0) > 0:
        fail_samples = [r for r in results if r["verdict"] == "fail"]
        # 按n_jumps升序排序（越小越接近通过阈值）
        fail_samples.sort(key=lambda x: x.get("n_jumps", 999))
        sampled.extend(fail_samples[:buckets["fail_near_threshold"]])

    # Bucket 2: multiple_reasons - 多原因标记
    if buckets.get("multiple_reasons", 0) > 0:
        multi_reason = [r for r in results if len(r.get("flag_reasons", "").split("|")) >= 2]
        multi_reason.sort(key=lambda x: len(x.get("flag_reasons", "").split("|")), reverse=True)
        sampled.extend(multi_reason[:buckets["multiple_reasons"]])

    # Bucket 3: pass_random - 随机通过样本
    if buckets.get("pass_random", 0) > 0:
        pass_samples = [r for r in results if r["verdict"] == "pass" and r not in sampled]
        sampled.extend(random.sample(pass_samples, min(buckets["pass_random"], len(pass_samples))))

    # Bucket 4: fail_random - 随机失败样本
    if buckets.get("fail_random", 0) > 0:
        fail_samples = [r for r in results if r["verdict"] == "fail" and r not in sampled]
        sampled.extend(random.sample(fail_samples, min(buckets["fail_random"], len(fail_samples))))

    # 去重并限制总数
    seen = set()
    unique_sampled = []
    for s in sampled:
        if s["sample_id"] not in seen:
            seen.add(s["sample_id"])
            unique_sampled.append(s)

    return unique_sampled[:total]

def extract_videos(samples: list[dict[str, Any]], output_dir: Path) -> None:
    """从tar文件中提取视频到output_dir/。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = []

    for sample in samples:
        sample_id = sample["sample_id"]
        tar_path = Path(sample["tar_path"])

        if not tar_path.exists():
            skipped.append(f"{sample_id}: tar not found")
            continue

        try:
            with tarfile.open(tar_path, "r") as tf:
                # 查找sample_id.mp4
                video_name = f"{sample_id}.mp4"
                try:
                    member = tf.getmember(video_name)
                    f = tf.extractfile(member)
                    if f:
                        output_path = output_dir / f"{sample_id}.mp4"
                        output_path.write_bytes(f.read())
                except KeyError:
                    skipped.append(f"{sample_id}: video not in tar")
        except Exception as e:
            skipped.append(f"{sample_id}: {e}")

    if skipped:
        print(f"WARNING: Skipped {len(skipped)} videos:", file=sys.stderr)
        for msg in skipped[:10]:  # 只打印前10个
            print(f"  {msg}", file=sys.stderr)

def generate_review_list(
    samples: list[dict[str, Any]],
    output_path: Path,
    video_dir: Path
) -> None:
    """生成review_list.csv。"""
    rows = []
    for s in samples:
        video_path = f"videos/{s['sample_id']}.mp4"
        if not (video_dir / f"{s['sample_id']}.mp4").exists():
            video_path = "MISSING"

        rows.append({
            "sample_id": s["sample_id"],
            "group": s.get("group", ""),
            "tar_path": s.get("tar_path", ""),
            "auto_verdict": s["verdict"],
            "flag_reasons": s.get("flag_reasons", ""),
            "n_jumps": s.get("n_jumps", ""),
            "caption_len": s.get("caption_len", ""),
            "black_frame_ratio": s.get("black_frame_ratio", ""),
            "scene_cuts": s.get("scene_cuts", ""),
            "caption_text": s.get("caption_text", ""),
            "video_path": video_path
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")

def generate_template(samples: list[dict[str, Any]], output_path: Path) -> None:
    """生成decisions_template.csv。"""
    rows = []
    for s in samples:
        rows.append({
            "sample_id": s["sample_id"],
            "auto_verdict": s["verdict"],
            "human_verdict": "",
            "video_quality": "",
            "trajectory_quality": "",
            "primary_issue": "",
            "notes": ""
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description="Export samples for human review")
    p.add_argument("--stage1-jsonl", nargs="+", required=True, help="Stage 1 result files (supports glob)")
    p.add_argument("--stage2-jsonl", nargs="*", default=[], help="Stage 2 result files (optional)")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--total-samples", type=int, default=1000, help="Total samples to export")
    p.add_argument("--sampling-strategy", default="balanced", choices=["balanced"], help="Sampling strategy")
    args = p.parse_args()

    # 展开glob模式
    stage1_paths = []
    for pattern in args.stage1_jsonl:
        stage1_paths.extend([Path(p) for p in glob.glob(pattern)])

    stage2_paths = []
    for pattern in args.stage2_jsonl:
        stage2_paths.extend([Path(p) for p in glob.glob(pattern)])

    # 加载数据
    print(f"Loading Stage 1 results from {len(stage1_paths)} files...")
    results = load_and_merge_results(stage1_paths, stage2_paths)
    print(f"Loaded {len(results)} samples")

    # 采样
    config = {
        "total_samples": args.total_samples,
        "buckets": {
            "fail_near_threshold": int(args.total_samples * 0.4),
            "multiple_reasons": int(args.total_samples * 0.2),
            "pass_random": int(args.total_samples * 0.3),
            "fail_random": int(args.total_samples * 0.1)
        }
    }
    print(f"Sampling {args.total_samples} samples...")
    sampled = balanced_sampling(results, config)
    print(f"Selected {len(sampled)} samples")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"

    # 提取视频
    print(f"Extracting videos to {video_dir}...")
    extract_videos(sampled, video_dir)

    # 生成CSV
    print("Generating CSVs...")
    generate_review_list(sampled, args.output_dir / "review_list.csv", video_dir)
    generate_template(sampled, args.output_dir / "decisions_template.csv")

    # 生成采样报告
    report_path = args.output_dir / "sampling_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Sampling Report\n")
        f.write(f"===============\n")
        f.write(f"Total samples loaded: {len(results)}\n")
        f.write(f"Total samples selected: {len(sampled)}\n")
        f.write(f"Pass samples: {sum(1 for s in sampled if s['verdict'] == 'pass')}\n")
        f.write(f"Fail samples: {sum(1 for s in sampled if s['verdict'] == 'fail')}\n")

    print(f"Done! Output in {args.output_dir}")

if __name__ == "__main__":
    main()
