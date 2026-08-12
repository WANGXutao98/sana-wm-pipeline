#!/usr/bin/env python3
# scripts/apply_human_review.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import argparse
import glob
from jinja2 import Template

SUMMARY_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Final QC Summary Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .pass { color: #5cb85c; font-weight: bold; }
        .fail { color: #d9534f; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Final QC Summary Report</h1>

    <h2>Final Statistics</h2>
    <table>
        <tr><td>Total samples</td><td>{{ total }}</td></tr>
        <tr><td class="pass">Pass samples</td><td class="pass">{{ pass_count }} ({{ pass_pct }}%)</td></tr>
        <tr><td class="fail">Fail samples</td><td class="fail">{{ fail_count }} ({{ fail_pct }}%)</td></tr>
        <tr><td>Human reviewed</td><td>{{ human_reviewed_count }} ({{ human_reviewed_pct }}%)</td></tr>
    </table>

    <h2>Human Review Impact</h2>
    <table>
        <tr><th>Change Type</th><th>Count</th></tr>
        <tr><td>Auto Fail → Human Pass</td><td>{{ auto_fail_human_pass }}</td></tr>
        <tr><td>Auto Pass → Human Fail</td><td>{{ auto_pass_human_fail }}</td></tr>
        <tr><td>No change</td><td>{{ no_change }}</td></tr>
    </table>

    <h2>Ready for Stage 3</h2>
    <p>Pass samples ({{ pass_count }}) are ready for Stage 3 GPU evaluation.</p>
    <p>Use <code>manifests/pass.txt</code> as input.</p>
</body>
</html>
"""

def merge_decisions(
    stage1_results: list[dict[str, Any]],
    human_review: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """将人工决策合并到Stage 1结果中。"""
    # 构建人工审查字典
    human_dict = {r["sample_id"]: r for r in human_review}

    merged = []
    for sample in stage1_results:
        sample_copy = sample.copy()
        sample_id = sample["sample_id"]

        if sample_id in human_dict:
            # 人工决策覆盖自动决策
            human = human_dict[sample_id]
            sample_copy["verdict"] = human["human_verdict"]
            sample_copy["human_reviewed"] = True
            sample_copy["human_feedback"] = {
                "auto_verdict": human["auto_verdict"],
                "video_quality": human.get("video_quality"),
                "trajectory_quality": human.get("trajectory_quality"),
                "primary_issue": human["primary_issue"],
                "notes": human.get("notes", "")
            }
        else:
            # 保持自动决策
            sample_copy["human_reviewed"] = False

        merged.append(sample_copy)

    return merged

def generate_manifests(
    merged_results: list[dict[str, Any]],
    manifests_dir: Path
) -> None:
    """生成pass/fail/human_reviewed清单文件。"""
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # pass.txt
    pass_samples = [r["sample_id"] for r in merged_results if r["verdict"] == "pass"]
    (manifests_dir / "pass.txt").write_text("\n".join(pass_samples) + "\n")

    # fail.txt
    fail_samples = [r["sample_id"] for r in merged_results if r["verdict"] == "fail"]
    (manifests_dir / "fail.txt").write_text("\n".join(fail_samples) + "\n")

    # human_reviewed.txt
    human_reviewed = [r["sample_id"] for r in merged_results if r.get("human_reviewed", False)]
    (manifests_dir / "human_reviewed.txt").write_text("\n".join(human_reviewed) + "\n")

def generate_summary_report(
    merged_results: list[dict[str, Any]],
    output_path: Path
) -> None:
    """生成最终汇总HTML报告。"""
    total = len(merged_results)
    pass_count = sum(1 for r in merged_results if r["verdict"] == "pass")
    fail_count = total - pass_count
    human_reviewed_count = sum(1 for r in merged_results if r.get("human_reviewed", False))

    # 计算人工审查影响
    auto_fail_human_pass = 0
    auto_pass_human_fail = 0
    no_change = 0

    for r in merged_results:
        if r.get("human_reviewed", False):
            auto_verdict = r.get("human_feedback", {}).get("auto_verdict")
            human_verdict = r["verdict"]

            if auto_verdict == "fail" and human_verdict == "pass":
                auto_fail_human_pass += 1
            elif auto_verdict == "pass" and human_verdict == "fail":
                auto_pass_human_fail += 1
            else:
                no_change += 1

    # 渲染HTML
    template = Template(SUMMARY_HTML_TEMPLATE)
    html = template.render(
        total=total,
        pass_count=pass_count,
        pass_pct=round(100 * pass_count / total, 1) if total > 0 else 0,
        fail_count=fail_count,
        fail_pct=round(100 * fail_count / total, 1) if total > 0 else 0,
        human_reviewed_count=human_reviewed_count,
        human_reviewed_pct=round(100 * human_reviewed_count / total, 1) if total > 0 else 0,
        auto_fail_human_pass=auto_fail_human_pass,
        auto_pass_human_fail=auto_pass_human_fail,
        no_change=no_change
    )

    output_path.write_text(html, encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description="Apply human review decisions to Stage 1 results")
    p.add_argument("--stage1-jsonl", nargs="+", required=True, help="Stage 1 result files (supports glob)")
    p.add_argument("--human-review", type=Path, required=True, help="human_review_results.jsonl from import script")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    args = p.parse_args()

    # 展开glob模式
    stage1_paths = []
    for pattern in args.stage1_jsonl:
        stage1_paths.extend([Path(p) for p in glob.glob(pattern)])

    # 加载Stage 1结果
    print(f"Loading Stage 1 results from {len(stage1_paths)} files...")
    stage1_results = []
    for path in stage1_paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    stage1_results.append(json.loads(line))
    print(f"Loaded {len(stage1_results)} samples")

    # 加载人工审查结果
    print(f"Loading human review from {args.human_review}...")
    human_review = []
    with open(args.human_review) as f:
        for line in f:
            if line.strip():
                human_review.append(json.loads(line))
    print(f"Loaded {len(human_review)} human reviews")

    # 合并决策
    print("Merging decisions...")
    merged = merge_decisions(stage1_results, human_review)

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 写入合并后的JSONL
    merged_jsonl = args.output_dir / "stage1_results_merged.jsonl"
    print(f"Writing {merged_jsonl}...")
    with open(merged_jsonl, "w") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 生成manifests
    manifests_dir = args.output_dir / "manifests"
    print(f"Generating manifests in {manifests_dir}...")
    generate_manifests(merged, manifests_dir)

    # 生成summary报告
    summary_path = args.output_dir / "summary_report.html"
    print(f"Generating {summary_path}...")
    generate_summary_report(merged, summary_path)

    # 打印统计
    pass_count = sum(1 for r in merged if r["verdict"] == "pass")
    print(f"\nDone! Output in {args.output_dir}")
    print(f"Final pass count: {pass_count}")
    print(f"Ready for Stage 3: manifests/pass.txt")

if __name__ == "__main__":
    main()
