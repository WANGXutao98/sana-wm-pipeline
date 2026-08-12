#!/usr/bin/env python3
# scripts/import_review_results.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Any
import sys
import json
import argparse
from datetime import datetime
from jinja2 import Template

VALID_VERDICTS = {"pass", "fail"}
VALID_ISSUES = {
    "trajectory_minor_jump", "trajectory_major_jump", "video_blurry",
    "video_artifacts", "caption_mismatch", "caption_too_vague",
    "black_frames", "scene_cut_abrupt", "multiple_issues", "no_issue", "other"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Human Review Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .warning { color: #d9534f; }
        .success { color: #5cb85c; }
    </style>
</head>
<body>
    <h1>Human Review Analysis Report</h1>

    <h2>Overall Statistics</h2>
    <table>
        <tr><td>Total samples</td><td>{{ total }}</td></tr>
        <tr><td>Reviewed</td><td>{{ reviewed }} ({{ reviewed_pct }}%)</td></tr>
        <tr><td>Not reviewed</td><td>{{ not_reviewed }} ({{ not_reviewed_pct }}%)</td></tr>
    </table>

    <h2>Agreement Analysis</h2>
    <table>
        <tr><th>Auto Verdict</th><th>Human Verdict</th><th>Count</th><th>%</th></tr>
        {% for row in agreement_table %}
        <tr>
            <td>{{ row.auto }}</td>
            <td>{{ row.human }}</td>
            <td>{{ row.count }}</td>
            <td>{{ row.pct }}%</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Disagreement: Auto Fail → Human Pass ({{ auto_fail_human_pass_count }} samples)</h2>
    <p>These are potential false positives (too strict thresholds).</p>
    <table>
        <tr><th>Primary Issue</th><th>Count</th></tr>
        {% for issue, count in auto_fail_human_pass_issues %}
        <tr><td>{{ issue }}</td><td>{{ count }}</td></tr>
        {% endfor %}
    </table>

    <h2>Disagreement: Auto Pass → Human Fail ({{ auto_pass_human_fail_count }} samples)</h2>
    <p>These are false negatives (missed issues).</p>
    <table>
        <tr><th>Primary Issue</th><th>Count</th></tr>
        {% for issue, count in auto_pass_human_fail_issues %}
        <tr><td>{{ issue }}</td><td>{{ count }}</td></tr>
        {% endfor %}
    </table>
</body>
</html>
"""

def validate_decisions(
    decisions: pd.DataFrame,
    review_list: pd.DataFrame
) -> list[str]:
    """验证decisions_filled.csv，返回错误列表。"""
    errors = []

    # 检查sample_id是否在review_list中
    review_ids = set(review_list["sample_id"])
    for idx, row in decisions.iterrows():
        sid = row["sample_id"]
        if sid not in review_ids:
            errors.append(f"Row {idx}: sample_id '{sid}' not in review_list")

    # 检查human_verdict
    for idx, row in decisions.iterrows():
        verdict = row.get("human_verdict", "")
        if pd.notna(verdict) and verdict and verdict not in VALID_VERDICTS:
            errors.append(f"Row {idx}: invalid verdict '{verdict}', must be pass/fail")

    # 检查primary_issue（警告而非错误）
    for idx, row in decisions.iterrows():
        issue = row.get("primary_issue", "")
        if pd.notna(issue) and issue and issue not in VALID_ISSUES:
            print(f"WARNING: Row {idx}: unknown primary_issue '{issue}'", file=sys.stderr)

    return errors

def generate_jsonl(
    review_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    output_path: Path,
    reviewer: str
) -> None:
    """生成human_review_results.jsonl。"""
    # 合并数据
    merged = review_df.merge(decisions_df, on="sample_id", suffixes=("_review", "_decision"))

    results = []
    review_date = datetime.now().strftime("%Y-%m-%d")

    for _, row in merged.iterrows():
        # 如果human_verdict为空，使用auto_verdict
        human_verdict = row.get("human_verdict", "")
        if pd.isna(human_verdict) or not human_verdict:
            human_verdict = row["auto_verdict_decision"]

        result = {
            "sample_id": row["sample_id"],
            "auto_verdict": row["auto_verdict_decision"],
            "human_verdict": human_verdict,
            "video_quality": row.get("video_quality", "") or None,
            "trajectory_quality": row.get("trajectory_quality", "") or None,
            "primary_issue": row.get("primary_issue", "") or None,
            "notes": row.get("notes", "") or "",
            "reviewer": reviewer,
            "review_date": review_date
        }
        results.append(result)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_disagreement_report(
    review_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    output_path: Path
) -> None:
    """生成HTML disagreement报告。"""
    # 合并数据
    merged = review_df.merge(decisions_df, on="sample_id", suffixes=("_review", "_decision"))

    # 填充空的human_verdict为auto_verdict
    merged["human_verdict_filled"] = merged["human_verdict"].fillna(merged["auto_verdict_decision"])
    merged["human_verdict_filled"] = merged.apply(
        lambda row: row["auto_verdict_decision"] if not row["human_verdict_filled"] else row["human_verdict_filled"],
        axis=1
    )

    # 统计
    total = len(merged)
    reviewed = len(merged[merged["human_verdict"].notna() & (merged["human_verdict"] != "")])
    not_reviewed = total - reviewed

    # Agreement分析
    agreement_data = []
    for auto in ["pass", "fail"]:
        for human in ["pass", "fail"]:
            count = len(merged[(merged["auto_verdict_decision"] == auto) & (merged["human_verdict_filled"] == human)])
            pct = round(100 * count / total, 1) if total > 0 else 0
            agreement_data.append({"auto": auto, "human": human, "count": count, "pct": pct})

    # Disagreement分析
    auto_fail_human_pass = merged[(merged["auto_verdict_decision"] == "fail") & (merged["human_verdict_filled"] == "pass")]
    auto_pass_human_fail = merged[(merged["auto_verdict_decision"] == "pass") & (merged["human_verdict_filled"] == "fail")]

    auto_fail_human_pass_issues = auto_fail_human_pass["primary_issue"].value_counts().items()
    auto_pass_human_fail_issues = auto_pass_human_fail["primary_issue"].value_counts().items()

    # 渲染HTML
    template = Template(HTML_TEMPLATE)
    html = template.render(
        total=total,
        reviewed=reviewed,
        reviewed_pct=round(100 * reviewed / total, 1) if total > 0 else 0,
        not_reviewed=not_reviewed,
        not_reviewed_pct=round(100 * not_reviewed / total, 1) if total > 0 else 0,
        agreement_table=agreement_data,
        auto_fail_human_pass_count=len(auto_fail_human_pass),
        auto_fail_human_pass_issues=auto_fail_human_pass_issues,
        auto_pass_human_fail_count=len(auto_pass_human_fail),
        auto_pass_human_fail_issues=auto_pass_human_fail_issues
    )

    output_path.write_text(html, encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description="Import and validate human review results")
    p.add_argument("--review-list", type=Path, required=True, help="Original review_list.csv")
    p.add_argument("--decisions", type=Path, required=True, help="Filled decisions_filled.csv")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--reviewer", default="batch1", help="Reviewer identifier")
    args = p.parse_args()

    # 加载CSV
    print(f"Loading review list from {args.review_list}...")
    review_df = pd.read_csv(args.review_list)
    print(f"Loading decisions from {args.decisions}...")
    decisions_df = pd.read_csv(args.decisions)

    # 验证
    print("Validating decisions...")
    errors = validate_decisions(decisions_df, review_df)
    if errors:
        print(f"ERROR: Found {len(errors)} validation errors:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)
    print("Validation passed!")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 生成JSONL
    jsonl_path = args.output_dir / "human_review_results.jsonl"
    print(f"Generating {jsonl_path}...")
    generate_jsonl(review_df, decisions_df, jsonl_path, args.reviewer)

    # 生成报告
    report_path = args.output_dir / "disagreement_report.html"
    print(f"Generating {report_path}...")
    generate_disagreement_report(review_df, decisions_df, report_path)

    print(f"Done! Output in {args.output_dir}")

if __name__ == "__main__":
    main()
