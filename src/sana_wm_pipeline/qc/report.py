"""Merge Stage 1/2/3 results and generate manifests + HTML report."""
from __future__ import annotations
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np


def merge_results(
    stage1_jsonl: Path,
    stage2_jsonl: Optional[Path] = None,
    stage3_jsonl: Optional[Path] = None,
) -> list[dict]:
    records: dict[str, dict] = {}
    ordered: list[str] = []
    with open(stage1_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            records[sid] = rec
            ordered.append(sid)

    for jsonl, key in [(stage2_jsonl, "stage2"), (stage3_jsonl, "stage3")]:
        if jsonl and Path(jsonl).exists():
            with open(jsonl) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    sid = rec["sample_id"]
                    if sid in records:
                        records[sid][key] = rec.get(key)

    # Stage 3 verdict upgrade: table6 rejected → fail
    for sid in ordered:
        rec = records[sid]
        s3 = rec.get("stage3") or {}
        if s3.get("table6_accepted") is False and rec.get("verdict") != "fail":
            rec["verdict"] = "fail"
            rec.setdefault("flag_reasons", []).extend(s3.get("reasons", ["table6_rejected"]))

    return [records[sid] for sid in ordered]


def write_manifests(results: list[dict], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    mdir = output_dir / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    pass_lines, reject_lines, review_lines = [], [], []
    for rec in results:
        sid = rec["sample_id"]
        verdict = rec.get("verdict", "pass")
        reasons = rec.get("flag_reasons", [])
        if verdict == "pass":
            pass_lines.append(sid)
        elif verdict == "fail":
            reject_lines.append(sid)
        elif verdict == "flag":
            review_lines.append(f"{sid}\t{'|'.join(reasons)}")
    for fname, lines in [("pass.txt", pass_lines), ("reject.txt", reject_lines), ("human_review.txt", review_lines)]:
        (mdir / fname).write_text(("\n".join(lines) + "\n") if lines else "")


def _svg_hist(values: list[float], w: int = 280, h: int = 60) -> str:
    if not values:
        return f'<svg width="{w}" height="{h}"></svg>'
    arr = np.array(values, dtype=float)
    counts, _ = np.histogram(arr, bins=min(20, max(2, len(set(values)))))
    mx = max(counts) or 1
    bw = w / len(counts)
    bars = "".join(
        f'<rect x="{i*bw:.1f}" y="{h - max(1, int(c/mx*h))}" '
        f'width="{max(1,bw-1):.1f}" height="{max(1,int(c/mx*h))}" fill="#4a9"/>'
        for i, c in enumerate(counts)
    )
    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'


def write_html_report(results: list[dict], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gs = defaultdict(lambda: {"pass": 0, "flag": 0, "fail": 0, "n_jumps": [], "dover": [], "flow": []})
    for rec in results:
        g, v = rec.get("group", "?"), rec.get("verdict", "pass")
        gs[g][v] += 1
        m = rec.get("metrics", {})
        if "n_jumps" in m:
            gs[g]["n_jumps"].append(m["n_jumps"])
        s3 = rec.get("stage3") or {}
        if s3.get("dover") is not None:
            gs[g]["dover"].append(s3["dover"])
        if s3.get("unimatch_flow") is not None:
            gs[g]["flow"].append(s3["unimatch_flow"])

    total = len(results)
    n_pass = sum(1 for r in results if r.get("verdict") == "pass")
    n_flag = sum(1 for r in results if r.get("verdict") == "flag")
    n_fail = sum(1 for r in results if r.get("verdict") == "fail")

    rows = []
    for g, s in sorted(gs.items()):
        gt = s["pass"] + s["flag"] + s["fail"]
        pct = 100 * s["pass"] / gt if gt else 0
        rows.append(
            f"<tr><td>{g}</td><td>{gt}</td><td>{s['pass']} ({pct:.1f}%)</td>"
            f"<td>{s['flag']}</td><td>{s['fail']}</td>"
            f"<td>{_svg_hist(s['n_jumps'])}</td>"
            f"<td>{_svg_hist(s['dover'])}</td>"
            f"<td>{_svg_hist(s['flow'])}</td></tr>"
        )

    flag_rows = "".join(
        f"<tr><td>{r['sample_id']}</td><td>{r.get('group','')}</td>"
        f"<td>{'<br>'.join(r.get('flag_reasons', []))}</td></tr>"
        for r in itertools.islice((r for r in results if r.get("verdict") == "flag"), 200)
    )

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>SANA-WM QC Report</title>
<style>body{{font-family:monospace;margin:20px;background:#1a1a2e;color:#eee}}
h1,h2{{color:#a8daff}}table{{border-collapse:collapse;width:100%;margin-bottom:20px}}
th,td{{border:1px solid #444;padding:6px 10px}}th{{background:#2a2a4a}}
.s{{display:inline-block;margin:10px 20px;font-size:1.3em}}
.p{{color:#4a9}}.fl{{color:#fa3}}.fa{{color:#f44}}</style></head><body>
<h1>SANA-WM Pipeline QC Report</h1>
<div>
  <span class="s">Total <b>{total}</b></span>
  <span class="s p">Pass <b>{n_pass}</b> ({100*n_pass/max(total,1):.1f}%)</span>
  <span class="s fl">Flag <b>{n_flag}</b> ({100*n_flag/max(total,1):.1f}%)</span>
  <span class="s fa">Fail <b>{n_fail}</b> ({100*n_fail/max(total,1):.1f}%)</span>
</div>
<h2>Per-Group Summary</h2>
<table><tr><th>Group</th><th>Total</th><th>Pass</th><th>Flag</th><th>Fail</th>
<th>Jump Dist</th><th>DOVER Dist</th><th>UniMatch Flow Dist</th></tr>
{"".join(rows)}</table>
<h2>Human Review Queue (flag samples)</h2>
<table><tr><th>sample_id</th><th>group</th><th>flag_reasons</th></tr>
{flag_rows}</table></body></html>"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def run_report(
    stage1_jsonl: Path,
    stage2_jsonl: Optional[Path],
    stage3_jsonl: Optional[Path],
    output_dir: Path,
) -> None:
    results = merge_results(stage1_jsonl, stage2_jsonl, stage3_jsonl)
    write_manifests(results, output_dir)
    write_html_report(results, output_dir)
