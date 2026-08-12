# Human Review System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现人工审查系统的3个Python脚本，支持在Stage 1+2和Stage 3之间进行人工质量评估

**Architecture:** 三个独立脚本形成完整工作流：export_for_review.py负责采样和视频提取，import_review_results.py负责验证和分析，apply_human_review.py负责合并决策到最终清单

**Tech Stack:** Python 3.9+, pandas, tarfile, av (PyAV), jinja2 (HTML报告)

## Global Constraints

- Python 3.9+
- 复用现有的tar提取逻辑（stage1_fast.py的_extract_samples_from_tar）
- 复用现有的group_config.py中的阈值配置
- 所有脚本放在 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/`
- 测试文件放在 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/tests/`
- CSV格式：UTF-8编码，逗号分隔
- JSONL格式：每行一个JSON对象
- 遵循DRY、YAGNI、TDD原则

---

### Task 1: Export Script - 采样和视频提取

**Files:**
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/export_for_review.py`
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/tests/test_export_for_review.py`
- Read: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage1_fast.py`
- Read: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/group_config.py`

**Interfaces:**
- Consumes: stage1_results.jsonl, stage2_results.jsonl（可选）, tar文件
- Produces: 
  - `review_list.csv`: 包含sample_id, group, tar_path, auto_verdict, flag_reasons, n_jumps, caption_len, black_frame_ratio, scene_cuts, caption_text, video_path
  - `videos/`: 提取的mp4文件
  - `decisions_template.csv`: 包含sample_id, auto_verdict, human_verdict, video_quality, trajectory_quality, primary_issue, notes
  - `sampling_report.txt`: 采样统计信息

- [ ] **Step 1: 写加载和合并Stage 1+2结果的测试**

```python
# tests/test_export_for_review.py
import json
from pathlib import Path
import pytest
from scripts.export_for_review import load_and_merge_results

def test_load_stage1_only(tmp_path):
    s1 = tmp_path / "stage1.jsonl"
    s1.write_text(json.dumps({"sample_id": "A", "verdict": "pass", "n_jumps": 1}) + "
")
    
    results = load_and_merge_results([s1], [])
    assert len(results) == 1
    assert results[0]["sample_id"] == "A"
    assert results[0]["verdict"] == "pass"

def test_merge_stage1_and_stage2(tmp_path):
    s1 = tmp_path / "stage1.jsonl"
    s2 = tmp_path / "stage2.jsonl"
    s1.write_text(json.dumps({"sample_id": "A", "verdict": "pass", "n_jumps": 1}) + "
")
    s2.write_text(json.dumps({"sample_id": "A", "black_frame_ratio": 0.02, "scene_cuts": 1}) + "
")
    
    results = load_and_merge_results([s1], [s2])
    assert len(results) == 1
    assert results[0]["sample_id"] == "A"
    assert results[0]["black_frame_ratio"] == 0.02
    assert results[0]["scene_cuts"] == 1
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_export_for_review.py::test_load_stage1_only -v
```

预期：FAIL with "ModuleNotFoundError: No module named 'scripts.export_for_review'"

- [ ] **Step 3: 实现load_and_merge_results函数**

```python
# scripts/export_for_review.py
from __future__ import annotations
import json, glob
from pathlib import Path
from typing import Any

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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_export_for_review.py::test_load_stage1_only -v
python -m pytest tests/test_export_for_review.py::test_merge_stage1_and_stage2 -v
```

预期：两个测试都PASS

- [ ] **Step 5: 写平衡采样策略的测试**

```python
# tests/test_export_for_review.py（追加）
from scripts.export_for_review import balanced_sampling

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
```

- [ ] **Step 6: 运行测试验证失败**

```bash
python -m pytest tests/test_export_for_review.py::test_balanced_sampling_fail_near_threshold -v
```

预期：FAIL with "function not defined"

- [ ] **Step 7: 实现balanced_sampling函数**

```python
# scripts/export_for_review.py（追加）
import random

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
```

- [ ] **Step 8: 运行测试验证通过**

```bash
python -m pytest tests/test_export_for_review.py::test_balanced_sampling_fail_near_threshold -v
python -m pytest tests/test_export_for_review.py::test_balanced_sampling_multiple_reasons -v
```

预期：两个测试都PASS

- [ ] **Step 9: 写视频提取的测试**

```python
# tests/test_export_for_review.py（追加）
import tarfile
from scripts.export_for_review import extract_videos

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
```

- [ ] **Step 10: 运行测试验证失败**

```bash
python -m pytest tests/test_export_for_review.py::test_extract_videos -v
```

预期：FAIL with "function not defined"

- [ ] **Step 11: 实现extract_videos函数**

```python
# scripts/export_for_review.py（追加）
import sys
import tarfile

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
```

- [ ] **Step 12: 运行测试验证通过**

```bash
python -m pytest tests/test_export_for_review.py::test_extract_videos -v
```

预期：PASS

- [ ] **Step 13: 写CSV生成的测试**

```python
# tests/test_export_for_review.py（追加）
import pandas as pd
from scripts.export_for_review import generate_review_list, generate_template

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
    
    generate_review_list(samples, output_path, tmp_path / "videos")
    
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
```

- [ ] **Step 14: 运行测试验证失败**

```bash
python -m pytest tests/test_export_for_review.py::test_generate_review_list -v
```

预期：FAIL with "function not defined"

- [ ] **Step 15: 实现CSV生成函数**

```python
# scripts/export_for_review.py（追加）
import pandas as pd

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
```

- [ ] **Step 16: 运行测试验证通过**

```bash
python -m pytest tests/test_export_for_review.py::test_generate_review_list -v
python -m pytest tests/test_export_for_review.py::test_generate_template -v
```

预期：两个测试都PASS

- [ ] **Step 17: 实现CLI主函数**

```python
# scripts/export_for_review.py（追加）
import argparse

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
        f.write(f"Sampling Report
")
        f.write(f"===============
")
        f.write(f"Total samples loaded: {len(results)}
")
        f.write(f"Total samples selected: {len(sampled)}
")
        f.write(f"Pass samples: {sum(1 for s in sampled if s['verdict'] == 'pass')}
")
        f.write(f"Fail samples: {sum(1 for s in sampled if s['verdict'] == 'fail')}
")
    
    print(f"Done! Output in {args.output_dir}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 18: 手动测试脚本**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python scripts/export_for_review.py --help
```

预期：显示帮助信息

- [ ] **Step 19: 提交Task 1**

```bash
git add scripts/export_for_review.py tests/test_export_for_review.py
git commit -m "feat: add export_for_review.py script with balanced sampling"
```
```

我将在下一条消息继续提供Task 2和Task 3的内容...

# 第2部分：Task 2 - Import Script

```markdown
---

### Task 2: Import Script - 验证和分析

**Files:**
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/import_review_results.py`
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/tests/test_import_review_results.py`

**Interfaces:**
- Consumes: `review_list.csv`, `decisions_filled.csv`
- Produces: 
  - `human_review_results.jsonl`: 标准化的人工审查结果，每行包含{"sample_id", "auto_verdict", "human_verdict", "video_quality", "trajectory_quality", "primary_issue", "notes", "reviewer", "review_date"}
  - `disagreement_report.html`: 自动vs人工决策分析报告

- [ ] **Step 1: 写验证函数的测试**

```python
# tests/test_import_review_results.py
import pandas as pd
import pytest
from pathlib import Path
from scripts.import_review_results import validate_decisions

def test_validate_decisions_valid(tmp_path):
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
    assert len(errors) == 0  # 空verdict是允许的，使用auto_verdict

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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_import_review_results.py::test_validate_decisions_valid -v
```

预期：FAIL with "ModuleNotFoundError"

- [ ] **Step 3: 实现validate_decisions函数**

```python
# scripts/import_review_results.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Any
import sys

VALID_VERDICTS = {"pass", "fail"}
VALID_ISSUES = {
    "trajectory_minor_jump", "trajectory_major_jump", "video_blurry",
    "video_artifacts", "caption_mismatch", "caption_too_vague",
    "black_frames", "scene_cut_abrupt", "multiple_issues", "no_issue", "other"
}

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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/test_import_review_results.py::test_validate_decisions_valid -v
python -m pytest tests/test_import_review_results.py::test_validate_decisions_invalid_verdict -v
python -m pytest tests/test_import_review_results.py::test_validate_decisions_unknown_sample -v
```

预期：所有测试都PASS

- [ ] **Step 5: 写生成JSONL的测试**

```python
# tests/test_import_review_results.py（追加）
import json
from scripts.import_review_results import generate_jsonl

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
    assert lines[1]["human_verdict"] == "pass"  # 从auto_verdict继承
```

- [ ] **Step 6: 运行测试验证失败**

```bash
python -m pytest tests/test_import_review_results.py::test_generate_jsonl -v
```

预期：FAIL with "function not defined"

- [ ] **Step 7: 实现generate_jsonl函数**

```python
# scripts/import_review_results.py（追加）
import json
from datetime import datetime

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
            f.write(json.dumps(r, ensure_ascii=False) + "
")
```

- [ ] **Step 8: 运行测试验证通过**

```bash
python -m pytest tests/test_import_review_results.py::test_generate_jsonl -v
```

预期：PASS

- [ ] **Step 9: 写生成disagreement report的测试**

```python
# tests/test_import_review_results.py（追加）
from scripts.import_review_results import generate_disagreement_report

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
```

- [ ] **Step 10: 运行测试验证失败**

```bash
python -m pytest tests/test_import_review_results.py::test_generate_disagreement_report -v
```

预期：FAIL with "function not defined"

- [ ] **Step 11: 实现generate_disagreement_report函数**

```python
# scripts/import_review_results.py（追加）

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

def generate_disagreement_report(
    review_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    output_path: Path
) -> None:
    """生成HTML disagreement报告。"""
    from jinja2 import Template
    
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
```

- [ ] **Step 12: 运行测试验证通过**

```bash
python -m pytest tests/test_import_review_results.py::test_generate_disagreement_report -v
```

预期：PASS

- [ ] **Step 13: 实现CLI主函数**

```python
# scripts/import_review_results.py（追加）
import argparse

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
```

- [ ] **Step 14: 手动测试脚本**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python scripts/import_review_results.py --help
```

预期：显示帮助信息

- [ ] **Step 15: 提交Task 2**

```bash
git add scripts/import_review_results.py tests/test_import_review_results.py
git commit -m "feat: add import_review_results.py with validation and reporting"
```
```

继续下一部分...

# 第3部分：Task 3 - Apply Script

```markdown
---

### Task 3: Apply Script - 合并决策到最终清单

**Files:**
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/apply_human_review.py`
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/tests/test_apply_human_review.py`

**Interfaces:**
- Consumes: 
  - stage1_results.jsonl（原始Stage 1结果）
  - human_review_results.jsonl（来自import脚本）
- Produces: 
  - `stage1_results_merged.jsonl`: 合并了人工决策的Stage 1结果
  - `manifests/pass.txt`: 最终通过清单（用于Stage 3输入）
  - `manifests/fail.txt`: 最终失败清单
  - `manifests/human_reviewed.txt`: 人工审查过的样本清单
  - `summary_report.html`: 最终统计报告

- [ ] **Step 1: 写合并决策的测试**

```python
# tests/test_apply_human_review.py
import json
from pathlib import Path
import pytest
from scripts.apply_human_review import merge_decisions

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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_apply_human_review.py::test_merge_decisions_override -v
```

预期：FAIL with "ModuleNotFoundError"

- [ ] **Step 3: 实现merge_decisions函数**

```python
# scripts/apply_human_review.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/test_apply_human_review.py::test_merge_decisions_override -v
python -m pytest tests/test_apply_human_review.py::test_merge_decisions_no_human_review -v
```

预期：两个测试都PASS

- [ ] **Step 5: 写生成manifests的测试**

```python
# tests/test_apply_human_review.py（追加）
from scripts.apply_human_review import generate_manifests

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
    pass_ids = pass_file.read_text().strip().split("
")
    assert set(pass_ids) == {"A", "C"}
    
    # 检查fail.txt
    fail_file = manifests_dir / "fail.txt"
    assert fail_file.exists()
    fail_ids = fail_file.read_text().strip().split("
")
    assert fail_ids == ["B"]
    
    # 检查human_reviewed.txt
    human_file = manifests_dir / "human_reviewed.txt"
    assert human_file.exists()
    human_ids = human_file.read_text().strip().split("
")
    assert human_ids == ["A"]
```

- [ ] **Step 6: 运行测试验证失败**

```bash
python -m pytest tests/test_apply_human_review.py::test_generate_manifests -v
```

预期：FAIL with "function not defined"

- [ ] **Step 7: 实现generate_manifests函数**

```python
# scripts/apply_human_review.py（追加）

def generate_manifests(
    merged_results: list[dict[str, Any]],
    manifests_dir: Path
) -> None:
    """生成pass/fail/human_reviewed清单文件。"""
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # pass.txt
    pass_samples = [r["sample_id"] for r in merged_results if r["verdict"] == "pass"]
    (manifests_dir / "pass.txt").write_text("
".join(pass_samples) + "
")
    
    # fail.txt
    fail_samples = [r["sample_id"] for r in merged_results if r["verdict"] == "fail"]
    (manifests_dir / "fail.txt").write_text("
".join(fail_samples) + "
")
    
    # human_reviewed.txt
    human_reviewed = [r["sample_id"] for r in merged_results if r.get("human_reviewed", False)]
    (manifests_dir / "human_reviewed.txt").write_text("
".join(human_reviewed) + "
")
```

- [ ] **Step 8: 运行测试验证通过**

```bash
python -m pytest tests/test_apply_human_review.py::test_generate_manifests -v
```

预期：PASS

- [ ] **Step 9: 写生成summary report的测试**

```python
# tests/test_apply_human_review.py（追加）
from scripts.apply_human_review import generate_summary_report

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
```

- [ ] **Step 10: 运行测试验证失败**

```bash
python -m pytest tests/test_apply_human_review.py::test_generate_summary_report -v
```

预期：FAIL with "function not defined"

- [ ] **Step 11: 实现generate_summary_report函数**

```python
# scripts/apply_human_review.py（追加）

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

def generate_summary_report(
    merged_results: list[dict[str, Any]],
    output_path: Path
) -> None:
    """生成最终汇总HTML报告。"""
    from jinja2 import Template
    
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
```

- [ ] **Step 12: 运行测试验证通过**

```bash
python -m pytest tests/test_apply_human_review.py::test_generate_summary_report -v
```

预期：PASS

- [ ] **Step 13: 实现CLI主函数**

```python
# scripts/apply_human_review.py（追加）
import argparse
import glob

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
            f.write(json.dumps(record, ensure_ascii=False) + "
")
    
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
    print(f"
Done! Output in {args.output_dir}")
    print(f"Final pass count: {pass_count}")
    print(f"Ready for Stage 3: manifests/pass.txt")

if __name__ == "__main__":
    main()
```

- [ ] **Step 14: 手动测试脚本**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python scripts/apply_human_review.py --help
```

预期：显示帮助信息

- [ ] **Step 15: 运行所有测试**

```bash
python -m pytest tests/test_apply_human_review.py -v
```

预期：所有测试都PASS

- [ ] **Step 16: 提交Task 3**

```bash
git add scripts/apply_human_review.py tests/test_apply_human_review.py
git commit -m "feat: add apply_human_review.py to merge decisions into final manifests"
```
```

继续下一部分...

# 第4部分：Task 4 - 人工审查操作手册

```markdown
---

### Task 4: 人工审查操作手册

**Files:**
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/human_review_manual.md`

**Interfaces:**
- Consumes: 无（文档任务）
- Produces: 完整的审查人员操作手册

- [ ] **Step 1: 创建操作手册**

```markdown
# 人工审查操作手册

## 概述

本手册面向人工审查人员，指导如何审查SANA-WM QC系统Stage 1+2筛选出的视频样本。

---

## 审查目标

在Stage 1+2自动筛选之后、Stage 3 GPU评估之前，人工审查1,000-3,000个边界样本，判断是否应该通过质检进入下一阶段。

**审查重点：**
- 自动筛选标记为失败但接近阈值的样本（可能是误判）
- 自动筛选标记为通过的随机样本（验证漏检）
- 存在多个问题标记的复杂样本

---

## 环境准备

### 所需工具

1. **VLC Media Player** - 视频播放
   - 下载：https://www.videolan.org/
   - 支持快捷键：空格（播放/暂停）、方向键（快进/快退）

2. **Excel / LibreOffice Calc / Google Sheets** - 填写审查结果
   - 推荐Excel或LibreOffice以保证CSV兼容性

3. **文本编辑器（可选）** - 查看review_list.csv的详细指标

### 获取审查包

技术人员会提供一个审查包目录，包含：
```
human_review_batch1/
├── review_list.csv           # 完整样本信息（只读参考）
├── decisions_template.csv    # 待填写的审查表格
├── videos/                   # 提取的视频文件
│   ├── sample_001.mp4
│   ├── sample_002.mp4
│   └── ...
└── sampling_report.txt       # 采样统计信息
```

---

## 审查流程

### Step 1: 打开文件

1. 用Excel打开 `decisions_template.csv`
2. 用文本编辑器或Excel打开 `review_list.csv`（作为参考）
3. 在VLC中打开 `videos/` 目录

### Step 2: 逐个审查样本

对于 `decisions_template.csv` 中的每一行：

#### 2.1 找到对应视频

- 根据 `sample_id` 列（例如 "DL3DV_001"）
- 在 `videos/` 目录中找到对应的 `.mp4` 文件
- 用VLC播放

#### 2.2 查看自动判断

- `auto_verdict` 列：自动系统的判断（pass/fail）
- 在 `review_list.csv` 中查看：
  - `flag_reasons`: 失败原因（如 "trajectory_jump|black_frames"）
  - `n_jumps`: 轨迹跳变次数
  - `caption_text`: 字幕文本
  - 其他指标：`black_frame_ratio`, `scene_cuts` 等

#### 2.3 观看视频（10-20秒）

**关注点：**
- **轨迹质量**：相机运动是否平滑？是否有突然跳跃？
- **视频质量**：画面是否清晰？是否有模糊、卡顿、伪影？
- **内容连贯性**：是否有突兀的场景切换？是否有黑屏？
- **字幕匹配**：字幕描述是否与视频内容一致？

#### 2.4 填写审查结果

在 `decisions_template.csv` 中填写以下列：

**必填列：**

1. **human_verdict**（必填）
   - `pass`: 该样本应该通过质检
   - `fail`: 该样本应该被淘汰
   - 留空：使用自动判断（不确定时可留空）

2. **primary_issue**（必填）
   - 从以下选项中选择主要问题：
   
   ```
   trajectory_minor_jump    - 小的轨迹跳变（可接受范围内）
   trajectory_major_jump    - 大的轨迹跳变（明显不连续）
   video_blurry             - 视频模糊
   video_artifacts          - 视频有伪影/故障
   caption_mismatch         - 字幕与内容不匹配
   caption_too_vague        - 字幕过于笼统
   black_frames             - 包含黑屏帧
   scene_cut_abrupt         - 突兀的场景切换
   multiple_issues          - 多个问题同时存在
   no_issue                 - 没有发现问题
   other                    - 其他（请在notes中说明）
   ```

**可选列：**

3. **video_quality**（可选）
   - `good`: 清晰、流畅、无伪影
   - `acceptable`: 有小问题但可用
   - `poor`: 模糊、卡顿、严重伪影

4. **trajectory_quality**（可选）
   - `good`: 平滑、合理
   - `acceptable`: 有小跳变但可接受
   - `poor`: 大跳变、不连续

5. **notes**（可选）
   - 额外说明，特别是选择 `other` 或不寻常情况时

### Step 3: 保存文件

- 定期保存 `decisions_template.csv`（建议每审查50个样本保存一次）
- 完成后另存为 `decisions_filled.csv`

---

## 判断标准

### 何时选择 PASS

即使自动系统标记为 `fail`，以下情况应该选择 `pass`：

1. **轨迹跳变很小**
   - 自动系统可能对阈值过于严格
   - 小的相机位置调整（< 0.5米）是可接受的
   - 判断：如果人眼观看感觉平滑，即使有小跳变也可通过

2. **字幕略短但足够描述**
   - 自动系统设置最小长度40字符
   - 如果字幕虽短但描述清晰（如"Person walking in a room"），可通过

3. **黑屏很少**
   - 1-2帧黑屏通常是编码问题，不影响整体质量

### 何时选择 FAIL

即使自动系统标记为 `pass`，以下情况应该选择 `fail`：

1. **视频模糊严重**
   - 自动系统目前无法检测模糊
   - 如果画面大部分时间模糊到无法识别物体，应淘汰

2. **字幕与内容明显不符**
   - 例如：字幕说"indoor bedroom"但视频是"outdoor street"

3. **场景内容不适合训练**
   - 完全静止的画面（无相机运动）
   - 纯文字屏幕/GUI界面
   - 严重的畸变或失真

### 何时留空（使用自动判断）

- 不确定时可以跳过（留空 `human_verdict`）
- 系统会自动使用 `auto_verdict` 的值
- 建议：如果观看15秒后仍无法判断，留空并继续下一个

---

## 审查技巧

### 1. 批量模式

**推荐流程：**
- 在VLC中将所有视频加入播放列表
- 逐个播放（VLC快捷键：N = 下一个，P = 上一个）
- 在Excel中同步填写

### 2. 加速播放

对于长视频（>30秒）：
- VLC快捷键：`]` 加速，`[` 减速
- 建议1.5-2倍速观看，重点关注轨迹和场景切换

### 3. 分批审查

建议分时段审查：
- 每次审查100-200个样本（约1-2小时）
- 休息10-15分钟后继续
- 避免疲劳导致判断标准飘移

### 4. 双人交叉验证（可选）

对于不确定的样本：
- 标记在notes列
- 两位审查人员独立判断
- 讨论后达成一致

---

## 常见问题

### Q1: 自动系统标记了3次跳变，但我只看到1次？

A: 可能的原因：
- 自动系统对小位移变化敏感，人眼可能忽略
- 如果你认为整体平滑，可以标记为 `pass` + `trajectory_minor_jump`

### Q2: 视频无法播放或损坏？

A: 
- 在 `review_list.csv` 中检查 `video_path` 是否为 "MISSING"
- 如果是，基于其他指标判断，或留空使用自动判断
- 记录在notes中："video corrupted, judged by metrics"

### Q3: 我对某些样本完全无法判断？

A: 
- 留空 `human_verdict`，系统会使用 `auto_verdict`
- 不强制要求100%完成率，90%以上即可

### Q4: 发现新的问题类型，不在枚举列表中？

A: 
- `primary_issue` 选择 `other`
- 在 `notes` 中详细描述问题
- 通知技术人员，可能需要更新自动检测逻辑

---

## 质量保证

### 自查清单

完成后检查：
- [ ] 至少90%的样本填写了 `human_verdict`
- [ ] 所有填写了 `human_verdict` 的行都填写了 `primary_issue`
- [ ] `human_verdict` 只包含 `pass` 或 `fail`（或留空）
- [ ] 文件另存为 `decisions_filled.csv`
- [ ] CSV文件编码为UTF-8（Excel默认可能是GBK，需检查）

### 提交前验证

技术人员会运行验证脚本：
```bash
python scripts/import_review_results.py \
  --review-list human_review_batch1/review_list.csv \
  --decisions human_review_batch1/decisions_filled.csv \
  --output-dir human_review_batch1/analysis
```

如果有错误，会返回具体行号和问题。

---

## 工作量估算

**单个样本：** 1-2分钟
- 10秒观看视频
- 5秒参考指标
- 30秒填写表格

**1000个样本：** 约20-30小时
- 建议2人并行，每人500个
- 分5-6个工作时段完成
- 总时长：2个工作日

---

## 示例

### 示例1: 自动fail → 人工pass

**review_list.csv:**
```
sample_id: DL3DV_001
auto_verdict: fail
flag_reasons: trajectory_jump
n_jumps: 3
caption_text: Person walking through a living room
```

**观察：**
- 视频播放平滑，只有2个很小的位移
- 画面清晰，内容合理

**填写 decisions_filled.csv:**
```
sample_id: DL3DV_001
auto_verdict: fail
human_verdict: pass
video_quality: good
trajectory_quality: acceptable
primary_issue: trajectory_minor_jump
notes: Small jumps but overall smooth trajectory
```

### 示例2: 自动pass → 人工fail

**review_list.csv:**
```
sample_id: RealEstate_050
auto_verdict: pass
n_jumps: 1
caption_text: Camera moving through modern apartment
```

**观察：**
- 视频严重模糊，无法识别房间细节
- 自动系统未检测到模糊

**填写 decisions_filled.csv:**
```
sample_id: RealEstate_050
auto_verdict: pass
human_verdict: fail
video_quality: poor
trajectory_quality: good
primary_issue: video_blurry
notes: Severely blurred, unusable for training
```

### 示例3: 不确定 → 留空

**review_list.csv:**
```
sample_id: Sekai_010
auto_verdict: fail
flag_reasons: multiple_issues
n_jumps: 5
scene_cuts: 2
```

**观察：**
- 有一些问题，但难以判断严重程度
- 不确定是否应该通过

**填写 decisions_filled.csv:**
```
sample_id: Sekai_010
auto_verdict: fail
human_verdict: 
video_quality: 
trajectory_quality: 
primary_issue: 
notes: Uncertain, using auto verdict
```

---

## 联系方式

**技术支持：**
- 审查过程中遇到问题，联系：[技术负责人姓名/邮箱]
- 文件格式问题、工具使用问题

**数据质量讨论：**
- 判断标准不明确
- 发现系统性问题
- 建议改进自动检测逻辑

---

## 附录：Primary Issue 完整列表

| 代码 | 中文说明 | 使用场景 |
|------|---------|---------|
| trajectory_minor_jump | 轻微轨迹跳变 | 有小的位移但整体平滑 |
| trajectory_major_jump | 严重轨迹跳变 | 明显的不连续、传送 |
| video_blurry | 视频模糊 | 画面不清晰，无法识别细节 |
| video_artifacts | 视频伪影 | 编码错误、花屏、色块 |
| caption_mismatch | 字幕不匹配 | 描述与视频内容不符 |
| caption_too_vague | 字幕过于笼统 | 如"A video"这种无效描述 |
| black_frames | 黑屏帧 | 包含大量黑色或纯色帧 |
| scene_cut_abrupt | 突兀场景切换 | 不同场景拼接在一起 |
| multiple_issues | 多个问题 | 同时存在2个以上明显问题 |
| no_issue | 无问题 | 审查后认为完全正常 |
| other | 其他 | 以上都不适用，需在notes说明 |

---

## 变更历史

| 日期 | 版本 | 变更说明 |
|------|------|---------|
| 2026-07-07 | 1.0 | 初始版本 |
```

- [ ] **Step 2: 验证手册完整性**

手动检查手册是否包含：
- [ ] 审查目标清晰
- [ ] 工具准备说明完整
- [ ] 逐步操作流程详细
- [ ] 判断标准明确
- [ ] 常见问题有解答
- [ ] 示例充分
- [ ] Primary issue枚举完整

- [ ] **Step 3: 创建快速参考卡片**

```markdown
# 人工审查快速参考卡

## 必填字段

1. **human_verdict**: `pass` / `fail` / 留空
2. **primary_issue**: 从11个选项中选择

## Primary Issue 速查

- `trajectory_minor_jump` - 小跳变
- `trajectory_major_jump` - 大跳变  
- `video_blurry` - 模糊
- `video_artifacts` - 伪影
- `caption_mismatch` - 字幕不符
- `caption_too_vague` - 字幕过泛
- `black_frames` - 黑屏
- `scene_cut_abrupt` - 场景切换
- `multiple_issues` - 多问题
- `no_issue` - 无问题
- `other` - 其他（需notes说明）

## 判断原则

**Pass**: 小瑕疵、可训练、人眼看平滑
**Fail**: 严重模糊、字幕不符、内容不适合
**留空**: 不确定时使用自动判断

## VLC快捷键

- 空格：播放/暂停
- N：下一个
- ]：加速
- [：减速
```

保存为：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/human_review_quick_reference.md`

- [ ] **Step 4: 提交文档**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add docs/human_review_manual.md docs/human_review_quick_reference.md
git commit -m "docs: add human review manual and quick reference"
```

---

## 执行计划完成

所有4个任务已完成：
- ✅ Task 1: export_for_review.py（采样和视频提取）
- ✅ Task 2: import_review_results.py（验证和分析）
- ✅ Task 3: apply_human_review.py（合并决策）
- ✅ Task 4: 人工审查操作手册

**最终交付物：**
1. 3个Python脚本（带完整测试）
2. 人工审查操作手册
3. 快速参考卡片

**下一步：**
使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 执行此计划。
```

---

**完整计划内容结束。**

**文件保存位置：**
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/superpowers/plans/2026-07-07-human-review-system.md
```

请你手动创建这个文件，将以上所有4部分的内容合并保存。创建完成后告诉我，我将进行错误复盘并继续执行计划。