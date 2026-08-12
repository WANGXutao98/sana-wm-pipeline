# Human Review System for Stage 1+2 QC Results

**Date:** 2026-07-02  
**Status:** Design Approved  
**Owner:** David Wang

---

## Overview

A minimal human review system to enable manual quality assessment between Stage 1+2 and Stage 3 of the SANA QC pipeline. The system exports QC results with videos for offline review, collects structured human feedback, and merges decisions back into the pipeline.

---

## Goals

### Primary Goal
Enable human reviewers to make pass/fail decisions on 1,000-3,000 borderline samples after Stage 1+2 automated screening, before investing GPU resources in Stage 3.

### Secondary Goals
- Collect structured feedback to identify automation threshold issues
- Discover blind spots in Stage 1+2 detection logic
- Provide quantifiable data for future rule improvements

---

## Non-Goals

- Real-time review interface (Web UI or interactive CLI)
- Iterative rule refinement (C-style approach with multiple rounds)
- Integration with external labeling platforms
- Automated retraining of ML models

---

## Context

### Current Pipeline
```
Data (140K samples, 7 groups)
    ↓
Stage 1: Fast checks (trajectory, caption, files)
Stage 2: Deep checks (black frames, scene cuts)
    ↓
Results: pass (~7%) / fail (~93%)
    ↓
??? Human review needed here ???
    ↓
Stage 3: GPU evaluation (DOVER, UniMatch, Qwen VLM)
    → 40-50 GPU hours for full dataset
```

### Problem
- Stage 1+2 uses hard thresholds that may be too strict or too loose
- Some borderline samples need human judgment
- Running Stage 3 on all "pass" samples is expensive
- Need to validate automation quality before full Stage 3 run

---

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1+2 Results                         │
│              (stage1_results.jsonl + stage2_results.jsonl)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  Export Script      │
            │  - Sampling         │
            │  - Video extraction │
            └─────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   Human Review Bundle      │
         │   - review_list.csv        │
         │   - videos/                │
         │   - decisions_template.csv │
         └────────────┬───────────────┘
                      │
                      ▼
              [Human Reviewer]
              (VLC + Excel)
              (2 work days)
                      │
                      ▼
         ┌────────────────────────────┐
         │   decisions_filled.csv     │
         └────────────┬───────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  Import Script      │
            │  - Validation       │
            │  - Analysis         │
            └─────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │ human_review_results.jsonl │
         │ disagreement_report.html   │
         └────────────┬───────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  Apply Script       │
            │  - Merge decisions  │
            └─────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  Final Manifests           │
         │  - manifests/pass.txt      │
         │  - manifests/fail.txt      │
         │  → Input for Stage 3       │
         └────────────────────────────┘
```

---

## Component Details

### 1. Sampling Strategy

**Goal:** Select 1,000-3,000 most valuable samples for human review

**Strategy: Balanced Sampling**

```python
sampling_config = {
    "total_samples": 1000,
    "per_group_ratio": True,  # Proportional to group size
    "priority_buckets": [
        ("fail_near_threshold", 400),   # Borderline samples
        ("multiple_reasons", 200),       # Complex cases
        ("pass_random", 300),            # Quality validation
        ("fail_random", 100),            # Sanity check
    ]
}
```

**Bucket Definitions:**

- **fail_near_threshold**: `verdict=fail` but metrics are close to pass thresholds
  - Example: `n_jumps=3` when threshold is `≤2`
- **multiple_reasons**: Samples with 2+ flag reasons
  - Example: `trajectory_jump + black_frame_ratio`
- **pass_random**: Random samples from `verdict=pass` to validate false negatives
- **fail_random**: Random samples from `verdict=fail` to validate obvious failures

---

### 2. Data Formats

#### review_list.csv

Contains all information reviewers need to make decisions.

**Columns:**
```csv
sample_id,group,tar_path,auto_verdict,flag_reasons,n_jumps,caption_len,black_frame_ratio,scene_cuts,caption_text,video_path
```

**Example:**
```csv
sample_id,group,tar_path,auto_verdict,flag_reasons,n_jumps,caption_len,black_frame_ratio,scene_cuts,caption_text,video_path
DL3DV_001,DL3DV,/data/shard-001.tar,fail,"trajectory_jump",3,45,0.01,0,"Person walking in room",videos/DL3DV_001.mp4
RealEstate_050,RealEstate10K,/data/shard-050.tar,pass,"",1,52,0.00,1,"Camera moving through apartment",videos/RealEstate_050.mp4
```

**Field Descriptions:**
- `sample_id`: Unique identifier
- `group`: Dataset group (DL3DV, RealEstate10K, etc.)
- `tar_path`: Original tar file (for traceability)
- `auto_verdict`: Automated decision (pass/fail)
- `flag_reasons`: Pipe-separated failure reasons (e.g., "trajectory_jump|black_frames")
- `n_jumps, caption_len, ...`: Key metric values
- `caption_text`: Full caption text
- `video_path`: Relative path to extracted video file

---

#### decisions_template.csv

Template for human reviewers to fill out.

**Columns:**
```csv
sample_id,auto_verdict,human_verdict,video_quality,trajectory_quality,primary_issue,notes
```

**Filled Example:**
```csv
sample_id,auto_verdict,human_verdict,video_quality,trajectory_quality,primary_issue,notes
DL3DV_001,fail,pass,good,acceptable,trajectory_minor,Small jump but overall smooth
RealEstate_050,pass,fail,poor,good,video_blurry,Severely blurred video
Sekai_010,fail,fail,good,poor,trajectory_major,Multiple large jumps
OmniWorld_005,pass,fail,good,good,caption_mismatch,Caption says indoor but video is outdoor
```

**Field Definitions:**

**Required Fields:**
- `sample_id`: Must match review_list.csv
- `auto_verdict`: Copied from review_list (for reference)
- `human_verdict`: **[pass | fail]** - Human decision
- `primary_issue`: **[enum]** - Main problem category (see below)

**Optional Fields:**
- `video_quality`: **[good | acceptable | poor | N/A]**
  - good: Clear, smooth, no artifacts
  - acceptable: Minor issues but usable
  - poor: Blurry, stuttering, severe artifacts
- `trajectory_quality`: **[good | acceptable | poor | N/A]**
  - good: Smooth, reasonable
  - acceptable: Small jumps but acceptable
  - poor: Large jumps, discontinuous
- `notes`: Free text for additional context

**Primary Issue Enum:**
```
trajectory_minor_jump    - Small trajectory jump
trajectory_major_jump    - Large trajectory jump
video_blurry             - Video is blurry
video_artifacts          - Video has artifacts/glitches
caption_mismatch         - Caption doesn't match content
caption_too_vague        - Caption is too generic
black_frames             - Contains black frames
scene_cut_abrupt         - Abrupt scene transitions
multiple_issues          - Multiple problems
no_issue                 - No problem found
other                    - Other (explain in notes)
```

---

#### human_review_results.jsonl

Standardized output after import.

**Format:**
```jsonl
{"sample_id": "DL3DV_001", "auto_verdict": "fail", "human_verdict": "pass", "video_quality": "good", "trajectory_quality": "acceptable", "primary_issue": "trajectory_minor", "notes": "Small jump but overall smooth", "reviewer": "batch1", "review_date": "2026-07-02"}
{"sample_id": "RealEstate_050", "auto_verdict": "pass", "human_verdict": "fail", "video_quality": "poor", "trajectory_quality": "good", "primary_issue": "video_blurry", "notes": "Severely blurred video", "reviewer": "batch1", "review_date": "2026-07-02"}
```

---

### 3. Export Script

**File:** `scripts/export_for_review.py`

**Inputs:**
- `--stage1-jsonl`: Path(s) to stage1_results.jsonl (supports glob)
- `--stage2-jsonl`: Path(s) to stage2_results.jsonl (optional, supports glob)
- `--output-dir`: Output directory for review bundle
- `--total-samples`: Total number of samples to export (default: 1000)
- `--sampling-strategy`: Sampling strategy (default: "balanced")

**Outputs:**
- `review_list.csv`: Complete information table
- `videos/`: Extracted mp4 files (named by sample_id)
- `decisions_template.csv`: Template for human reviewers
- `sampling_report.txt`: Statistics about sampling

**Key Functions:**

1. **Load and merge Stage 1+2 results**
   ```python
   def load_results(stage1_paths, stage2_paths):
       # Merge stage1 and stage2 by sample_id
       # stage2 adds: video_T, black_frame_ratio, scene_cuts, traj_frozen
   ```

2. **Sample selection**
   ```python
   def balanced_sampling(results, config):
       # Bucket 1: fail_near_threshold
       #   - Calculate distance to threshold for each metric
       #   - Select closest to threshold
       # Bucket 2: multiple_reasons
       #   - Count flag_reasons, select samples with 2+
       # Bucket 3: pass_random
       # Bucket 4: fail_random
   ```

3. **Video extraction**
   ```python
   def extract_videos(samples, output_dir):
       # For each sample:
       #   - Open tar file
       #   - Extract sample_id.mp4
       #   - Copy to output_dir/videos/
       # Handle corrupted tars (use existing recovery logic)
   ```

4. **Generate CSVs**
   ```python
   def generate_review_list(samples, output_path):
       # Write review_list.csv with all columns
   
   def generate_template(samples, output_path):
       # Write decisions_template.csv
       # Include sample_id, auto_verdict columns pre-filled
       # Other columns empty for human input
   ```

**Usage:**
```bash
python scripts/export_for_review.py \
  --stage1-jsonl qc_output/full_*/stage1_results.jsonl \
  --stage2-jsonl qc_output/full_*/stage2_results.jsonl \
  --output-dir human_review_batch1 \
  --total-samples 1000 \
  --sampling-strategy balanced
```

---

### 4. Import Script

**File:** `scripts/import_review_results.py`

**Inputs:**
- `--review-list`: Original review_list.csv
- `--decisions`: Filled decisions_filled.csv
- `--output-dir`: Output directory

**Outputs:**
- `human_review_results.jsonl`: Standardized results
- `disagreement_report.html`: Analysis of auto vs human differences

**Key Functions:**

1. **Validation**
   ```python
   def validate_decisions(decisions_df):
       # Check all sample_ids exist in review_list
       # Check human_verdict is pass/fail (not empty)
       # Check primary_issue is valid enum
       # Report validation errors
   ```

2. **Generate disagreement report**
   ```python
   def generate_disagreement_report(review_df, decisions_df):
       # Overall stats:
       #   - Total reviewed, % completion
       #   - Agreement rate
       # Confusion matrix:
       #   - auto_pass → human_pass/fail
       #   - auto_fail → human_pass/fail
       # Disagreement breakdown:
       #   - auto_fail → human_pass: list primary_issues
       #   - auto_pass → human_fail: list primary_issues
       # Metric analysis (for auto_fail → human_pass):
       #   - Distribution of n_jumps, caption_len, etc.
       #   - Suggest threshold adjustments
   ```

**Disagreement Report Structure:**

```html
<h1>Human Review Analysis Report</h1>

<h2>Overall Statistics</h2>
<table>
  <tr><td>Total samples</td><td>1000</td></tr>
  <tr><td>Reviewed</td><td>950 (95%)</td></tr>
  <tr><td>Not reviewed (empty)</td><td>50 (5%)</td></tr>
</table>

<h2>Agreement Analysis</h2>
<table>
  <tr><th>Auto Verdict</th><th>Human Verdict</th><th>Count</th><th>%</th></tr>
  <tr><td>pass</td><td>pass</td><td>280</td><td>29.5%</td></tr>
  <tr><td>fail</td><td>fail</td><td>520</td><td>54.7%</td></tr>
  <tr><td>pass</td><td>fail</td><td>30</td><td>3.2%</td></tr>
  <tr><td>fail</td><td>pass</td><td>120</td><td>12.6%</td></tr>
</table>

<h2>Disagreement: Auto Fail → Human Pass (120 samples)</h2>
<p>These are potential false positives (too strict thresholds).</p>
<table>
  <tr><th>Primary Issue</th><th>Count</th><th>Avg Metric</th><th>Current Threshold</th></tr>
  <tr><td>trajectory_minor</td><td>80</td><td>n_jumps=3.2</td><td>≤2</td></tr>
  <tr><td>caption_too_short</td><td>25</td><td>caption_len=38</td><td>≥40</td></tr>
  <tr><td>black_frames</td><td>15</td><td>black_ratio=0.06</td><td>≤0.05</td></tr>
</table>
<p><strong>Recommendation:</strong> Consider relaxing n_jumps threshold to ≤4</p>

<h2>Disagreement: Auto Pass → Human Fail (30 samples)</h2>
<p>These are false negatives (missed issues).</p>
<table>
  <tr><th>Primary Issue</th><th>Count</th></tr>
  <tr><td>video_blurry</td><td>18</td></tr>
  <tr><td>caption_mismatch</td><td>12</td></tr>
</table>
<p><strong>Recommendation:</strong> Add blur detection to Stage 1</p>
```

**Usage:**
```bash
python scripts/import_review_results.py \
  --review-list human_review_batch1/review_list.csv \
  --decisions human_review_batch1/decisions_filled.csv \
  --output-dir human_review_batch1/analysis
```

---

### 5. Apply Script

**File:** `scripts/apply_human_review.py`

**Inputs:**
- `--stage1-jsonl`: Original stage1_results.jsonl files
- `--human-review`: human_review_results.jsonl from import script
- `--output-dir`: Output directory

**Outputs:**
- `stage1_results_merged.jsonl`: Stage 1 results with human decisions merged
- `manifests/pass.txt`: Final pass list (for Stage 3 input)
- `manifests/fail.txt`: Final fail list
- `manifests/human_reviewed.txt`: List of human-reviewed samples
- `summary_report.html`: Final statistics

**Key Logic:**
```python
def merge_decisions(stage1_results, human_review):
    human_dict = {r['sample_id']: r for r in human_review}
    
    for sample in stage1_results:
        if sample['sample_id'] in human_dict:
            # Human decision overrides auto
            human = human_dict[sample['sample_id']]
            sample['verdict'] = human['human_verdict']
            sample['human_reviewed'] = True
            sample['human_feedback'] = {
                'auto_verdict': human['auto_verdict'],
                'video_quality': human.get('video_quality'),
                'trajectory_quality': human.get('trajectory_quality'),
                'primary_issue': human['primary_issue'],
                'notes': human.get('notes', '')
            }
        else:
            # Keep automated decision
            sample['human_reviewed'] = False
    
    return stage1_results
```

**Usage:**
```bash
python scripts/apply_human_review.py \
  --stage1-jsonl qc_output/full_*/stage1_results.jsonl \
  --human-review human_review_batch1/analysis/human_review_results.jsonl \
  --output-dir qc_output/final_with_human_review
```

---

## Complete Workflow

### Step 1: Run Stage 1+2 on full dataset
```bash
# Run Stage 1+2 for all 7 groups (~6.5 hours total)
# See CURRENT_STATUS.md for specific commands
```

### Step 2: Export samples for review
```bash
cd /root/work/david_work/sana_wm_qc

python scripts/export_for_review.py \
  --stage1-jsonl qc_output/full_*/stage1_results.jsonl \
  --stage2-jsonl qc_output/full_*/stage2_results.jsonl \
  --output-dir human_review_batch1 \
  --total-samples 1000 \
  --sampling-strategy balanced

# Output:
#   human_review_batch1/review_list.csv
#   human_review_batch1/videos/ (1000 mp4 files)
#   human_review_batch1/decisions_template.csv
#   human_review_batch1/sampling_report.txt
```

### Step 3: Human review (2 work days)
```bash
# 1. Open VLC, batch load videos from human_review_batch1/videos/
# 2. Open review_list.csv to check auto_verdict and metrics
# 3. Open decisions_template.csv in Excel
# 4. For each sample:
#    - Watch video (10-20 seconds)
#    - Check auto_verdict and flag_reasons
#    - Fill human_verdict (pass/fail)
#    - Fill primary_issue (from enum list)
#    - Optionally fill video_quality, trajectory_quality
#    - Optionally add notes
# 5. Save as decisions_filled.csv
```

**Tips for reviewers:**
- Focus on human_verdict and primary_issue (required fields)
- video_quality and trajectory_quality are optional but helpful
- Use notes sparingly (only for unusual cases)
- Can skip samples by leaving human_verdict empty (will use auto decision)

### Step 4: Import and analyze
```bash
python scripts/import_review_results.py \
  --review-list human_review_batch1/review_list.csv \
  --decisions human_review_batch1/decisions_filled.csv \
  --output-dir human_review_batch1/analysis

# Output:
#   human_review_batch1/analysis/human_review_results.jsonl
#   human_review_batch1/analysis/disagreement_report.html

# View analysis
open human_review_batch1/analysis/disagreement_report.html
```

### Step 5: Apply decisions
```bash
python scripts/apply_human_review.py \
  --stage1-jsonl qc_output/full_*/stage1_results.jsonl \
  --human-review human_review_batch1/analysis/human_review_results.jsonl \
  --output-dir qc_output/final_with_human_review

# Output:
#   qc_output/final_with_human_review/stage1_results_merged.jsonl
#   qc_output/final_with_human_review/manifests/pass.txt
#   qc_output/final_with_human_review/manifests/fail.txt
#   qc_output/final_with_human_review/summary_report.html

# Use for Stage 3
cat qc_output/final_with_human_review/manifests/pass.txt | wc -l
# → Number of samples approved for Stage 3
```

---

## Error Handling

### Export Phase

**Corrupted tar files:**
- Use existing tar recovery logic (`stage1_fast.py`)
- Skip samples that cannot be extracted
- Log to `skipped_samples.txt` with reasons

**Missing videos:**
- If mp4 cannot be extracted, still include in review_list
- Set video_path to "MISSING"
- Reviewer can judge based on metrics alone or skip

### Import Phase

**Validation errors:**
- `human_verdict` is empty → Mark as "not reviewed", use auto_verdict
- `human_verdict` not in [pass, fail] → Error, list problematic rows
- `sample_id` not in review_list → Error, list unmatched IDs
- `primary_issue` not in enum → Warning, accept but flag in report

**Partial completion:**
- If only 800/1000 samples reviewed → OK, use auto_verdict for remaining 200
- Report completion rate in disagreement_report

### Apply Phase

**Multiple review batches:**
- If importing multiple human_review results → Later overrides earlier
- Track review_date to determine precedence

**Missing samples:**
- Samples not in any human_review → Use auto_verdict
- No error, just note in summary_report

---

## Success Metrics

### Immediate Metrics (from disagreement_report)

1. **Completion Rate**: % of samples actually reviewed
   - Target: >90%

2. **Agreement Rate**: % where auto_verdict = human_verdict
   - Baseline expectation: 80-90%
   - Lower rate suggests thresholds need adjustment

3. **False Positive Rate**: auto_fail but human_pass
   - High rate (>15%) → Thresholds too strict

4. **False Negative Rate**: auto_pass but human_fail
   - High rate (>5%) → Missing detection logic

### Long-term Metrics (for future iterations)

5. **Issue Coverage**: % of distinct primary_issues identified
   - Goal: Discover automation blind spots

6. **Threshold Insights**: For trajectory_minor cases, what's the actual acceptable n_jumps range?
   - Inform future threshold tuning

---

## Future Extensions (Out of Scope)

### Extension 1: Web UI
If review volume increases, build a simple web interface:
- Video player integrated
- Metrics displayed side-by-side
- Keyboard shortcuts for quick decisions
- Real-time progress tracking

### Extension 2: Multi-round Refinement
After identifying threshold issues:
- Adjust thresholds in group_config.py
- Re-run Stage 1 (fast, uses cached tar contents)
- Export new samples at new boundaries
- Validate improvements

### Extension 3: Active Learning
Use human feedback to train an ML model:
- Features: All Stage 1+2 metrics
- Labels: Human verdicts
- Model predicts pass/fail probability
- Reduces future human review burden

---

## Implementation Estimates

**Development Time:**
- Export script: 2-3 hours
- Import script: 1-2 hours
- Apply script: 1 hour
- Testing and documentation: 2 hours
- **Total: 6-8 hours**

**Human Review Time (per 1000 samples):**
- Average time per sample: 1-2 minutes (10s video + judgment)
- Total: 20-30 hours
- With 2 reviewers in parallel: 10-15 hours (~2 work days)

**End-to-End Timeline:**
- Stage 1+2 full run: 1 day
- Export: 1 hour
- Human review: 2 days
- Import + Apply: 30 minutes
- **Total: ~4 days**

---

## Open Questions

None - design is complete and approved.

---

## References

- Stage 1 implementation: `src/sana_wm_pipeline/qc/stage1_fast.py`
- Stage 2 implementation: `src/sana_wm_pipeline/qc/stage2_deep.py`
- Group config: `src/sana_wm_pipeline/qc/group_config.py`
- Current status: `sana_qc_cmcc_pack/CURRENT_STATUS.md`
