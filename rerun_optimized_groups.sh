#!/bin/bash
set -e

# Rerun Stage 1 for groups with optimized configurations
# - DL3DV: min_caption_len 50 → 0
# - OmniWorld: check_camera_words disabled
# - sekai-game-drone: check_camera_words disabled
# - sekai-game-walking: check_camera_words disabled

DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/repair_done"
OUTPUT_BASE="qc_full_output"

# Set PYTHONPATH to find sana_wm_pipeline module
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

echo "========================================"
echo "  Rerunning Stage 1 with Optimized Config"
echo "  Started at: $(date)"
echo "========================================"
echo ""

# Function to rerun a group
rerun_group() {
  local dir_name=$1
  local group_name=$2
  local est_time=$3
  local reason=$4

  wds_dir=$(echo "$dir_name" | sed 's/final_//')
  tar_root="$DATA_ROOT/$dir_name/$wds_dir"
  output_dir="$OUTPUT_BASE/$(basename $dir_name | sed 's/final_wds-//')"

  echo "----------------------------------------"
  echo "Group: $group_name"
  echo "Reason: $reason"
  echo "Source: $tar_root"
  echo "Output: $output_dir"
  echo "Est. time: $est_time"
  echo "Start: $(date)"
  echo "----------------------------------------"

  # Backup old results
  if [ -d "$output_dir" ]; then
    backup_dir="${output_dir}.backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up to: $backup_dir"
    mv "$output_dir" "$backup_dir"
  fi

  python scripts/run_qc.py \
    --tar-root "$tar_root" \
    --group "$group_name" \
    --output-dir "$output_dir" \
    --n-workers 16 \
    --read-video-frames

  if [ $? -eq 0 ]; then
    echo "✅ $group_name completed at $(date)"

    # Show statistics
    if [ -f "$output_dir/stage1_results.jsonl" ]; then
      total=$(wc -l < "$output_dir/stage1_results.jsonl")
      pass=$(wc -l < "$output_dir/manifests/pass.txt" 2>/dev/null || echo 0)
      fail=$(wc -l < "$output_dir/manifests/fail.txt" 2>/dev/null || echo 0)
      flag=$(wc -l < "$output_dir/manifests/human_review.txt" 2>/dev/null || echo 0)

      echo "  Total: $total, Pass: $pass, Fail: $fail, Flag: $flag"
    fi
  else
    echo "❌ $group_name FAILED at $(date)"
  fi

  echo ""
}

# Rerun DL3DV (min_caption_len: 50 → 0)
rerun_group "final_wds-DL3DV-ALL-2K" "wds-DL3DV-ALL-2K" "30min" "Allow no caption (min_caption_len=0)"

# Rerun RealEstate10K (max_jumps_flag: 0 → 3)
rerun_group "final_wds-RealEstate10K-360p" "wds-RealEstate10K-360p" "20min" "Relax n_jumps threshold (max_jumps_flag=3)"

# Rerun OmniWorld (check_camera_words: true → false)
rerun_group "final_wds-OmniWorld-Game" "wds-OmniWorld-Game" "15min" "Disable camera words check"

# Rerun sekai-game-drone (check_camera_words: true → false)
rerun_group "final_wds-sekai-game-drone" "wds-sekai-game-drone" "5min" "Disable camera words check"

# Rerun sekai-game-walking (check_camera_words: true → false)
rerun_group "final_wds-sekai-game-walking" "wds-sekai-game-walking" "10min" "Disable camera words check"

echo "========================================"
echo "  All Optimized Groups Rerun Complete"
echo "  Finished at: $(date)"
echo "========================================"

# Generate comparison report
echo ""
echo "=== Comparison: Before vs After ==="
echo ""

# You can manually compare with backup directories
echo "Before results are in: ${OUTPUT_BASE}/*.backup_*"
echo "After results are in: ${OUTPUT_BASE}/*"
echo ""
echo "Run this to compare:"
echo "  for group in DL3DV-ALL-2K OmniWorld-Game sekai-game-drone sekai-game-walking; do"
echo "    echo \"=== \$group ===\";"
echo "    echo -n \"Before: \"; jq -r '.verdict' \${OUTPUT_BASE}/\${group}.backup_*/stage1_results.jsonl 2>/dev/null | sort | uniq -c;"
echo "    echo -n \"After:  \"; jq -r '.verdict' \${OUTPUT_BASE}/\${group}/stage1_results.jsonl 2>/dev/null | sort | uniq -c;"
echo "    echo \"\";"
echo "  done"
