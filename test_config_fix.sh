#!/bin/bash
set -e

# Quick test to verify config changes are working
# Tests DL3DV with a small sample

cd /root/work/david_work/sana_wm_qc

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/repair_done"

echo "========================================"
echo "  Quick Config Test (DL3DV sample)"
echo "  Started at: $(date)"
echo "========================================"
echo ""

# Test with just 1% of DL3DV data
python scripts/run_qc.py \
  --tar-root "$DATA_ROOT/final_wds-DL3DV-ALL-2K/wds-DL3DV-ALL-2K" \
  --group wds-DL3DV-ALL-2K \
  --output-dir qc_test_output/DL3DV-test \
  --n-workers 4 \
  --sample-frac 0.01 \
  --skip-stage2

echo ""
echo "========================================"
echo "  Test Complete"
echo "========================================"

# Check results
if [ -f "qc_test_output/DL3DV-test/stage1_results.jsonl" ]; then
  echo ""
  echo "=== Results ==="

  # Correct statistics using jq
  total=$(wc -l < qc_test_output/DL3DV-test/stage1_results.jsonl)
  pass=$(jq -r 'select(.verdict=="pass")' qc_test_output/DL3DV-test/stage1_results.jsonl | wc -l)
  fail=$(jq -r 'select(.verdict=="fail")' qc_test_output/DL3DV-test/stage1_results.jsonl | wc -l)
  flag=$(jq -r 'select(.verdict=="flag")' qc_test_output/DL3DV-test/stage1_results.jsonl | wc -l)

  echo "Total: $total"
  echo "Pass: $pass ($(awk "BEGIN {printf \"%.1f\", $pass*100/$total}")%)"
  echo "Fail: $fail ($(awk "BEGIN {printf \"%.1f\", $fail*100/$total}")%)"
  echo "Flag: $flag ($(awk "BEGIN {printf \"%.1f\", $flag*100/$total}")%)"
  echo ""

  # Expected: ~70-85% pass (with relaxed n_jumps)
  pass_rate=$(awk "BEGIN {printf \"%.0f\", $pass*100/$total}")
  if [ "$pass_rate" -gt 60 ]; then
    echo "✅ SUCCESS! Pass rate is $pass_rate% (expected ~70-85%)"
    echo "Config changes are working correctly!"
    echo ""
    echo "Top 5 flag reasons:"
    jq -r 'select(.verdict=="flag") | .flag_reasons[]' qc_test_output/DL3DV-test/stage1_results.jsonl | sort | uniq -c | sort -rn | head -5
  else
    echo "❌ FAILED! Pass rate is only $pass_rate% (expected ~70-85%)"
    echo "Config changes are NOT working!"
    echo ""
    echo "Top flag reasons:"
    jq -r 'select(.verdict=="flag") | .flag_reasons[]' qc_test_output/DL3DV-test/stage1_results.jsonl | sort | uniq -c | sort -rn | head -5
  fi
fi
