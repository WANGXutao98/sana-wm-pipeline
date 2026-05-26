#!/usr/bin/env bash
# E2E smoke: 1 clip per source -> shard written.
# Heavy stages (VIPE / Pi3X / Qwen-VL / FCGS) are stubbed via env vars so
# this can run on a CPU-only host or in CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG=${CONFIG:-configs/pipeline.yaml}
SOURCES=${SOURCES:-configs/sources.yaml}

python -m sana_wm_pipeline.orchestrate.ray_pipeline \
  --config "$CONFIG" \
  --sources "$SOURCES" \
  --smoke --in-process

OUT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['out_root'])")
N=$(ls "$OUT_ROOT"/shard-*.tar 2>/dev/null | wc -l)
echo "Found $N shard(s) in $OUT_ROOT"
test "$N" -ge 1
