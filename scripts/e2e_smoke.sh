#!/usr/bin/env bash
# E2E smoke: 1 clip per source -> shard written.
#
# Two modes:
#   SANA_WM_POSE_STUB=1 (default) — fake pose artifact, no ML models needed.
#       Tests pipeline plumbing (Stage01 ffmpeg + Stage02 stub + Stage03-06).
#   SANA_WM_POSE_STUB=0            — real VIPE + Pi3X + MoGe-2 invoked.
#       Requires models downloaded; see docs/DATASETS.md §E for weights.
#
# Prerequisites (minimal, stub mode):
#   - conda env: source .../miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
#   - ffmpeg with libx264: pip install --no-user static-ffmpeg && python -c "import static_ffmpeg; static_ffmpeg.add_paths()"
#     (or system ffmpeg if libx264 is present)
#   - Video data at local_path_example configured in configs/sources.yaml
#
# Prerequisites (full mode, SANA_WM_POSE_STUB=0):
#   - VIPE installed: bash scripts/00_setup_vipe.sh
#   - Pi3X weights:   huggingface-cli download yyfz233/Pi3X --local-dir /mnt/afs/davidwang/models/pi3x
#   - MoGe-2 weights: huggingface-cli download microsoft/MoGe --local-dir /mnt/afs/davidwang/models/moge2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default: stub heavy ML stages so pipeline plumbing can be tested without model weights.
export SANA_WM_POSE_STUB=${SANA_WM_POSE_STUB:-1}
echo "[smoke] SANA_WM_POSE_STUB=$SANA_WM_POSE_STUB"

CONFIG=${CONFIG:-configs/pipeline.yaml}
SOURCES=${SOURCES:-configs/sources.yaml}

# Ensure ffmpeg is on PATH.  If using static-ffmpeg, add_paths() was called at
# import time; this export covers subprocess invocations from normalize.py.
python -c "
try:
    import static_ffmpeg; static_ffmpeg.add_paths()
    print('[smoke] ffmpeg: using static-ffmpeg')
except ImportError:
    print('[smoke] ffmpeg: using system ffmpeg')
" 2>/dev/null || true

python -m sana_wm_pipeline.orchestrate.ray_pipeline \
  --config "$CONFIG" \
  --sources "$SOURCES" \
  --smoke --in-process

OUT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['out_root'])")
N=$(ls "$OUT_ROOT"/shard-*.tar 2>/dev/null | wc -l)
echo "[smoke] Found $N shard(s) in $OUT_ROOT"
test "$N" -ge 1
echo "[smoke] PASS"
