#!/usr/bin/env bash
# Download a tiny SpatialVID-HQ subset for local H100 smoke validation.
#
# Prerequisites:
#   1. Install huggingface_hub CLI:  pip install --no-user "huggingface_hub[cli]"
#   2. Log in:                       huggingface-cli login    (one-time)
#   3. Visit https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ
#      and click "Agree and access repository" (gated under CC-BY-NC-SA 4.0)
#
# Result:
#   ~15 GB of SpatialVID-HQ group_0001 in $LOCAL_DIR;
#   then we copy 10 random clips into raw_root for the pipeline.
set -euo pipefail

LOCAL_DIR=${LOCAL_DIR:-/mnt/afs/davidwang/data/spatialvid_hq}
RAW_ROOT=${RAW_ROOT:-/mnt/afs/davidwang/workspace/data/sana_wm/raw/spatialvid_hq}
N_CLIPS=${N_CLIPS:-10}

echo "[smoke] Downloading SpatialVID-HQ group_0001 -> $LOCAL_DIR"
mkdir -p "$LOCAL_DIR" "$RAW_ROOT"

huggingface-cli download SpatialVID/SpatialVID-HQ \
  --repo-type dataset \
  --include "group_0001/*" \
  --local-dir "$LOCAL_DIR"

echo "[smoke] Selecting $N_CLIPS clip(s) -> $RAW_ROOT"
# group_0001 layout: group_0001/videos/*.mp4 (approx)
# We pick the first N mp4 files we find.
mapfile -t CLIPS < <(find "$LOCAL_DIR/group_0001" -type f -name "*.mp4" | sort | head -n "$N_CLIPS")
if [ ${#CLIPS[@]} -eq 0 ]; then
  echo "[smoke] No mp4 found under $LOCAL_DIR/group_0001 — inspect the layout manually:"
  find "$LOCAL_DIR/group_0001" -maxdepth 3 -type d
  exit 1
fi
for v in "${CLIPS[@]}"; do
  cp -v "$v" "$RAW_ROOT/"
done

echo "[smoke] Done. $N_CLIPS clip(s) at $RAW_ROOT"
echo "        Now run:  bash scripts/e2e_smoke.sh"
