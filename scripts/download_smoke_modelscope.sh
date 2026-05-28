#!/usr/bin/env bash
# Download a tiny SpatialVID-HQ subset for local H100 smoke validation.
#
# Prerequisites:
#   1. Install the ModelScope CLI:  pip install --no-user modelscope
#   2. (Optional) Log in:           modelscope login --token <YOUR_MS_TOKEN>
#      Only needed if the dataset is gated for your account. If download fails
#      with an auth error, log in and accept the license on the dataset page.
#   3. Visit https://www.modelscope.cn/datasets/SpatialVID/SpatialVID-HQ
#      and accept the license (CC-BY-NC-SA 4.0)
#
# Repo layout (same as HF mirror):
#   videos/group_0001.tar.gz        (~14 GB of mp4 clips, packed)
#   annotations/group_0001.tar.gz   (~1.5 GB)
#   depths/group_0001.tar.gz
#   data/train/SpatialVID_HQ_metadata.csv
#
# Result:
#   ~14 GB videos/group_0001.tar.gz in $LOCAL_DIR, extracted in place;
#   then we copy $N_CLIPS clips into $RAW_ROOT for the pipeline.
set -euo pipefail

LOCAL_DIR=${LOCAL_DIR:-/mnt/afs/davidwang/workspace/data/spatialvid_hq}
RAW_ROOT=${RAW_ROOT:-/mnt/afs/davidwang/workspace/data/sana_wm/raw/spatialvid_hq}
N_CLIPS=${N_CLIPS:-10}
GROUP=${GROUP:-group_0001}

echo "[smoke] Downloading SpatialVID-HQ videos/$GROUP.tar.gz -> $LOCAL_DIR"
mkdir -p "$LOCAL_DIR" "$RAW_ROOT"

modelscope download \
  --dataset SpatialVID/SpatialVID-HQ \
  --include "videos/$GROUP.tar.gz" \
  --local_dir "$LOCAL_DIR"

TARBALL="$LOCAL_DIR/videos/$GROUP.tar.gz"
if [ ! -f "$TARBALL" ]; then
  echo "[smoke] Expected tarball not found: $TARBALL"
  echo "[smoke] Inspect what was downloaded:"
  find "$LOCAL_DIR" -maxdepth 3 -type f | head -n 50
  exit 1
fi

echo "[smoke] Extracting $TARBALL"
tar -xzf "$TARBALL" -C "$LOCAL_DIR"

echo "[smoke] Selecting $N_CLIPS clip(s) -> $RAW_ROOT"
# After extraction the mp4 clips live somewhere under $LOCAL_DIR; search recursively.
mapfile -t CLIPS < <(find "$LOCAL_DIR" -type f -name "*.mp4" | sort | head -n "$N_CLIPS")
if [ ${#CLIPS[@]} -eq 0 ]; then
  echo "[smoke] No mp4 found under $LOCAL_DIR after extraction — inspect the layout manually:"
  find "$LOCAL_DIR" -maxdepth 3 -type d
  exit 1
fi
for v in "${CLIPS[@]}"; do
  cp -v "$v" "$RAW_ROOT/"
done

echo "[smoke] Done. $N_CLIPS clip(s) at $RAW_ROOT"
echo "        Now run:  bash scripts/e2e_smoke.sh"