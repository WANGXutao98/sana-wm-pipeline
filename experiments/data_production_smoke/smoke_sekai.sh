#!/bin/bash
# Sekai-Real-Walking-HQ C.1 Smoke Test（单样本 Default 模式）
# 用法：bash experiments/data_production_smoke/smoke_sekai.sh
# CMCC 机器上运行，依赖 activate_sana_wm.sh 已存在
set -euo pipefail

export NEW_BASE=/root/work/david_work
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full"
export OUT_BASE="$NEW_BASE/sekai_smoke"
export GROUP="wds-sekai-real-walking-hq"
export SHARD_IDX=0

source "$NEW_BASE/activate_sana_wm.sh"
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 确保 sana_wm_pipeline 已安装 + psutil 存在
PYTHONNOUSERSITE=1 "$ENV_DIR/bin/pip" install --no-user -e "$PROJ_DIR" \
  --no-deps --no-build-isolation --quiet
PYTHONNOUSERSITE=1 "$ENV_DIR/bin/pip" install --no-user psutil --quiet

python -c "
import sana_wm_pipeline; print('sana_wm_pipeline ✓')
import vipe_ext;          print('vipe_ext ✓')
import vipe;              print('vipe ✓')
import torch;             print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} ✓')
"

mkdir -p "$OUT_BASE"
cd "$PROJ_DIR"

# ── Stage 0: 解包 1 个 sekai 样本 ────────────────────────────────────────────
echo "=== Stage 0: prepare sample ==="
python experiments/data_production_smoke/prepare_jdvbbfb.py \
  --local-root "$DATA_ROOT" --group "$GROUP" --shard-idx "$SHARD_IDX" \
  --sample-limit 1 --out-base "$OUT_BASE"

export SCENE_DIR="$(find "$OUT_BASE" -mindepth 1 -maxdepth 1 -type d \
  -exec test -f '{}/video.mp4' \; -print | sort | tail -1)"
export SCENE_ID="$(basename "$SCENE_DIR")"
echo "scene: $SCENE_DIR  id: $SCENE_ID"

export NORM_VIDEO="${SCENE_DIR}/normalized.mp4"
export VIPE_WORK="${SCENE_DIR}/vipe_work_default"
export ARTIFACT_JSON="${VIPE_WORK}/pose_artifact_default.json"
mkdir -p "$VIPE_WORK"

# ── Stage 1: 归一化 → 1280×720 @16fps ────────────────────────────────────────
echo "=== Stage 1: normalize ==="
python -c "
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
info = normalize_video(Path('$SCENE_DIR/video.mp4'), Path('$NORM_VIDEO'))
print(f'Normalized: {info.n_frames} frames @ {info.fps}fps ({info.width}x{info.height})')
"

# ── Stage 2: VIPE SLAM（Pi3X + MoGe-2，约 5-20 分钟）────────────────────────
echo "=== Stage 2: VIPE SLAM (Pi3X + MoGe-2) ==="
python -c "
import json
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_default import run_default
art = run_default(Path('$NORM_VIDEO'), Path('$VIPE_WORK'))
print(f'Poses {art.poses_c2w.shape}  Intr {art.intrinsics.shape}')
Path('$ARTIFACT_JSON').write_text(json.dumps({
    'poses_c2w': art.poses_c2w.tolist(),
    'intrinsics': art.intrinsics.tolist(),
    'scale_per_frame': art.scale_per_frame.tolist(),
}))
print('Stage 2 DONE')
" 2>&1 | tee /tmp/stage2_sekai.log

# ── Stage 6: 打包 WebDataset shard ───────────────────────────────────────────
echo "=== Stage 6: pack shard ==="
export SHARDS_DIR="$OUT_BASE/shards_default"
mkdir -p "$SHARDS_DIR"
export SHARD="${SHARDS_DIR}/shard-000001.tar"
python - <<PYEOF
import io, json, numpy as np, tarfile
from pathlib import Path
scene_id = "$SCENE_ID"
art   = json.loads(Path("$ARTIFACT_JSON").read_text())
poses = np.array(art["poses_c2w"],       np.float32)
intr  = np.array(art["intrinsics"],      np.float32)
scale = np.array(art["scale_per_frame"], np.float32)
cap_path = Path("$SCENE_DIR/caption.txt")
cap = cap_path.read_text() if cap_path.exists() else "no caption"
def add_npy(tf, key, arr):
    b = io.BytesIO(); np.save(b, arr); raw = b.getvalue()
    ti = tarfile.TarInfo(f"{scene_id}.{key}"); ti.size = len(raw); tf.addfile(ti, io.BytesIO(raw))
with tarfile.open("$SHARD", "w") as tf:
    vb = Path("$NORM_VIDEO").read_bytes()
    ti = tarfile.TarInfo(f"{scene_id}.mp4"); ti.size = len(vb); tf.addfile(ti, io.BytesIO(vb))
    add_npy(tf, "poses_c2w.npy",  poses)
    add_npy(tf, "intrinsics.npy", intr)
    add_npy(tf, "scale.npy",      scale)
    cb = cap.encode(); ti = tarfile.TarInfo(f"{scene_id}.caption.txt"); ti.size = len(cb); tf.addfile(ti, io.BytesIO(cb))
    meta = json.dumps({"scene_id": scene_id, "T": len(poses), "mode": "default",
                       "dataset": "jdvbbfb-v3-full", "group": "$GROUP"}).encode()
    ti = tarfile.TarInfo(f"{scene_id}.meta.json"); ti.size = len(meta); tf.addfile(ti, io.BytesIO(meta))
print(f"Shard written: $SHARD  ({len(poses)} frames)")
PYEOF

# ── Schema check ──────────────────────────────────────────────────────────────
echo "=== Schema check ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema --shards-dir "$SHARDS_DIR"

echo ""
echo "✓ Sekai Smoke Test PASSED: $SCENE_ID"
