#!/usr/bin/env bash
# Default 模式端到端：jdvbbfb-v3-full 单样本 → WebDataset shard + ATE 评估
#
# 用法：
#   bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh \
#     <group> <shard_idx> [<out_base>]
# 例：
#   bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh wds-DL3DV-ALL-2K 0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 环境（CMCC 部署时由 sed 重写为 <YOUR_BASE>）──────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export DISABLE_XFORMERS=1
export VIPE_EXT_JIT=1

GROUP="${1:?Usage: $0 <group> <shard_idx> [out_base]}"
SHARD_IDX="${2:?Usage: $0 <group> <shard_idx> [out_base]}"
OUT_BASE="${3:-/mnt/afs/davidwang/workspace/data/jdvbbfb_smoke}"
SHARDS_DIR="${OUT_BASE}/shards_default"
mkdir -p "${OUT_BASE}" "${SHARDS_DIR}"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo " jdvbbfb Default E2E: group=${GROUP} shard=${SHARD_IDX}"
echo "========================================================================"

# ── Stage 0: 拉取一个样本 → scene 目录 ───────────────────────────────────────
# 数据来源：若 JDVBBFB_LOCAL_ROOT 已设（CMCC，数据在 externalstorage）→ 读本地；
#           否则走 HF 流式（H100 开发）。
echo "=== Stage 0: prepare 1 sample from ${GROUP} shard ${SHARD_IDX} ==="
if [ -n "${JDVBBFB_LOCAL_ROOT:-}" ]; then
  SRC_ARGS=(--local-root "${JDVBBFB_LOCAL_ROOT}")
  echo "  source: LOCAL ${JDVBBFB_LOCAL_ROOT}"
else
  SRC_ARGS=(--repo junchaoh-cs/jdvbbfb-v3-full)
  echo "  source: HF stream"
fi
python experiments/data_production_smoke/prepare_jdvbbfb.py \
  "${SRC_ARGS[@]}" --group "${GROUP}" --shard-idx "${SHARD_IDX}" \
  --sample-limit 1 --out-base "${OUT_BASE}"

# 取刚写出的 scene 目录（最新修改的、含 video.mp4 的子目录）
SCENE_DIR="$(find "${OUT_BASE}" -mindepth 1 -maxdepth 1 -type d -name "${GROUP#wds-}*" \
             -exec test -f '{}/video.mp4' \; -print | sort | tail -1)"
SCENE_ID="$(basename "${SCENE_DIR}")"
echo "scene: ${SCENE_DIR}"

# ── Stage 1: normalize → 1280x720 @16fps ─────────────────────────────────────
echo "=== Stage 1: normalize ==="
NORM_VIDEO="${SCENE_DIR}/normalized.mp4"
if [ ! -f "${NORM_VIDEO}" ]; then
  python - <<PYEOF
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
info = normalize_video(Path("${SCENE_DIR}/video.mp4"), Path("${NORM_VIDEO}"))
print(f"Normalized: {info.n_frames} frames @ {info.fps}fps ({info.width}x{info.height})")
PYEOF
fi

# ── Stage 2: Default mode VIPE SLAM (Pi3X + MoGe-2) ──────────────────────────
echo "=== Stage 2: Default mode (Pi3X + MoGe-2 + VIPE) ==="
VIPE_WORK="${SCENE_DIR}/vipe_work_default"
mkdir -p "${VIPE_WORK}"
ARTIFACT_JSON="${VIPE_WORK}/pose_artifact_default.json"
if [ ! -f "${ARTIFACT_JSON}" ]; then
  python - <<PYEOF
import json
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_default import run_default
art = run_default(Path("${NORM_VIDEO}"), Path("${VIPE_WORK}"))
print(f"Poses {art.poses_c2w.shape}  Intr {art.intrinsics.shape}")
Path("${ARTIFACT_JSON}").write_text(json.dumps({
    "poses_c2w": art.poses_c2w.tolist(),
    "intrinsics": art.intrinsics.tolist(),
    "scale_per_frame": art.scale_per_frame.tolist(),
}))
PYEOF
fi

# ── Stage 6: pack WebDataset shard ───────────────────────────────────────────
echo "=== Stage 6: pack shard ==="
SHARD="${SHARDS_DIR}/shard-000001.tar"
python - <<PYEOF
import io, json, numpy as np, tarfile
from pathlib import Path
scene_id="${SCENE_ID}"
art=json.loads(Path("${ARTIFACT_JSON}").read_text())
poses=np.array(art["poses_c2w"],np.float32)
intr=np.array(art["intrinsics"],np.float32)        # (T,1,4)
scale=np.array(art["scale_per_frame"],np.float32)
cap=Path("${SCENE_DIR}/caption.txt").read_text()
def add_npy(tf,key,arr):
    b=io.BytesIO(); np.save(b,arr); raw=b.getvalue()
    ti=tarfile.TarInfo(f"{scene_id}.{key}"); ti.size=len(raw); tf.addfile(ti,io.BytesIO(raw))
with tarfile.open("${SHARD}","w") as tf:
    vb=Path("${NORM_VIDEO}").read_bytes()
    ti=tarfile.TarInfo(f"{scene_id}.mp4"); ti.size=len(vb); tf.addfile(ti,io.BytesIO(vb))
    add_npy(tf,"poses_c2w.npy",poses)
    add_npy(tf,"intrinsics.npy",intr)
    add_npy(tf,"scale.npy",scale)
    cb=cap.encode(); ti=tarfile.TarInfo(f"{scene_id}.caption.txt"); ti.size=len(cb); tf.addfile(ti,io.BytesIO(cb))
    meta=json.dumps({"scene_id":scene_id,"T":len(poses),"mode":"default","dataset":"jdvbbfb-v3-full","group":"${GROUP}"}).encode()
    ti=tarfile.TarInfo(f"{scene_id}.meta.json"); ti.size=len(meta); tf.addfile(ti,io.BytesIO(meta))
print(f"Shard: ${SHARD}")
PYEOF

# ── Schema check + Pose eval (vs gt_poses.npy) ───────────────────────────────
echo "=== Schema check ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema --shards-dir "${SHARDS_DIR}"
echo "=== Pose eval (vs GT c2w) ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval --shards-dir "${SHARDS_DIR}" \
  --scenes-dir "${OUT_BASE}" --out-dir "${SHARDS_DIR}/eval_output" || \
  echo "[note] pose-eval needs meta.scene_id == scene dir name"

echo "✓ jdvbbfb Default E2E 完成: ${SCENE_ID}"
