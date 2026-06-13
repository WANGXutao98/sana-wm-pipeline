#!/usr/bin/env bash
# Default 模式端到端：OmniWorld 单场景 → WebDataset shard
#
# 与 run_e2e_gtdepth.sh 的区别：
#   - Stage 2 使用 mode_default（Pi3X + MoGe-2 + VIPE SLAM），不需要 GT depth
#   - Shards 输出到 shards_default/（而非 shards_gtdepth/）
#   - pose_artifact 保存为 vipe_work_default/pose_artifact_default.json
#   - intrinsics shape 为 (T, 1, 4) NVD 格式，按原样存入 shard
#
# 用法（推荐）：
#   bash experiments/data_production_smoke/run_e2e_default_omniworld.sh \
#     <DATA_ROOT>/annotations/OmniWorld-Game/<scene_id> \
#     <DATA_ROOT>/videos/OmniWorld-Game/<scene_id>
#
# 若只有一个参数（annotations 和 videos 在同一目录）：
#   bash experiments/data_production_smoke/run_e2e_default_omniworld.sh <unified_scene_dir>
#
# 前置：
#   - normalized.mp4 已存在（由 run_e2e_gtdepth.sh Stage 0+1 或 prepare_omniworld.py 生成）
#   - conda env sana_wm 已激活（或由此脚本激活）
#   - SANA_WM_PI3X_WEIGHTS / SANA_WM_MOGE2_WEIGHTS 已指向对应权重目录
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 环境 ──────────────────────────────────────────────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export DISABLE_XFORMERS=1
export VIPE_EXT_JIT=1   # pre-built slam_ext API mismatch; JIT recompiles on first run (~2min)

ANNOT_DIR="${1:?Usage: $0 <annotations_scene_dir> [<video_scene_dir>]}"
VIDEO_DIR="${2:-${ANNOT_DIR}}"   # 若未提供则与 annotations 相同
SCENE_ID="$(basename "${ANNOT_DIR}")"
WORK_BASE="/mnt/afs/davidwang/workspace/data/omniworld_smoke"
PREP_DIR="${WORK_BASE}/${SCENE_ID}"
SHARDS_DIR="${WORK_BASE}/shards_default"
VIPE_WORK_DEFAULT="${PREP_DIR}/vipe_work_default"
NORM_VIDEO="${PREP_DIR}/normalized.mp4"
CAPTION_FILE="${PREP_DIR}/caption.txt"

mkdir -p "${PREP_DIR}" "${SHARDS_DIR}" "${VIPE_WORK_DEFAULT}"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo " Default E2E (Pi3X + MoGe-2 + VIPE SLAM): ${SCENE_ID}"
echo "  annotations: ${ANNOT_DIR}"
echo "  videos:      ${VIDEO_DIR}"
echo "  shards out:  ${SHARDS_DIR}"
echo "========================================================================"

# ── Stage 0: 检查 normalized.mp4 是否已存在（不重跑 prepare）────────────────
echo ""
echo "=== Stage 0: check normalized.mp4 ==="
if [ ! -f "${NORM_VIDEO}" ]; then
  echo "[ERROR] normalized.mp4 not found at ${NORM_VIDEO}" >&2
  echo "[ERROR] Run run_e2e_gtdepth.sh first (or prepare_omniworld.py + normalize) to produce it." >&2
  exit 1
fi
echo "normalized.mp4 found: ${NORM_VIDEO}"

# ── Stage 2: Default mode VIPE SLAM（Pi3X + MoGe-2）────────────────────────
echo ""
echo "=== Stage 2: Default mode VIPE SLAM (Pi3X + MoGe-2) ==="
ARTIFACT_JSON="${VIPE_WORK_DEFAULT}/pose_artifact_default.json"

if [ ! -f "${ARTIFACT_JSON}" ]; then
  python - <<PYEOF
import json, numpy as np
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_default import run_default

clip    = Path("${NORM_VIDEO}")
workdir = Path("${VIPE_WORK_DEFAULT}")

print(f"Running Default mode on: {clip}")
artifact = run_default(clip, workdir)

print(f"Poses:      {artifact.poses_c2w.shape}")
print(f"Intrinsics: {artifact.intrinsics.shape}")
print(f"Scale mean: {artifact.scale_per_frame.mean():.4f}  std={artifact.scale_per_frame.std():.4f}")

(workdir / "pose_artifact_default.json").write_text(json.dumps({
    "poses_c2w":       artifact.poses_c2w.tolist(),
    "intrinsics":      artifact.intrinsics.tolist(),
    "scale_per_frame": artifact.scale_per_frame.tolist(),
}))
print(f"Saved: ${ARTIFACT_JSON}")
PYEOF
else
  echo "pose_artifact_default.json already exists, skipping VIPE."
fi

# ── Stage 5: Caption（复用已有 caption.txt）─────────────────────────────────
echo ""
echo "=== Stage 5: caption ==="
if [ ! -f "${CAPTION_FILE}" ]; then
  echo "Synthetic indoor scene rendered by OmniWorld-Game engine." \
    > "${CAPTION_FILE}"
  echo "Created stub caption: ${CAPTION_FILE}"
else
  echo "Reusing existing caption: ${CAPTION_FILE}"
fi

# ── Stage 6: pack → WebDataset shard ────────────────────────────────────────
# 文件命名格式：{scene_id}.{suffix}，符合 verify_and_eval.py REQUIRED_SUFFIXES：
#   mp4 / poses_c2w.npy / intrinsics.npy / scale.npy / caption.txt / meta.json
#
# intrinsics shape：mode_default 返回 (T, 1, 4) NVD 格式，按原样存入 shard
echo ""
echo "=== Stage 6: pack WebDataset shard ==="
SHARD="${SHARDS_DIR}/shard-000001.tar"

python - <<PYEOF
import io, json, numpy as np, tarfile
from pathlib import Path

scene_id   = "${SCENE_ID}"
norm_video = Path("${NORM_VIDEO}")
vipe_work  = Path("${VIPE_WORK_DEFAULT}")
caption_p  = Path("${CAPTION_FILE}")
shard      = Path("${SHARD}")

art = json.loads((vipe_work / "pose_artifact_default.json").read_text())
poses_c2w  = np.array(art["poses_c2w"],       dtype=np.float32)   # (T, 4, 4)
intrinsics = np.array(art["intrinsics"],       dtype=np.float32)   # (T, 1, 4) NVD format
scale      = np.array(art["scale_per_frame"],  dtype=np.float32)   # (T,)
T = len(poses_c2w)

print(f"poses_c2w:  {poses_c2w.shape}")
print(f"intrinsics: {intrinsics.shape}  (NVD format, stored as-is)")
print(f"scale:      {scale.shape}")

def add_npy(tf, key, arr):
    # key e.g. "poses_c2w.npy"  →  tar name "{scene_id}.{key}"
    buf = io.BytesIO(); np.save(buf, arr); raw = buf.getvalue()
    ti = tarfile.TarInfo(f"{scene_id}.{key}"); ti.size = len(raw)
    tf.addfile(ti, io.BytesIO(raw))

with tarfile.open(shard, "w") as tf:
    # video: {scene_id}.mp4
    vbytes = norm_video.read_bytes()
    ti = tarfile.TarInfo(f"{scene_id}.mp4"); ti.size = len(vbytes)
    tf.addfile(ti, io.BytesIO(vbytes))

    add_npy(tf, "poses_c2w.npy",  poses_c2w)
    add_npy(tf, "intrinsics.npy", intrinsics)   # (T, 1, 4) as-is
    add_npy(tf, "scale.npy",      scale)

    # caption: {scene_id}.caption.txt
    cbytes = caption_p.read_bytes()
    ti = tarfile.TarInfo(f"{scene_id}.caption.txt"); ti.size = len(cbytes)
    tf.addfile(ti, io.BytesIO(cbytes))

    # meta: {scene_id}.meta.json
    meta = {"scene_id": scene_id, "T": T, "mode": "default", "dataset": "OmniWorld"}
    mbytes = json.dumps(meta).encode()
    ti = tarfile.TarInfo(f"{scene_id}.meta.json"); ti.size = len(mbytes)
    tf.addfile(ti, io.BytesIO(mbytes))

print(f"Shard: {shard}  ({shard.stat().st_size/1e6:.1f} MB)")
with tarfile.open(shard) as tf:
    print(f"Contents: {[m.name for m in tf.getmembers()]}")
PYEOF

# ── Schema check ─────────────────────────────────────────────────────────────
echo ""
echo "=== Schema check ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir "${SHARDS_DIR}"

# ── Pose eval（与 gt_poses.npy 对比）─────────────────────────────────────────
echo ""
echo "=== Pose eval (vs gt_poses.npy) ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir "${SHARDS_DIR}" \
  --scenes-dir "${WORK_BASE}" \
  --out-dir "${SHARDS_DIR}/eval_output"

echo ""
echo "========================================================================"
echo " Default E2E 完成 ✓"
echo "  Scene:  ${SCENE_ID}"
echo "  Shard:  ${SHARDS_DIR}/shard-000001.tar"
echo "  Eval:   ${SHARDS_DIR}/eval_output/"
echo "========================================================================"
