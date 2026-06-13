#!/usr/bin/env bash
# GT-depth 模式端到端：OmniWorld 单场景 → WebDataset shard
#
# 用法：
#   bash experiments/data_production_smoke/run_e2e_gtdepth.sh \
#     /mnt/afs/davidwang/data/omniworld/OmniWorld-Game/<scene_id>
#
# 前置：
#   - conda env sana_wm 已激活（或由此脚本激活）
#   - SANA_WM_MOGE2_WEIGHTS 已指向 MoGe-2 权重
#   - 场景目录含 color/ depth/ camera/ fps.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 环境 ──────────────────────────────────────────────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export DISABLE_XFORMERS=1

SCENE_DIR="${1:?Usage: $0 <omniworld_scene_dir>}"
SCENE_ID="$(basename "${SCENE_DIR}")"
WORK_BASE="/mnt/afs/davidwang/workspace/data/omniworld_smoke"
PREP_DIR="${WORK_BASE}/${SCENE_ID}"
SHARDS_DIR="${WORK_BASE}/shards_gtdepth"
VIPE_WORK="${PREP_DIR}/vipe_work"

mkdir -p "${PREP_DIR}" "${SHARDS_DIR}" "${VIPE_WORK}"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo " GT-depth E2E: ${SCENE_ID}"
echo "========================================================================"

# ── Stage 0: 准备 OmniWorld 场景 ─────────────────────────────────────────────
echo ""
echo "=== Stage 0: prepare OmniWorld scene ==="
python experiments/data_production_smoke/prepare_omniworld.py \
  --scene-dir "${SCENE_DIR}" \
  --out-dir   "${PREP_DIR}"

VIDEO="${PREP_DIR}/video.mp4"
GT_DEPTH="${PREP_DIR}/gt_depth.npy"

if [ ! -f "${VIDEO}" ]; then
  echo "[ERROR] video.mp4 not found after prepare_omniworld.py" >&2; exit 1
fi
if [ ! -f "${GT_DEPTH}" ]; then
  echo "[ERROR] gt_depth.npy not found after prepare_omniworld.py" >&2; exit 1
fi

# ── Stage 1: normalize（统一分辨率/帧率）────────────────────────────────────
echo ""
echo "=== Stage 1: normalize video ==="
NORM_VIDEO="${PREP_DIR}/normalized.mp4"
if [ ! -f "${NORM_VIDEO}" ]; then
  python - <<PYEOF
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
info = normalize_video(Path("${VIDEO}"), Path("${NORM_VIDEO}"))
print(f"Normalized: {info.n_frames} frames @ {info.fps}fps  ({info.width}x{info.height})")
PYEOF
else
  echo "Already normalized: ${NORM_VIDEO}"
fi

# ── Stage 1b: 将 GT depth 重采样到归一化后帧率（16fps）──────────────────────
# normalize 后视频帧数与原始帧率不同，GT depth 必须按时间戳对齐
echo ""
echo "=== Stage 1b: resample GT depth to 16fps ==="
NORM_DEPTH="${PREP_DIR}/gt_depth_16fps.npy"
ORIG_FPS_FILE="${PREP_DIR}/orig_fps.txt"
if [ ! -f "${NORM_DEPTH}" ]; then
  python - <<PYEOF
import cv2, numpy as np
from pathlib import Path

# 归一化后的帧数
cap = cv2.VideoCapture("${NORM_VIDEO}")
T_norm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# 原始帧数 / fps
d_orig = np.load("${GT_DEPTH}")  # (T_orig, H, W)
T_orig = len(d_orig)
orig_fps_file = Path("${ORIG_FPS_FILE}")
orig_fps = float(orig_fps_file.read_text().strip()) if orig_fps_file.exists() else 30.0

# 按时间戳映射：norm 帧 i (16fps) → orig 帧 j
# j = round(i / 16.0 * orig_fps)
target_fps = 16.0
t_norm = np.arange(T_norm) / target_fps         # 归一化帧时间戳 (秒)
t_orig = np.arange(T_orig) / orig_fps            # 原始帧时间戳 (秒)
indices = np.round(np.interp(t_norm, t_orig, np.arange(T_orig))).astype(int)
indices = np.clip(indices, 0, T_orig - 1)
d_resampled = d_orig[indices]                    # (T_norm, H, W)

np.save("${NORM_DEPTH}", d_resampled)
print(f"GT depth resampled: {T_orig} frames@{orig_fps}fps -> {T_norm} frames@{target_fps}fps")
print(f"Saved: ${NORM_DEPTH}  shape={d_resampled.shape}")
PYEOF
else
  echo "Already resampled: ${NORM_DEPTH}"
fi

# ── Stage 2: GT-depth VIPE SLAM ─────────────────────────────────────────────
echo ""
echo "=== Stage 2: GT-depth VIPE SLAM (MoGe-2 + VIPE) ==="
ARTIFACT_JSON="${VIPE_WORK}/pose_artifact.json"

if [ ! -f "${ARTIFACT_JSON}" ]; then
  python - <<PYEOF
import json, numpy as np
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_gtdepth import run_gtdepth

clip     = Path("${NORM_VIDEO}")
gt_depth = Path("${NORM_DEPTH}")   # 16fps 对齐后的 GT depth
work_dir = Path("${VIPE_WORK}")
out_json = work_dir / "pose_artifact.json"

print(f"GT depth shape (16fps): {np.load(str(gt_depth)).shape}")
artifact = run_gtdepth(clip, gt_depth, work_dir)

print(f"Poses:      {artifact.poses_c2w.shape}")
print(f"Intrinsics: {artifact.intrinsics.shape}")
print(f"Scale:      mean={artifact.scale_per_frame.mean():.4f}  "
      f"std={artifact.scale_per_frame.std():.4f}")

out_json.write_text(json.dumps({
    "poses_c2w":       artifact.poses_c2w.tolist(),
    "intrinsics":      artifact.intrinsics.tolist(),
    "scale_per_frame": artifact.scale_per_frame.tolist(),
}))
print(f"Pose artifact: {out_json}")
PYEOF
else
  echo "pose_artifact.json already exists, skipping VIPE"
fi

# ── Stage 5: caption（stub）─────────────────────────────────────────────────
echo ""
echo "=== Stage 5: stub caption ==="
CAPTION_FILE="${PREP_DIR}/caption.txt"
if [ ! -f "${CAPTION_FILE}" ]; then
  echo "Indoor synthetic scene from OmniWorld-Game with ground-truth depth." \
    > "${CAPTION_FILE}"
fi

# ── Stage 6: pack → WebDataset shard ────────────────────────────────────────
echo ""
echo "=== Stage 6: pack WebDataset shard ==="
SHARD="${SHARDS_DIR}/shard-000001.tar"

python - <<PYEOF
import io, json, numpy as np, tarfile
from pathlib import Path

scene_id   = "${SCENE_ID}"
norm_video = Path("${NORM_VIDEO}")
vipe_work  = Path("${VIPE_WORK}")
gt_depth_p = Path("${GT_DEPTH}")
caption_p  = Path("${CAPTION_FILE}")
shards_dir = Path("${SHARDS_DIR}")
shard      = Path("${SHARD}")

art = json.loads((vipe_work / "pose_artifact.json").read_text())
poses_c2w      = np.array(art["poses_c2w"],       dtype=np.float32)
intrinsics_nvd = np.array(art["intrinsics"],       dtype=np.float32)
scale_pf       = np.array(art["scale_per_frame"],  dtype=np.float32)
T = len(poses_c2w)

gt_depth = np.load(str(gt_depth_p)).astype(np.float32)
gt_depth_ds = gt_depth[:T, ::4, ::4]

def add_npy(tf, name, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    raw = buf.getvalue()
    ti = tarfile.TarInfo(f"{scene_id}/{name}")
    ti.size = len(raw)
    tf.addfile(ti, io.BytesIO(raw))

with tarfile.open(shard, "w") as tf:
    vbytes = norm_video.read_bytes()
    ti = tarfile.TarInfo(f"{scene_id}/video.mp4"); ti.size = len(vbytes)
    tf.addfile(ti, io.BytesIO(vbytes))

    add_npy(tf, "poses_c2w.npy",        poses_c2w)
    add_npy(tf, "intrinsics_NVD.npy",   intrinsics_nvd)
    add_npy(tf, "scale_per_frame.npy",  scale_pf)
    add_npy(tf, "depth_downsampled.npy", gt_depth_ds)

    cbytes = caption_p.read_bytes()
    ti = tarfile.TarInfo(f"{scene_id}/caption.txt"); ti.size = len(cbytes)
    tf.addfile(ti, io.BytesIO(cbytes))

    meta = {"scene_id": scene_id, "T": T, "mode": "gt_depth", "dataset": "OmniWorld"}
    mbytes = json.dumps(meta).encode()
    ti = tarfile.TarInfo(f"{scene_id}/meta.json"); ti.size = len(mbytes)
    tf.addfile(ti, io.BytesIO(mbytes))

print(f"Shard: {shard}  ({shard.stat().st_size/1e6:.1f} MB)")
members = [m.name.split('/')[-1] for m in tarfile.open(shard).getmembers()]
print(f"Contents: {members}")
PYEOF

# ── Schema check ─────────────────────────────────────────────────────────────
echo ""
echo "=== Schema check ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir "${SHARDS_DIR}"

echo ""
echo "========================================================================"
echo " GT-depth E2E 完成 ✓"
echo "  Scene:  ${SCENE_ID}"
echo "  Shard:  ${SHARDS_DIR}/shard-000001.tar"
echo ""
echo "下一步（SANA-WM 推理）："
echo "  python experiments/data_production_smoke/run_sana_wm_inference.py \\"
echo "    --shards-dir ${SHARDS_DIR} \\"
echo "    --sana-dir /mnt/afs/davidwang/workspace/Sana \\"
echo "    --output-dir /mnt/afs/davidwang/workspace/data/sana_wm_results_gtdepth \\"
echo "    --sample-limit 1"
echo "========================================================================"
