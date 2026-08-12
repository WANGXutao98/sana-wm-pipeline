#!/bin/bash
# ── 基础路径 ──────────────────────────────────────────────────────────────────
export NEW_BASE="${NEW_BASE:-/root/work/david_work}"
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"

# DATA_ROOT / OUT_BASE 可被启动脚本（双参数）外部 export 覆盖，未设时用默认
export DATA_ROOT="${DATA_ROOT:-/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full}"
export OUT_BASE="${OUT_BASE:-/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output}"

# ── 模型权重路径 ──────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"

# ── conda 环境激活（唯一真源；远程 SSH 也只 source 本文件）────────────────────
source "$NEW_BASE/activate_sana_wm.sh"

# ── 离线模式（CMCC 无外网）────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── 显存优化：减少 CUDA allocator 碎片 ────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 批次 1 优先 group（仅在不传 --groups 时作默认队列）────────────────────────
BATCH1_GROUPS=(
    "wds-sekai-real-walking-hq"
    "wds-DL3DV-ALL-2K"
    "wds-SpatialVID-hq"
)
export BATCH1_GROUPS

# 每个输出 shard 最多样本数
export SAMPLES_PER_OUTPUT_SHARD=200
