#!/usr/bin/env bash
# 订正版 VIPE 对比实验：Method A (metric3d-small) vs Method B (Pi3X+MoGe-2 cached)
# 严格对齐 SANA-WM 论文 App. B.1
#
# 用法:
#   bash experiments/vipe_comparison/run_corrected.sh
#   bash experiments/vipe_comparison/run_corrected.sh fr2  # 仅跑 fr2/desk

set -euo pipefail

cd "$(dirname "$0")/../.."   # 切换到 sana_wm_pipeline/

export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

SEQ_FILTER=${1:-"all"}   # "fr1" / "fr2" / "all"

# ── 序列列表 ──────────────────────────────────────────────────────────────────
declare -A SEQ_VIDEOS=(
    [fr1]="experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4"
    [fr2]="experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk/video.mp4"
)
declare -A SEQ_CACHE=(
    [fr1]="experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz"
    [fr2]="experiments/vipe_comparison/results/cache_pi3x_moge2_fr2_desk.npz"
)
declare -A SEQ_RESULTS=(
    [fr1]="experiments/vipe_comparison/results"
    [fr2]="experiments/vipe_comparison/results_fr2"
)

for SEQ in fr1 fr2; do
    if [[ "$SEQ_FILTER" != "all" && "$SEQ_FILTER" != "$SEQ" ]]; then
        continue
    fi

    VIDEO="${SEQ_VIDEOS[$SEQ]}"
    CACHE="${SEQ_CACHE[$SEQ]}"
    RESULTS="${SEQ_RESULTS[$SEQ]}"

    if [[ ! -f "$VIDEO" ]]; then
        echo "[prepare] $SEQ: video not found, running prepare_tum.py..."
        python experiments/vipe_comparison/prepare_tum.py \
            --seq "freiburg${SEQ#fr}_desk" \
            --out experiments/vipe_comparison/data \
            2>&1 | tee "/tmp/prepare_${SEQ}.log"
    fi
    if [[ ! -f "$VIDEO" ]]; then
        echo "[SKIP] $SEQ: video still not found after prepare, skipping"
        continue
    fi

    echo "======================================================="
    echo "SEQ: $SEQ"
    echo "======================================================="

    # ── Step 1: 预计算深度缓存 ─────────────────────────────────────────────
    if [[ ! -f "$CACHE" ]]; then
        echo "[Step 1] Precomputing Pi3X+MoGe-2 depths for $SEQ..."
        python experiments/vipe_comparison/precompute_pi3x_depths.py \
            --video "$VIDEO" \
            --out   "$CACHE" \
            2>&1 | tee "/tmp/precompute_${SEQ}.log"
        echo "[Step 1] Done: $CACHE"
    else
        echo "[Step 1] Cache exists, skipping: $CACHE"
    fi

    # ── Step 2: Method A — VIPE + metric3d-small ───────────────────────────
    A_OUT="${RESULTS}/method_A_m3d"
    if [[ ! -f "${A_OUT}/pose/video.npz" ]]; then
        echo "[Step 2] Method A (metric3d-small) for $SEQ..."
        vipe infer "$VIDEO" \
            --pipeline vipe_metric3d_small \
            --output "${A_OUT}" \
            2>&1 | tee "/tmp/method_A_${SEQ}.log"
        echo "[Step 2] Done: ${A_OUT}"
    else
        echo "[Step 2] Method A result exists, skipping"
    fi

    # ── Step 3: Method B — VIPE + cached Pi3X+MoGe-2 ─────────────────────
    B_OUT="${RESULTS}/method_B_cached"
    if [[ ! -f "${B_OUT}/pose/video.npz" ]]; then
        echo "[Step 3] Method B (cached Pi3X+MoGe-2) for $SEQ..."
        export SANA_WM_CACHED_DEPTH_PATH="$CACHE"
        vipe infer "$VIDEO" \
            --pipeline vipe_cached_depth \
            --output "${B_OUT}" \
            2>&1 | tee "/tmp/method_B_${SEQ}.log"
        echo "[Step 3] Done: ${B_OUT}"
    else
        echo "[Step 3] Method B result exists, skipping"
    fi

    # ── Step 4: 评测 ────────────────────────────────────────────────────────
    echo "[Step 4] Evaluating $SEQ..."
    SEQ_DATA_DIR="experiments/vipe_comparison/data/rgbd_dataset_freiburg${SEQ#fr}_desk"
    python experiments/vipe_comparison/evaluate.py \
        --seq "$SEQ_DATA_DIR" \
        --results "$RESULTS" \
        2>&1 | tee "/tmp/eval_${SEQ}.log"

    echo "[Done] $SEQ complete."
done

echo ""
echo "All done. Logs in /tmp/precompute_*.log, /tmp/method_*.log, /tmp/eval_*.log"
