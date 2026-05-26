#!/bin/bash
# SLURM template for Stage-02 pose annotation — by far the heaviest stage.
# Configured for 64×H100 cluster (CMCC), modify --array bound to match
# total shard count.  See docker-images/cmcc/docs for cluster-specific paths.
#SBATCH --job-name=sanawm_pose
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --array=0-3999%200          # 200 concurrent tasks, 4000 shards
#SBATCH --time=24:00:00
#SBATCH --output=logs/pose_%A_%a.log
#SBATCH --error=logs/pose_%A_%a.err

set -euo pipefail

# Persistent paths only (machine has 3 h auto-reboot wiping /root/.local).
export SANA_WM_ROOT=/mnt/afs/davidwang/workspace/sana_wm_pipeline
export CONDA_ENV=/mnt/afs/davidwang/miniconda3/envs/sana_wm

source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${SANA_WM_ROOT}

python -m sana_wm_pipeline.stage02_pose.run_shard \
  --manifest manifests/stage01_passed.parquet \
  --shard ${SLURM_ARRAY_TASK_ID} \
  --num-shards 4000 \
  --out work/stage02/${SLURM_ARRAY_TASK_ID}
