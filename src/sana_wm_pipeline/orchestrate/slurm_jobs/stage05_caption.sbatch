#!/bin/bash
#SBATCH --job-name=sanawm_caption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-999%100
#SBATCH --time=8:00:00
#SBATCH --output=logs/caption_%A_%a.log

set -euo pipefail
export SANA_WM_ROOT=/mnt/afs/davidwang/workspace/sana_wm_pipeline
export CONDA_ENV=/mnt/afs/davidwang/miniconda3/envs/sana_wm
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${SANA_WM_ROOT}

python -m sana_wm_pipeline.stage05_caption.run_caption_shard \
  --manifest manifests/stage04_passed.parquet \
  --shard ${SLURM_ARRAY_TASK_ID} \
  --num-shards 1000
