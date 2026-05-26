#!/bin/bash
#SBATCH --job-name=sanawm_normalize
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-999%50
#SBATCH --time=4:00:00
#SBATCH --output=logs/normalize_%A_%a.log

set -euo pipefail
export SANA_WM_ROOT=/mnt/afs/davidwang/workspace/sana_wm_pipeline
export CONDA_ENV=/mnt/afs/davidwang/miniconda3/envs/sana_wm
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${SANA_WM_ROOT}

python -m sana_wm_pipeline.stage01_ingest.run_normalize_shard \
  --manifest manifests/raw.parquet \
  --shard ${SLURM_ARRAY_TASK_ID} \
  --num-shards 1000
