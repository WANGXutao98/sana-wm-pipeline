#!/bin/bash
# Stage3 冒烟测试 - 2s 分块版本

VIDEO="/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001/00eb7564-d5e8-54a1-b8bd-52ab85334924.mp4"
OUTPUT="/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl"

source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_qc
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 创建临时目录
TMP_DIR="/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_test"
mkdir -p "$TMP_DIR"
ln -sf "$VIDEO" "$TMP_DIR/test.mp4"

# 运行测试
python scripts/stage3_batch_minimal.py \
  --input_dir "$TMP_DIR" \
  --output "$OUTPUT" \
  --device cuda

# 输出结果
echo "=== 2s 分块版本结果 ==="
cat "$OUTPUT"
