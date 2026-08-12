#!/bin/bash
# 快速执行脚本 - Stage 3 冒烟测试

echo "=========================================="
echo "Stage 3 冒烟测试"
echo "=========================================="
echo ""

# 激活环境
echo "[1/4] 激活 conda 环境..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm_qc_env

# 设置环境变量
echo "[2/4] 设置环境变量..."
export TORCH_HOME=/root/work/david_work/cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 进入工作目录
cd /root/work/david_work/sana_qc_pipeline

# 运行验证脚本
echo "[3/4] 验证文件定位逻辑..."
python scripts/verify_file_location.py /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output

echo ""
echo "[4/4] 运行 Stage 3 冒烟测试..."
python scripts/run_stage3_cmcc.py \
  --stage1-jsonl /root/work/david_work/qc_output_new/smoke_test_manifest.jsonl \
  --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output-dir /root/work/david_work/sana_qc_pipeline/scripts/stage3_smoke_test \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "查看输出："
echo "  ls -lh /root/work/david_work/sana_qc_pipeline/scripts/stage3_smoke_test/"
echo ""
echo "查看结果："
echo "  cat /root/work/david_work/sana_qc_pipeline/scripts/stage3_smoke_test/stage3_worker000.jsonl | jq ."
