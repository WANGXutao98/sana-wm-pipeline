#!/bin/bash
# 最终执行命令 - 复制到中移动机器使用

# ============================================================================
# 中移动机器 WebDataset 统计分析 - 执行命令
# ============================================================================

# 路径配置
DATA_DIR=~/work/filestorage/shangaoooooo/davidwang/repair_done
OUTPUT_DIR=~/work/analysis_results

# ============================================================================
# 执行分析
# ============================================================================

python3 analyze_wds_stats.py \
  --input-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --verbose

# ============================================================================
# 查看结果
# ============================================================================

echo ""
echo "=========================================="
echo "生成的报告文件："
echo "=========================================="
ls -lh "${OUTPUT_DIR}/"

echo ""
echo "=========================================="
echo "查看 Markdown 报告："
echo "=========================================="
cat "${OUTPUT_DIR}/comparison_report.md"

echo ""
echo "=========================================="
echo "查看 JSON 统计（格式化）："
echo "=========================================="
python3 -m json.tool "${OUTPUT_DIR}/dataset_statistics.json"

# ============================================================================
# 提取关键信息
# ============================================================================

echo ""
echo "=========================================="
echo "关键统计信息："
echo "=========================================="
python3 << 'EOPYTHON'
import json

with open("~/work/analysis_results/dataset_statistics.json".replace("~",
    __import__("os").path.expanduser("~"))) as f:
    data = json.load(f)

print(f"总样本数: {data['total_samples']:,}")
print(f"总数据量: {data['total_bytes'] / (1024**3):.2f} GB")
print(f"总 Shard 数: {data['total_shards']:,}")

for ds_name, ds_data in data['datasets'].items():
    print(f"\n【{ds_name}】")
    print(f"  样本数: {ds_data['total_samples']:,}")
    print(f"  数据量: {ds_data['size_gb']:.2f} GB")
    print(f"  Worker 数: {ds_data['workers']}")
EOPYTHON
