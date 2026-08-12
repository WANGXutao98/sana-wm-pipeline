#!/bin/bash
# 中移动机器上直接运行此脚本
# 用途：统计 final_wds-SpatialVID-hq 的数据量和样本数

echo "=========================================="
echo "WebDataset 数据统计"
echo "=========================================="
echo ""

# 数据路径
DATA_DIR=~/work/filestorage/shangaoooooo/davidwang/repair_done
OUTPUT_DIR=~/work/analysis_results

echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 检查工具是否存在
if [ ! -f "analyze_wds_stats.py" ]; then
    echo "❌ 错误: 找不到 analyze_wds_stats.py"
    echo "请先上传此文件到当前目录"
    exit 1
fi

# 检查数据目录是否存在
if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ 错误: 数据目录不存在: ${DATA_DIR}"
    exit 1
fi

echo "开始分析..."
echo ""

# 运行分析
python3 analyze_wds_stats.py \
  --input-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --verbose

echo ""
echo "=========================================="
echo "分析完成！"
echo "=========================================="
echo ""
echo "生成的报告文件："
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "查看 Markdown 报告："
echo "  cat ${OUTPUT_DIR}/comparison_report.md"
echo ""
echo "查看 JSON 统计："
echo "  cat ${OUTPUT_DIR}/dataset_statistics.json | python3 -m json.tool"
echo ""
