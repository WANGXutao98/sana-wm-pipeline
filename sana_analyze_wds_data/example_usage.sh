#!/bin/bash
# WebDataset 统计分析工具 - 使用示例

echo "=========================================="
echo "WebDataset 统计分析工具 - 使用示例"
echo "=========================================="
echo ""

# 检查 Python 版本
echo "1. 检查 Python 版本..."
python3 --version
echo ""

# 基础用法示例
echo "2. 基础用法示例："
echo "   python3 analyze_wds_stats.py --input-dir /path/to/jdvbbfb_output"
echo ""

# 带对比的用法
echo "3. 与原始数据对比："
echo "   python3 analyze_wds_stats.py \\"
echo "     --input-dir /path/to/jdvbbfb_output \\"
echo "     --original-stats original_stats.json \\"
echo "     --output-dir ./analysis_results"
echo ""

# 详细模式
echo "4. 详细模式："
echo "   python3 analyze_wds_stats.py \\"
echo "     --input-dir /path/to/jdvbbfb_output \\"
echo "     --verbose"
echo ""

# 远程服务器使用
echo "5. 远程 GPU 服务器使用："
echo "   # 上传脚本到服务器"
echo "   scp -P 10523 analyze_wds_stats.py root@180.184.148.133:/mnt/afs/davidwang/workspace/"
echo ""
echo "   # 在服务器上运行"
echo "   ssh -p 10523 root@180.184.148.133"
echo "   cd /mnt/afs/davidwang/workspace"
echo "   python3 analyze_wds_stats.py --input-dir ./jdvbbfb_output"
echo ""

# 查看帮助
echo "6. 查看完整帮助："
echo "   python3 analyze_wds_stats.py --help"
echo ""

echo "=========================================="
echo "完整文档请参考 README_analyze_wds_stats.md"
echo "=========================================="
