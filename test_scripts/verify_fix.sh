#!/bin/bash
# 快速验证 Qwen 思维链修复是否生效

set -e

echo "=========================================="
echo "Qwen 思维链修复验证脚本"
echo "=========================================="
echo ""

# 检查修改是否应用
echo "[1] 检查代码修改..."
if grep -q "enable_thinking=False" /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py; then
    echo "✅ 代码已修改"
else
    echo "❌ 代码未修改，请先运行修复"
    exit 1
fi

# 显示修改内容
echo ""
echo "[2] 修改内容预览:"
echo "---"
grep -A 5 "enable_thinking=False" /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py | head -6
echo "---"

# 运行测试脚本
echo ""
echo "[3] 运行验证测试..."
echo "测试 1: 禁用思维链效果验证"
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/test_scripts
python test_qwen_disable_thinking.py 2>&1 | tee /tmp/verify_fix_output.log

# 检查结果
echo ""
echo "[4] 结果分析..."
if grep -q "禁用思维链.*思维链: ✅ 无" /tmp/verify_fix_output.log; then
    echo "✅ 修复成功！思维链已被抑制"

    # 提取性能数据
    echo ""
    echo "性能对比:"
    grep -E "(启用思维链|禁用思维链|加速比)" /tmp/verify_fix_output.log | tail -8
else
    echo "⚠️  请检查测试输出"
fi

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 如果测试通过，在 CMCC 机器上小批量测试"
echo "2. 预期性能提升: 5-10x"
echo "3. 详细报告: /tmp/verify_fix_output.log"
