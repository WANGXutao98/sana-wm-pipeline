#!/bin/bash
# Stage 3 快速验证脚本
# 用途：在 CMCC 机器上验证所有代码是否正常工作

set -e  # 遇到错误立即退出

echo "============================================"
echo "Stage 3 代码验证脚本"
echo "============================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在 CMCC 机器上
if [ ! -d "/root/work" ]; then
    echo -e "${YELLOW}⚠️  警告：不在 CMCC 机器上，某些检查可能失败${NC}"
fi

# 步骤 1：检查 Python 环境
echo "[1/6] 检查 Python 环境..."
if command -v python &> /dev/null; then
    python_version=$(python --version 2>&1)
    echo -e "  ${GREEN}✓${NC} Python 版本: $python_version"
else
    echo -e "  ${RED}✗${NC} Python 未安装"
    exit 1
fi

# 步骤 2：检查依赖库
echo ""
echo "[2/6] 检查依赖库..."
dependencies=("torch" "numpy" "av" "transformers")
for dep in "${dependencies[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $dep"
    else
        echo -e "  ${RED}✗${NC} $dep 未安装"
    fi
done

# 步骤 3：检查脚本文件
echo ""
echo "[3/6] 检查脚本文件..."
scripts=(
    "scripts/data_loader_cmcc.py"
    "scripts/run_stage3_smoke_test.py"
    "scripts/run_stage3_cmcc_full.py"
    "scripts/stage3_worker.py"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo -e "  ${GREEN}✓${NC} $script"
    else
        echo -e "  ${RED}✗${NC} $script 不存在"
    fi
done

# 步骤 4：检查核心代码
echo ""
echo "[4/6] 检查核心代码..."
core_file="src/sana_wm_pipeline/qc/stage3_gpu.py"
if [ -f "$core_file" ]; then
    echo -e "  ${GREEN}✓${NC} $core_file 存在"

    # 检查 Qwen 思维链修复
    if grep -q "enable_thinking=False" "$core_file"; then
        echo -e "  ${GREEN}✓${NC} Qwen 思维链修复已应用"
    else
        echo -e "  ${YELLOW}⚠${NC}  Qwen 思维链修复未检测到（可能导致推理慢）"
    fi
else
    echo -e "  ${RED}✗${NC} $core_file 不存在"
fi

# 步骤 5：检查数据目录（CMCC 专用）
echo ""
echo "[5/6] 检查数据目录（CMCC 机器专用）..."
data_root="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"
if [ -d "$data_root" ]; then
    echo -e "  ${GREEN}✓${NC} 数据根目录存在: $data_root"

    # 检查是否有数据集目录
    dataset_count=$(find "$data_root" -maxdepth 1 -name "final_wds-*" -type d 2>/dev/null | wc -l)
    echo -e "  ${GREEN}✓${NC} 找到 $dataset_count 个数据集目录"

    # 检查 CSV 文件
    csv_file="$data_root/sample_completeness.csv"
    if [ -f "$csv_file" ]; then
        echo -e "  ${GREEN}✓${NC} sample_completeness.csv 存在"
        sample_count=$(wc -l < "$csv_file")
        echo -e "  ${GREEN}✓${NC} CSV 包含 $((sample_count - 1)) 个样本（不含表头）"
    else
        echo -e "  ${YELLOW}⚠${NC}  sample_completeness.csv 不存在"
    fi
else
    echo -e "  ${YELLOW}⚠${NC}  数据根目录不存在（跳过，可能不在 CMCC 机器上）"
fi

# 步骤 6：语法检查
echo ""
echo "[6/6] Python 语法检查..."
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        if python -m py_compile "$script" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $script 语法正确"
        else
            echo -e "  ${RED}✗${NC} $script 语法错误"
        fi
    fi
done

# 总结
echo ""
echo "============================================"
echo "验证完成！"
echo "============================================"
echo ""
echo "下一步："
echo "1. 在 CMCC 机器上，测试数据加载器："
echo "   python scripts/data_loader_cmcc.py $data_root $data_root/sample_completeness.csv"
echo ""
echo "2. 运行冒烟测试（需要 GPU）："
echo "   python scripts/run_stage3_smoke_test.py \\"
echo "     --sample-id \"<从数据加载器输出中选一个>\" \\"
echo "     --data-root $data_root \\"
echo "     --completeness-csv $data_root/sample_completeness.csv \\"
echo "     --gpu-id 0"
echo ""
echo "3. 小批量测试（100 样本，2 GPU）："
echo "   python scripts/run_stage3_cmcc_full.py \\"
echo "     --data-root $data_root \\"
echo "     --completeness-csv $data_root/sample_completeness.csv \\"
echo "     --output-dir /tmp/stage3_test \\"
echo "     --num-gpus 2 \\"
echo "     --max-samples 100"
echo ""
echo "详细文档："
echo "  - STAGE3_CMCC_执行指南_2026-08-07.md"
echo "  - STAGE3_CMCC_数据结构分析_2026-08-07.md"
echo ""
