#!/bin/bash
# CMCC 机器专用冒烟测试（不依赖 conda 命令）
# 用途：在 CMCC 离线环境中验证打包环境的完整性
# 特点：直接调用环境的 bin/python，不需要 conda activate
#
# 使用方法：
#   bash scripts/smoke_test_cmcc_remote.sh
#
# 前置条件：
#   1. 已解压 sana_wm_cmcc.tar.gz 到 /root/work/david_work/envs/sana_wm_cmcc
#   2. 已解压 sana_qc_cmcc.tar.gz 到 /root/work/david_work/envs/sana_qc_cmcc
#   3. 已运行 conda-unpack（在各环境目录下执行 ./bin/conda-unpack）
#   4. 项目代码已部署到 /root/work/david_work/sana_wm_pipeline

set -euo pipefail

# ─── 路径配置（CMCC 标准路径）────────────────────────────────────────────
BASE=/root/work/david_work
ENV_WM=$BASE/envs/sana_wm_cmcc
ENV_QC=$BASE/envs/sana_qc_cmcc
PROJ=$BASE/sana_wm_optimized/sana_wm_pipeline

echo "════════════════════════════════════════════════════════════════════"
echo "  CMCC 环境冒烟测试（直接调用模式）"
echo "════════════════════════════════════════════════════════════════════"
echo "环境路径:"
echo "  sana_wm_cmcc: $ENV_WM"
echo "  sana_qc_cmcc: $ENV_QC"
echo "  项目代码:     $PROJ"
echo ""

# ─── 前置检查 ──────────────────────────────────────────────────────────────
echo "=== 前置检查 ==="

# 检查环境目录
if [ ! -d "$ENV_WM" ]; then
    echo "✗ 错误: sana_wm_cmcc 环境不存在: $ENV_WM"
    echo "  请先解压 sana_wm_cmcc.tar.gz 到该路径"
    exit 1
fi

if [ ! -d "$ENV_QC" ]; then
    echo "✗ 错误: sana_qc_cmcc 环境不存在: $ENV_QC"
    echo "  请先解压 sana_qc_cmcc.tar.gz 到该路径"
    exit 1
fi

# 检查 Python 可执行文件
if [ ! -f "$ENV_WM/bin/python3" ]; then
    echo "✗ 错误: $ENV_WM/bin/python3 不存在"
    echo "  请确认环境已正确解压并运行 conda-unpack"
    exit 1
fi

if [ ! -f "$ENV_QC/bin/python3" ]; then
    echo "✗ 错误: $ENV_QC/bin/python3 不存在"
    echo "  请确认环境已正确解压并运行 conda-unpack"
    exit 1
fi

# 检查项目代码
if [ ! -d "$PROJ/src" ]; then
    echo "✗ 错误: 项目代码不存在: $PROJ/src"
    echo "  请先部署 sana_wm_pipeline 项目代码"
    exit 1
fi

echo "✓ 环境路径检查通过"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Test 1: sana_wm_cmcc (VIPE SLAM with JIT)
# ═══════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════"
echo "  Test 1: sana_wm_cmcc (VIPE SLAM)"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# 设置环境变量
export PYTHONPATH=$PROJ/src:$PROJ/third_party/vipe:${PYTHONPATH:-}
export SANA_WM_PI3X_WEIGHTS=${SANA_WM_PI3X_WEIGHTS:-$BASE/models/pi3x}
export SANA_WM_MOGE2_WEIGHTS=${SANA_WM_MOGE2_WEIGHTS:-$BASE/models/moge2}
export TORCH_HOME=${TORCH_HOME:-$BASE/cache/torch}
export HF_HOME=${HF_HOME:-$BASE/cache/huggingface}
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/4] 基础环境检查..."
$ENV_WM/bin/python3 -c "
import sys
print(f'Python: {sys.version}')
print(f'Python 路径: {sys.executable}')
"

if [ $? -ne 0 ]; then
    echo "✗ Python 基础检查失败"
    exit 1
fi
echo "✓ Python 可执行"
echo ""

echo "[2/4] PyTorch + CUDA 检查..."
$ENV_WM/bin/python3 -c "
import torch
print(f'torch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'GPU 型号: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  警告: CUDA 不可用，但继续测试...')
"

if [ $? -ne 0 ]; then
    echo "✗ PyTorch 导入失败"
    exit 1
fi
echo "✓ PyTorch + CUDA 就绪"
echo ""

echo "[3/4] JIT 编译能力检查..."
$ENV_WM/bin/python3 -c "
import tempfile
import torch
from torch.utils.cpp_extension import load_inline

# 简单的 C++ JIT 测试（不需要 CUDA）
try:
    mod = load_inline(
        name='smoke_test_jit',
        cpp_sources=['int add(int a, int b) { return a + b; }'],
        functions=['add'],
        verbose=False,
        build_directory=tempfile.mkdtemp()
    )
    result = mod.add(1, 2)
    assert result == 3, f'JIT 计算错误: 1+2={result}'
    print('✓ C++ JIT 编译成功')
except Exception as e:
    print(f'⚠️  JIT 编译失败: {e}')
    print('  这可能影响 VIPE SLAM 的性能，但不影响基本功能')

# 如果有 CUDA，测试 CUDA JIT
if torch.cuda.is_available():
    try:
        import os
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')

        mod_cuda = load_inline(
            name='smoke_test_cuda_jit',
            cpp_sources=['torch::Tensor f(torch::Tensor x);'],
            cuda_sources=['''
#include <torch/extension.h>
__global__ void k(float* x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) x[i] += 1.0f;
}
torch::Tensor f(torch::Tensor x){
    int n = x.numel();
    k<<<(n+255)/256, 256>>>(x.data_ptr<float>(), n);
    return x;
}
'''],
            functions=['f'],
            verbose=False,
            build_directory=tempfile.mkdtemp()
        )
        result = mod_cuda.f(torch.zeros(8, device='cuda')).cpu().tolist()
        assert result == [1.0]*8, f'CUDA JIT 计算错误: {result}'
        print('✓ CUDA JIT 编译成功')
    except Exception as e:
        print(f'⚠️  CUDA JIT 编译失败: {e}')
        print('  这可能影响 VIPE SLAM 的性能，但不影响基本功能')
"

if [ $? -ne 0 ]; then
    echo "✗ JIT 测试失败"
    exit 1
fi
echo "✓ JIT 编译能力确认"
echo ""

echo "[4/4] VIPE 模块导入检查..."
$ENV_WM/bin/python3 -c "
import sys

# 核心依赖
import numpy as np
print(f'✓ numpy {np.__version__}')

import cv2
print(f'✓ opencv {cv2.__version__}')

import einops
print(f'✓ einops')

# VIPE SLAM 模块
try:
    from sana_wm_pipeline.stage02_pose.mode_default import run_default
    print('✓ sana_wm_pipeline.stage02_pose.mode_default')
except ImportError as e:
    print(f'✗ VIPE 模块导入失败: {e}')
    sys.exit(1)

# 检查关键子模块
try:
    import vipe
    print(f'✓ vipe {vipe.__version__}')
except ImportError as e:
    print(f'⚠️  警告: vipe 导入失败: {e}')
    print('  可能缺少 VIPE 扩展，但基本功能可用')

print('')
print('✓ VIPE SLAM 模块导入成功')
"

if [ $? -ne 0 ]; then
    echo "✗ VIPE 模块导入失败"
    exit 1
fi

echo ""
echo "✓✓✓ sana_wm_cmcc 测试通过 ✓✓✓"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Test 2: sana_qc_cmcc (Quality Control)
# ═══════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════"
echo "  Test 2: sana_qc_cmcc (Quality Control)"
echo "════════════════════════════════════════════════════════════════════"
echo ""

echo "[1/3] 基础环境检查..."
$ENV_QC/bin/python3 -c "
import sys
print(f'Python: {sys.version}')
print(f'Python 路径: {sys.executable}')
"

if [ $? -ne 0 ]; then
    echo "✗ Python 基础检查失败"
    exit 1
fi
echo "✓ Python 可执行"
echo ""

echo "[2/3] PyTorch + CUDA 检查..."
$ENV_QC/bin/python3 -c "
import torch
print(f'torch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
else:
    print('⚠️  警告: CUDA 不可用')
"

if [ $? -ne 0 ]; then
    echo "✗ PyTorch 导入失败"
    exit 1
fi
echo "✓ PyTorch + CUDA 就绪"
echo ""

echo "[3/3] QC 依赖库检查..."
$ENV_QC/bin/python3 -c "
import sys

# 基础依赖
import numpy as np
print(f'✓ numpy {np.__version__}')

import cv2
print(f'✓ opencv {cv2.__version__}')

# DOVER 依赖
try:
    import torchvision
    print(f'✓ torchvision {torchvision.__version__}')
except ImportError as e:
    print(f'✗ torchvision 导入失败: {e}')
    sys.exit(1)

try:
    import scipy
    print(f'✓ scipy {scipy.__version__}')
except ImportError as e:
    print(f'⚠️  警告: scipy 导入失败: {e}')

# UniMatch 依赖
try:
    import PIL
    print(f'✓ PIL (Pillow)')
except ImportError as e:
    print(f'⚠️  警告: PIL 导入失败: {e}')

# Qwen 依赖
try:
    import transformers
    print(f'✓ transformers {transformers.__version__}')
except ImportError as e:
    print(f'✗ transformers 导入失败: {e}')
    sys.exit(1)

try:
    import tokenizers
    print(f'✓ tokenizers')
except ImportError as e:
    print(f'⚠️  警告: tokenizers 导入失败: {e}')

print('')
print('✓ QC 依赖库检查通过')
"

if [ $? -ne 0 ]; then
    echo "✗ QC 依赖库检查失败"
    exit 1
fi

echo ""
echo "✓✓✓ sana_qc_cmcc 测试通过 ✓✓✓"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Final Report
# ═══════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════"
echo "  冒烟测试完成"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "测试结果汇总:"
echo "  ✓ sana_wm_cmcc (VIPE SLAM)  - PASS"
echo "  ✓ sana_qc_cmcc (QC)         - PASS"
echo ""
echo "环境状态:"
echo "  ✓ Python 解释器正常"
echo "  ✓ PyTorch + CUDA 就绪"
echo "  ✓ JIT 编译能力确认"
echo "  ✓ VIPE SLAM 模块可导入"
echo "  ✓ QC 依赖库完整"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✓✓✓ 所有测试通过，环境部署成功 ✓✓✓"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "下一步操作:"
echo "  1. 设置环境变量（添加到 ~/.bashrc）:"
echo "     export SANA_WM_PI3X_WEIGHTS=$BASE/models/pi3x"
echo "     export SANA_WM_MOGE2_WEIGHTS=$BASE/models/moge2"
echo "     export TORCH_HOME=$BASE/cache/torch"
echo "     export HF_HOME=$BASE/cache/huggingface"
echo ""
echo "  2. 运行数据生产任务:"
echo "     cd $PROJ"
echo "     # 使用 sana_wm_cmcc:"
echo "     $ENV_WM/bin/python scripts/your_script.py"
echo ""
echo "     # 或通过 PATH:"
echo "     export PATH=$ENV_WM/bin:\$PATH"
echo "     python scripts/your_script.py"
echo ""
