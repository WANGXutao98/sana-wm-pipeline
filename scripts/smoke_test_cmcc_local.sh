#!/bin/bash
# 本地验证 CMCC 环境（在打包前）
set -euo pipefail

CONDA_BASE=/mnt/afs/davidwang/miniconda3
PROJ=/mnt/afs/davidwang/workspace/sana_wm_pipeline

echo "════════════════════════════════════════════════════════════════════"
echo "  本地 CMCC 环境冒烟测试"
echo "════════════════════════════════════════════════════════════════════"

# ─── Test 1: sana_wm_cmcc ──────────────────────────────────────────────────
echo ""
echo "=== Test 1: sana_wm_cmcc (VIPE SLAM) ==="
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate sana_wm_cmcc

export PYTHONPATH=$PROJ/src:${PYTHONPATH:-}
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface

python3 -c "
import torch
print(f'✓ torch {torch.__version__} (cuda: {torch.cuda.is_available()})')

# JIT 快速验证
import tempfile
from torch.utils.cpp_extension import load_inline
mod = load_inline(
    name='quick_jit',
    cpp_sources=['int add(int a, int b) { return a + b; }'],
    functions=['add'],
    verbose=False,
    build_directory=tempfile.mkdtemp()
)
assert mod.add(1, 2) == 3
print('✓ JIT compilation works')

# VIPE 导入测试
from sana_wm_pipeline.stage02_pose.mode_default import run_default
print('✓ VIPE imports work')
"

if [ $? -eq 0 ]; then
    echo "✓ sana_wm_cmcc PASS"
else
    echo "✗ sana_wm_cmcc FAIL"
    exit 1
fi

# ─── Test 2: sana_qc_cmcc ──────────────────────────────────────────────────
echo ""
echo "=== Test 2: sana_qc_cmcc (QC) ==="
conda activate sana_qc_cmcc

python3 -c "
import torch
print(f'✓ torch {torch.__version__} (cuda: {torch.cuda.is_available()})')

# QC 模块测试
try:
    import torchvision
    print(f'✓ torchvision {torchvision.__version__}')
except ImportError as e:
    print(f'✗ torchvision import failed: {e}')
    exit(1)

try:
    import transformers
    print(f'✓ transformers {transformers.__version__}')
except ImportError as e:
    print(f'✗ transformers import failed: {e}')
    exit(1)

print('✓ QC dependencies OK')
"

if [ $? -eq 0 ]; then
    echo "✓ sana_qc_cmcc PASS"
else
    echo "✗ sana_qc_cmcc FAIL"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✓✓✓ 本地冒烟测试全部通过 ✓✓✓"
echo "════════════════════════════════════════════════════════════════════"
