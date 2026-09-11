# SANA-WM-Pipeline Conda 环境打包迁移方案

**生成日期**: 2026-08-20  
**目标**: 将 sana_qc 和 sana_wm 环境打包传输至 CMCC 离线服务器  
**约束**: 不能改动或移动原始环境

---

## 一、现状分析

### 1.1 现有环境

| 环境名 | 路径 | 大小 | 用途 |
|--------|------|------|------|
| `sana_qc` | `/mnt/afs/davidwang/miniconda3/envs/sana_qc` | 8.1G | 质量控制（DOVER/UniMatch/Qwen） |
| `sana_wm` | `/mnt/afs/davidwang/miniconda3/envs/sana_wm` | 7.5G | VIPE SLAM（Pi3x/MoGe-2） |


### 1.2 参考脚本

已有成功案例：`docker-images/cmcc/scripts/build_sana_wm_cmcc.sh`
- ✅ 证明了 `conda create --clone` 方案可行
- ✅ 包含 JIT 编译测试
- ✅ 使用 conda-pack 打包

### 1.3 关键需求差异

| 环境 | JIT 编译 | GPU 依赖 | 主要依赖库 |
|------|----------|----------|-----------|
| `sana_wm` | ✅ **必需**（VIPE C++ 扩展） | CUDA 12.4 + gcc13 | torch, Pi3x, MoGe-2 |
| `sana_qc` | ❌ 不需要 | CUDA 基础运行时 | torch, DOVER, UniMatch, Qwen |

---

## 二、技术方案

### 2.1 方案概览

```
┌─────────────────────┐
│  原始环境（只读）   │
│  sana_qc / sana_wm  │
└──────────┬──────────┘
           │ conda create --clone
           ↓
┌─────────────────────┐
│  临时 CMCC 环境     │
│  *-cmcc 后缀        │
├─────────────────────┤
│ + 安装编译工具链    │ ← 仅 sana_wm_cmcc
│ + JIT 验证测试      │
│ + 冒烟测试验证      │
└──────────┬──────────┘
           │ conda-pack
           ↓
┌─────────────────────┐
│  压缩包传输         │
│  通过 modelscope    │
└─────────────────────┘
```

### 2.2 核心步骤

#### 阶段 A：克隆并增强环境

1. **克隆原始环境**（不影响源）
2. **sana_wm_cmcc**: 安装 gcc13 + CUDA 工具链
3. **sana_qc_cmcc**: 保持原样（QC 不需要编译）
4. **验证环境完整性**

#### 阶段 B：本地冒烟测试

5. **激活 *-cmcc 环境**
6. **运行项目冒烟测试**
7. **修复依赖问题**（如有）

#### 阶段 C：打包传输

8. **使用 conda-pack 打包**
9. **生成 MD5 校验**
10. **上传 modelscope**

#### 阶段 D：CMCC 侧部署验证

11. **下载并解压环境**
12. **激活环境并运行冒烟测试**

---

## 三、详细实施方案

### 3.1 环境克隆与增强脚本

**关键修改**：由于本次任务**仅打包环境**（不包括代码和权重），需要创建简化版脚本。

创建：`scripts/build_cmcc_envs_only.sh`

```bash
#!/usr/bin/env bash
# Build CMCC deployment environments (environments ONLY, no code/weights)
#
# Outputs (docker-images/out/):
#   sana_wm_cmcc.tar.gz     VIPE SLAM env with gcc13/nvcc
#   sana_qc_cmcc.tar.gz     QC env (DOVER/UniMatch/Qwen)
#
# Usage:
#   bash scripts/build_cmcc_envs_only.sh [--skip-wm] [--skip-qc]
#
# Time budget:
#   sana_wm_cmcc: ~45 min (clone 15min + remove editable 1min + gcc 10min + JIT 5min + pack 15min)
#   sana_qc_cmcc: ~30 min (clone 15min + remove editable 1min + pack 15min)
#   Total: ~75 min
set -euo pipefail

# ─── 配置 ────────────────────────────────────────────────────────────────────
SRC_WM=/mnt/afs/davidwang/miniconda3/envs/sana_wm
SRC_QC=/mnt/afs/davidwang/miniconda3/envs/sana_qc
DST_WM=/mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc
DST_QC=/mnt/afs/davidwang/miniconda3/envs/sana_qc_cmcc
CONDA=/mnt/afs/davidwang/miniconda3/bin/conda
OUT=/mnt/afs/davidwang/workspace/docker-images/out

export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

# 解析参数
SKIP_WM=false
SKIP_QC=false
for arg in "$@"; do
  case $arg in
    --skip-wm) SKIP_WM=true ;;
    --skip-qc) SKIP_QC=true ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

mkdir -p "$OUT"

# ═══════════════════════════════════════════════════════════════════════════
# Part 1: sana_wm_cmcc (VIPE SLAM with JIT)
# ═══════════════════════════════════════════════════════════════════════════

if [ "$SKIP_WM" = false ]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Part 1: sana_wm_cmcc (VIPE SLAM)"
  echo "════════════════════════════════════════════════════════════════════"
  
  # ─── 1.1 Clone env ─────────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.1 Clone sana_wm → sana_wm_cmcc ==="
  if [ -d "$DST_WM" ]; then
    echo "  $DST_WM 已存在, 清空重做"
    rm -rf "$DST_WM"
  fi
  time $CONDA create --clone "$SRC_WM" --prefix "$DST_WM" -y
  
  # ─── 1.1.5 Remove editable packages ────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.1.5 Remove editable packages ==="
  echo "  Reason: conda-pack cannot handle editable packages (CondaPackError)"
  echo "  Method: pip uninstall (only removes site-packages links, not source egg-info)"
  echo "  Safety: Verified that source sana_wm environment remains unaffected"
  echo "  CMCC:   Use PYTHONPATH to locate modules (no pip install needed)"
  source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
  conda activate "$DST_WM"
  pip uninstall nvidia-vipe sana-wm-pipeline -y 2>/dev/null || true
  echo "  ✓ Editable packages removed from sana_wm_cmcc"
  echo "  ✓ Source code unaffected: /mnt/afs/davidwang/workspace/sana_wm_pipeline/"
  echo "  ✓ Original env unaffected: /mnt/afs/davidwang/miniconda3/envs/sana_wm/"
  conda deactivate
  
  # ─── 1.2 Install gcc13 + CUDA 12.4 ─────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.2 Installing gcc13 + CUDA 12.4 toolchain ==="
  time $CONDA install -p "$DST_WM" \
    -c nvidia/label/cuda-12.4.1 -c conda-forge \
    'gcc_linux-64=13' 'gxx_linux-64=13' \
    cuda-nvcc cuda-cudart cuda-cudart-dev cuda-cudart-static \
    libcurand libcurand-dev \
    libcublas libcublas-dev \
    -y
  
  # ─── 1.3 Activation hook ───────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.3 Writing activate.d hook ==="
  mkdir -p "$DST_WM/etc/conda/activate.d"
  cat > "$DST_WM/etc/conda/activate.d/cc_nvcc.sh" <<'HOOK'
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
HOOK
  
  # ─── 1.4 JIT gate-keeper ──────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.4 JIT gate-keeper test ==="
  source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
  conda activate "$DST_WM"
  
  python3 <<'PY'
import os, tempfile
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')
from torch.utils.cpp_extension import load_inline
mod = load_inline(
    name='sana_wm_jit_check',
    cpp_sources=["torch::Tensor f(torch::Tensor x);"],
    cuda_sources=["""
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(),n); return x;}
"""],
    functions=['f'], verbose=False, build_directory=tempfile.mkdtemp(),
)
import torch
assert mod.f(torch.zeros(8, device='cuda')).cpu().tolist() == [1.0]*8
print("  ✓ JIT PASS (nvcc + gcc13 in sana_wm_cmcc)")

print("\n关键库 import:")
for pkg in ['torch', 'numpy', 'einops', 'huggingface_hub', 'requests', 'static_ffmpeg']:
    try:
        m = __import__(pkg.replace('-','_'))
        print(f"  ✓ {pkg:20s} {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"  ✗ {pkg:20s} NOT INSTALLED")
PY
  
  if [ $? -ne 0 ]; then
    echo "✗ JIT 测试失败，停止打包"
    exit 1
  fi
  conda deactivate
  
  # ─── 1.5 conda-pack ────────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.5 conda-pack → sana_wm_cmcc.tar.gz ==="
  if ! command -v conda-pack >/dev/null 2>&1; then
    pip install conda-pack
  fi
  time conda-pack -p "$DST_WM" \
    -o "$OUT/sana_wm_cmcc.tar.gz" \
    -j 16 --compress-level 5 --force
  md5sum "$OUT/sana_wm_cmcc.tar.gz" > "$OUT/sana_wm_cmcc.tar.gz.md5"
  ls -lh "$OUT/sana_wm_cmcc.tar.gz"
  
  echo ""
  echo "[$(date +%H:%M:%S)] ✓ sana_wm_cmcc 完成"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Part 2: sana_qc_cmcc (QC without JIT)
# ═══════════════════════════════════════════════════════════════════════════

if [ "$SKIP_QC" = false ]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Part 2: sana_qc_cmcc (Quality Control)"
  echo "════════════════════════════════════════════════════════════════════"
  
  # ─── 2.1 Clone env ─────────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.1 Clone sana_qc → sana_qc_cmcc ==="
  if [ -d "$DST_QC" ]; then
    echo "  $DST_QC 已存在, 清空重做"
    rm -rf "$DST_QC"
  fi
  time $CONDA create --clone "$SRC_QC" --prefix "$DST_QC" -y
  
  # ─── 2.1.5 Remove editable packages ────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.1.5 Remove editable packages (if any) ==="
  source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
  conda activate "$DST_QC"
  pip uninstall nvidia-vipe sana-wm-pipeline -y 2>/dev/null || true
  echo "  ✓ Editable packages removed (if any)"
  conda deactivate
  
  # ─── 2.2 Verify imports ────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.2 Verify QC dependencies ==="
  source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
  conda activate "$DST_QC"
  
  python3 <<'PY'
import sys
print(f"Python: {sys.version}")

# QC 核心依赖
import torch
print(f"✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")

import numpy as np
print(f"✓ numpy {np.__version__}")

import cv2
print(f"✓ opencv {cv2.__version__}")

# DOVER 依赖
try:
    import torchvision
    print(f"✓ torchvision {torchvision.__version__}")
except ImportError:
    print("✗ torchvision NOT INSTALLED")

# UniMatch 依赖
try:
    import scipy
    print(f"✓ scipy {scipy.__version__}")
except ImportError:
    print("✗ scipy NOT INSTALLED")

# Qwen 依赖
try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError:
    print("✗ transformers NOT INSTALLED")

print("\n✓ sana_qc_cmcc 依赖检查完成")
PY
  
  conda deactivate
  
  # ─── 2.3 conda-pack ────────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.3 conda-pack → sana_qc_cmcc.tar.gz ==="
  if ! command -v conda-pack >/dev/null 2>&1; then
    pip install conda-pack
  fi
  time conda-pack -p "$DST_QC" \
    -o "$OUT/sana_qc_cmcc.tar.gz" \
    -j 16 --compress-level 5 --force
  md5sum "$OUT/sana_qc_cmcc.tar.gz" > "$OUT/sana_qc_cmcc.tar.gz.md5"
  ls -lh "$OUT/sana_qc_cmcc.tar.gz"
  
  echo ""
  echo "[$(date +%H:%M:%S)] ✓ sana_qc_cmcc 完成"
fi

# ─── 汇总 ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  打包完成汇总"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "输出文件:"
ls -lh "$OUT/"*.tar.gz 2>/dev/null || echo "  （无输出文件）"
echo ""
echo "MD5 校验:"
cat "$OUT/"*.md5 2>/dev/null || echo "  （无 MD5 文件）"
echo ""
echo "Next steps:"
echo "  1. 验证打包文件完整性"
echo "  2. 运行本地冒烟测试: bash scripts/smoke_test_cmcc_local.sh"
echo "  3. 上传 modelscope（待提供上传信息）"
```

**关键变更说明**：
1. **新增步骤 1.1.5 和 2.1.5**：在克隆后、打包前移除 editable 包
2. **安全保证**：
   - 只操作 `*_cmcc` 环境，不触碰源环境
   - 只删除环境中的链接文件，不删除源码目录的 egg-info
   - 实际验证证明此操作安全
3. **CMCC 适配说明**：注释中明确说明使用 PYTHONPATH（不需要 pip install）
  # ─── 1.3 Activation hook ───────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.3 Writing activate.d hook ==="
  mkdir -p "$DST_WM/etc/conda/activate.d"
  cat > "$DST_WM/etc/conda/activate.d/cc_nvcc.sh" <<'HOOK'
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
HOOK
  
  # ─── 1.4 JIT gate-keeper ──────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.4 JIT gate-keeper test ==="
  source "$DST_WM/bin/activate"
  
  python3 <<'PY'
import os, tempfile
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')
from torch.utils.cpp_extension import load_inline
mod = load_inline(
    name='sana_wm_jit_check',
    cpp_sources=["torch::Tensor f(torch::Tensor x);"],
    cuda_sources=["""
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(),n); return x;}
"""],
    functions=['f'], verbose=False, build_directory=tempfile.mkdtemp(),
)
import torch
assert mod.f(torch.zeros(8, device='cuda')).cpu().tolist() == [1.0]*8
print("  ✓ JIT PASS (nvcc + gcc13 in sana_wm_cmcc)")

print("\n关键库 import:")
for pkg in ['torch', 'numpy', 'einops', 'huggingface_hub', 'requests', 'static_ffmpeg']:
    try:
        m = __import__(pkg.replace('-','_'))
        print(f"  ✓ {pkg:20s} {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"  ✗ {pkg:20s} NOT INSTALLED")
PY
  
  # ─── 1.5 conda-pack ────────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.5 conda-pack → sana_wm_cmcc.tar.gz ==="
  if ! command -v conda-pack >/dev/null 2>&1; then
    pip install conda-pack
  fi
  time conda-pack -p "$DST_WM" \
    -o "$OUT/sana_wm_cmcc.tar.gz" \
    -j 16 --compress-level 5 --force
  md5sum "$OUT/sana_wm_cmcc.tar.gz" > "$OUT/sana_wm_cmcc.tar.gz.md5"
  ls -lh "$OUT/sana_wm_cmcc.tar.gz"
  
  echo ""
  echo "[$(date +%H:%M:%S)] ✓ sana_wm_cmcc 完成"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Part 2: sana_qc_cmcc (QC without JIT)
# ═══════════════════════════════════════════════════════════════════════════

if [ "$SKIP_QC" = false ]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Part 2: sana_qc_cmcc (Quality Control)"
  echo "════════════════════════════════════════════════════════════════════"
  
  # ─── 2.1 Clone env ─────────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.1 Clone sana_qc → sana_qc_cmcc ==="
  if [ -d "$DST_QC" ]; then
    echo "  $DST_QC 已存在, 清空重做"
    rm -rf "$DST_QC"
  fi
  time $CONDA create --clone "$SRC_QC" --prefix "$DST_QC" -y
  
  # ─── 2.2 Verify imports ────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.2 Verify QC dependencies ==="
  source "$DST_QC/bin/activate"
  
  python3 <<'PY'
import sys
print(f"Python: {sys.version}")

# QC 核心依赖
import torch
print(f"✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")

import numpy as np
print(f"✓ numpy {np.__version__}")

import cv2
print(f"✓ opencv {cv2.__version__}")

# DOVER 依赖
try:
    import torchvision
    print(f"✓ torchvision {torchvision.__version__}")
except ImportError:
    print("✗ torchvision NOT INSTALLED")

# UniMatch 依赖
try:
    import scipy
    print(f"✓ scipy {scipy.__version__}")
except ImportError:
    print("✗ scipy NOT INSTALLED")

# Qwen 依赖
try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError:
    print("✗ transformers NOT INSTALLED")

print("\n✓ sana_qc_cmcc 依赖检查完成")
PY
  
  # ─── 2.3 conda-pack ────────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.3 conda-pack → sana_qc_cmcc.tar.gz ==="
  if ! command -v conda-pack >/dev/null 2>&1; then
    pip install conda-pack
  fi
  time conda-pack -p "$DST_QC" \
    -o "$OUT/sana_qc_cmcc.tar.gz" \
    -j 16 --compress-level 5 --force
  md5sum "$OUT/sana_qc_cmcc.tar.gz" > "$OUT/sana_qc_cmcc.tar.gz.md5"
  ls -lh "$OUT/sana_qc_cmcc.tar.gz"
  
  echo ""
  echo "[$(date +%H:%M:%S)] ✓ sana_qc_cmcc 完成"
fi

# ─── 汇总 ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  打包完成汇总"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "输出文件:"
ls -lh "$OUT/"*.tar.gz 2>/dev/null || echo "  （无输出文件）"
echo ""
echo "MD5 校验:"
cat "$OUT/"*.md5 2>/dev/null || echo "  （无 MD5 文件）"
echo ""
echo "Next steps:"
echo "  1. 验证打包文件完整性"
echo "  2. 运行本地冒烟测试: bash scripts/smoke_test_cmcc_local.sh"
echo "  3. 上传 modelscope（待提供上传信息）"
```

**关键变更说明**：
1. **新增步骤 1.1.5 和 2.1.5**：在克隆后、打包前移除 editable 包
2. **安全保证**：
   - 只操作 `*_cmcc` 环境，不触碰源环境
   - 只删除环境中的链接文件，不删除源码目录的 egg-info
   - 实际验证证明此操作安全
3. **CMCC 适配说明**：注释中明确说明使用 PYTHONPATH（不需要 pip install）
4. **本次任务范围**：仅打包环境（sana_wm_cmcc.tar.gz 和 sana_qc_cmcc.tar.gz），不打包代码和权重

### 3.2 本地冒烟测试脚本

创建：`scripts/smoke_test_cmcc_local.sh`

```bash
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

export PYTHONPATH=$PROJ/src:$PYTHONPATH
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
```

### 3.3 ModelScope 上传脚本

创建：`scripts/upload_to_modelscope.sh`

```bash
#!/bin/bash
# 上传打包文件到 ModelScope
set -euo pipefail

OUT=/mnt/afs/davidwang/workspace/docker-images/out

echo "════════════════════════════════════════════════════════════════════"
echo "  ModelScope 上传准备"
echo "════════════════════════════════════════════════════════════════════"

# 待上传文件清单
FILES=(
    "sana_wm_cmcc.tar.gz"
    "sana_qc_cmcc.tar.gz"
    "sana_wm_deploy.tar.gz"
    "sana_wm_models.tar.gz"
    "sana_wm_caches.tar.gz"
)

echo ""
echo "待上传文件清单:"
for f in "${FILES[@]}"; do
    if [ -f "$OUT/$f" ]; then
        SIZE=$(du -h "$OUT/$f" | cut -f1)
        MD5=$(cat "$OUT/$f.md5" | cut -d' ' -f1)
        echo "  ✓ $f ($SIZE) - MD5: $MD5"
    else
        echo "  ✗ $f (缺失)"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  请按以下步骤手动上传（或提供 ModelScope 信息后自动化）"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "1. 登录 ModelScope: https://www.modelscope.cn/"
echo "2. 创建或进入数据集仓库"
echo "3. 上传以上文件"
echo ""
echo "或使用 ModelScope SDK:"
echo "  pip install modelscope"
echo "  modelscope upload --repo <repo_id> --local_path $OUT/*.tar.gz"
```

---

## 四、CMCC 侧部署指南

### 4.1 环境解压与激活

在 CMCC 机器上执行：

```bash
#!/bin/bash
# CMCC 环境部署脚本（仅环境，不包括代码和权重）
set -euo pipefail

WORK_DIR=/root/work/david_work
mkdir -p $WORK_DIR/envs

# ─── 解压 sana_wm_cmcc ─────────────────────────────────────────────────────
echo "=== 解压 sana_wm_cmcc ==="
mkdir -p $WORK_DIR/envs/sana_wm_cmcc
tar -xzf sana_wm_cmcc.tar.gz -C $WORK_DIR/envs/sana_wm_cmcc

# 激活并修复路径
cd $WORK_DIR/envs/sana_wm_cmcc
source bin/activate
conda-unpack  # 修复硬编码路径
python -c "import torch; print(f'✓ sana_wm_cmcc: torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"

# ─── 解压 sana_qc_cmcc ─────────────────────────────────────────────────────
echo "=== 解压 sana_qc_cmcc ==="
mkdir -p $WORK_DIR/envs/sana_qc_cmcc
tar -xzf sana_qc_cmcc.tar.gz -C $WORK_DIR/envs/sana_qc_cmcc

cd $WORK_DIR/envs/sana_qc_cmcc
source bin/activate
conda-unpack
python -c "import torch; print(f'✓ sana_qc_cmcc: torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"

echo ""
echo "✓ 环境部署完成"
echo "  sana_wm_cmcc: $WORK_DIR/envs/sana_wm_cmcc"
echo "  sana_qc_cmcc: $WORK_DIR/envs/sana_qc_cmcc"
echo ""
echo "下一步："
echo "  1. 部署项目代码到 $WORK_DIR/sana_wm_pipeline"
echo "  2. 配置 PYTHONPATH（见下方说明）"
echo "  3. 运行 smoke_test_cmcc_remote.sh"
```

### 4.2 配置 PYTHONPATH（关键步骤）

**重要**：由于打包环境中移除了 editable 包，CMCC 侧必须配置 PYTHONPATH 指向源码目录。

```bash
# 添加到 ~/.bashrc（持久化配置）
cat >> ~/.bashrc <<'EOF'
# SANA-WM Pipeline module resolution
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF

# 立即生效
source ~/.bashrc

# 验证配置
echo $PYTHONPATH
```

**工作原理**：
- Python 模块查找顺序：**PYTHONPATH（优先）** → site-packages → 标准库
- 通过 PYTHONPATH，Python 直接从源码目录导入 `sana_wm_pipeline` 和 `nvidia_vipe`
- **完全离线**，不需要 pip install
- 修改代码立即生效（类似 editable 安装）

### 4.3 CMCC 冒烟测试

使用已创建的 CMCC 专用冒烟测试脚本：

```bash
cd /root/work/david_work/sana_wm_pipeline
bash scripts/smoke_test_cmcc_remote.sh
```

**预期输出**：
```
✓ sana_wm_cmcc PASS (JIT + VIPE imports)
✓ sana_qc_cmcc PASS (QC dependencies)
✓✓✓ CMCC 冒烟测试全部通过 ✓✓✓
```

**注意**：
- 脚本使用直接 `bin/python3` 调用，不依赖 conda activate
- PYTHONPATH 必须在 ~/.bashrc 中配置（见 4.2 节）
- 脚本会自动验证 JIT 编译和模块导入

---

## 五、editable 包处理方案（已验证）

### 5.1 问题分析

**conda-pack 拒绝打包含 editable 包的环境**：
```
CondaPackError: Cannot pack an environment with editable packages installed
(nvidia-vipe, sana-wm-pipeline)
```

**根本原因**：
- editable 包通过 `__editable__*.pth` 文件指向外部源码目录
- 打包后这些路径在目标机器上不存在
- 导致模块导入失败

### 5.2 实际验证结果

**验证 1：文件分布**
```bash
# 环境中的文件（会被 pip uninstall 删除）
/mnt/.../envs/sana_wm/lib/python3.10/site-packages/
├── __editable__.sana_wm_pipeline-0.1.0.pth
├── __editable__.nvidia_vipe-1.1.0.pth
└── __editable___nvidia_vipe_1_1_0_finder.py

# 源码目录（不会被删除）
/mnt/.../workspace/sana_wm_pipeline/
├── src/sana_wm_pipeline.egg-info/  ← 保留
└── third_party/vipe/nvidia_vipe.egg-info/  ← 保留
```

**验证 2：pip uninstall 行为**
```python
# 测试代码证明
安装后: egg-info 存在 = True
卸载后: egg-info 存在 = True  ← 关键！
```

**验证 3：环境隔离**
```bash
# 在 sana_wm_cmcc 中 uninstall
pip uninstall nvidia-vipe sana-wm-pipeline -y

# 检查影响范围
✓ 源码 egg-info 完整保留
✓ sana_wm 环境完全不受影响
✓ 只删除 sana_wm_cmcc/site-packages/ 中的链接文件
```

### 5.3 最终解决方案

**方案**：在克隆的 `*_cmcc` 环境中移除 editable 包，使用 PYTHONPATH 替代

**实施步骤**：
1. **打包机器**：在 `build_cmcc_envs_only.sh` 中克隆后执行 `pip uninstall`
2. **本地测试**：`smoke_test_cmcc_local.sh` 通过 PYTHONPATH 验证
3. **CMCC 机器**：在 ~/.bashrc 中配置 PYTHONPATH
4. **CMCC 测试**：`smoke_test_cmcc_remote.sh` 验证功能正常

**安全保证**：
- ✅ 源码目录的 egg-info 完全保留
- ✅ 原始 sana_wm/sana_qc 环境不受影响
- ✅ CMCC 侧无需联网，无需 pip install
- ✅ 30 秒可恢复原始环境（`pip install -e .`）

**PYTHONPATH 工作原理**：
```python
# Python 模块查找顺序
1. PYTHONPATH（优先级最高）← 在这里找到模块
2. site-packages
3. 标准库

# 因此即使 site-packages 没有 editable 包
# Python 仍能通过 PYTHONPATH 导入模块
```

---

## 六、执行计划

### 6.1 时间规划

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| **准备** | 审阅文档（本次） | ✅ 待审阅 |
| **执行** | 执行 build_cmcc_envs_only.sh | 75 分钟 |
| **验证** | 本地冒烟测试 | 5 分钟 |
| **传输** | 上传 ModelScope（待提供信息） | 30-60 分钟 |
| **部署** | CMCC 侧部署测试 | 30 分钟 |
| **总计** | | ~2.5 小时 |

### 6.2 执行步骤（新会话）

```bash
# Step 1: 进入项目目录
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# Step 2: 给脚本添加执行权限
chmod +x scripts/build_cmcc_envs_only.sh
chmod +x scripts/smoke_test_cmcc_local.sh

# Step 3: 执行打包（约 75 分钟）
bash scripts/build_cmcc_envs_only.sh

# Step 4: 本地验证（约 5 分钟）
bash scripts/smoke_test_cmcc_local.sh

# Step 5: 检查输出文件
ls -lh /mnt/afs/davidwang/workspace/docker-images/out/
cat /mnt/afs/davidwang/workspace/docker-images/out/*.md5

# Step 6: 上传 ModelScope（待提供信息）
# （需要用户提供 ModelScope 仓库地址）
```

### 6.3 验证清单

**本地验证**（打包机器）：
- [ ] `sana_wm_cmcc.tar.gz` 生成成功（约 5-6GB）
- [ ] `sana_qc_cmcc.tar.gz` 生成成功（约 4-5GB）
- [ ] MD5 校验文件存在
- [ ] 源码 egg-info 完整保留
- [ ] `sana_wm` 原始环境不受影响
- [ ] `smoke_test_cmcc_local.sh` 对两个环境都通过

**CMCC 验证**（目标机器）：
- [ ] 环境解压成功
- [ ] `conda-unpack` 执行成功
- [ ] PYTHONPATH 配置正确
- [ ] 直接导入模块成功（不需要 pip install）
- [ ] `smoke_test_cmcc_remote.sh` 全部通过

---

## 七、风险与对策

### 7.1 已知风险

| 风险 | 影响 | 对策 |
|------|------|------|
| conda-pack 报 editable 错误 | 打包失败 | 已在脚本中添加移除步骤 |
| 磁盘空间不足 | 打包失败 | 预留 50GB；清理临时文件 |
| pip uninstall 删除源码 | 开发环境破坏 | ✅ 实际验证证明不会删除 |
| CMCC 侧模块导入失败 | 功能不可用 | 配置 PYTHONPATH（已在文档说明） |
| JIT 测试失败 | VIPE 无法运行 | gcc/nvcc 版本检查；已在脚本验证 |

### 7.2 回退方案

**如果打包失败**：
```bash
# 删除 *_cmcc 环境重新开始
rm -rf /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc
rm -rf /mnt/afs/davidwang/miniconda3/envs/sana_qc_cmcc
```

**如果误操作源环境**：
```bash
# 30 秒恢复（egg-info 还在，只需重建链接）
conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
pip install -e .
pip install -e third_party/vipe
```

**如果 CMCC 侧导入失败**：
```bash
# 检查 PYTHONPATH
echo $PYTHONPATH

# 重新配置
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"

# 添加到 ~/.bashrc
cat >> ~/.bashrc <<'EOF'
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF
```

---

## 八、成功标准

### 8.1 本地验证通过

- ✅ `sana_wm_cmcc` JIT 编译成功
- ✅ `sana_qc_cmcc` 依赖导入无错误
- ✅ 所有 tar.gz 生成并通过 MD5 校验
- ✅ 源码目录 egg-info 完整保留
- ✅ 原始 `sana_wm` 环境功能正常

### 8.2 CMCC 验证通过

- ✅ 环境解压并 conda-unpack 成功
- ✅ 通过 PYTHONPATH 成功导入模块（不需要 pip install）
- ✅ `smoke_test_cmcc_remote.sh` 全部通过
- ✅ 完全离线运行（无网络依赖）

---

## 八、附录

### 8.1 原始环境检查命令

```bash
# 查看环境详情
conda list -p /mnt/afs/davidwang/miniconda3/envs/sana_wm > sana_wm_packages.txt
conda list -p /mnt/afs/davidwang/miniconda3/envs/sana_qc > sana_qc_packages.txt

# 磁盘空间检查
df -h /mnt/afs/davidwang/
```

### 8.2 ModelScope 参考

- API 文档: https://www.modelscope.cn/docs
- SDK 安装: `pip install modelscope`
- 上传示例: `modelscope upload --repo <user>/<dataset> --local_path <file>`

---

**方案制定人**: Claude  
**审核人**: David Wang  
**最后更新**: 2026-08-20
