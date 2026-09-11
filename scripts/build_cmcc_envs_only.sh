#!/usr/bin/env bash
# Build CMCC conda environments ONLY (no code/weights/caches)
#
# Outputs (docker-images/out/):
#   sana_wm_cmcc.tar.gz     VIPE SLAM env with gcc13/nvcc
#   sana_qc_cmcc.tar.gz     QC env (DOVER/UniMatch/Qwen)
#
# Usage:
#   bash scripts/build_cmcc_envs_only.sh [--skip-wm] [--skip-qc]
#
# Time budget:
#   sana_wm_cmcc: ~45 min (clone 15min + gcc 10min + pack 15min)
#   sana_qc_cmcc: ~30 min (clone 15min + pack 15min)
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

  if [ $? -ne 0 ]; then
    echo "✗ 依赖检查失败，停止打包"
    exit 1
  fi

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
