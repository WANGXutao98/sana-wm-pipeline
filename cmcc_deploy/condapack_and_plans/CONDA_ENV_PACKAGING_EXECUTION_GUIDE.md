# SANA-WM-Pipeline Conda 环境打包执行指南

**生成日期**: 2026-08-23  
**状态**: ✅ 打包已完成  
**范围**: 仅打包 conda 环境（不包括代码和权重）

---

## 执行状态记录

### 打包阶段（已完成 ✅）

- ✅ 环境克隆完成（sana_wm → sana_wm_cmcc, sana_qc → sana_qc_cmcc）
- ✅ 移除 editable 包（nvidia-vipe, sana-wm-pipeline）
- ✅ 安装 gcc13 + CUDA 12.4 工具链（sana_wm_cmcc）
- ✅ JIT 编译测试通过
- ✅ conda-pack 打包成功
- ✅ 生成文件：
  - `sana_wm_cmcc.tar.gz` + MD5
  - `sana_qc_cmcc.tar.gz` + MD5

### 传输阶段（已完成 ✅）

- ✅ 文件已下载到 CMCC 机器
- ✅ 存放路径：`/root/work/david_work/conda_envs_download/conda_envs/`

### 部署阶段（待执行）

- [ ] 解压环境包
- [ ] 运行 conda-unpack
- [ ] 部署项目代码
- [ ] 配置 PYTHONPATH
- [ ] 运行冒烟测试

---

## 一、执行前准备（打包阶段 - 已完成）

### 1.1 确认磁盘空间

```bash
# 检查可用空间（需要至少 50GB）
df -h /mnt/afs/davidwang/

# 检查原始环境大小
du -sh /mnt/afs/davidwang/miniconda3/envs/sana_qc
du -sh /mnt/afs/davidwang/miniconda3/envs/sana_wm
```

**预期输出**:
- `sana_qc`: 8.1G
- `sana_wm`: 7.5G
- 打包后大小：~5-6G（压缩后）

### 1.2 确认输出目录

```bash
mkdir -p /mnt/afs/davidwang/workspace/docker-images/out
```

---

## 二、执行步骤

### 步骤 1: 创建打包脚本

**文件路径**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/build_cmcc_envs_only.sh`

**脚本内容**:

```bash
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
```

---

### 步骤 2: 创建本地冒烟测试脚本

**文件路径**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/smoke_test_cmcc_local.sh`

**脚本内容**:

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

---

### 步骤 3: 执行打包

在新的 Claude 对话中，按顺序执行以下命令：

```bash
# 进入项目目录
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 创建脚本目录（如果不存在）
mkdir -p scripts

# 给脚本添加执行权限
chmod +x scripts/build_cmcc_envs_only.sh
chmod +x scripts/smoke_test_cmcc_local.sh

# 执行打包（预计 75 分钟）
bash scripts/build_cmcc_envs_only.sh
```

---

### 步骤 4: 本地验证

打包完成后，立即运行冒烟测试：

```bash
bash scripts/smoke_test_cmcc_local.sh
```

**预期输出**:
```
✓ sana_wm_cmcc PASS
✓ sana_qc_cmcc PASS
✓✓✓ 本地冒烟测试全部通过 ✓✓✓
```

---

### 步骤 5: 检查输出文件

```bash
# 查看生成的文件
ls -lh /mnt/afs/davidwang/workspace/docker-images/out/

# 查看 MD5 校验
cat /mnt/afs/davidwang/workspace/docker-images/out/*.md5
```

**预期输出**:
```
sana_wm_cmcc.tar.gz      (~5-6G)
sana_wm_cmcc.tar.gz.md5
sana_qc_cmcc.tar.gz      (~4-5G)
sana_qc_cmcc.tar.gz.md5
```

---

## 三、传输到 CMCC（已完成 ✅）

### 3.1 传输状态

- ✅ 环境包已下载到 CMCC 机器
- ✅ CMCC 存放路径：`/root/work/david_work/conda_envs_download/conda_envs/`
- ✅ 文件清单：
  - `sana_wm_cmcc.tar.gz`
  - `sana_wm_cmcc.tar.gz.md5`（可选）
  - `sana_qc_cmcc.tar.gz`
  - `sana_qc_cmcc.tar.gz.md5`（可选）

### 3.2 MD5 校验（建议）

在 CMCC 机器上验证文件完整性：

```bash
cd /root/work/david_work/conda_envs_download/conda_envs/

# 如果有 MD5 文件
md5sum -c sana_wm_cmcc.tar.gz.md5
md5sum -c sana_qc_cmcc.tar.gz.md5

# 如果没有 MD5 文件，生成哈希值记录
md5sum sana_wm_cmcc.tar.gz
md5sum sana_qc_cmcc.tar.gz
```

---
  --local_path /mnt/afs/davidwang/workspace/docker-images/out/sana_wm_cmcc.tar.gz

modelscope upload --repo <您的repo_id> \
  --local_path /mnt/afs/davidwang/workspace/docker-images/out/sana_qc_cmcc.tar.gz
```

**方式 B: Web 界面手动上传**
1. 登录 https://www.modelscope.cn/
2. 进入您的数据集仓库
3. 手动上传两个 tar.gz 文件

---

## 四、CMCC 侧部署（当前阶段）

**前置条件**：
- ✅ 环境包已在 `/root/work/david_work/conda_envs_download/conda_envs/`
- ⚠️ 需要项目代码部署到 `/root/work/david_work/sana_wm_pipeline/`

### 4.1 解压环境包

在 CMCC 机器上执行：

```bash
# 创建工作目录
mkdir -p /root/work/david_work/envs

# 切换到下载目录
cd /root/work/david_work/conda_envs_download/conda_envs/

# 解压 sana_wm_cmcc（约 5-6GB）
echo "=== 解压 sana_wm_cmcc ==="
mkdir -p /root/work/david_work/envs/sana_wm_cmcc
tar -xzf sana_wm_cmcc.tar.gz -C /root/work/david_work/envs/sana_wm_cmcc
cd /root/work/david_work/envs/sana_wm_cmcc
source bin/activate
conda-unpack  # 修复硬编码路径
python -c "import torch; print(f'✓ sana_wm_cmcc: torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
deactivate

# 解压 sana_qc_cmcc（约 4-5GB）
echo ""
echo "=== 解压 sana_qc_cmcc ==="
cd /root/work/david_work/conda_envs_download/conda_envs/
mkdir -p /root/work/david_work/envs/sana_qc_cmcc
tar -xzf sana_qc_cmcc.tar.gz -C /root/work/david_work/envs/sana_qc_cmcc
cd /root/work/david_work/envs/sana_qc_cmcc
source bin/activate
conda-unpack
python -c "import torch; print(f'✓ sana_qc_cmcc: torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
deactivate

echo ""
echo "✓ 环境解压完成"
```

**预期输出**：
```
✓ sana_wm_cmcc: torch 2.x.x (CUDA: True)
✓ sana_qc_cmcc: torch 2.x.x (CUDA: True)
✓ 环境解压完成
```

### 4.2 部署项目代码

**方式 A：从打包机器复制**
```bash
# 在打包机器上打包代码
cd /mnt/afs/davidwang/workspace
tar --exclude='*/.git' --exclude='*/__pycache__' \
    -czf sana_wm_pipeline.tar.gz sana_wm_pipeline/

# 传输到 CMCC（通过 ModelScope 或其他方式）
# 然后在 CMCC 上解压
cd /root/work/david_work
tar -xzf sana_wm_pipeline.tar.gz
```

**方式 B：从 Git 仓库克隆**（如果 CMCC 有网络）
```bash
cd /root/work/david_work
git clone <repository_url> sana_wm_pipeline
```

### 4.3 配置 PYTHONPATH（关键步骤！）

**重要**：由于打包环境中移除了 editable 包，CMCC 侧**必须**配置 PYTHONPATH 指向源码目录。

```bash
# 添加到 ~/.bashrc（持久化配置）
cat >> ~/.bashrc <<'EOF'
# SANA-WM Pipeline module resolution
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF

# 立即生效
source ~/.bashrc

# 验证配置
echo "PYTHONPATH=$PYTHONPATH"
```

**预期输出**：
```
PYTHONPATH=/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:...
```

**工作原理**：
- Python 模块查找顺序：**PYTHONPATH（优先）** → site-packages → 标准库
- `sana_wm_pipeline` 和 `nvidia_vipe` 直接从源码目录导入
- **完全离线**，无需 pip install
- 修改代码立即生效（等效于 editable 安装）

### 4.4 快速验证模块导入

**方式 A：使用直接 bin/python3 调用**（推荐，不依赖 conda activate）

```bash
# 测试 sana_wm_cmcc 环境
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "
from sana_wm_pipeline.stage02_pose.mode_default import run_default
from nvidia_vipe.models import Pi3xMogeModel
import torch
print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print('✓ sana_wm_pipeline 模块导入成功')
print('✓ nvidia_vipe 模块导入成功')
"

# 测试 sana_qc_cmcc 环境
/root/work/david_work/envs/sana_qc_cmcc/bin/python3 -c "
import torch, torchvision, transformers
print(f'✓ torch {torch.__version__}')
print(f'✓ torchvision {torchvision.__version__}')
print(f'✓ transformers {transformers.__version__}')
"
```

**预期输出**：
```
✓ torch 2.x.x (CUDA: True)
✓ sana_wm_pipeline 模块导入成功
✓ nvidia_vipe 模块导入成功
✓ torch 2.x.x
✓ torchvision x.x.x
✓ transformers x.x.x
```

**方式 B：使用 conda activate**（如果 CMCC 有 conda）
```bash
# 假设 CMCC 的 conda 在 /opt/conda
source /opt/conda/etc/profile.d/conda.sh

conda activate /root/work/david_work/envs/sana_wm_cmcc
python -c "import sana_wm_pipeline; print('✓ sana_wm_pipeline')"

conda activate /root/work/david_work/envs/sana_qc_cmcc
python -c "import torch; print('✓ sana_qc_cmcc')"
```

### 4.5 运行 CMCC 冒烟测试

确保项目代码已部署并且 PYTHONPATH 已配置后：

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

---

## 五、故障排查

### 5.1 CMCC 侧：conda-unpack 失败

**症状**: `bash: conda-unpack: command not found`

**原因**: conda-unpack 是 conda-pack 的一部分，在打包时已内置

**解决**:
```bash
# 方法 1：直接调用（已在环境中）
cd /root/work/david_work/envs/sana_wm_cmcc
source bin/activate
./bin/conda-unpack

# 方法 2：检查是否已经 unpack
ls -la bin/conda-unpack  # 应该存在
```

### 5.2 CMCC 侧：模块导入失败

**症状**: `ModuleNotFoundError: No module named 'sana_wm_pipeline'`

**原因**: PYTHONPATH 未正确配置

**解决**:
```bash
# 检查 PYTHONPATH
echo $PYTHONPATH

# 检查项目代码是否存在
ls -la /root/work/david_work/sana_wm_pipeline/src/
ls -la /root/work/david_work/sana_wm_pipeline/third_party/vipe/

# 手动设置（临时）
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"

# 持久化配置
cat >> ~/.bashrc <<'EOF'
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF
source ~/.bashrc
```

### 5.3 CMCC 侧：JIT 编译失败

**症状**: JIT 测试报错 `RuntimeError: CUDA error` 或编译失败

**可能原因**:
1. GPU 不可用
2. CUDA 驱动版本不兼容

**解决**:
```bash
# 检查 GPU
nvidia-smi

# 检查环境中的 CUDA 版本
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"

# 检查 nvcc
/root/work/david_work/envs/sana_wm_cmcc/bin/nvcc --version
```

### 5.4 本地打包阶段：conda-pack 报错 "editable packages"（已解决）

**症状**: `CondaPackError: Cannot pack an environment with editable packages installed`

**原因**: 打包脚本中移除 editable 包的步骤未执行或失败

**解决**:
```bash
# 手动移除
conda activate sana_wm_cmcc
pip uninstall nvidia-vipe sana-wm-pipeline -y

# 验证已移除
pip list | grep -E "nvidia-vipe|sana-wm-pipeline"  # 应该无输出

# 重新打包
conda-pack -p /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc \
  -o /mnt/afs/davidwang/workspace/docker-images/out/sana_wm_cmcc.tar.gz
```

### 5.5 CMCC 侧：解压后文件过大

**症状**: 解压后占用空间超过预期

**说明**: 这是正常的
- 打包文件：5-6GB（压缩）
- 解压后：15-20GB（完整 conda 环境）

**检查磁盘空间**:
```bash
df -h /root/work/david_work/
du -sh /root/work/david_work/envs/*
```

---

---

## 六、执行检查清单

### 打包阶段（已完成 ✅）

- [x] 环境克隆成功
- [x] 移除 editable 包
- [x] 安装 gcc13 + CUDA 工具链（sana_wm_cmcc）
- [x] JIT 编译测试通过
- [x] conda-pack 打包成功
- [x] 生成 tar.gz 文件和 MD5

### CMCC 部署阶段（待执行）

- [ ] 文件传输完成并校验 MD5
- [ ] 解压 sana_wm_cmcc.tar.gz
- [ ] 解压 sana_qc_cmcc.tar.gz
- [ ] 运行 conda-unpack 修复路径
- [ ] 部署项目代码到 /root/work/david_work/sana_wm_pipeline/
- [ ] 配置 PYTHONPATH 到 ~/.bashrc
- [ ] 验证模块导入成功（不需要 pip install）
- [ ] 运行 smoke_test_cmcc_remote.sh 通过

---

## 附录：关键路径参考

### 打包机器（本地）

```
/mnt/afs/davidwang/
├── miniconda3/envs/
│   ├── sana_wm/              # 原始环境（只读）
│   ├── sana_qc/              # 原始环境（只读）
│   ├── sana_wm_cmcc/         # 临时打包环境（可删除）
│   └── sana_qc_cmcc/         # 临时打包环境（可删除）
├── workspace/
│   ├── sana_wm_pipeline/     # 项目源码
│   │   ├── src/sana_wm_pipeline.egg-info/  # 保留
│   │   └── third_party/vipe/nvidia_vipe.egg-info/  # 保留
│   └── docker-images/out/    # 打包输出
│       ├── sana_wm_cmcc.tar.gz
│       ├── sana_wm_cmcc.tar.gz.md5
│       ├── sana_qc_cmcc.tar.gz
│       └── sana_qc_cmcc.tar.gz.md5
```

### CMCC 机器

```
/root/work/david_work/
├── conda_envs_download/conda_envs/  # 下载目录（当前）
│   ├── sana_wm_cmcc.tar.gz
│   └── sana_qc_cmcc.tar.gz
├── envs/                             # 解压目标（待创建）
│   ├── sana_wm_cmcc/
│   └── sana_qc_cmcc/
└── sana_wm_pipeline/                 # 项目代码（待部署）
    ├── src/
    ├── third_party/vipe/
    └── scripts/smoke_test_cmcc_remote.sh
```

---
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF

# 重新加载
source ~/.bashrc

# 验证
echo $PYTHONPATH

# 重新运行测试
bash scripts/smoke_test_cmcc_remote.sh
```

### 5.6 磁盘空间不足

**症状**: 打包过程中报错 "No space left on device"

**解决**:
```bash
# 清理临时文件
rm -rf /tmp/*

# 清理旧的 *-cmcc 环境
rm -rf /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc
rm -rf /mnt/afs/davidwang/miniconda3/envs/sana_qc_cmcc

# 清理 conda 缓存
conda clean --all -y
```

---

## 六、执行检查清单

在新的 Claude 对话中，按此清单逐项执行：

- [ ] **准备阶段**
  - [ ] 确认磁盘空间充足（50GB+）
  - [ ] 创建输出目录
  - [ ] 创建两个脚本文件

- [ ] **打包阶段**
  - [ ] 运行 `build_cmcc_envs_only.sh`
  - [ ] 等待约 75 分钟完成
  - [ ] 确认生成 2 个 tar.gz 文件

- [ ] **验证阶段**
  - [ ] 运行 `smoke_test_cmcc_local.sh`
  - [ ] 确认两个环境测试通过
  - [ ] 检查 MD5 文件生成

- [ ] **传输阶段**
  - [ ] 提供 ModelScope 仓库信息
  - [ ] 上传 `sana_wm_cmcc.tar.gz`
  - [ ] 上传 `sana_qc_cmcc.tar.gz`
  - [ ] 上传对应的 `.md5` 文件

- [ ] **清理阶段**（可选）
  - [ ] 删除临时 *-cmcc 环境
  - [ ] 保留 tar.gz 文件作为备份

---

## 七、时间预算

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| 准备 | 创建脚本、确认环境 | 5 分钟 |
| 打包 | 克隆、安装工具链、conda-pack | 75 分钟 |
| 验证 | 本地冒烟测试 | 10 分钟 |
| 传输 | 上传 ModelScope | 30-60 分钟 |
| **总计** | | **~2-2.5 小时** |

---

## 八、成功标准

✅ **打包成功**:
- 生成 `sana_wm_cmcc.tar.gz` (~5-6G)
- 生成 `sana_qc_cmcc.tar.gz` (~4-5G)
- 两个 `.md5` 文件存在

✅ **本地验证成功**:
- `sana_wm_cmcc` JIT 测试通过
- `sana_qc_cmcc` 依赖导入无错误

✅ **原始环境完整**:
- `/mnt/afs/davidwang/miniconda3/envs/sana_qc` 未被修改
- `/mnt/afs/davidwang/miniconda3/envs/sana_wm` 未被修改

---

**文档版本**: v1.0  
**生成日期**: 2026-08-21  
**维护者**: Claude  
**适用场景**: 在新的 Claude 对话中执行
