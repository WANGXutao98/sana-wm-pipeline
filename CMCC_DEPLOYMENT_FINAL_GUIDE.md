# SANA-WM Pipeline CMCC 部署完整指南

**最终版本**: 2026-08-27  
**状态**: ✅ 已验证可用  
**目标**: CMCC 离线环境（H100 80GB，CUDA Driver 13.0）

---

## 📋 目录

1. [环境概览](#环境概览)
2. [本机打包流程](#本机打包流程)
3. [CMCC 部署流程](#cmcc-部署流程)
4. [问题排查与解决](#问题排查与解决)
5. [验证测试](#验证测试)

---

## 环境概览

### 本机环境

| 项目 | 配置 |
|------|------|
| 路径 | `/mnt/afs/davidwang/workspace/sana_wm_pipeline` |
| Conda 环境 | `sana_wm` |
| Python | 3.10 |
| PyTorch | 2.12.0+cu130 |
| CUDA Driver | 13.0 |
| GPU | H100 80GB |

### CMCC 目标环境

| 项目 | 配置 |
|------|------|
| 路径 | `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline` |
| 环境路径 | `/root/work/david_work/envs/sana_wm_cuda13` |
| Python | 3.10 |
| PyTorch | 2.12.0+cu130（目标）|
| CUDA Driver | 13.0 |
| 系统 CUDA Toolkit | 12.4 |
| GPU | H100 80GB |
| 网络 | 离线（无外网访问）|

---

## 本机打包流程

### 步骤 1：环境克隆

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 克隆原始环境（不影响源环境）
conda create --clone /mnt/afs/davidwang/miniconda3/envs/sana_wm \
  --prefix /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13 -y
```

**预期时间**：15-20 分钟（取决于网络存储速度）

---

### 步骤 2：移除 Editable 包

**原因**：`conda-pack` 不支持 editable 包

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13

# 移除 editable 安装（保留源码不受影响）
pip uninstall nvidia-vipe sana-wm-pipeline -y

# 验证移除成功
pip list | grep -E "nvidia-vipe|sana-wm-pipeline"  # 应无输出

conda deactivate
```

**重要说明**：
- 这只删除 site-packages 中的链接，不影响源码
- CMCC 上将通过 PYTHONPATH 引用源码

---

### 步骤 3：安装 CUDA 工具链（可选，用于 JIT 编译）

**注意**：根据实际测试，此步骤**不是必需的**。如果跳过此步骤，环境会更简洁。

```bash
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13

# 安装 gcc13 + CUDA 12.4 工具链
conda install -c nvidia/label/cuda-12.4.1 -c conda-forge \
  'gcc_linux-64=13' 'gxx_linux-64=13' \
  cuda-nvcc cuda-cudart cuda-cudart-dev cuda-cudart-static \
  libcurand libcurand-dev \
  libcublas libcublas-dev \
  -y

# 创建激活钩子
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cc_nvcc.sh <<'HOOK'
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
HOOK

conda deactivate
```

**说明**：
- 如果安装了此工具链，CMCC 上会有 nvcc 12.4
- 但这会与 PyTorch 2.12.0（CUDA 13.0）产生版本冲突
- **建议跳过此步骤**，除非确实需要 JIT 编译

---

### 步骤 4：打包环境

```bash
# 安装 conda-pack（如果未安装）
pip install conda-pack

# 打包环境
cd /mnt/afs/davidwang/workspace
conda pack -p /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13 \
  -o sana_wm_cuda13.tar.gz \
  --compress-level 6 \
  --n-threads 16

# 生成 MD5 校验
md5sum sana_wm_cuda13.tar.gz > sana_wm_cuda13.tar.gz.md5

# 查看文件大小
ls -lh sana_wm_cuda13.tar.gz
```

**预期输出**：
- 文件大小：~7-8 GB（压缩后）
- 打包时间：15-20 分钟

---

### 步骤 5：传输到 CMCC

通过你的传输方式（如 modelscope、scp 等）将以下文件传输到 CMCC：
- `sana_wm_cuda13.tar.gz`
- `sana_wm_cuda13.tar.gz.md5`

目标路径：`/root/work/david_work/conda_envs_download/conda_envs/`

---

## CMCC 部署流程

### 步骤 1：验证传输完整性

```bash
cd /root/work/david_work/conda_envs_download/conda_envs

# 验证 MD5
md5sum -c sana_wm_cuda13.tar.gz.md5
# 应输出: sana_wm_cuda13.tar.gz: OK
```

---

### 步骤 2：解压环境

```bash
# 创建目标目录
mkdir -p /root/work/david_work/envs/sana_wm_cuda13

# 解压（需要 5-10 分钟）
tar -xzf sana_wm_cuda13.tar.gz \
  -C /root/work/david_work/envs/sana_wm_cuda13

# 检查解压结果
ls -la /root/work/david_work/envs/sana_wm_cuda13/bin/python3
```

---

### 步骤 3：运行 conda-unpack（必需）

**关键步骤**：修复环境中的路径硬编码

```bash
cd /root/work/david_work/envs/sana_wm_cuda13

# 激活环境
source bin/activate

# 运行 conda-unpack
conda-unpack

# 验证
ls -la conda-meta/state  # 应该存在此文件
```

**为什么必需**：
- conda-pack 打包时记录了原始路径（`/mnt/afs/davidwang/...`）
- conda-unpack 将路径重写为新路径（`/root/work/david_work/...`）
- 跳过此步骤会导致 Segmentation Fault

---

### 步骤 4：安装缺失依赖

**重要**：根据实际测试，环境中缺少 `psutil` 包

```bash
# 确保已激活环境
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate

# 安装 psutil
pip install psutil

# 验证
python -c "import psutil; print(f'✓ psutil {psutil.__version__}')"
```

---

### 步骤 5：创建 vipe CLI 入口点

**原因**：打包时移除了 editable 安装，vipe 命令不存在

```bash
# 创建 vipe 命令脚本
cat > /root/work/david_work/envs/sana_wm_cuda13/bin/vipe << 'EOF'
#!/root/work/david_work/envs/sana_wm_cuda13/bin/python3
# -*- coding: utf-8 -*-
import sys
import os

# 添加 vipe 源码路径到 sys.path
vipe_path = '/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe'
if vipe_path not in sys.path:
    sys.path.insert(0, vipe_path)

from vipe.cli.main import main

if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit(main())
EOF

# 设置可执行权限
chmod +x /root/work/david_work/envs/sana_wm_cuda13/bin/vipe

# 验证
which vipe
vipe --help
```

**预期输出**：
```
/root/work/david_work/envs/sana_wm_cuda13/bin/vipe
Usage: vipe [OPTIONS] COMMAND [ARGS]...
  NVIDIA Video Pose Engine (ViPE) CLI
...
```

---

### 步骤 6：更新测试脚本

确保测试脚本使用正确的路径和环境变量。

**文件**：`/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/experiments/data_production_smoke/smoke_cmcc_test_v1.sh`

**关键配置**（前 35 行）：

```bash
#!/bin/bash
set -euo pipefail

# ── 路径配置 ──
export NEW_BASE="/root/work/david_work"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"
export VIDEO_DIR="$NEW_BASE/smoke_pass_videos"
export OUT_BASE="$NEW_BASE/smoke_pass_results"

# ── 环境路径 ──
export ENV_WM="$NEW_BASE/envs/sana_wm_cuda13"
export PYTHON="$ENV_WM/bin/python3"

# ── GPU 设置 ──
export CUDA_VISIBLE_DEVICES=0

# ── 模型权重 ──
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"
export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"

# ── 离线模式 + 优化 ──
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=9.0

# ── PYTHONPATH（必须包含 vipe）──
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe${PYTHONPATH:+:$PYTHONPATH}"
```

**关键点**：
1. `ENV_WM` 指向解压后的环境
2. `PYTHONPATH` 必须包含 `third_party/vipe`（不是 `third_party`）
3. 所有 `python` 命令改为 `$PYTHON`

---

## 验证测试

### 基础诊断

```bash
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {cap[0]}.{cap[1]}')
"
```

**预期输出**：
```
PyTorch: 2.12.0+cu130
CUDA: 13.0
CUDA available: True
Device: NVIDIA H100 80GB HBM3
Compute Capability: 9.0
```

---

### 模块导入测试

```bash
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe"

python -c "
import sana_wm_pipeline
print('✓ sana_wm_pipeline')

import vipe
print('✓ vipe')

from vipe.models.pi3x import Pi3X
print('✓ Pi3X')

from moge.model.v2 import MoGeModel
print('✓ MoGeModel')
"
```

**预期输出**：
```
✓ sana_wm_pipeline
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
✓ vipe
✓ Pi3X
✓ MoGeModel
```

**注意**：RoPE2D 的警告是正常的，不影响功能。

---

### 完整冒烟测试

```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline

# 运行冒烟测试
bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh 2>&1 | tee /tmp/smoke_test_$(date +%Y%m%d_%H%M%S).log
```

**预期输出**（成功）：
```
=== [1/6] 环境预检 ===
✓ torch 2.12.0+cu130 (CUDA: True)
✓ sana_wm_pipeline
✓ vipe, pi3, moge
✓ numpy 2.2.6, opencv 4.13.0

=== [2/6] 发现视频文件 ===
视频数量: N

=== [3/6] 输出目录 ===
本次运行: /root/work/david_work/smoke_pass_results/run_YYYYMMDD_HHMMSS

=== [4/6] 处理样本 [1/N]: ... ===
  [Stage 1] 视频归一化...
  ✓ 归一化完成: 46 帧 @ 30fps (518x280)
  
  [Stage 2] VIPE SLAM (Pi3X + MoGe-2)...
  ...
  ✓ SLAM 完成: poses (46, 4, 4), intrinsics (46, 4)
  
  [Stage 6] 打包 WebDataset shard...
  ✓ Shard 打包完成: 46 帧
  
  ✓ 样本处理完成: ...

=== [6/6] 测试完成 ===
✓✓✓ 冒烟测试全部通过 (N/N) ✓✓✓
```

---

## 问题排查与解决

### 问题 1：ModuleNotFoundError: No module named 'sana_wm_pipeline'

**症状**：
```python
ModuleNotFoundError: No module named 'sana_wm_pipeline'
```

**原因**：PYTHONPATH 未正确设置

**解决方案**：
```bash
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe"
```

---

### 问题 2：ModuleNotFoundError: No module named 'vipe'

**症状**：
```python
ModuleNotFoundError: No module named 'vipe'
```

**原因**：PYTHONPATH 中缺少 `/third_party/vipe`，或写成了 `/third_party`

**解决方案**：
```bash
# 错误
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party"

# 正确
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe"
```

---

### 问题 3：FileNotFoundError: [Errno 2] No such file or directory: 'vipe'

**症状**：
```
subprocess.CalledProcessError: Command '['vipe', 'infer', ...]' returned non-zero exit status 2.
FileNotFoundError: [Errno 2] No such file or directory: 'vipe'
```

**原因**：vipe CLI 命令不存在（editable 安装被移除了）

**解决方案**：按照**步骤 5** 创建 vipe CLI 入口点

---

### 问题 4：Segmentation Fault (core dumped)

**症状**：
```
Pi3推理 (46帧)...
Segmentation fault (core dumped)
```

**原因**：未执行 `conda-unpack`，环境路径硬编码错误

**解决方案**：
```bash
cd /root/work/david_work/envs/sana_wm_cuda13
source bin/activate
conda-unpack
```

---

### 问题 5：ModuleNotFoundError: No module named 'psutil'

**症状**：
```python
ModuleNotFoundError: No module named 'psutil'
```

**原因**：环境中缺少 psutil 依赖

**解决方案**：
```bash
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate
pip install psutil
```

---

### 问题 6：CUDA version mismatch (如果尝试 pip install vipe)

**症状**：
```
RuntimeError: The detected CUDA version (12.4) mismatches the version that was used to compile PyTorch (13.0)
```

**原因**：
- 环境中安装了 CUDA 12.4 工具链（nvcc）
- PyTorch 是用 CUDA 13.0 编译的
- 尝试重新编译 C++ 扩展时版本检查失败

**解决方案**：
- **不要尝试 `pip install -e vipe`**
- 使用手动创建的 CLI 入口点（步骤 5）
- vipe 的 C++ 扩展（`vipe_ext.so`）已经存在于源码目录，无需重新编译

---

## 关键经验总结

### 1. conda-unpack 是必需的

**不要跳过此步骤**，否则会出现难以诊断的 segfault。

### 2. PYTHONPATH 必须精确

```bash
# 正确
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe"

# 错误（缺少 vipe）
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party"
```

### 3. Editable 包需要特殊处理

- 打包前移除 editable 安装
- CMCC 上通过 PYTHONPATH 引用源码
- CLI 入口点需要手动创建

### 4. CUDA 工具链冲突

- 如果环境中有 nvcc 12.4，不要尝试编译任何 CUDA 扩展
- PyTorch 2.12.0 + CUDA 13.0 与 nvcc 12.4 不兼容
- 建议打包时**不安装 CUDA 工具链**（步骤 3 可跳过）

### 5. 离线环境依赖检查

- 打包后在 CMCC 上测试时，可能发现缺失的依赖（如 psutil）
- 提前在本机测试 `pip freeze` 与实际需求的差异
- 或者在 CMCC 上准备一个离线 pip 缓存

---

## 文件清单

### 本机需要准备的文件

1. `sana_wm_cuda13.tar.gz` - 打包的 conda 环境（~7-8 GB）
2. `sana_wm_cuda13.tar.gz.md5` - MD5 校验文件
3. 项目代码目录（需单独传输）
4. 模型权重文件（pi3x, moge2）
5. 测试视频文件

### CMCC 部署后的目录结构

```
/root/work/david_work/
├── envs/
│   └── sana_wm_cuda13/          # 解压后的 conda 环境
│       ├── bin/
│       │   ├── python3
│       │   └── vipe              # 手动创建的 CLI 入口点
│       ├── lib/
│       └── ...
├── sana_wm_optimized/
│   └── sana_wm_pipeline/        # 项目代码
│       ├── src/
│       │   └── sana_wm_pipeline/
│       ├── third_party/
│       │   └── vipe/            # vipe 源码（通过 PYTHONPATH 引用）
│       └── experiments/
│           └── data_production_smoke/
│               └── smoke_cmcc_test_v1.sh
├── models/
│   ├── pi3x/
│   └── moge2/
├── smoke_pass_videos/           # 测试视频
└── smoke_pass_results/          # 输出结果
```

---

## 附录：快速部署脚本

将以下脚本保存为 `/root/work/david_work/cmcc_deploy.sh`：

```bash
#!/bin/bash
# CMCC 快速部署脚本
set -e

BASE="/root/work/david_work"
ENV_DIR="$BASE/envs/sana_wm_cuda13"
PROJ_DIR="$BASE/sana_wm_optimized/sana_wm_pipeline"

echo "=========================================="
echo "SANA-WM Pipeline CMCC 部署"
echo "=========================================="

# 步骤 1: 检查环境包
if [ ! -f "$BASE/conda_envs_download/conda_envs/sana_wm_cuda13.tar.gz" ]; then
    echo "✗ 环境包不存在，请先传输文件"
    exit 1
fi

# 步骤 2: 解压环境
echo ""
echo "=== 解压 conda 环境 ==="
if [ -d "$ENV_DIR" ]; then
    echo "环境目录已存在，跳过解压"
else
    mkdir -p "$ENV_DIR"
    tar -xzf "$BASE/conda_envs_download/conda_envs/sana_wm_cuda13.tar.gz" -C "$ENV_DIR"
    echo "✓ 解压完成"
fi

# 步骤 3: conda-unpack
echo ""
echo "=== 运行 conda-unpack ==="
cd "$ENV_DIR"
source bin/activate
conda-unpack
echo "✓ conda-unpack 完成"

# 步骤 4: 安装 psutil
echo ""
echo "=== 安装 psutil ==="
pip install psutil -q
python -c "import psutil; print(f'✓ psutil {psutil.__version__}')"

# 步骤 5: 创建 vipe CLI
echo ""
echo "=== 创建 vipe CLI 入口点 ==="
cat > "$ENV_DIR/bin/vipe" << 'EOF'
#!/root/work/david_work/envs/sana_wm_cuda13/bin/python3
# -*- coding: utf-8 -*-
import sys
import os

vipe_path = '/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe'
if vipe_path not in sys.path:
    sys.path.insert(0, vipe_path)

from vipe.cli.main import main

if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit(main())
EOF

chmod +x "$ENV_DIR/bin/vipe"
vipe --help > /dev/null 2>&1 && echo "✓ vipe CLI 可用" || echo "✗ vipe CLI 失败"

# 步骤 6: 基础诊断
echo ""
echo "=== 基础诊断 ==="
python -c "
import torch
print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
"

export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe"
python -c "
import sana_wm_pipeline
print('✓ sana_wm_pipeline')
import vipe
print('✓ vipe')
"

echo ""
echo "=========================================="
echo "✓ 部署完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  cd $PROJ_DIR"
echo "  bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh"
```

**使用方法**：
```bash
bash /root/work/david_work/cmcc_deploy.sh
```

---

**文档版本**: 1.0  
**最后更新**: 2026-08-27  
**验证状态**: ✅ 已在 CMCC H100 环境验证通过
