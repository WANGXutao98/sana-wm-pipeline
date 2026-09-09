# SANA-WM Pipeline CMCC 部署完整指南（最终交接版）

**文档版本**: 2.0  
**创建日期**: 2026-08-27  
**状态**: ✅ 已验证可用  
**目标**: CMCC 离线环境（H100 80GB，CUDA Driver 13.0）  
**适用人员**: 接手部署工作的新成员

---

## 📋 快速导航

| 章节 | 用途 | 预计时间 |
|------|------|----------|
| [前置准备](#前置准备) | 了解需要准备的所有文件 | 5 分钟 |
| [本机打包流程](#本机打包流程) | 从零开始打包环境 | 60-90 分钟 |
| [CMCC 部署流程](#cmcc-部署流程) | 在 CMCC 上部署环境 | 30-45 分钟 |
| [验证测试](#验证测试) | 验证部署是否成功 | 10-15 分钟 |
| [问题排查](#问题排查与解决) | 遇到问题时查阅 | 按需 |
| [快速部署脚本](#附录快速部署脚本) | 一键自动化部署 | 20 分钟 |

---

## 前置准备

### 需要准备的文件清单

#### 1. Conda 环境包（本文档会教你如何打包）

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `sana_wm_cuda13.tar.gz` | ~7-8 GB | 打包的 Python 环境 |
| `sana_wm_cuda13.tar.gz.md5` | ~100 字节 | 完整性校验文件 |

#### 2. 项目代码（需单独传输）

- **源路径**（本机）：`/mnt/afs/davidwang/workspace/sana_wm_pipeline`
- **目标路径**（CMCC）：`/root/work/david_work/sana_wm_optimized/sana_wm_pipeline`
- **传输方式**：tar + scp 或你的传输方式
- **大小**：约 2-3 GB（包含 third_party）

#### 3. 模型权重文件

| 模型 | 大小 | 源路径（本机） | 目标路径（CMCC） |
|------|------|---------------|-----------------|
| Pi3X | ~1 GB | `/mnt/afs/davidwang/models/pi3x` | `/root/work/david_work/models/pi3x` |
| MoGe-2 | ~2 GB | `/mnt/afs/davidwang/models/moge2` | `/root/work/david_work/models/moge2` |

#### 4. 测试数据（可选，用于验证）

- **测试视频**：放到 `/root/work/david_work/smoke_pass_videos/`
- **格式**：Apple Vision Pro 录制的 `.mp4` 空间视频
- **建议**：准备 1-2 个视频即可，每个 50-200 MB

---

### 环境概览

#### 本机环境（打包侧）

| 项目 | 配置 | 备注 |
|------|------|------|
| 主机名 | 任意 | - |
| 路径 | `/mnt/afs/davidwang/workspace/sana_wm_pipeline` | AFS 网络存储 |
| Conda 环境 | `sana_wm` | 源环境（不会被修改）|
| Python | 3.10.20 | - |
| PyTorch | 2.12.0+cu130 | 预编译版本 |
| CUDA Driver | 13.0 | - |
| GPU | H100 80GB | - |

#### CMCC 目标环境（部署侧）

| 项目 | 配置 | 备注 |
|------|------|------|
| 主机 | CMCC 训练服务器 | - |
| 项目路径 | `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline` | 注意比本机多一层 |
| 环境路径 | `/root/work/david_work/envs/sana_wm_cuda13` | 解压后的 conda 环境 |
| Python | 3.10.20 | 来自打包环境 |
| PyTorch | 2.12.0+cu130 | 来自打包环境 |
| CUDA Driver | 13.0 | 系统自带 |
| 系统 CUDA Toolkit | 12.4 | ⚠️ 注意：与 PyTorch 不匹配 |
| GPU | H100 80GB | - |
| 网络 | 离线 | 无外网访问 |

---

## 本机打包流程

**总耗时**：60-90 分钟  
**执行机器**：本机（有 `sana_wm` 环境的机器）

---

### 步骤 1：环境克隆（~15-20 分钟）

```bash
# 进入项目目录
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 克隆原始环境到新环境（不影响源环境）
conda create --clone /mnt/afs/davidwang/miniconda3/envs/sana_wm \
  --prefix /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13 -y
```

**预期输出**：
```
Source:      /mnt/afs/davidwang/miniconda3/envs/sana_wm
Destination: /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13
Packages: 237
Files: 36824
...
done
```

**注意事项**：
- 如果看起来"卡住"（无输出），不要中断，这是正常的 I/O 密集型操作
- 可以在另一个终端运行 `watch -n 30 "du -sh /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13"` 监控进度

---

### 步骤 2：移除 Editable 包（~1 分钟）

**为什么需要**：`conda-pack` 不支持 editable 包（`pip install -e .` 安装的包）

```bash
# 激活新环境
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13

# 移除 editable 安装（保留源码不受影响）
pip uninstall nvidia-vipe sana-wm-pipeline -y

# 验证移除成功（应该无输出）
pip list | grep -E "nvidia-vipe|sana-wm-pipeline"

# 退出环境
conda deactivate
```

**重要说明**：
- 这只删除 site-packages 中的符号链接，**不影响源码**
- 源码仍在 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/` 下
- CMCC 上将通过 `PYTHONPATH` 环境变量引用源码

---

### 步骤 3：（可选）安装 CUDA 工具链

**⚠️ 重要决策点**：是否需要 JIT 编译能力？

#### 选项 A：跳过此步骤（推荐）

**适用场景**：
- 你不需要在 CMCC 上编译任何 CUDA 扩展
- vipe 的 C++ 扩展（`vipe_ext.so`）已经预编译好

**优点**：
- 环境更简洁
- 避免 CUDA 版本冲突问题
- 打包时间更短

**执行**：直接跳到步骤 4

#### 选项 B：安装 CUDA 12.4 工具链（不推荐）

**适用场景**：
- 你确实需要在 CMCC 上 JIT 编译 CUDA 代码
- 接受可能的版本冲突风险

**缺点**：
- CUDA 12.4 与 PyTorch 2.12.0（CUDA 13.0）不兼容
- 不能用 `pip install -e vipe`（会触发版本检查失败）

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

**建议**：除非有明确需求，否则**跳过此步骤**。

---

### 步骤 4：打包环境（~15-20 分钟）

```bash
# 安装 conda-pack（如果未安装）
pip install conda-pack

# 切换到工作目录
cd /mnt/afs/davidwang/workspace

# 打包环境
conda pack -p /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13 \
  -o sana_wm_cuda13.tar.gz \
  --compress-level 6 \
  --n-threads 16

# 生成 MD5 校验文件
md5sum sana_wm_cuda13.tar.gz > sana_wm_cuda13.tar.gz.md5

# 查看文件信息
ls -lh sana_wm_cuda13.tar.gz
cat sana_wm_cuda13.tar.gz.md5
```

**预期输出**：
```
-rw-r--r-- 1 user group 7.2G Aug 27 12:00 sana_wm_cuda13.tar.gz
abcdef1234567890abcdef1234567890  sana_wm_cuda13.tar.gz
```

**打包时间参考**：
- 7.5 GB 环境 → 约 7.2 GB 压缩包
- 打包时间：15-20 分钟（取决于 CPU 和磁盘速度）

---

### 步骤 5：打包项目代码

```bash
cd /mnt/afs/davidwang/workspace

# 打包项目代码（包含 third_party/vipe）
tar -czf sana_wm_pipeline_code.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='vipe_results' \
  sana_wm_pipeline/

# 查看大小
ls -lh sana_wm_pipeline_code.tar.gz
```

---

### 步骤 6：传输到 CMCC

**需要传输的文件**：

1. `sana_wm_cuda13.tar.gz`（~7.2 GB）
2. `sana_wm_cuda13.tar.gz.md5`（~100 字节）
3. `sana_wm_pipeline_code.tar.gz`（~2-3 GB）
4. 模型权重（pi3x, moge2）
5. 测试视频（可选）

**CMCC 目标路径**：
```
/root/work/david_work/conda_envs_download/conda_envs/sana_wm_cuda13.tar.gz
/root/work/david_work/conda_envs_download/conda_envs/sana_wm_cuda13.tar.gz.md5
/root/work/david_work/sana_wm_pipeline_code.tar.gz
/root/work/david_work/models/  （模型权重）
/root/work/david_work/smoke_pass_videos/  （测试视频）
```

**传输方式示例**（根据你的实际方式调整）：
```bash
# 方式 1: scp（如果有网络连接）
scp sana_wm_cuda13.tar.gz* user@cmcc-host:/root/work/david_work/conda_envs_download/conda_envs/

# 方式 2: ModelScope 或其他云存储
# （按照你的实际传输流程操作）
```

---

## CMCC 部署流程

**总耗时**：30-45 分钟  
**执行机器**：CMCC 训练服务器

---

### 步骤 0：创建必要目录

```bash
# 创建所有需要的目录
mkdir -p /root/work/david_work/conda_envs_download/conda_envs
mkdir -p /root/work/david_work/envs
mkdir -p /root/work/david_work/sana_wm_optimized
mkdir -p /root/work/david_work/models/pi3x
mkdir -p /root/work/david_work/models/moge2
mkdir -p /root/work/david_work/cache/torch
mkdir -p /root/work/david_work/cache/huggingface
mkdir -p /root/work/david_work/smoke_pass_videos
mkdir -p /root/work/david_work/smoke_pass_results
```

---

### 步骤 1：验证传输完整性

```bash
cd /root/work/david_work/conda_envs_download/conda_envs

# 验证 MD5（确保文件传输没有损坏）
md5sum -c sana_wm_cuda13.tar.gz.md5
```

**预期输出**：
```
sana_wm_cuda13.tar.gz: OK
```

**如果失败**：
- 输出会显示 `FAILED`
- 需要重新传输文件

---

### 步骤 2：解压环境（~5-10 分钟）

```bash
# 创建目标目录
mkdir -p /root/work/david_work/envs/sana_wm_cuda13

# 解压（需要 5-10 分钟）
cd /root/work/david_work/conda_envs_download/conda_envs
tar -xzf sana_wm_cuda13.tar.gz \
  -C /root/work/david_work/envs/sana_wm_cuda13

# 检查解压结果
ls -la /root/work/david_work/envs/sana_wm_cuda13/bin/python3
```

**预期输出**：
```
-rwxr-xr-x 1 root root 29304 Aug 27 12:00 /root/work/david_work/envs/sana_wm_cuda13/bin/python3
```

---

### 步骤 3：运行 conda-unpack（必需，~1 分钟）

**⚠️ 关键步骤**：这是最容易被忽略但**必须执行**的步骤

```bash
cd /root/work/david_work/envs/sana_wm_cuda13

# 激活环境
source bin/activate

# 运行 conda-unpack（修复路径硬编码）
conda-unpack

# 验证执行成功
ls -la conda-meta/state
```

**预期输出**：
```
-rw-r--r-- 1 root root 4 Aug 27 12:05 conda-meta/state
```

**为什么必需**：
- `conda-pack` 打包时记录了原始路径（`/mnt/afs/davidwang/...`）
- `conda-unpack` 将这些路径重写为新路径（`/root/work/david_work/...`）
- **跳过此步骤会导致 Segmentation Fault**（难以诊断）

---

### 步骤 4：解压项目代码

```bash
cd /root/work/david_work

# 解压项目代码
tar -xzf sana_wm_pipeline_code.tar.gz

# 移动到正确位置（注意多一层 sana_wm_optimized）
mkdir -p sana_wm_optimized
mv sana_wm_pipeline sana_wm_optimized/

# 验证目录结构
ls -la sana_wm_optimized/sana_wm_pipeline/src/
ls -la sana_wm_optimized/sana_wm_pipeline/third_party/vipe/
```

---

### 步骤 5：安装缺失依赖

**⚠️ 重要**：根据实际测试，环境中缺少 `psutil` 包

```bash
# 确保已激活环境
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate

# 安装 psutil（离线环境需要提前准备 wheel 文件）
pip install psutil

# 如果离线环境无法安装，需要提前准备：
# 本机执行: pip download psutil -d /tmp/wheels
# 然后传输 wheel 文件到 CMCC，执行:
# pip install /path/to/psutil-*.whl

# 验证安装成功
python -c "import psutil; print(f'✓ psutil {psutil.__version__}')"
```

**预期输出**：
```
✓ psutil 6.1.0
```

---

### 步骤 6：创建 vipe CLI 入口点

**为什么需要**：打包时移除了 editable 安装，`vipe` 命令不存在

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
vipe --help | head -5
```

**预期输出**：
```
/root/work/david_work/envs/sana_wm_cuda13/bin/vipe
Usage: vipe [OPTIONS] COMMAND [ARGS]...

  NVIDIA Video Pose Engine (ViPE) CLI

Options:
```

---

### 步骤 7：放置模型权重和测试视频

```bash
# 假设模型权重已传输到 /root/work/david_work/
# 移动到正确位置
mv pi3x_weights/* /root/work/david_work/models/pi3x/
mv moge2_weights/* /root/work/david_work/models/moge2/

# 验证模型文件
ls -lh /root/work/david_work/models/pi3x/
ls -lh /root/work/david_work/models/moge2/

# 放置测试视频
# cp your_test_videos/*.mp4 /root/work/david_work/smoke_pass_videos/
```

---

### 步骤 8：验证测试脚本配置

**文件路径**：`/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/experiments/data_production_smoke/smoke_cmcc_test_v1.sh`

**验证关键配置**：

```bash
# 检查脚本的前 35 行配置
head -n 35 /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/experiments/data_production_smoke/smoke_cmcc_test_v1.sh
```

**必须包含的配置**：

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

**如果配置不正确，需要修改**：参考本文档的"附录：完整测试脚本模板"

---

## 验证测试

### 测试 1：基础环境诊断（~1 分钟）

```bash
# 激活环境
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate

# 运行诊断
python << 'EOF'
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {cap[0]}.{cap[1]}')
    
    # 测试基本 CUDA 操作
    x = torch.randn(10, 10, device='cuda')
    y = x @ x.T
    print(f'✓ CUDA 矩阵运算正常')
EOF
```

**预期输出**：
```
PyTorch: 2.12.0+cu130
CUDA: 13.0
CUDA available: True
Device: NVIDIA H100 80GB HBM3
Compute Capability: 9.0
✓ CUDA 矩阵运算正常
```

**如果 CUDA 不可用**：
- 检查 `nvidia-smi` 是否能看到 GPU
- 检查 CUDA Driver 版本：`nvidia-smi` 右上角显示
- 检查环境变量 `CUDA_VISIBLE_DEVICES`

---

### 测试 2：模块导入测试（~1 分钟）

```bash
# 设置 PYTHONPATH
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe"

# 激活环境（如果还没激活）
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate

# 测试模块导入
python << 'EOF'
print("测试模块导入...")

import sana_wm_pipeline
print('✓ sana_wm_pipeline')

import vipe
print('✓ vipe')

from vipe.models.pi3x import Pi3X
print('✓ Pi3X')

from moge.model.v2 import MoGeModel
print('✓ MoGeModel')

import psutil
print('✓ psutil')

print("\n所有核心模块导入成功！")
EOF
```

**预期输出**：
```
测试模块导入...
✓ sana_wm_pipeline
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
✓ vipe
✓ Pi3X
✓ MoGeModel
✓ psutil

所有核心模块导入成功！
```

**注意**：
- `RoPE2D` 的警告是**正常的**，不影响功能
- 表示使用 PyTorch 实现而非 CUDA 编译版本

---

### 测试 3：完整冒烟测试（~5-10 分钟）

```bash
# 进入项目目录
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline

# 运行冒烟测试（保存日志）
bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh 2>&1 | tee /tmp/smoke_test_$(date +%Y%m%d_%H%M%S).log
```

**预期输出**（成功）：

```
==========================================
CMCC 冒烟测试 - Pass Videos
==========================================
项目目录: /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
视频目录: /root/work/david_work/smoke_pass_videos
输出目录: /root/work/david_work/smoke_pass_results
Python 环境: /root/work/david_work/envs/sana_wm_cuda13

=== [1/6] 环境预检 ===
Python: 3.10.20 | packaged by conda-forge | ...
✓ torch 2.12.0+cu130 (CUDA: True)
✓ sana_wm_pipeline
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
✓ vipe, pi3, moge
✓ numpy 2.2.6, opencv 4.13.0

=== [2/6] 发现视频文件 ===
视频数量: 2
  [1/2] video1 (150M)
  [2/2] video2 (180M)

=== [3/6] 输出目录 ===
本次运行: /root/work/david_work/smoke_pass_results/run_20260827_143022

=== [4/6] 处理样本 [1/2]: video1 ===
  [Stage 1] 视频归一化...
  ✓ 归一化完成: 46 帧 @ 30fps (518x280)
  
  [Stage 2] VIPE SLAM (Pi3X + MoGe-2)...
[mode_default] Phase A: 深度预计算
  读取视频: .../normalized.mp4
  采样帧数: 46
  Pi3推理 (46帧)...
[_real.py] Loading Pi3X from /root/work/david_work/models/pi3x...
[_real.py] Pi3X loaded successfully
  MoGe-2推理 (46帧)...
[_real.py] Loading MoGe-2 from /root/work/david_work/models/moge2/model.pt...
[_real.py] MoGe-2 loaded successfully
  深度融合...
  ✅ 预计算完成: fused(46, 280, 518), scale~1.137
[mode_default] Phase B: VIPE SLAM
2026-08-27 14:31:15,234 - vipe - INFO - Processing video .../normalized.mp4...
2026-08-27 14:31:32,451 - vipe - INFO - SLAM completed successfully
  ✓ SLAM 完成: poses (46, 4, 4), intrinsics (46, 4)
  
  [Stage 6] 打包 WebDataset shard...
  ✓ Shard 打包完成: 46 帧
  
  ✓ 样本处理完成: video1

=== [4/6] 处理样本 [2/2]: video2 ===
  ...（类似输出）
  ✓ 样本处理完成: video2

=== [5/6] 生成测试报告 ===
...

=== [6/6] 测试完成 ===
报告文件: /root/work/david_work/smoke_pass_results/run_20260827_143022/smoke_test_report.txt

✓✓✓ 冒烟测试全部通过 (2/2) ✓✓✓
```

**测试时间**：
- 单个视频（46 帧）：约 3-5 分钟
- 2 个视频：约 6-10 分钟

---

## 问题排查与解决

### 问题诊断流程图

```
遇到错误
    ↓
查看错误类型
    ↓
    ├─→ ModuleNotFoundError → 查看问题 1、2、5
    ├─→ FileNotFoundError (vipe) → 查看问题 3
    ├─→ Segmentation Fault → 查看问题 4
    ├─→ CUDA version mismatch → 查看问题 6
    └─→ ImportError (undefined symbol) → 查看问题 7
```

---

### 问题 1：ModuleNotFoundError: No module named 'sana_wm_pipeline'

**症状**：
```python
Traceback (most recent call last):
  ...
ModuleNotFoundError: No module named 'sana_wm_pipeline'
```

**原因**：PYTHONPATH 未正确设置或不包含 `src` 目录

**诊断**：
```bash
echo $PYTHONPATH
# 应该包含：/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src
```

**解决方案**：
```bash
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe"

# 验证
python -c "import sana_wm_pipeline; print('✓ 成功')"
```

---

### 问题 2：ModuleNotFoundError: No module named 'vipe'

**症状**：
```python
Traceback (most recent call last):
  ...
ModuleNotFoundError: No module named 'vipe'
```

**原因**：PYTHONPATH 中缺少 `/third_party/vipe`，或错误地写成了 `/third_party`

**诊断**：
```bash
echo $PYTHONPATH
# 应该包含：.../third_party/vipe（注意有 /vipe 后缀）
```

**解决方案**：
```bash
# ❌ 错误（缺少 vipe）
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party"

# ✅ 正确（包含 vipe）
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

**诊断**：
```bash
which vipe
# 如果输出为空或报错，说明 vipe 命令不存在
```

**解决方案**：按照**步骤 6** 创建 vipe CLI 入口点

```bash
# 快速创建
cat > /root/work/david_work/envs/sana_wm_cuda13/bin/vipe << 'EOF'
#!/root/work/david_work/envs/sana_wm_cuda13/bin/python3
import sys
vipe_path = '/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe'
if vipe_path not in sys.path:
    sys.path.insert(0, vipe_path)
from vipe.cli.main import main
if __name__ == '__main__':
    sys.exit(main())
EOF

chmod +x /root/work/david_work/envs/sana_wm_cuda13/bin/vipe

# 验证
vipe --help
```

---

### 问题 4：Segmentation Fault (core dumped)

**症状**：
```
  Pi3推理 (46帧)...
Segmentation fault (core dumped)
```

**原因**：未执行 `conda-unpack`，环境路径硬编码错误

**诊断**：
```bash
# 检查是否执行过 conda-unpack
ls -la /root/work/david_work/envs/sana_wm_cuda13/conda-meta/state

# 如果文件不存在，说明未执行 conda-unpack
```

**解决方案**：
```bash
cd /root/work/david_work/envs/sana_wm_cuda13
source bin/activate
conda-unpack

# 验证
ls -la conda-meta/state  # 应该存在

# 重新运行测试
```

**为什么会 segfault**：
- conda-pack 打包时硬编码了本机路径（`/mnt/afs/davidwang/...`）
- Python 加载的某些动态库（.so 文件）包含绝对路径引用
- 未执行 conda-unpack 导致路径不存在，库加载失败，触发 segfault

---

### 问题 5：ModuleNotFoundError: No module named 'psutil'

**症状**：
```python
Traceback (most recent call last):
  ...
  import psutil
ModuleNotFoundError: No module named 'psutil'
```

**原因**：打包的环境中缺少 `psutil` 依赖（rerun-sdk 需要）

**解决方案**：
```bash
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate

# 在线安装
pip install psutil

# 离线安装（需要提前准备 wheel 文件）
# 本机执行: pip download psutil -d /tmp/wheels
# 传输到 CMCC 后: pip install /path/to/psutil-*.whl

# 验证
python -c "import psutil; print('✓ psutil 可用')"
```

---

### 问题 6：CUDA version mismatch（如果尝试 pip install vipe）

**症状**：
```
RuntimeError: ('The detected CUDA version (%s) mismatches the version that was used to compile PyTorch (%s). Please make sure to use the same CUDA versions.', '12.4', '13.0')
```

**发生场景**：尝试 `pip install -e vipe` 时

**原因**：
- 环境中安装了 CUDA 12.4 工具链（nvcc）
- PyTorch 是用 CUDA 13.0 编译的
- pip install 尝试编译 C++ 扩展时版本检查失败

**解决方案**：
- **不要尝试 `pip install -e vipe`**
- 使用手动创建的 CLI 入口点（步骤 6）
- vipe 的 C++ 扩展（`vipe_ext.so`）已经存在于源码目录，无需重新编译

**如果确实需要重新编译**（不推荐）：
1. 卸载 CUDA 12.4 工具链
2. 安装 CUDA 13.0 工具链（需要从外网下载）
3. 或者在本机重新打包一个没有 CUDA 工具链的环境

---

### 问题 7：ABI 不兼容 - ImportError: undefined symbol

**症状**：
```python
ImportError: /path/to/vipe_ext.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEi
```
或其他涉及 `torch`、`c10`、`_Z` 开头的 C++ 符号未定义错误。

**原因**：PyTorch C++ ABI 不兼容
- PyTorch 有两个 ABI 版本：旧 ABI (`_GLIBCXX_USE_CXX11_ABI=0`) 和新 ABI (`_GLIBCXX_USE_CXX11_ABI=1`)
- vipe 的 C++ 扩展（`vipe_ext.so`）必须与 PyTorch 使用相同的 ABI 编译
- ABI 不匹配会导致运行时符号链接失败

**诊断**：
```bash
# 检查 PyTorch 的 ABI 版本
python -c "import torch; print(f'PyTorch CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
# True = 新 ABI, False = 旧 ABI

# 检查 vipe_ext.so 依赖的符号
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe
nm -D vipe_ext.cpython-310-x86_64-linux-gnu.so | grep torch | head -10
```

**根本原因**：
- 本机打包时使用的 PyTorch ABI 版本与 CMCC 不同
- 或者 vipe 扩展在不同 ABI 环境下编译

**解决方案 A：重新打包使用 CXX11 ABI 的 PyTorch（推荐）**

如果当前 PyTorch 使用旧 ABI，但需要新 ABI：

```bash
# 在本机执行（重新准备环境）
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

# 1. 检查当前 ABI
python -c "import torch; print(f'当前 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"

# 2. 如果是 False（旧 ABI），需要重新安装 PyTorch
pip uninstall torch torchvision torchaudio -y

# 3. 安装新 ABI 版本（需要外网或提前下载 wheel）
pip install torch==2.12.0+cu130.cxx11.abi \
  torchvision==0.17.0+cu130.cxx11.abi \
  torchaudio==2.12.0+cu130.cxx11.abi \
  --extra-index-url https://download.pytorch.org/whl/cu130

# 或者离线安装（如果已下载 wheel 文件）
pip install torch-2.12.0+cu130.cxx11.abi-*.whl \
  torchvision-0.17.0+cu130.cxx11.abi-*.whl \
  torchaudio-2.12.0+cu130.cxx11.abi-*.whl

# 4. 验证新 ABI
python -c "import torch; print(f'新 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
# 应该输出：新 ABI: True

# 5. 重新编译 vipe 扩展
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe
rm -f vipe_ext.*.so
pip install -e . --no-build-isolation

# 6. 测试 vipe 扩展加载
python -c "import vipe; print('✓ vipe 扩展加载成功')"

# 7. 重新执行完整打包流程（本文档"本机打包流程"章节）
```

**解决方案 B：CMCC 上重新编译 vipe 扩展（需要 CUDA 工具链）**

如果不想重新打包，可以在 CMCC 上直接重新编译扩展：

```bash
# CMCC 操作
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate

# 前提：需要先安装 CUDA 工具链（参考本文档步骤 3 选项 B）
# 如果没有工具链，这个方案不可行

cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe

# 1. 清理旧的编译产物
rm -f vipe_ext.*.so
find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# 2. 重新编译
export TORCH_CUDA_ARCH_LIST=9.0
python setup.py build_ext --inplace

# 3. 验证
python -c "
import sys
sys.path.insert(0, '.')
from vipe.slam.vipe_ext import *
print('✓ vipe 扩展加载成功')
"
```

**预防措施（打包前检查）**：

```bash
# 在本机打包前执行此检查
echo "=== ABI 兼容性检查 ==="

# 1. 检查 PyTorch ABI
python -c "import torch; print(f'PyTorch CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"

# 2. 检查 vipe 扩展文件
VIPE_EXT=$(find /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe -name "vipe_ext.*.so")
if [ -n "$VIPE_EXT" ]; then
    echo "✓ vipe 扩展存在: $VIPE_EXT"
    ls -lh "$VIPE_EXT"
else
    echo "✗ vipe 扩展不存在，需要编译"
fi

# 3. 测试扩展加载
python -c "
import sys
sys.path.insert(0, '/mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe')
try:
    import vipe
    print('✓ vipe 模块加载正常')
except ImportError as e:
    print(f'✗ vipe 模块加载失败: {e}')
    exit(1)
"

echo "=== 检查完成 ==="
```

**关键点**：
- ABI 问题**只影响 C++ 扩展**，纯 Python 代码不受影响
- 如果错误信息包含 `undefined symbol` 且符号名包含 `_Z`、`torch`、`c10`，99% 是 ABI 不兼容
- **最简单的方案**：确保本机和 CMCC 使用相同 ABI 版本的 PyTorch
- 如果 vipe 不需要 C++ 加速，可以设置 `VIPE_EXT_JIT=0` 使用纯 Python 实现（性能较低）

---

## 关键经验总结

### 1. 必须执行的步骤（不能跳过）

| 步骤 | 为什么必需 | 后果 |
|------|-----------|------|
| conda-unpack | 修复路径硬编码 | Segmentation Fault |
| 创建 vipe CLI | vipe 命令被移除了 | FileNotFoundError: 'vipe' |
| 设置 PYTHONPATH | 模块通过路径引用 | ModuleNotFoundError |
| 安装 psutil | rerun-sdk 依赖 | ModuleNotFoundError: 'psutil' |
| ABI 一致性检查 | PyTorch 与 C++ 扩展必须同 ABI | ImportError: undefined symbol |

### 2. PYTHONPATH 的正确写法

```bash
# ✅ 正确
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe"

# ❌ 错误 1：缺少 src
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline"

# ❌ 错误 2：缺少 vipe 后缀
export PYTHONPATH="...:$PROJ_DIR/third_party"

# ✅ 简化写法（定义变量后）
export PROJ_DIR="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline"
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe"
```

### 3. Editable 包处理策略

| 阶段 | vipe 安装方式 | sana_wm_pipeline 安装方式 |
|------|--------------|-------------------------|
| 本机（开发） | editable (`pip install -e .`) | editable |
| 打包前 | 移除 (`pip uninstall`) | 移除 |
| CMCC（生产） | PYTHONPATH 引用源码 | PYTHONPATH 引用源码 |
| CLI 入口点 | 手动创建脚本 | 不需要（Python 模块） |

### 4. CUDA 工具链决策树

```
需要在 CMCC 上编译 CUDA 扩展吗？
    ├─→ 否（推荐）：跳过步骤 3，不安装 CUDA 工具链
    │   优点：简洁，无版本冲突
    │   缺点：不能 JIT 编译（但实际不需要）
    │
    └─→ 是（不推荐）：安装 CUDA 12.4 工具链
        优点：可以 JIT 编译
        缺点：与 PyTorch 2.12.0 冲突，不能 pip install vipe
```

### 5. 离线环境依赖管理

**问题**：打包后才发现缺少某个包（如 psutil）

**预防方法**：

```bash
# 本机：在打包前检查完整依赖
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm_cuda13
pip freeze > /tmp/requirements.txt

# 对比原始环境
conda activate sana_wm
pip freeze > /tmp/requirements_original.txt

# 找出差异
diff /tmp/requirements.txt /tmp/requirements_original.txt
```

**补救方法**：

```bash
# 本机：下载缺失包的 wheel 文件
pip download psutil -d /tmp/offline_packages

# 传输到 CMCC 后安装
pip install /tmp/offline_packages/psutil-*.whl
```

---

## 文件清单

### 本机需要准备的文件

| 类别 | 文件 | 大小 | 说明 |
|------|------|------|------|
| **Conda 环境** | `sana_wm_cuda13.tar.gz` | ~7-8 GB | 打包的 Python 环境 |
| | `sana_wm_cuda13.tar.gz.md5` | ~100 bytes | MD5 校验文件 |
| **项目代码** | `sana_wm_pipeline_code.tar.gz` | ~2-3 GB | 项目源码（包含 third_party/vipe）|
| **模型权重** | `pi3x/` | ~1 GB | Pi3X 模型权重目录 |
| | `moge2/` | ~2 GB | MoGe-2 模型权重目录 |
| **测试数据** | `*.mp4` | 50-200 MB/个 | 测试视频（可选）|
| **离线依赖**（可选）| `psutil-*.whl` | ~500 KB | psutil 的 wheel 文件 |

**总大小**：约 12-15 GB

---

### CMCC 部署后的完整目录结构

```
/root/work/david_work/
├── conda_envs_download/
│   └── conda_envs/
│       ├── sana_wm_cuda13.tar.gz          # 传输的环境包（部署后可删除）
│       └── sana_wm_cuda13.tar.gz.md5      # MD5 文件
│
├── envs/
│   └── sana_wm_cuda13/                    # ⭐ 解压后的 conda 环境
│       ├── bin/
│       │   ├── python3                    # Python 解释器
│       │   ├── vipe                       # ⭐ 手动创建的 CLI 入口点
│       │   └── ...
│       ├── lib/
│       │   └── python3.10/
│       │       └── site-packages/         # Python 包
│       ├── conda-meta/
│       │   └── state                      # ⭐ conda-unpack 执行标记
│       └── ...
│
├── sana_wm_optimized/
│   └── sana_wm_pipeline/                  # ⭐ 项目代码目录
│       ├── src/
│       │   └── sana_wm_pipeline/          # ⭐ 主项目代码（PYTHONPATH）
│       │       ├── __init__.py
│       │       ├── stage01_ingest/
│       │       ├── stage02_pose/
│       │       └── ...
│       ├── third_party/
│       │   └── vipe/                      # ⭐ vipe 源码（PYTHONPATH）
│       │       ├── vipe/
│       │       │   ├── __init__.py
│       │       │   ├── cli/
│       │       │   │   └── main.py        # vipe CLI 实现
│       │       │   ├── models/
│       │       │   └── ...
│       │       └── vipe_ext.cpython-310-x86_64-linux-gnu.so  # C++ 扩展
│       ├── experiments/
│       │   └── data_production_smoke/
│       │       └── smoke_cmcc_test_v1.sh  # ⭐ 冒烟测试脚本
│       └── scripts/
│
├── models/
│   ├── pi3x/                              # ⭐ Pi3X 模型权重
│   │   ├── model.pt
│   │   └── ...
│   └── moge2/                             # ⭐ MoGe-2 模型权重
│       ├── model.pt
│       └── ...
│
├── cache/
│   ├── torch/                             # PyTorch 缓存（自动创建）
│   └── huggingface/                       # HuggingFace 缓存（自动创建）
│
├── smoke_pass_videos/                     # ⭐ 测试视频输入目录
│   ├── test_video_1.mp4
│   └── test_video_2.mp4
│
└── smoke_pass_results/                    # 测试结果输出目录（自动创建）
    └── run_20260827_HHMMSS/
        ├── video1/
        │   ├── normalized.mp4
        │   ├── vipe_work_default/
        │   └── shards/
        └── smoke_test_report.txt
```

**关键路径说明**：

| 路径 | 用途 | 来源 |
|------|------|------|
| `/root/work/david_work/envs/sana_wm_cuda13/bin/python3` | Python 解释器 | conda 环境 |
| `/root/work/david_work/envs/sana_wm_cuda13/bin/vipe` | vipe CLI | 手动创建 |
| `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src` | 项目代码 | 传输 |
| `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe` | vipe 源码 | 传输 |
| `/root/work/david_work/models/pi3x` | Pi3X 权重 | 传输 |
| `/root/work/david_work/models/moge2` | MoGe-2 权重 | 传输 |

---

## 附录：快速部署脚本

将以下脚本保存为 `/root/work/david_work/cmcc_deploy.sh`：

```bash
#!/bin/bash
# CMCC 快速部署脚本
# 用途：自动化执行所有部署步骤
# 用法：bash /root/work/david_work/cmcc_deploy.sh

set -e  # 遇到错误立即退出

BASE="/root/work/david_work"
ENV_DIR="$BASE/envs/sana_wm_cuda13"
PROJ_DIR="$BASE/sana_wm_optimized/sana_wm_pipeline"
ENV_PACKAGE="$BASE/conda_envs_download/conda_envs/sana_wm_cuda13.tar.gz"

echo "=========================================="
echo "SANA-WM Pipeline CMCC 自动化部署"
echo "=========================================="
echo ""
echo "目标环境路径: $ENV_DIR"
echo "项目代码路径: $PROJ_DIR"
echo ""

# ============================================================================
# 步骤 1: 检查前置条件
# ============================================================================
echo "=== [1/7] 检查前置条件 ==="

if [ ! -f "$ENV_PACKAGE" ]; then
    echo "✗ 环境包不存在: $ENV_PACKAGE"
    echo "请先传输 sana_wm_cuda13.tar.gz 到 CMCC"
    exit 1
fi
echo "✓ 环境包存在"

if [ ! -d "$PROJ_DIR" ]; then
    echo "✗ 项目代码不存在: $PROJ_DIR"
    echo "请先传输并解压项目代码"
    exit 1
fi
echo "✓ 项目代码存在"

echo ""

# ============================================================================
# 步骤 2: 解压环境
# ============================================================================
echo "=== [2/7] 解压 conda 环境 ==="

if [ -d "$ENV_DIR" ]; then
    echo "⚠️  环境目录已存在，跳过解压"
    echo "   如需重新解压，请先删除: rm -rf $ENV_DIR"
else
    echo "创建目标目录..."
    mkdir -p "$ENV_DIR"
    
    echo "解压环境包（需要 5-10 分钟）..."
    tar -xzf "$ENV_PACKAGE" -C "$ENV_DIR"
    echo "✓ 解压完成"
fi

echo ""

# ============================================================================
# 步骤 3: 运行 conda-unpack（关键步骤）
# ============================================================================
echo "=== [3/7] 运行 conda-unpack（修复路径）==="

cd "$ENV_DIR"
source bin/activate

if [ -f "conda-meta/state" ]; then
    echo "⚠️  conda-unpack 已执行过（state 文件存在）"
else
    echo "执行 conda-unpack..."
    conda-unpack
    echo "✓ conda-unpack 完成"
fi

echo ""

# ============================================================================
# 步骤 4: 安装缺失依赖
# ============================================================================
echo "=== [4/7] 安装缺失依赖 ==="

echo "检查 psutil..."
if python -c "import psutil" 2>/dev/null; then
    echo "✓ psutil 已安装"
else
    echo "安装 psutil..."
    pip install psutil -q
    python -c "import psutil; print(f'✓ psutil {psutil.__version__} 安装成功')"
fi

echo ""

# ============================================================================
# 步骤 5: 创建 vipe CLI 入口点
# ============================================================================
echo "=== [5/7] 创建 vipe CLI 入口点 ==="

VIPE_CLI="$ENV_DIR/bin/vipe"

if [ -f "$VIPE_CLI" ] && [ -x "$VIPE_CLI" ]; then
    echo "⚠️  vipe CLI 已存在"
else
    echo "创建 vipe 命令脚本..."
    cat > "$VIPE_CLI" << 'EOF'
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

    chmod +x "$VIPE_CLI"
    echo "✓ vipe CLI 创建完成"
fi

# 验证 vipe 命令
if vipe --help > /dev/null 2>&1; then
    echo "✓ vipe CLI 可用"
else
    echo "✗ vipe CLI 验证失败"
    exit 1
fi

echo ""

# ============================================================================
# 步骤 6: 基础诊断
# ============================================================================
echo "=== [6/7] 基础诊断 ==="

echo "测试 PyTorch CUDA..."
python << 'PYEOF'
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    # 测试基本运算
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'✓ CUDA 运算正常')
else:
    print('✗ CUDA 不可用')
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "✗ PyTorch CUDA 测试失败"
    exit 1
fi

echo ""
echo "测试模块导入..."
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe"
python << 'PYEOF'
try:
    import sana_wm_pipeline
    print('✓ sana_wm_pipeline')
    
    import vipe
    print('✓ vipe')
    
    from vipe.models.pi3x import Pi3X
    print('✓ Pi3X')
    
    from moge.model.v2 import MoGeModel
    print('✓ MoGeModel')
except ImportError as e:
    print(f'✗ 模块导入失败: {e}')
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "✗ 模块导入测试失败"
    exit 1
fi

echo ""

# ============================================================================
# 步骤 7: 完成
# ============================================================================
echo "=== [7/7] 部署完成 ==="
echo ""
echo "=========================================="
echo "✓✓✓ 部署成功！ ✓✓✓"
echo "=========================================="
echo ""
echo "环境信息："
echo "  Python: $(python --version)"
echo "  环境路径: $ENV_DIR"
echo "  项目路径: $PROJ_DIR"
echo ""
echo "下一步操作："
echo ""
echo "1. 检查模型权重是否已放置："
echo "   ls -lh $BASE/models/pi3x/"
echo "   ls -lh $BASE/models/moge2/"
echo ""
echo "2. 检查测试视频是否已放置："
echo "   ls -lh $BASE/smoke_pass_videos/"
echo ""
echo "3. 运行冒烟测试："
echo "   cd $PROJ_DIR"
echo "   bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh"
echo ""
echo "如有问题，请查阅 CMCC_DEPLOYMENT_FINAL_GUIDE.md 的"问题排查与解决"章节"
echo ""
```

**使用方法**：

```bash
# 1. 保存脚本
cat > /root/work/david_work/cmcc_deploy.sh << 'EOF'
# （粘贴上面的脚本内容）
EOF

# 2. 设置可执行权限
chmod +x /root/work/david_work/cmcc_deploy.sh

# 3. 执行部署
bash /root/work/david_work/cmcc_deploy.sh
```

**脚本特点**：
- ✅ 自动化所有部署步骤
- ✅ 检查前置条件（文件是否存在）
- ✅ 幂等性（可以重复执行，自动跳过已完成的步骤）
- ✅ 错误处理（任何步骤失败会立即退出）
- ✅ 清晰的进度提示

---

## 附录：完整测试脚本模板

如果你的 `smoke_cmcc_test_v1.sh` 需要更新，可以参考以下模板：

**文件**：`/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/experiments/data_production_smoke/smoke_cmcc_test_v1.sh`

```bash
#!/bin/bash
# CMCC 冒烟测试脚本（Pass Videos）
# 用途：验证 CMCC 环境部署成功，确保核心功能可用
# 用法：bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# 配置部分（根据实际环境修改）
# ═══════════════════════════════════════════════════════════════════════════

# ── 路径配置 ──────────────────────────────────────────────────────────────
export NEW_BASE="/root/work/david_work"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"
export VIDEO_DIR="$NEW_BASE/smoke_pass_videos"
export OUT_BASE="$NEW_BASE/smoke_pass_results"

# ── 环境路径 ──────────────────────────────────────────────────────────────
export ENV_WM="$NEW_BASE/envs/sana_wm_cuda13"
export PYTHON="$ENV_WM/bin/python3"

# ── GPU 设置 ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ── 模型权重 ──────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"
export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"

# ── 离线模式 + 优化 ───────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=9.0

# ── PYTHONPATH（必须包含 vipe）────────────────────────────────────────────
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe${PYTHONPATH:+:$PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════════════
# 测试脚本开始
# ═══════════════════════════════════════════════════════════════════════════

echo "=========================================="
echo "CMCC 冒烟测试 - Pass Videos"
echo "=========================================="
echo "项目目录: $PROJ_DIR"
echo "视频目录: $VIDEO_DIR"
echo "输出目录: $OUT_BASE"
echo "Python 环境: $ENV_WM"
echo ""

# ── 预检：环境验证 ────────────────────────────────────────────────────────
echo "=== [1/6] 环境预检 ==="
$PYTHON -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')

import sana_wm_pipeline
print('✓ sana_wm_pipeline')

import vipe
from vipe.models.pi3x import Pi3X
from moge.model.v2 import MoGeModel
print('✓ vipe, pi3, moge')

import numpy as np, cv2
print(f'✓ numpy {np.__version__}, opencv {cv2.__version__}')
"

if [ $? -ne 0 ]; then
    echo "✗ 环境预检失败，请检查依赖"
    exit 1
fi

echo ""

# ── 检查视频目录 ──────────────────────────────────────────────────────────
if [ ! -d "$VIDEO_DIR" ]; then
    echo "✗ 视频目录不存在: $VIDEO_DIR"
    exit 1
fi

VIDEO_FILES=($(find "$VIDEO_DIR" -name "*.mp4" | sort))
VIDEO_COUNT=${#VIDEO_FILES[@]}

if [ $VIDEO_COUNT -eq 0 ]; then
    echo "✗ 视频目录为空: $VIDEO_DIR"
    exit 1
fi

echo "=== [2/6] 发现视频文件 ==="
echo "视频数量: $VIDEO_COUNT"
for i in "${!VIDEO_FILES[@]}"; do
    VIDEO="${VIDEO_FILES[$i]}"
    BASENAME=$(basename "$VIDEO" .mp4)
    SIZE=$(du -sh "$VIDEO" | cut -f1)
    echo "  [$((i+1))/$VIDEO_COUNT] $BASENAME ($SIZE)"
done
echo ""

# ── 创建输出目录 ──────────────────────────────────────────────────────────
mkdir -p "$OUT_BASE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUT_BASE/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "=== [3/6] 输出目录 ==="
echo "本次运行: $RUN_DIR"
echo ""

# ── 批量处理视频 ──────────────────────────────────────────────────────────
cd "$PROJ_DIR"

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_VIDEOS=()

for i in "${!VIDEO_FILES[@]}"; do
    VIDEO="${VIDEO_FILES[$i]}"
    BASENAME=$(basename "$VIDEO" .mp4)
    SCENE_DIR="$RUN_DIR/$BASENAME"
    mkdir -p "$SCENE_DIR"

    echo "=== [4/6] 处理样本 [$((i+1))/$VIDEO_COUNT]: $BASENAME ==="

    # 复制原始视频到工作目录
    cp "$VIDEO" "$SCENE_DIR/video.mp4"

    # Stage 1: 归一化
    echo "  [Stage 1] 视频归一化..."
    NORM_VIDEO="$SCENE_DIR/normalized.mp4"
    $PYTHON -c "
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
try:
    info = normalize_video(Path('$SCENE_DIR/video.mp4'), Path('$NORM_VIDEO'))
    print(f'  ✓ 归一化完成: {info.n_frames} 帧 @ {info.fps}fps ({info.width}x{info.height})')
except Exception as e:
    print(f'  ✗ 归一化失败: {e}')
    exit(1)
" 2>&1 | tee "$SCENE_DIR/stage1.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 1 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 1)")
        continue
    fi

    # Stage 2: VIPE SLAM
    echo "  [Stage 2] VIPE SLAM (Pi3X + MoGe-2)..."
    VIPE_WORK="$SCENE_DIR/vipe_work_default"
    ARTIFACT_JSON="$VIPE_WORK/pose_artifact_default.json"
    mkdir -p "$VIPE_WORK"

    $PYTHON -c "
import json
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_default import run_default

try:
    art = run_default(Path('$NORM_VIDEO'), Path('$VIPE_WORK'))
    print(f'  ✓ SLAM 完成: poses {art.poses_c2w.shape}, intrinsics {art.intrinsics.shape}')

    # 保存结果
    Path('$ARTIFACT_JSON').write_text(json.dumps({
        'poses_c2w': art.poses_c2w.tolist(),
        'intrinsics': art.intrinsics.tolist(),
        'scale_per_frame': art.scale_per_frame.tolist(),
    }))
except Exception as e:
    print(f'  ✗ SLAM 失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" </dev/null 2>&1 | tee "$SCENE_DIR/stage2.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 2 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 2)")
        continue
    fi

    # Stage 6: 打包 WebDataset
    echo "  [Stage 6] 打包 WebDataset shard..."
    SHARDS_DIR="$SCENE_DIR/shards"
    mkdir -p "$SHARDS_DIR"
    SHARD="$SHARDS_DIR/$BASENAME.tar"

    $PYTHON - <<PYEOF
import io, json, numpy as np, tarfile
from pathlib import Path

try:
    scene_id = "$BASENAME"
    art = json.loads(Path("$ARTIFACT_JSON").read_text())
    poses = np.array(art["poses_c2w"], np.float32)
    intr = np.array(art["intrinsics"], np.float32)
    scale = np.array(art["scale_per_frame"], np.float32)

    def add_npy(tf, key, arr):
        b = io.BytesIO()
        np.save(b, arr)
        raw = b.getvalue()
        ti = tarfile.TarInfo(f"{scene_id}.{key}")
        ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))

    with tarfile.open("$SHARD", "w") as tf:
        # 视频
        vb = Path("$NORM_VIDEO").read_bytes()
        ti = tarfile.TarInfo(f"{scene_id}.mp4")
        ti.size = len(vb)
        tf.addfile(ti, io.BytesIO(vb))

        # Pose 数据
        add_npy(tf, "poses_c2w.npy", poses)
        add_npy(tf, "intrinsics.npy", intr)
        add_npy(tf, "scale.npy", scale)

        # Caption（占位）
        cap = "CMCC smoke test video"
        cb = cap.encode()
        ti = tarfile.TarInfo(f"{scene_id}.caption.txt")
        ti.size = len(cb)
        tf.addfile(ti, io.BytesIO(cb))

        # Metadata
        meta = json.dumps({
            "scene_id": scene_id,
            "T": len(poses),
            "mode": "default",
            "dataset": "cmcc_smoke_pass",
            "group": "smoke-test"
        }).encode()
        ti = tarfile.TarInfo(f"{scene_id}.meta.json")
        ti.size = len(meta)
        tf.addfile(ti, io.BytesIO(meta))

    print(f'  ✓ Shard 打包完成: {len(poses)} 帧')
except Exception as e:
    print(f'  ✗ 打包失败: {e}')
    exit(1)
PYEOF

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 6 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 6)")
        continue
    fi

    echo "  ✓ 样本处理完成: $BASENAME"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo ""
done

# ── 生成测试报告 ──────────────────────────────────────────────────────────
echo "=== [5/6] 生成测试报告 ==="

REPORT_FILE="$RUN_DIR/smoke_test_report.txt"
cat > "$REPORT_FILE" <<EOF
CMCC 冒烟测试报告
==================

运行时间: $(date)
输出目录: $RUN_DIR

测试结果
--------
总样本数: $VIDEO_COUNT
成功: $SUCCESS_COUNT
失败: $FAIL_COUNT

EOF

if [ $FAIL_COUNT -gt 0 ]; then
    echo "失败样本:" >> "$REPORT_FILE"
    for failed in "${FAILED_VIDEOS[@]}"; do
        echo "  - $failed" >> "$REPORT_FILE"
    done
    echo "" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" <<EOF
详细日志
--------
每个样本的详细日志保存在对应目录下:
  - stage1.log: 视频归一化日志
  - stage2.log: VIPE SLAM 日志

EOF

cat "$REPORT_FILE"

# ── 最终结果 ──────────────────────────────────────────────────────────────
echo ""
echo "=== [6/6] 测试完成 ==="
echo "报告文件: $REPORT_FILE"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓✓✓ 冒烟测试全部通过 ($SUCCESS_COUNT/$VIDEO_COUNT) ✓✓✓"
    exit 0
else
    echo "✗✗✗ 冒烟测试部分失败 (成功: $SUCCESS_COUNT, 失败: $FAIL_COUNT) ✗✗✗"
    exit 1
fi
```

---

## 常见问题（FAQ）

### Q1：打包环境时可以跳过步骤 3（CUDA 工具链）吗？

**A1**：**推荐跳过**。根据实际测试，CUDA 工具链不是必需的，而且会引入版本冲突问题。vipe 的 C++ 扩展已经预编译好，不需要在 CMCC 上重新编译。

---

### Q2：为什么 CMCC 路径比本机多一层 `sana_wm_optimized`？

**A2**：这是 CMCC 的目录组织方式。如果你的 CMCC 环境路径不同，请在部署时相应修改所有脚本中的路径变量。

---

### Q3：如果 CMCC 是完全离线环境，如何安装 psutil？

**A3**：需要提前准备 wheel 文件：

```bash
# 本机：下载 psutil 及其依赖
pip download psutil -d /tmp/offline_packages

# 打包成 tar
tar -czf psutil_offline.tar.gz -C /tmp offline_packages

# 传输到 CMCC 后：
tar -xzf psutil_offline.tar.gz
pip install /tmp/offline_packages/*.whl
```

---

### Q4：能否使用更新的 PyTorch 版本（如 2.5+）？

**A4**：可以，但需要注意：
1. 确保本机和 CMCC 使用相同的 PyTorch 版本
2. CUDA 版本要匹配（CUDA 13.0 → PyTorch cu130）
3. 重新测试所有功能，特别是 vipe 和模型推理

---

### Q5：部署完成后，如何清理不需要的文件以节省空间？

**A5**：

```bash
# 1. 删除传输的压缩包（部署完成后）
rm -f /root/work/david_work/conda_envs_download/conda_envs/sana_wm_cuda13.tar.gz
rm -f /root/work/david_work/sana_wm_pipeline_code.tar.gz

# 2. 清理 PyTorch 缓存（如果空间紧张）
rm -rf /root/work/david_work/cache/torch/*

# 3. 清理测试结果（保留最近的）
cd /root/work/david_work/smoke_pass_results
ls -t | tail -n +6 | xargs -r rm -rf  # 保留最近 5 次
```

---

### Q6：如何验证模型权重文件是否完整？

**A6**：

```bash
# 检查 Pi3X 权重
source /root/work/david_work/envs/sana_wm_cuda13/bin/activate
python << 'EOF'
import torch
from pathlib import Path

pi3x_path = Path('/root/work/david_work/models/pi3x')
# 根据实际的权重文件名调整
model_file = pi3x_path / 'model.pt'

if not model_file.exists():
    print(f'✗ Pi3X 权重文件不存在: {model_file}')
else:
    print(f'✓ Pi3X 权重文件存在')
    # 尝试加载
    try:
        state_dict = torch.load(model_file, map_location='cpu')
        print(f'✓ Pi3X 权重文件可以正常加载')
    except Exception as e:
        print(f'✗ Pi3X 权重文件损坏: {e}')
EOF

# MoGe-2 同理
```

---

## 交接检查清单

### 交接前（交接人准备）

- [ ] 完整阅读本文档
- [ ] 在本机成功打包环境
- [ ] 传输所有文件到 CMCC
- [ ] 在 CMCC 上成功部署并通过冒烟测试
- [ ] 记录任何文档中未提及的问题和解决方案
- [ ] 准备演示环境（可选）

### 交接时（双方确认）

- [ ] 交接人演示完整部署流程（可选）
- [ ] 接收人了解目录结构和关键路径
- [ ] 接收人了解最常见的 3 个问题及解决方案
- [ ] 确认所有文件传输方式和权限
- [ ] 确认联系方式（遇到问题时）

### 交接后（接收人验证）

- [ ] 独立阅读本文档
- [ ] 在 CMCC 上独立运行冒烟测试
- [ ] 尝试解决一个模拟问题（如删除 vipe CLI 后重新创建）
- [ ] 熟悉快速部署脚本的使用
- [ ] 有问题及时与交接人沟通

---

## 联系与支持

**文档维护人**：David Wang  
**最后更新**：2026-08-27  
**文档版本**：2.0（最终交接版）

**反馈建议**：
- 如发现文档中的错误或不清楚的地方，请及时反馈
- 如遇到文档中未覆盖的问题，记录后补充到文档

---

**祝部署顺利！**
