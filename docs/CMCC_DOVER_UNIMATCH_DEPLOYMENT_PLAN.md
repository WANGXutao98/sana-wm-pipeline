# CMCC 机器部署 DOVER+UniMatch 管线方案

**制定日期**: 2026-09-02  
**目标**: 在 CMCC 离线环境跑通 Stage3 视频质量筛选流水线  
**参考文档**:
- `CMCC_DEPLOYMENT_FINAL_GUIDE.md` (VIPE SLAM 已验证可用)
- `SESSION_PROGRESS_2026-08-16_DOVER_PIPELINE.md` (本机 DOVER+UniMatch 已验证可用)

---

## 📋 部署概览

### 目标环境对比

| 项目 | 本机环境 | CMCC 环境 | 状态 |
|------|---------|-----------|------|
| **VIPE SLAM** | ✅ 可用 | ✅ 已验证 | 冒烟测试通过 |
| **DOVER+UniMatch** | ✅ 可用 | ❓ 待部署 | **本次任务** |

### 环境配置对比

| 配置项 | 本机 | CMCC 目标 |
|--------|------|-----------|
| Conda 环境 | `sana_qc` (从 sana_wm 克隆) | `sana_qc_cmcc` (tar 文件已存在) |
| 环境路径 | `/mnt/afs/davidwang/miniconda3/envs/sana_qc` | `/root/work/david_work/envs/sana_qc_cmcc` |
| DOVER 模型 | `/mnt/afs/.../models/DOVER` | `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER` ✅ |
| UniMatch 模型 | `/mnt/afs/.../models/unimatch` | `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/unimatch` ✅ |
| GPU | H100 80GB, CUDA 13.0 | H100 80GB, CUDA 13.0 ✅ |

---

## 🔍 需要收集的 CMCC 机器信息

### 优先级 P0（必须确认）

```bash
# 1. 检查 sana_qc_cmcc 环境 tar 包是否存在
ls -lh /root/work/david_work/conda_envs_download/conda_envs/sana_qc_cmcc.tar.gz
md5sum /root/work/david_work/conda_envs_download/conda_envs/sana_qc_cmcc.tar.gz

# 2. 检查模型文件是否完整（路径已更新）
ls -lh /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth
ls -lh /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth

# 3. 检查 DOVER/UniMatch 源码是否存在（路径已更新）
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER/dover/
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/unimatch/unimatch/

# 4. 检查项目代码是否已同步
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/scripts/stage3_batch_minimal.py
```

### 优先级 P1（环境诊断）

```bash
# 5. 如果 sana_qc_cmcc 环境已解压，检查关键依赖
source /root/work/david_work/envs/sana_qc_cmcc/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import decord; print(f'decord: {decord.__version__}')"
python -c "import cv2; print(f'opencv: {cv2.__version__}')"

# 6. 检查是否缺少 DOVER/UniMatch 依赖
pip list | grep -E "timm|einops|ftfy|regex|tensorboardX"
```

### 优先级 P2（测试数据）

```bash
# 7. 准备测试视频（至少 1 个样本用于冒烟测试）
ls -lh /root/work/david_work/smoke_pass_videos/*.mp4 | head -n 3
```

---

## 🚀 部署步骤

### 阶段 1: 环境准备

#### 步骤 1.1: 解压 sana_qc_cmcc 环境

```bash
# 创建目标目录
mkdir -p /root/work/david_work/envs/sana_qc_cmcc

# 解压环境（预计 5-10 分钟）
tar -xzf /root/work/david_work/conda_envs_download/conda_envs/sana_qc_cmcc.tar.gz \
  -C /root/work/david_work/envs/sana_qc_cmcc

# 验证解压结果
ls -la /root/work/david_work/envs/sana_qc_cmcc/bin/python3
```

#### 步骤 1.2: 运行 conda-unpack（必须）

```bash
cd /root/work/david_work/envs/sana_qc_cmcc
source bin/activate

# 修复环境路径硬编码
conda-unpack

# 验证
ls -la conda-meta/state  # 应该存在此文件
```

#### 步骤 1.3: 诊断依赖完整性

```bash
# 确保已激活环境
source /root/work/david_work/envs/sana_qc_cmcc/bin/activate

# 测试核心依赖
python -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')

import decord
print(f'✓ decord {decord.__version__}')

import cv2
print(f'✓ opencv {cv2.__version__}')

import numpy as np
print(f'✓ numpy {np.__version__}')
"
```

**预期输出**:
```
✓ PyTorch 2.12.0+cu130
✓ CUDA available: True
✓ decord 0.x.x
✓ opencv 4.x.x
✓ numpy 2.x.x
```

#### 步骤 1.4: 补充缺失依赖（如需要）

根据本机 `sana_qc` 环境，DOVER+UniMatch 需要以下额外依赖：

```bash
# 如果环境中缺少这些包，需要安装（离线环境需提前准备 wheel 文件）
pip list | grep -E "timm|einops|ftfy|regex|tensorboardX|psutil"

# 如果缺少，记录需要从本机打包的 wheel 列表：
# - timm
# - einops
# - ftfy
# - regex
# - tensorboardX
# - psutil
```

**⚠️ 重要**: 如果发现缺少依赖，需要在本机提前打包：

```bash
# 【本机操作】打包缺失依赖
conda activate sana_qc
pip download timm einops ftfy regex tensorboardX psutil \
  --dest /mnt/afs/davidwang/workspace/sana_wm_pipeline/offline_wheels \
  --platform manylinux2014_x86_64 --python-version 310 --only-binary=:all:

# 传输到 CMCC 后安装
# pip install --no-index --find-links=/path/to/offline_wheels timm einops ftfy regex tensorboardX psutil
```

---

### 阶段 2: 模型配置验证

#### 步骤 2.1: 检查 DOVER 模型完整性

```bash
# 验证 DOVER 权重文件（路径已更新）
ls -lh /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth
# 预期大小: ~800MB

# 验证 DOVER 源码结构（路径已更新）
ls /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER/dover/
# 应包含: models/, datasets/, __init__.py 等
```

#### 步骤 2.2: 检查 UniMatch 模型完整性

```bash
# 验证 UniMatch 权重文件（路径已更新）
ls -lh /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth
# 预期大小: ~50-100MB

# 验证 UniMatch 源码结构（路径已更新）
ls /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/unimatch/unimatch/
# 应包含: backbone.py, geometry.py, __init__.py 等
```

#### 步骤 2.3: 测试模型加载

```bash
source /root/work/david_work/envs/sana_qc_cmcc/bin/activate

# 路径已更新：模型在项目目录下
export DOVER_PATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER"
export UNIMATCH_PATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/unimatch"
export PYTHONPATH="$DOVER_PATH:$UNIMATCH_PATH:$PYTHONPATH"

# 测试 DOVER 导入
python -c "
import sys
sys.path.insert(0, '$DOVER_PATH')
from dover.models import DOVER
print('✓ DOVER 模型加载成功')
"

# 测试 UniMatch 导入
python -c "
import sys
sys.path.insert(0, '$UNIMATCH_PATH')
from unimatch.unimatch import UniMatch
print('✓ UniMatch 模型加载成功')
"
```

---

### 阶段 3: 脚本适配

#### 步骤 3.1: 创建 CMCC 配置文件

创建配置文件 `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/experiments/stage3_smoke/smoke_dover_cmcc.sh`:

```bash
#!/bin/bash
set -euo pipefail

# ── 路径配置 ──
export NEW_BASE="/root/work/david_work"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"
export VIDEO_DIR="$NEW_BASE/smoke_pass_videos"
export OUT_DIR="$NEW_BASE/stage3_smoke_results"

# ── 环境路径 ──
export ENV_QC="$NEW_BASE/envs/sana_qc_cmcc"
export PYTHON="$ENV_QC/bin/python3"

# ── GPU 设置 ──
export CUDA_VISIBLE_DEVICES=0

# ── 模型权重与源码路径（路径已更新：在项目目录下）──
export DOVER_PATH="$PROJ_DIR/models/DOVER"
export UNIMATCH_PATH="$PROJ_DIR/models/unimatch"
export DOVER_WEIGHTS="$DOVER_PATH/pretrained_weights/DOVER.pth"
export UNIMATCH_WEIGHTS="$UNIMATCH_PATH/pretrained/gmflow-scale2-regrefine6-mixdata.pth"

# ── 离线模式 ──
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── PYTHONPATH（必须包含 DOVER 和 UniMatch）──
export PYTHONPATH="$PROJ_DIR/src:$DOVER_PATH:$UNIMATCH_PATH${PYTHONPATH:+:$PYTHONPATH}"

# ── 创建输出目录 ──
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "DOVER + UniMatch 冒烟测试 (CMCC)"
echo "=========================================="
echo "环境: $ENV_QC"
echo "视频目录: $VIDEO_DIR"
echo "输出目录: $OUT_DIR"
echo ""

# ── 环境预检 ──
echo "=== [1/5] 环境预检 ==="
$PYTHON -c "
import torch
print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')

import sys
sys.path.insert(0, '$DOVER_PATH')
sys.path.insert(0, '$UNIMATCH_PATH')

from dover.models import DOVER
print('✓ DOVER')

from unimatch.unimatch import UniMatch
print('✓ UniMatch')

import decord
print(f'✓ decord {decord.__version__}')

import cv2
print(f'✓ opencv {cv2.__version__}')
"

# ── 发现视频文件 ──
echo ""
echo "=== [2/5] 发现视频文件 ==="
VIDEO_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -type f -name "*.mp4" | wc -l)
echo "视频数量: $VIDEO_COUNT"

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "✗ 未找到测试视频"
    exit 1
fi

# ── 运行 stage3 脚本 ──
echo ""
echo "=== [3/5] 运行 DOVER+UniMatch 筛选 ==="
cd "$PROJ_DIR"

$PYTHON scripts/stage3_batch_minimal.py \
  --video_dir "$VIDEO_DIR" \
  --output_file "$OUT_DIR/stage3_smoke_results.jsonl" \
  --dover_path "$DOVER_PATH" \
  --unimatch_path "$UNIMATCH_PATH" \
  --dover_weights "$DOVER_WEIGHTS" \
  --unimatch_weights "$UNIMATCH_WEIGHTS" \
  --num_workers 4 \
  --resume

echo ""
echo "=== [4/5] 检查输出结果 ==="
if [ -f "$OUT_DIR/stage3_smoke_results.jsonl" ]; then
    LINE_COUNT=$(wc -l < "$OUT_DIR/stage3_smoke_results.jsonl")
    echo "✓ 输出文件存在: $LINE_COUNT 条结果"
    echo ""
    echo "前 3 条结果:"
    head -n 3 "$OUT_DIR/stage3_smoke_results.jsonl"
else
    echo "✗ 输出文件不存在"
    exit 1
fi

echo ""
echo "=== [5/5] 测试完成 ==="
echo "✓✓✓ DOVER+UniMatch 冒烟测试通过 ✓✓✓"
```

#### 步骤 3.2: 验证脚本存在

```bash
# 检查生产脚本是否存在
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/scripts/stage3_batch_minimal.py

# 如果不存在，需要从本机同步：
# scp /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/stage3_batch_minimal.py \
#   cmcc:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/scripts/
```

---

### 阶段 4: 冒烟测试

#### 步骤 4.1: 准备测试视频

```bash
# 从已有的 smoke_pass_videos 中挑选 1-3 个视频用于测试
ls -lh /root/work/david_work/smoke_pass_videos/*.mp4 | head -n 3

# 如果没有，需要从 VIPE 冒烟测试中复用视频
```

#### 步骤 4.2: 运行冒烟测试

```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline

# 给脚本添加执行权限
chmod +x experiments/stage3_smoke/smoke_dover_cmcc.sh

# 运行测试
bash experiments/stage3_smoke/smoke_dover_cmcc.sh 2>&1 | tee /tmp/dover_smoke_test_$(date +%Y%m%d_%H%M%S).log
```

#### 步骤 4.3: 验证输出

```bash
# 检查输出文件
cat /root/work/david_work/stage3_smoke_results/stage3_smoke_results.jsonl

# 预期格式：
# {"sample_id": "xxx", "dover_tqe": -0.03, "dover_aqe": 0.05, "dover_fused": 0.54, "unimatch_flow": 22.3, "pass": true, ...}
```

**成功标准**:
- ✅ DOVER 分数在合理范围（fused: 0.3-1.0）
- ✅ UniMatch 分数在合理范围（flow: 0-100）
- ✅ 无 OOM 错误（自动降采样到 720p）
- ✅ 无 segfault（conda-unpack 已执行）

---

### 阶段 5: 全量部署

#### 步骤 5.1: 创建生产运行脚本

```bash
#!/bin/bash
# /root/work/david_work/run_stage3_production.sh

set -euo pipefail

# 配置
export NEW_BASE="/root/work/david_work"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"
export VIDEO_DIR="$NEW_BASE/spatialvid_data"  # 替换为实际数据路径
export OUT_FILE="$NEW_BASE/stage3_results/stage3_production_$(date +%Y%m%d_%H%M%S).jsonl"

# 环境
export ENV_QC="$NEW_BASE/envs/sana_qc_cmcc"
export PYTHON="$ENV_QC/bin/python3"
export CUDA_VISIBLE_DEVICES=0

# 模型（路径已更新：在项目目录下）
export DOVER_PATH="$PROJ_DIR/models/DOVER"
export UNIMATCH_PATH="$PROJ_DIR/models/unimatch"
export DOVER_WEIGHTS="$DOVER_PATH/pretrained_weights/DOVER.pth"
export UNIMATCH_WEIGHTS="$UNIMATCH_PATH/pretrained/gmflow-scale2-regrefine6-mixdata.pth"

# 离线模式
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PYTHONPATH
export PYTHONPATH="$PROJ_DIR/src:$DOVER_PATH:$UNIMATCH_PATH${PYTHONPATH:+:$PYTHONPATH}"

# 创建输出目录
mkdir -p "$(dirname "$OUT_FILE")"

# 运行
cd "$PROJ_DIR"
source "$ENV_QC/bin/activate"

$PYTHON scripts/stage3_batch_minimal.py \
  --video_dir "$VIDEO_DIR" \
  --output_file "$OUT_FILE" \
  --dover_path "$DOVER_PATH" \
  --unimatch_path "$UNIMATCH_PATH" \
  --dover_weights "$DOVER_WEIGHTS" \
  --unimatch_weights "$UNIMATCH_WEIGHTS" \
  --num_workers 4 \
  --resume \
  2>&1 | tee "${OUT_FILE%.jsonl}.log"

echo "✓ 生产任务完成: $OUT_FILE"
```

#### 步骤 5.2: 后台运行

```bash
# 给脚本添加执行权限
chmod +x /root/work/david_work/run_stage3_production.sh

# 后台运行（支持断点续传）
nohup bash /root/work/david_work/run_stage3_production.sh > /tmp/stage3_production.nohup 2>&1 &

# 监控进度
tail -f /tmp/stage3_production.nohup
```

---

## ⚠️ 潜在问题与解决方案

### 问题 1: ModuleNotFoundError: No module named 'dover' 或 'unimatch'

**原因**: PYTHONPATH 未正确设置

**解决方案**:
```bash
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/unimatch:$PYTHONPATH"
```

### 问题 2: FileNotFoundError: DOVER.pth 或 gmflow-*.pth

**原因**: 权重文件路径错误或文件缺失

**解决方案**:
```bash
# 检查文件是否存在（路径已更新）
ls -lh /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth
ls -lh /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth

# 如果缺失，需要从本机传输
```

### 问题 3: CUDA out of memory

**原因**: 视频分辨率过高（>720p）

**解决方案**:
- `stage3_batch_minimal.py` 已内置自动降采样逻辑
- 确保代码版本是最新的（270 行版本）

### 问题 4: Segmentation Fault

**原因**: 未执行 `conda-unpack`

**解决方案**:
```bash
cd /root/work/david_work/envs/sana_qc_cmcc
source bin/activate
conda-unpack
```

### 问题 5: 缺少依赖包（timm, einops 等）

**原因**: sana_qc_cmcc 环境打包时遗漏

**解决方案**:
1. 在本机打包缺失依赖的 wheel 文件
2. 传输到 CMCC 后离线安装

---

## 📋 检查清单

### 部署前检查

- [ ] sana_qc_cmcc.tar.gz 已存在且 MD5 校验通过
- [ ] DOVER 模型权重文件存在（~800MB）
- [ ] UniMatch 模型权重文件存在（~50-100MB）
- [ ] DOVER 源码目录完整
- [ ] UniMatch 源码目录完整
- [ ] stage3_batch_minimal.py 脚本已同步
- [ ] 至少有 1-3 个测试视频用于冒烟测试

### 部署中检查

- [ ] conda-unpack 已执行
- [ ] 核心依赖导入成功（torch, decord, cv2）
- [ ] DOVER 模型加载成功
- [ ] UniMatch 模型加载成功
- [ ] PYTHONPATH 正确设置

### 冒烟测试检查

- [ ] 脚本运行无错误
- [ ] DOVER 分数在合理范围（0.3-1.0）
- [ ] UniMatch 分数在合理范围（0-100）
- [ ] 输出 JSONL 格式正确
- [ ] 无 OOM 错误
- [ ] 无 segfault

### 生产部署检查

- [ ] 数据路径配置正确
- [ ] 输出路径有写入权限
- [ ] 断点续传功能可用（--resume）
- [ ] 日志文件正常生成

---

## 📊 预期性能指标

基于本机测试结果：

| 指标 | 预期值 |
|------|--------|
| 单视频处理时间 | 10-30 秒（取决于长度和分辨率） |
| DOVER 分数范围 | 0.3-1.0（正常），-0.05 表示异常 |
| UniMatch 分数范围 | 0-100（像素/帧），3-80 为合格区间 |
| 通过率 | 60-80%（论文为 77%） |
| 显存占用 | <40GB（H100 80GB 足够） |

---

## 🎯 成功标准

冒烟测试通过标准：
- ✅ 至少处理 1 个视频成功
- ✅ DOVER 分数 > 0.3（非异常值 -0.05）
- ✅ UniMatch 分数在 0-100 范围
- ✅ 输出 JSONL 包含所有必需字段
- ✅ 无崩溃、无 OOM

生产部署成功标准：
- ✅ 可处理全量数据集
- ✅ 支持断点续传
- ✅ 通过率接近 60-80%
- ✅ 分数分布符合预期

---

## 📝 下一步行动

1. **信息收集**（你执行）：运行上述 P0/P1 诊断命令，将结果反馈给我
2. **环境部署**（你执行）：按照阶段 1 步骤解压环境并诊断
3. **脚本创建**（我协助）：根据诊断结果调整配置脚本
4. **冒烟测试**（你执行）：运行单视频测试验证
5. **问题诊断**（协作）：如有错误，根据错误信息排查
6. **生产部署**（你执行）：冒烟测试通过后启动全量任务

---

**文档版本**: 1.1  
**最后更新**: 2026-09-02  
**变更记录**: 更新模型路径为项目目录下（`$PROJ_DIR/models/DOVER` 和 `$PROJ_DIR/models/unimatch`）  
**审核状态**: ⏳ 等待 CMCC 环境信息确认
