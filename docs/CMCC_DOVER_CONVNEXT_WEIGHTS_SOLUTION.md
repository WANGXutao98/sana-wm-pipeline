# CMCC 部署问题定位与解决方案

**问题发现日期**: 2026-09-02  
**问题类型**: DOVER 模型初始化时尝试联网下载 backbone 权重  
**影响**: CMCC 离线环境无法运行 DOVER+UniMatch 管线

---

## 📋 问题现象

### 错误信息

```
Downloading: "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth" to /root/.cache/torch/hub/checkpoints/convnext_tiny_1k_224_ema.pth
...
urllib.error.URLError: <urlopen error Tunnel connection failed: 403 Forbidden>
```

### 调用栈

```python
File "scripts/stage3_batch_minimal_cmcc.py", line 49, in load_models
    dover = DOVER(**opt["model"]["args"])
File "models/DOVER/dover/models/evaluator.py", line 81, in __init__
    b = convnext_3d_tiny(pretrained=True)
File "models/DOVER/dover/models/conv_backbone.py", line 589, in convnext_3d_tiny
    checkpoint = torch.hub.load_state_dict_from_url(url=url, ...)
```

---

## 🔍 根因分析

### 问题根源

DOVER 模型使用 ConvNeXt 作为视觉 backbone，初始化时会调用：

```python
# dover/models/evaluator.py, line 81
b = convnext_3d_tiny(pretrained=True)

# dover/models/conv_backbone.py, line 589
url = model_urls['convnext_tiny_1k']  # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
```

### 缺失文件

| 文件名 | URL | 大小（估算） | 用途 |
|--------|-----|-------------|------|
| `convnext_tiny_1k_224_ema.pth` | https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth | ~110MB | ConvNeXt-Tiny ImageNet-1K 预训练权重 |

### 默认缓存路径

PyTorch Hub 默认缓存位置：
- **默认**: `~/.cache/torch/hub/checkpoints/`
- **可配置**: 通过 `TORCH_HOME` 环境变量指定

---

## ✅ 解决方案

### 方案概述

1. **本机下载** ConvNeXt 权重
2. **打包传输** 到 CMCC 机器
3. **设置环境变量** 让 DOVER 使用本地缓存
4. **验证** 模型加载成功

---

## 📦 步骤 1: 本机下载权重

### 方法 A: 通过 DOVER 触发下载（推荐）

```bash
# 在本机执行
conda activate sana_qc

# 创建临时测试脚本
cat > /tmp/test_dover.py << 'EOF'
import torch
import sys
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
from dover import DOVER

print("初始化 DOVER 模型（会自动下载 ConvNeXt 权重）...")
dover = DOVER()
print("✓ DOVER 加载成功")

# 检查缓存文件
import os
cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
if os.path.exists(cache_dir):
    print(f"\n✓ 缓存目录: {cache_dir}")
    for f in os.listdir(cache_dir):
        fpath = os.path.join(cache_dir, f)
        size_mb = os.path.getsize(fpath) / (1024**2)
        print(f"  - {f} ({size_mb:.1f} MB)")
else:
    print(f"✗ 缓存目录不存在: {cache_dir}")
EOF

# 运行测试脚本
python /tmp/test_dover.py
```

**预期输出**:
```
初始化 DOVER 模型（会自动下载 ConvNeXt 权重）...
Downloading: "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth" ...
✓ DOVER 加载成功

✓ 缓存目录: /home/davidwang/.cache/torch/hub/checkpoints
  - convnext_tiny_1k_224_ema.pth (109.1 MB)
```

### 方法 B: 直接下载（备选）

```bash
# 创建缓存目录
mkdir -p ~/.cache/torch/hub/checkpoints

# 下载 ConvNeXt 权重
cd ~/.cache/torch/hub/checkpoints
wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth

# 验证文件
ls -lh convnext_tiny_1k_224_ema.pth
# 预期大小: ~110MB
```

---

## 📦 步骤 2: 打包权重

```bash
# 在本机执行
cd ~/.cache/torch/hub

# 打包 checkpoints 目录
tar -czf /mnt/afs/davidwang/workspace/sana_wm_pipeline/torch_hub_checkpoints.tar.gz checkpoints/

# 生成 MD5 校验
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
md5sum torch_hub_checkpoints.tar.gz > torch_hub_checkpoints.tar.gz.md5

# 查看打包结果
ls -lh torch_hub_checkpoints.tar.gz
tar -tzf torch_hub_checkpoints.tar.gz
```

**预期输出**:
```
-rw-r--r-- 1 davidwang davidwang 105M Sep  2 15:00 torch_hub_checkpoints.tar.gz

checkpoints/
checkpoints/convnext_tiny_1k_224_ema.pth
```

---

## 📦 步骤 3: 传输到 CMCC

```bash
# 【本机执行】传输到 CMCC
scp /mnt/afs/davidwang/workspace/sana_wm_pipeline/torch_hub_checkpoints.tar.gz \
  cmcc:/root/work/david_work/cache/

scp /mnt/afs/davidwang/workspace/sana_wm_pipeline/torch_hub_checkpoints.tar.gz.md5 \
  cmcc:/root/work/david_work/cache/
```

---

## 📦 步骤 4: CMCC 解压并配置

```bash
# 【CMCC 执行】
cd /root/work/david_work/cache

# 验证传输完整性
md5sum -c torch_hub_checkpoints.tar.gz.md5

# 解压到 torch hub 目录
mkdir -p torch/hub
tar -xzf torch_hub_checkpoints.tar.gz -C torch/hub/

# 验证解压结果
ls -lh torch/hub/checkpoints/
# 应该看到: convnext_tiny_1k_224_ema.pth
```

---

## 🔧 步骤 5: 更新环境变量

### 更新冒烟测试脚本

在 `smoke_dover_cmcc_v2.sh` 中添加 `TORCH_HOME` 环境变量：

```bash
# 在 "── 离线模式 ──" 部分添加
export TORCH_HOME="/root/work/david_work/cache/torch"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 完整的环境变量配置

```bash
# CMCC 环境变量配置
export NEW_BASE="/root/work/david_work"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"
export ENV_QC="$NEW_BASE/envs/sana_qc_cmcc"

# 模型路径
export DOVER_PATH="$PROJ_DIR/models/DOVER"
export UNIMATCH_PATH="$PROJ_DIR/models/unimatch"

# 缓存路径（关键！）
export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"

# 离线模式
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PYTHONPATH
export PYTHONPATH="$PROJ_DIR/src:$DOVER_PATH:$UNIMATCH_PATH${PYTHONPATH:+:$PYTHONPATH}"
```

---

## ✅ 步骤 6: 验证

### 验证权重文件存在

```bash
# 【CMCC 执行】
ls -lh /root/work/david_work/cache/torch/hub/checkpoints/convnext_tiny_1k_224_ema.pth

# 预期输出:
# -rw-r--r-- 1 root root 110M ... convnext_tiny_1k_224_ema.pth
```

### 验证 DOVER 加载

```bash
# 【CMCC 执行】
source /root/work/david_work/envs/sana_qc_cmcc/bin/activate

export TORCH_HOME="/root/work/david_work/cache/torch"
export DOVER_PATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/models/DOVER"

python -c "
import sys
sys.path.insert(0, '$DOVER_PATH')
from dover import DOVER
print('初始化 DOVER...')
dover = DOVER()
print('✓ DOVER 加载成功（使用本地缓存）')
"
```

**预期输出**:
```
初始化 DOVER...
Using Imagenet 22K pretrain False
✓ DOVER 加载成功（使用本地缓存）
```

**不应该看到**: `Downloading: "https://..."`

---

## 🚀 步骤 7: 重新运行冒烟测试

```bash
# 【CMCC 执行】
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline

# 运行更新后的冒烟测试
bash experiments/stage3_smoke/smoke_dover_cmcc_v2.sh 2>&1 | tee /tmp/dover_smoke_final_$(date +%Y%m%d_%H%M%S).log
```

---

## 📊 预期结果

成功运行后应该看到：

```
=== [4/6] 运行 DOVER+UniMatch 筛选 ===
2026-09-02 15:30:00,000 - 发现 246 个视频
2026-09-02 15:30:00,001 - 加载模型...
Using Imagenet 22K pretrain False  # ← 不会再尝试下载
2026-09-02 15:30:05,123 - ✅ 模型加载完成

Stage3: 100%|████████████| 246/246 [XX:XX<00:00, X.XX it/s]

=== [5/6] 检查输出结果 ===
✓ 输出文件存在: 246 条结果

分数统计:
  DOVER fused: min=0.35, max=0.75, avg=0.54
  UniMatch flow: min=12.34, max=45.67, avg=28.90
  通过率: 180/246 (73.2%)

=== [6/6] 测试完成 ===
✓✓✓ DOVER+UniMatch 冒烟测试通过 ✓✓✓
```

---

## 📝 文件清单

### 本机需要准备的文件

| 文件 | 路径 | 大小 | 用途 |
|------|------|------|------|
| `torch_hub_checkpoints.tar.gz` | `/mnt/afs/davidwang/workspace/sana_wm_pipeline/` | ~105MB | ConvNeXt 权重打包 |
| `torch_hub_checkpoints.tar.gz.md5` | 同上 | 几百字节 | MD5 校验 |

### CMCC 部署后的目录结构

```
/root/work/david_work/
├── cache/
│   └── torch/
│       └── hub/
│           └── checkpoints/
│               └── convnext_tiny_1k_224_ema.pth  # ← 关键文件
├── envs/
│   └── sana_qc_cmcc/
├── sana_wm_optimized/
│   └── sana_wm_pipeline/
│       ├── models/
│       │   ├── DOVER/
│       │   └── unimatch/
│       └── scripts/
│           └── stage3_batch_minimal_cmcc.py
└── smoke_pass_videos/
```

---

## ⚠️ 常见问题

### 问题 1: 仍然尝试下载

**症状**: 看到 `Downloading: "https://..."`

**原因**: `TORCH_HOME` 环境变量未生效

**解决**:
```bash
# 确认环境变量
echo $TORCH_HOME
# 应输出: /root/work/david_work/cache/torch

# 如果为空，重新设置
export TORCH_HOME="/root/work/david_work/cache/torch"
```

### 问题 2: 找不到权重文件

**症状**: `FileNotFoundError: ...convnext_tiny_1k_224_ema.pth`

**原因**: 文件路径不正确

**解决**:
```bash
# 检查文件是否存在
ls -la $TORCH_HOME/hub/checkpoints/

# 确保路径正确
# 应该是: $TORCH_HOME/hub/checkpoints/convnext_tiny_1k_224_ema.pth
# 不是:   $TORCH_HOME/checkpoints/convnext_tiny_1k_224_ema.pth
```

### 问题 3: 权重文件损坏

**症状**: `RuntimeError: PytorchStreamReader failed reading zip archive`

**原因**: 传输过程中文件损坏

**解决**:
```bash
# 在 CMCC 验证 MD5
md5sum /root/work/david_work/cache/torch_hub_checkpoints.tar.gz

# 与本机的 MD5 对比
# 如果不一致，重新传输
```

---

## 🎯 核心要点

1. **DOVER 依赖 ConvNeXt**: DOVER 使用 ConvNeXt-Tiny 作为 backbone，初始化时会下载 ImageNet-1K 预训练权重
2. **torch.hub 缓存机制**: PyTorch Hub 默认缓存到 `~/.cache/torch/hub/checkpoints/`
3. **TORCH_HOME 环境变量**: 通过设置 `TORCH_HOME` 可以指定自定义缓存路径
4. **离线部署必需**: CMCC 无网环境必须提前打包所有权重文件
5. **一次配置，长期有效**: 权重文件部署后，所有后续任务都可复用

---

## 📚 相关文档

- [DOVER 官方仓库](https://github.com/VQAssessment/DOVER)
- [ConvNeXt 论文](https://arxiv.org/abs/2201.03545)
- [PyTorch Hub 文档](https://pytorch.org/docs/stable/hub.html)
- 本项目文档: `/docs/QC_MODEL_WEIGHTS.md`

---

**文档版本**: 1.0  
**最后更新**: 2026-09-02  
**验证状态**: ⏳ 等待 CMCC 验证
