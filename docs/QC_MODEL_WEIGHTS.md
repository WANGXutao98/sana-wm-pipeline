# SANA-WM QC 系统模型权重清单

**文档日期：** 2026-06-27  
**适用版本：** QC System v1.0 for CMCC Deployment

---

## 概述

QC Stage 3 需要三个深度学习模型：

| 模型 | 用途 | 大小 | 打包策略 |
|------|------|------|----------|
| **UniMatch** | 光流计算（替代 VMAF Motion） | ~200MB | 需打包 |
| **DOVER** | 视频质量评分 | ~400MB | pip 包管理，可能需打包权重 |
| **Qwen3.5-27B-VL** | VLM 实体计数+质量标记+caption改写 | ~55GB | **CMCC 已有，无需打包** |

---

## 模型 1: UniMatch 光流

### 基本信息

- **论文：** "Unifying Flow, Stereo and Depth Estimation" (TPAMI 2023)
- **GitHub：** https://github.com/autonomousvision/unimatch
- **任务：** 计算相邻帧间光流幅值均值（论文 Table 6 指标）
- **权重文件：** `gmflow-scale2-regrefine6-mixdata.pth`（~180MB）
- **配置：** feature_channels=128, num_scales=2, upsample_factor=4

### 加载方式

见 `src/sana_wm_pipeline/qc/stage3_gpu.py::load_unimatch_fn()`：

```python
import sys
sys.path.insert(0, str(Path(model_dir).parent))
from unimatch.unimatch import UniMatch
import torch

model = UniMatch(
    feature_channels=128, num_scales=2, upsample_factor=4,
    num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
    reg_refine=True, task="flow",
).to(device).eval()

ckpt = Path(model_dir) / "gmflow-scale2-regrefine6-mixdata.pth"
state = torch.load(ckpt, map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)
```

### 打包方案

**方法 A：作为项目子模块（推荐）**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party
git clone https://github.com/autonomousvision/unimatch.git
cd unimatch
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained_models/gmflow-scale2-regrefine6-mixdata.pth

# 打包
tar -czf unimatch.tar.gz unimatch/
md5sum unimatch.tar.gz
```

**方法 B：独立目录打包（CMCC 部署）**

```bash
# 假设已 clone 到 /tmp/unimatch
cd /tmp
tar -czf sana_wm_qc-unimatch.tar.gz \
  --transform 's,^unimatch,models/unimatch,' \
  unimatch/

# CMCC 部署时解压到：
# $NEW_BASE/models/unimatch/
#   ├── unimatch.py
#   ├── backbone.py
#   ├── ...（其他代码）
#   └── gmflow-scale2-regrefine6-mixdata.pth
```

**预期目录结构：**
```
models/unimatch/
├── unimatch/              # Python 包目录
│   ├── __init__.py
│   ├── unimatch.py        # UniMatch 类定义
│   ├── backbone.py
│   ├── transformer.py
│   └── geometry.py
└── gmflow-scale2-regrefine6-mixdata.pth  # 权重文件
```

**CMCC 使用方式：**
```bash
export UNIMATCH_DIR="/root/work/<USERSPACE>/models/unimatch"
python scripts/run_stage3_cmcc.py \
  --unimatch-dir "$UNIMATCH_DIR" \
  ...
```

### 验证命令

```python
import sys
sys.path.insert(0, "/path/to/models/unimatch")
from unimatch.unimatch import UniMatch
import torch

model = UniMatch(
    feature_channels=128, num_scales=2, upsample_factor=4,
    num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
    reg_refine=True, task="flow",
).cuda()

ckpt = torch.load("/path/to/models/unimatch/gmflow-scale2-regrefine6-mixdata.pth")
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
print("UniMatch loaded successfully")
```

---

## 模型 2: DOVER 视频质量评分

### 基本信息

- **论文：** "DOVER: A Method for Disentangled-Objective Video Quality Evaluation" (ICCV 2023)
- **PyPI：** `pip install dover`
- **任务：** 评估视频技术质量和美学质量综合评分（论文 Table 6 指标）
- **权重：** 由 pip 包自动下载到 `~/.cache/torch/hub/checkpoints/`

### 加载方式

见 `src/sana_wm_pipeline/qc/stage3_gpu.py::load_dover_fn()`：

```python
from dover import DOVER
import torch

model = DOVER().to(device).eval()

def dover_fn(frames_rgb: np.ndarray) -> float:
    # frames_rgb: (T, H, W, 3) uint8
    t = torch.from_numpy(frames_rgb).float()
    t = t.permute(0, 3, 1, 2).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        score = model(t)
    return float(score.mean().item())
```

### 打包方案

**检查 dover 包是否自带权重：**

```bash
pip install dover
python -c "from dover import DOVER; m = DOVER(); print('DOVER OK')"
# 首次运行会自动下载权重到 ~/.cache/torch/hub/checkpoints/
```

**如果需要打包权重（无网环境）：**

```bash
# 在有网机器上先触发下载
python -c "from dover import DOVER; DOVER().eval()"

# 打包缓存
tar -czf dover_weights.tar.gz \
  -C ~/.cache/torch/hub/checkpoints \
  $(ls ~/.cache/torch/hub/checkpoints/ | grep -i dover)

# CMCC 部署时解压到 $TORCH_HOME/hub/checkpoints/
```

**CMCC 使用方式：**
```bash
export TORCH_HOME="/root/work/<USERSPACE>/cache/torch"
# DOVER 会自动从 $TORCH_HOME/hub/checkpoints/ 加载权重
```

### 验证命令

```python
from dover import DOVER
import torch
import numpy as np

model = DOVER().cuda().eval()
# 模拟输入：10帧 224×224 RGB 视频
frames = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
t = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).cuda() / 255.0

with torch.no_grad():
    score = model(t)
print(f"DOVER score: {score.mean().item():.4f}")
```

---

## 模型 3: Qwen3.5-27B-VL

### 基本信息

- **模型：** Qwen/Qwen2.5-VL-27B-Instruct
- **用途：** VLM 实体计数（人/车/动物）+ 视觉质量标记 + caption 改写
- **大小：** ~55GB（BF16 权重）
- **CMCC 路径：** `/root/work/filestorage/shangaoooooo/davidwang/Qwen3.5-27B-VL/`（已确认）

### 加载方式

见 `src/sana_wm_pipeline/qc/stage3_gpu.py::load_qwen_fn()`：

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

processor = AutoProcessor.from_pretrained(model_dir)
```

### CMCC 使用方式

```bash
export QWEN_DIR="/root/work/filestorage/shangaoooooo/davidwang/Qwen3.5-27B-VL"
python scripts/run_stage3_cmcc.py \
  --qwen-dir "$QWEN_DIR" \
  ...
```

### 显存占用

- **BF16 加载：** ~54GB
- **H100 80GB：** 足够容纳 Qwen + UniMatch + DOVER 三个模型
- **并发策略：** 每张 GPU 独立加载完整模型，round-robin 分配样本

### 验证命令

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model_dir = "/root/work/filestorage/.../Qwen3.5-27B-VL"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
processor = AutoProcessor.from_pretrained(model_dir)
print(f"Qwen loaded, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
```

---

## 完整打包清单

### 需要打包的模型

| 文件名 | 内容 | 大小 | MD5（待计算） |
|--------|------|------|--------------|
| `sana_wm_qc-unimatch.tar.gz` | UniMatch 代码 + 权重 | ~200MB | `<待计算>` |
| `sana_wm_qc-dover.tar.gz` | DOVER 权重缓存（可选） | ~400MB | `<待计算>` |

### 无需打包的模型

- **Qwen3.5-27B-VL**：CMCC 已有，路径 `/root/work/filestorage/shangaoooooo/davidwang/Qwen3.5-27B-VL/`

---

## CMCC 部署后验证脚本

```bash
# 在 CMCC 机器上执行（假设 conda env 已激活）
source "$NEW_BASE/activate_sana_wm.sh"

python3 - <<'PY'
import sys, torch
sys.path.insert(0, "$NEW_BASE/models/unimatch")

# 验证 UniMatch
from unimatch.unimatch import UniMatch
um = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4,
              num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
              reg_refine=True, task="flow").cuda()
print("✓ UniMatch loaded")

# 验证 DOVER
from dover import DOVER
dover_model = DOVER().cuda()
print("✓ DOVER loaded")

# 验证 Qwen
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "$QWEN_DIR", torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
print("✓ Qwen3.5-27B-VL loaded")

print("\n=== GPU Memory ===")
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
PY
```

**预期输出：**
```
✓ UniMatch loaded
✓ DOVER loaded
✓ Qwen3.5-27B-VL loaded

=== GPU Memory ===
Allocated: 58.xx GB
Reserved: 60.xx GB
```

---

## 常见问题

### Q1: UniMatch 权重下载慢或失败

**A:** 使用镜像或手动下载：
```bash
# 手动下载
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained_models/gmflow-scale2-regrefine6-mixdata.pth

# 或从 HuggingFace 镜像（如有）
# ...
```

### Q2: DOVER 首次运行卡住

**A:** DOVER 首次运行会自动下载权重，无网环境需要预先打包缓存：
```bash
# 有网机器
python -c "from dover import DOVER; DOVER()"
tar -czf dover_cache.tar.gz ~/.cache/torch/hub/checkpoints/

# CMCC 机器
tar -xzf dover_cache.tar.gz -C $TORCH_HOME/hub/
```

### Q3: Qwen 加载报错找不到文件

**A:** 确认路径变量正确：
```bash
echo $QWEN_DIR
ls $QWEN_DIR/config.json  # 必须存在
```

### Q4: 三个模型同时加载 OOM

**A:** H100 80GB 理论足够，如果 OOM：
1. 确认没有其他进程占用 GPU
2. 检查 Qwen 是否正确使用 BF16（不是 FP32）
3. 降低 DOVER 或 UniMatch 的 batch size（如有）

---

*本文档对应 `docs/QC_DEPENDENCIES.md` 依赖清单。模型打包脚本见下一节。*
