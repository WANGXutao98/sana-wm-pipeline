# QC 系统依赖说明

**文档日期：** 2026-06-27  
**适用版本：** SANA-WM QC System v1.0

---

## 依赖分层架构

QC 系统采用三阶段设计，依赖按阶段递增：

```
Stage 1 (CPU 快速扫描)
  └─ 基础：Python 3.10+, numpy, scipy
  
Stage 2 (CPU 深度检测)
  └─ Stage 1 +
      └─ av (PyAV 视频解码)
      └─ scenedetect (场景切割)
  
Stage 3 (GPU 密集评估)
  └─ Stage 2 +
      └─ torch, torchvision (GPU 加速)
      └─ transformers (Qwen VLM)
      └─ dover (视频质量评分)
      └─ unimatch (光流计算)
      └─ PIL, einops
```

---

## 核心依赖清单

### Stage 1 依赖（纯 CPU，无额外包）

| 包名 | 版本 | 用途 | 安装来源 |
|------|------|------|----------|
| numpy | ≥1.24 | 数组计算（位姿矩阵、SO(3) 验证） | pip/conda |
| scipy | ≥1.10 | 科学计算（旋转矩阵检验） | pip/conda |

**说明：** Stage 1 的 11 项检查全部基于 NumPy 和标准库，无需额外依赖。

---

### Stage 2 依赖（CPU 深度检测）

| 包名 | 版本 | 用途 | 安装来源 |
|------|------|------|----------|
| av | ≥10.0 | 视频解码、帧数核验、黑帧检测 | `pip install av` |
| scenedetect | ≥0.6.0 | 场景切割检测（ContentDetector） | `pip install scenedetect[opencv]` |

**注意：**
- `av` 是 FFmpeg 的 Python 绑定，需要系统有 FFmpeg 库
- `scenedetect[opencv]` 会自动安装 opencv-python 作为后端
- 如果 conda 环境已有 opencv，可以只装 `scenedetect`

---

### Stage 3 依赖（GPU 密集型）

| 包名 | 版本 | 用途 | 安装来源 |
|------|------|------|----------|
| torch | ≥2.4 | PyTorch 深度学习框架（CUDA 加速） | pip/conda |
| torchvision | ≥0.19 | 视觉工具（与 torch 版本对应） | pip/conda |
| transformers | ≥4.45 | HuggingFace Transformers（加载 Qwen） | `pip install transformers` |
| einops | ≥0.7.0 | Tensor 操作工具 | `pip install einops` |
| Pillow | ≥10.0 | PIL 图像处理（Qwen 输入） | pip/conda |
| **dover** | latest | DOVER 视频质量评分模型 | `pip install dover` |
| **unimatch** | — | UniMatch 光流计算模型 | **需从 GitHub 安装** |

**UniMatch 安装：**
```bash
# 方法 1：作为子模块（推荐）
cd sana_wm_pipeline/third_party
git clone https://github.com/autonomousvision/unimatch.git
# 在 stage3_gpu.py 中通过 sys.path.insert 加载

# 方法 2：打包为独立目录
# 将 unimatch/ 目录放在 models/ 下，CMCC 部署时解压到指定路径
```

**Qwen3.5-27B-VL 权重：**
- **不需要** pip 安装（权重文件约 55GB）
- CMCC 已有：`/root/work/filestorage/.../Qwen3.5-27B-VL/`
- 使用 `transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir)`

---

## CMCC 部署特殊依赖

### conda-pack 环境打包

CMCC 无外网，需要在源机器打包完整 conda 环境，包含：

```bash
# 基础环境
conda create -n sana_wm python=3.10
conda activate sana_wm

# 安装所有依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy av scenedetect[opencv] transformers einops Pillow pyyaml tqdm dover

# 打包
conda install conda-pack -y
conda pack -n sana_wm -o sana_wm_qc-cmcc.tar.gz \
  --ignore-editable-packages \
  --compress-level 6
```

### 模型权重打包需求

| 模型 | 大小 | 打包策略 |
|------|------|----------|
| UniMatch | ~200MB | 需打包（GitHub clone） |
| DOVER | ~400MB | pip 包自带，无需单独打包 |
| Qwen3.5-27B-VL | ~55GB | **CMCC 已有，无需打包** |

---

## 依赖验证命令

### 验证 Stage 1+2 依赖

```bash
python3 -c "
import numpy, scipy, av, scenedetect
print('Stage 1+2 dependencies OK')
"
```

### 验证 Stage 3 依赖

```bash
python3 -c "
import torch, torchvision, transformers, einops, PIL
from dover import DOVER
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('Stage 3 dependencies OK')
"
```

### 验证 UniMatch

```bash
python3 -c "
import sys
sys.path.insert(0, '/path/to/unimatch')
from unimatch.unimatch import UniMatch
print('UniMatch OK')
"
```

---

## 版本兼容性说明

### Python 版本

- **必需：** Python 3.10+
- **推荐：** Python 3.10（与 sana_wm 环境一致）
- **不支持：** Python 3.9 及以下（transformers ≥4.45 需要 3.10+）

### PyTorch 版本

- **必需：** torch ≥2.4
- **CUDA：** 12.4+（H100 支持）
- **推荐：** torch 2.4.0+cu124

### NumPy 版本

- **兼容：** NumPy 1.x 和 2.x 均支持
- **注意：** 代码已修复 NumPy 2.x `np.eye(4, dtype=np.float32)` 语法

---

## 已知问题与解决方案

### 问题 1：av 安装失败

**症状：** `pip install av` 报错找不到 FFmpeg 库

**解决：**
```bash
# conda 环境
conda install -c conda-forge av

# 或使用 static-ffmpeg
pip install static-ffmpeg
```

### 问题 2：scenedetect 导入报错

**症状：** `ModuleNotFoundError: No module named 'cv2'`

**解决：**
```bash
pip install opencv-python-headless
# 或 conda install -c conda-forge opencv
```

### 问题 3：DOVER 安装失败

**症状：** `pip install dover` 找不到包

**解决：**
```bash
# 确认 PyPI 源可用，或使用镜像
pip install dover -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 4：transformers 版本过低

**症状：** `Qwen2_5_VLForConditionalGeneration` 类不存在

**解决：**
```bash
pip install --upgrade transformers>=4.45
```

---

## 打包清单汇总

CMCC 部署需要以下包：

1. **sana_wm_qc-cmcc.tar.gz**（约 4-5GB）
   - conda 环境，包含所有 pip 依赖
   - torch + CUDA 12.4
   - av, scenedetect, dover, transformers, 等

2. **sana_wm_qc-deploy.tar.gz**（约 50-100MB）
   - QC 源代码（src/sana_wm_pipeline/qc/）
   - CLI 脚本（scripts/run_qc.py, run_stage3_cmcc.py）
   - 文档（docs/QC_REVIEW_DESIGN.md）

3. **sana_wm_qc-models.tar.gz**（约 200MB）
   - UniMatch 权重 + 代码
   - DOVER 权重（如果 pip 包未自带）

4. **无需打包：**
   - Qwen3.5-27B-VL（CMCC 已有）
   - torch hub 缓存（Stage 3 不使用预训练视觉模型）

---

## 测试套件依赖

运行 QC 测试需要额外安装：

```bash
pip install pytest>=8 pytest-xdist
```

测试命令：
```bash
pytest tests/test_qc_*.py -v
```

---

*本文档对应 `requirements-qc.txt` 清单。如有疑问请联系 David Wang。*
