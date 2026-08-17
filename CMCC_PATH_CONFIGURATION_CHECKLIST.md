# CMCC 路径配置检查清单

**迁移区间**: `bea3193` → `b8fda66`  
**生成日期**: 2026-08-16  
**用途**: 离线迁移到 CMCC 服务器前必须修改的硬编码路径清单

---

## 一、必须修改的核心文件（3 个）

### 1.1 `src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py`

**行号**: 约 15-20 行  
**路径类型**: 模型权重路径（环境变量默认值）

```python
# 当前值（本机）
_PI3X_WEIGHTS = os.environ.get("SANA_WM_PI3X_WEIGHTS", "/mnt/afs/davidwang/models/pi3x")
_MOGE2_WEIGHTS = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "/mnt/afs/davidwang/models/moge2")

# 修改为（CMCC）
_PI3X_WEIGHTS = os.environ.get("SANA_WM_PI3X_WEIGHTS", "/root/work/david_work/models/pi3x")
_MOGE2_WEIGHTS = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "/root/work/david_work/models/moge2")
```

**影响**: Stage 2 位姿估计核心功能，若路径错误会直接报错无法运行。

---

### 1.2 `scripts/stage3_batch_minimal.py`

**行号**: 约 10-30 行  
**路径类型**: 模型路径（DOVER/UniMatch）+ 临时文件路径

```python
# 当前值（本机）
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch")

with open("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml") as f:

scorer = DOVER(
    "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth",
)

flow_estimator = UniMatchFlow(
    "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth",
)

tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4', dir='/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp')

# 修改为（CMCC）
sys.path.insert(0, "/root/work/david_work/models/DOVER")
sys.path.insert(0, "/root/work/david_work/models/unimatch")

with open("/root/work/david_work/models/DOVER/dover.yml") as f:

scorer = DOVER(
    "/root/work/david_work/models/DOVER/pretrained_weights/DOVER.pth",
)

flow_estimator = UniMatchFlow(
    "/root/work/david_work/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth",
)

tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4', dir='/root/work/david_work/tmp')
```

**影响**: Stage 3 质量评估核心脚本，路径错误会导致模型加载失败。

---

### 1.3 `scripts/stage3_test_5s.py`

**路径类型**: 同 `stage3_batch_minimal.py`，另有输出文件路径

```python
# 额外修改
output_file = Path("/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_5s.jsonl")
# 改为
output_file = Path("/root/work/david_work/tmp/stage3_smoke_5s.jsonl")
```

---

## 二、调试/验证脚本（8 个，可选修改）

这些文件用于本地测试，**CMCC 生产环境可不复制**，如需复制则必须改路径。

### 2.1 测试数据路径相关

| 文件 | 硬编码路径 | 路径类型 |
|------|-----------|---------|
| `scripts/analyze_smoke_results.py` | `/mnt/afs/davidwang/workspace/sana_test_data/smoke_result` | 测试数据目录 |
| `scripts/compare_scales.py` | `/mnt/afs/davidwang/workspace/sana_test_data/` | 测试数据目录 |
| `scripts/diagnose_depth_scale.py` | `/mnt/afs/davidwang/workspace/sana_test_data/` | 测试数据目录 |
| `scripts/verify_dover_fix.py` | `/mnt/afs/davidwang/workspace/data/spatialvid_001/` | 测试视频路径 |
| `scripts/verify_official_alignment.py` | `/mnt/afs/davidwang/workspace/data/spatialvid_001/` | 测试视频路径 |

### 2.2 Shell 脚本路径

| 文件 | 硬编码路径 | 路径类型 |
|------|-----------|---------|
| `scripts/run_smoke_test_comparison.sh` | `/mnt/afs/davidwang/workspace/sana_wm_pipeline`<br>`/mnt/afs/davidwang/miniconda3/` | 项目根目录<br>Conda 路径 |
| `scripts/smoke_test_2s.sh` | `/mnt/afs/davidwang/workspace/data/spatialvid_001/` | 测试数据路径 |
| `scripts/download_smoke_*.sh` | `/mnt/afs/davidwang/workspace/data/` | 数据下载路径 |

**建议**: 这些脚本仅用于开发调试，CMCC 生产环境可忽略。

---

## 三、环境变量配置（推荐方式）

### 3.1 CMCC 服务器环境变量设置

在 `~/.bashrc` 或启动脚本中添加：

```bash
# Stage 2 模型权重
export SANA_WM_PI3X_WEIGHTS=/root/work/david_work/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/root/work/david_work/models/moge2

# Stage 2 最大帧数限制（可选，默认 64）
export SANA_WM_MAX_FRAMES=64

# PyTorch 缓存路径（DOVER 需要 ConvNeXt 权重）
export TORCH_HOME=/root/work/david_work/cache/torch

# HuggingFace 缓存路径（如果使用 HF 模型）
export HF_HOME=/root/work/david_work/cache/huggingface
```

**优先级**: 环境变量 > 代码硬编码默认值。设置环境变量后无需修改代码。

---

## 四、路径映射表

| 路径类型 | 本机路径 | CMCC 路径 | 说明 |
|---------|---------|----------|------|
| **模型权重** | `/mnt/afs/davidwang/models/` | `/root/work/david_work/models/` | Pi3x, Moge2, DOVER, UniMatch |
| **项目根目录** | `/mnt/afs/davidwang/workspace/sana_wm_pipeline/` | `/root/work/david_work/sana_qc_pipeline/` | 代码仓库 |
| **数据输入** | `/mnt/afs/davidwang/workspace/data/` | `/root/work/filestorage/shangaoooooo/davidwang/` | 原始视频 |
| **数据输出** | `/mnt/afs/davidwang/workspace/sana_wm_pipeline/outputs/` | `/root/work/david_work/qc_output/` | 处理结果 |
| **临时文件** | `/mnt/afs/davidwang/workspace/data/*/tmp/` | `/root/work/david_work/tmp/` | 临时视频切片 |
| **缓存目录** | `~/.cache/torch/` | `/root/work/david_work/cache/torch/` | PyTorch 模型缓存 |
| **Conda 环境** | `/mnt/afs/davidwang/miniconda3/` | `/root/miniconda3/` 或 `/opt/conda/` | Conda 安装路径 |

---

## 五、快速修改脚本

### 5.1 批量替换路径（在 CMCC 上执行）

```bash
cd /root/work/david_work/sana_qc_pipeline

# 备份原文件
find src scripts -name "*.py" -exec cp {} {}.bak \;

# 替换模型路径
find src scripts -name "*.py" -exec sed -i \
  's|/mnt/afs/davidwang/models/|/root/work/david_work/models/|g' {} \;

# 替换项目根路径
find src scripts -name "*.py" -exec sed -i \
  's|/mnt/afs/davidwang/workspace/sana_wm_pipeline/|/root/work/david_work/sana_qc_pipeline/|g' {} \;

# 替换数据路径
find src scripts -name "*.py" -exec sed -i \
  's|/mnt/afs/davidwang/workspace/data/|/root/work/david_work/tmp/|g' {} \;

# 验证修改
grep -r "/mnt/afs/davidwang" src scripts --include="*.py" | wc -l
# 输出应为 0
```

### 5.2 仅修改核心文件（最小改动）

```bash
cd /root/work/david_work/sana_qc_pipeline

# 仅修改 3 个核心文件
sed -i 's|/mnt/afs/davidwang/models/|/root/work/david_work/models/|g' \
  src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py

sed -i 's|/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/|/root/work/david_work/models/|g' \
  scripts/stage3_batch_minimal.py \
  scripts/stage3_test_5s.py

sed -i 's|/mnt/afs/davidwang/workspace/data/.*/tmp|/root/work/david_work/tmp|g' \
  scripts/stage3_batch_minimal.py \
  scripts/stage3_test_5s.py
```

---

## 六、验证清单

### 6.1 路径配置验证

```bash
cd /root/work/david_work/sana_qc_pipeline

# 1. 检查模型路径是否存在
ls -l /root/work/david_work/models/pi3x
ls -l /root/work/david_work/models/moge2
ls -l /root/work/david_work/models/DOVER/pretrained_weights/DOVER.pth
ls -l /root/work/david_work/models/unimatch/pretrained/*.pth

# 2. 检查临时目录权限
mkdir -p /root/work/david_work/tmp
touch /root/work/david_work/tmp/test.txt && rm /root/work/david_work/tmp/test.txt

# 3. 验证环境变量
echo $SANA_WM_PI3X_WEIGHTS
echo $SANA_WM_MOGE2_WEIGHTS
echo $TORCH_HOME

# 4. Python 导入测试
python -c "from src.sana_wm_pipeline.sana_wm_data_clean.pose._real import _PI3X_WEIGHTS, _MOGE2_WEIGHTS; print(f'Pi3x: {_PI3X_WEIGHTS}\nMoge2: {_MOGE2_WEIGHTS}')"
```

### 6.2 功能验证

```bash
# 测试 Stage 2 模型加载（需要 GPU）
python -c "
import os
os.environ['SANA_WM_PI3X_WEIGHTS'] = '/root/work/david_work/models/pi3x'
os.environ['SANA_WM_MOGE2_WEIGHTS'] = '/root/work/david_work/models/moge2'
from src.sana_wm_pipeline.sana_wm_data_clean.pose.stage import stage2_default
print('✓ Stage 2 模块加载成功')
"

# 测试 Stage 3 脚本（dry-run）
python scripts/stage3_batch_minimal.py --help
```

---

## 七、常见问题

### Q1: 设置了环境变量但还是报路径错误？

**A**: 检查环境变量是否生效：
```bash
python -c "import os; print(os.environ.get('SANA_WM_PI3X_WEIGHTS', 'NOT SET'))"
```

如果显示 `NOT SET`，说明环境变量未加载。确保：
1. 已将 `export` 语句写入 `~/.bashrc`
2. 执行了 `source ~/.bashrc` 或重新登录
3. 在 Python 脚本执行前设置（如果通过 shell 脚本启动）

---

### Q2: DOVER 报错 `FileNotFoundError: convnext_base_1k_224_ema.pth`？

**A**: DOVER 会自动下载 ConvNeXt 权重到 `$TORCH_HOME`。CMCC 离线环境需提前准备：

```bash
# 本机下载
mkdir -p /tmp/torch_cache/hub/checkpoints
cd /tmp/torch_cache/hub/checkpoints
wget https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth

# 传输到 CMCC
scp -r /tmp/torch_cache/hub cmcc:/root/work/david_work/cache/torch/

# CMCC 上验证
ls -lh /root/work/david_work/cache/torch/hub/checkpoints/convnext_base_1k_224_ema.pth
```

---

### Q3: 是否需要修改 `run_worker.py` 中的路径？

**A**: **不需要**。`run_worker.py` 中的路径是命令行参数示例（注释中），实际使用时通过 `--data-root` 等参数指定，不受硬编码影响。

示例：
```python
# run_worker.py 中的注释（无需修改）
"""
Example:
    python run_worker.py \
        --data-root /root/work/externalstorage/.../jdvbbfb-v3-full \
        --out-base  /root/work/filestorage/jdvbbfb_output
"""
```

---

## 八、迁移优先级

| 优先级 | 文件 | 修改方式 | 不修改后果 |
|-------|------|---------|----------|
| **P0** | `src/.../pose/_real.py` | 环境变量 or 硬改 | Stage 2 无法运行 |
| **P0** | `scripts/stage3_batch_minimal.py` | 硬改 | Stage 3 无法运行 |
| **P0** | `scripts/stage3_test_5s.py` | 硬改 | 测试脚本无法运行 |
| P1 | `scripts/verify_*.py` | 硬改 | 仅影响验证，可跳过 |
| P2 | `scripts/run_smoke_test_comparison.sh` | 硬改 | 仅影响本地测试 |
| P3 | 其他 `scripts/*.py` | 可不复制 | 调试脚本，CMCC 用不到 |

**结论**: 核心修改仅需 3 个文件，其余脚本为调试工具，可选择性迁移。

---

## 九、推荐迁移方案

### 方案 A：环境变量优先（推荐）

1. 在 CMCC 上设置环境变量（Stage 2 模型路径）
2. 仅修改 2 个 Stage 3 脚本的硬编码路径
3. 其他调试脚本不复制

**优点**: 改动最小，维护简单。  
**适用**: 生产环境，长期使用。

---

### 方案 B：硬编码全部替换

1. 使用批量替换脚本修改所有文件
2. 包括调试脚本一并迁移

**优点**: 一劳永逸，无需管理环境变量。  
**缺点**: 改动大，容易漏改。  
**适用**: 完全离线环境，需要所有脚本。

---

**文档版本**: v1.0  
**生成时间**: 2026-08-16  
**维护者**: Claude (Ponytail Mode)
