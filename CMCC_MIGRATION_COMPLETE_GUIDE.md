# CMCC 离线迁移完整指南

**目标环境**: CMCC 算力服务器（无外网、无 GitHub 访问）  
**迁移方式**: 完整 tar 打包 + 手动传输  
**生成日期**: 2026-08-16

---

## 一、任务1：完整打包迁移方案评估

### 1.1 可行性分析

**✅ 方案可行**，但需注意以下关键点。

---

### 1.2 项目规模统计

| 项目 | 大小 | 说明 |
|------|------|------|
| **仓库总大小** | 51GB | 包含 .git 历史 |
| `.git` 目录 | 41GB | Git 历史（占 80%） |
| `models/DOVER/.git` | 47MB | submodule git |
| `models/unimatch/.git` | 22MB | submodule git |
| `third_party/vipe/.git` | 14MB | submodule git |
| **代码 + 配置** | ~10GB | 实际工作文件 |

**关键发现**:
- `.git` 占据 80% 空间，但离线环境无法用 git 操作
- `models/` 目录 512MB（DOVER + UniMatch），需保留
- `third_party/vipe` 仅 45MB，需完整保留

---

### 1.3 推荐打包策略

#### 方案 A：完整打包（含 .git，推荐）

**优点**:
- 保留完整历史，CMCC 侧可查看 commit、diff
- 可使用 `git show`、`git log` 等只读命令
- 未来若 CMCC 打通外网可直接 pull 更新

**缺点**:
- 打包体积 ~50GB（压缩后约 20-30GB）
- 传输时间较长

**打包命令**:
```bash
cd /mnt/afs/davidwang/workspace

# 完整打包（含 .git 和所有 submodule）
tar --exclude='outputs' \
    --exclude='results' \
    --exclude='cache' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.tar' \
    -czf sana_wm_pipeline_full_$(date +%Y%m%d).tar.gz \
    sana_wm_pipeline/

# 验证打包
tar -tzf sana_wm_pipeline_full_$(date +%Y%m%d).tar.gz | grep "third_party/vipe" | head -10
tar -tzf sana_wm_pipeline_full_$(date +%Y%m%d).tar.gz | grep "models/DOVER" | head -10

echo "打包完成:"
ls -lh sana_wm_pipeline_full_$(date +%Y%m%d).tar.gz
```

---

#### 方案 B：代码打包（不含 .git，备选）

**优点**:
- 体积小 ~10GB（压缩后约 3-5GB）
- 传输快

**缺点**:
- 丢失 git 历史，无法使用 git 命令
- 未来更新困难

**打包命令**:
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 使用 git archive（自动排除 .git）
git archive --format=tar.gz \
    --prefix=sana_wm_pipeline/ \
    -o ../sana_wm_pipeline_code_$(date +%Y%m%d).tar.gz \
    HEAD

# 手动添加 submodule（git archive 不含 submodule）
cd third_party/vipe
git archive --format=tar.gz \
    --prefix=sana_wm_pipeline/third_party/vipe/ \
    -o /tmp/vipe.tar.gz \
    HEAD

cd /mnt/afs/davidwang/workspace
tar -xzf sana_wm_pipeline_code_$(date +%Y%m%d).tar.gz
tar -xzf /tmp/vipe.tar.gz
tar -czf sana_wm_pipeline_code_final_$(date +%Y%m%d).tar.gz sana_wm_pipeline/
rm -rf sana_wm_pipeline/ /tmp/vipe.tar.gz
```

---

### 1.4 Submodule 处理方案

#### 问题：`git archive` 不包含 submodule

**解决方案 1：递归 tar（推荐）**
```bash
# 直接 tar 整个目录，包含所有 submodule 内容
cd /mnt/afs/davidwang/workspace
tar --exclude='.git' \
    --exclude='outputs' \
    --exclude='__pycache__' \
    -czf sana_wm_pipeline_with_submodules_$(date +%Y%m%d).tar.gz \
    sana_wm_pipeline/

# 验证 submodule 完整性
tar -tzf sana_wm_pipeline_with_submodules_*.tar.gz | grep "third_party/vipe/vipe/" | wc -l
# 应输出 >100（vipe 源码文件数）
```

**解决方案 2：单独打包 submodule**
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 主仓库
git archive -o /tmp/main.tar.gz HEAD

# Submodule
cd third_party/vipe
git archive --prefix=third_party/vipe/ -o /tmp/vipe.tar.gz HEAD

# 合并
mkdir -p /tmp/merged
cd /tmp/merged
tar -xzf /tmp/main.tar.gz
tar -xzf /tmp/vipe.tar.gz
tar -czf /tmp/sana_wm_pipeline_merged_$(date +%Y%m%d).tar.gz .
```

---

### 1.5 迁移风险点

| 风险点 | 影响 | 缓解措施 |
|--------|------|----------|
| **Submodule 丢失** | VIPE 模块无法使用，Stage 2 失败 | 验证 `third_party/vipe/` 完整性 |
| **模型权重未打包** | 需在 CMCC 重新下载（无外网则失败） | 单独打包 `models/` 或提前拷贝到 CMCC |
| **硬编码路径** | 解压后路径错误无法运行 | 见任务2清单，批量替换 |
| **Conda 环境** | CMCC 需重建环境 | 导出 `environment.yaml` 或打包整个 conda env |
| **Git 无法操作** | 不影响运行，仅影响开发体验 | 保留 .git 目录用于只读查询 |

---

### 1.6 传输与部署流程

#### 步骤 1：本机打包
```bash
cd /mnt/afs/davidwang/workspace

# 执行方案 A 打包命令（推荐）
tar --exclude='outputs' --exclude='__pycache__' \
    -czf sana_wm_pipeline_full_$(date +%Y%m%d).tar.gz \
    sana_wm_pipeline/

# 生成 MD5 校验
md5sum sana_wm_pipeline_full_*.tar.gz > sana_wm_pipeline_full.md5

# 输出文件信息
ls -lh sana_wm_pipeline_full_*
```

#### 步骤 2：传输到 CMCC（人工方式）
- 通过跳板机、U盘、内网文件传输系统等方式
- 验证传输完整性：`md5sum -c sana_wm_pipeline_full.md5`

#### 步骤 3：CMCC 侧解压部署
```bash
# 解压到工作目录
cd /root/work/david_work
tar -xzf sana_wm_pipeline_full_20260816.tar.gz

# 验证关键文件
ls -la sana_wm_pipeline/third_party/vipe/
ls -la sana_wm_pipeline/models/DOVER/
ls -la sana_wm_pipeline/src/sana_wm_pipeline/sana_wm_data_clean/

# 检查 submodule
python -c "import sys; sys.path.insert(0, 'sana_wm_pipeline/third_party'); from vipe.models import Pi3xMogeModel; print('✓ VIPE 可用')"
```

#### 步骤 4：路径配置（见任务2）

---

### 1.7 模型权重处理

**关键**: `models/` 目录包含 DOVER 和 UniMatch 权重（512MB），**必须**打包传输。

**验证命令**（CMCC 侧）:
```bash
cd /root/work/david_work/sana_wm_pipeline

ls -lh models/DOVER/pretrained_weights/DOVER.pth
# 应为 ~280MB

ls -lh models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth
# 应为 ~180MB

# 测试加载
python -c "
import torch
dover = torch.load('models/DOVER/pretrained_weights/DOVER.pth', map_location='cpu')
print(f'✓ DOVER 权重可加载，keys: {len(dover)}')
"
```

---

### 1.8 Conda 环境迁移

**问题**: Conda 环境无法直接打包传输（路径依赖强）。

**解决方案**:

**方式 1：导出依赖列表（推荐）**
```bash
# 本机导出
conda activate sana_wm
conda env export --no-builds > environment.yaml
pip list --format=freeze > requirements.txt

# CMCC 侧重建
conda env create -f environment.yaml -p /root/work/david_work/conda_envs/sana_wm
# 或
conda create -n sana_wm python=3.10
conda activate sana_wm
pip install -r requirements.txt
```

**方式 2：打包完整环境（备选，体积大）**
```bash
# 本机打包 conda env
conda pack -n sana_wm -o sana_wm_env.tar.gz

# CMCC 侧解压
mkdir -p /root/work/david_work/conda_envs/sana_wm
tar -xzf sana_wm_env.tar.gz -C /root/work/david_work/conda_envs/sana_wm
source /root/work/david_work/conda_envs/sana_wm/bin/activate
conda-unpack  # 修复路径引用
```

---

### 1.9 推荐最终方案

**组合方案（最佳）**:
1. 主仓库：方案 A（完整打包含 .git）
2. Conda 环境：导出 `environment.yaml`（CMCC 侧重建）
3. 验证清单：打包后验证 submodule、models、配置文件完整性

**预期工作量**:
- 本机打包：30 分钟
- 传输时间：1-3 小时（取决于网络）
- CMCC 部署：1 小时
- 路径配置：30 分钟
- 功能验证：1 小时
- **总计**: 4-6 小时

---

## 二、任务2：硬编码路径完整扫描结果

### 2.1 扫描统计

**全项目扫描结果**:
- **总扫描**: 2274 处路径引用
- **代码文件**: 248 处（77 个文件）
- **文档**: 2026 处（排除，不影响运行）

**路径类型分布**（仅代码文件）:
| 类型 | 数量 | 是否必须修改 |
|------|------|-------------|
| 模型权重路径 | 45 处 | ✅ 必须 |
| 数据集路径 | 89 处 | ⚠️ 部分必须 |
| 输出结果路径 | 34 处 | ⚠️ 部分必须 |
| Conda环境路径 | 28 处 | ❌ 可选（用环境变量） |
| 缓存路径 | 12 处 | ⚠️ 建议修改 |
| 其他 | 40 处 | ❌ 大多可忽略 |

---

### 2.2 P0 级别：必须修改的文件（3 个）

这些文件不修改将**直接导致运行失败**。

#### 2.2.1 `src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py`

**位置**: 行 15-16  
**路径类型**: 模型权重（环境变量默认值）  
**当前值**:
```python
_PI3X_WEIGHTS = os.environ.get("SANA_WM_PI3X_WEIGHTS", "/mnt/afs/davidwang/models/pi3x")
_MOGE2_WEIGHTS = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "/mnt/afs/davidwang/models/moge2")
```

**修改建议**:
```bash
# 方式 1：环境变量（推荐）
export SANA_WM_PI3X_WEIGHTS=/root/work/david_work/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/root/work/david_work/models/moge2

# 方式 2：硬改代码
sed -i 's|/mnt/afs/davidwang/models/|/root/work/david_work/models/|g' \
  src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py
```

---

#### 2.2.2 `scripts/stage3_batch_minimal.py`

**位置**: 行 10-30（多处）  
**路径类型**: 模型路径 + 临时文件目录  
**当前值**:
```python
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch")
with open("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml") as f:
scorer = DOVER("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth")
flow_estimator = UniMatchFlow("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth")
tmp_path = tempfile.mkstemp(suffix='.mp4', dir='/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp')
```

**修改建议**:
```bash
# 批量替换（CMCC 侧执行）
cd /root/work/david_work/sana_wm_pipeline

sed -i 's|/mnt/afs/davidwang/workspace/sana_wm_pipeline|/root/work/david_work/sana_wm_pipeline|g' \
  scripts/stage3_batch_minimal.py

sed -i 's|/mnt/afs/davidwang/workspace/data/.*/tmp|/root/work/david_work/tmp|g' \
  scripts/stage3_batch_minimal.py
```

---

#### 2.2.3 `scripts/stage3_test_5s.py`

**位置**: 同 `stage3_batch_minimal.py`，另有输出文件路径  
**路径类型**: 模型路径 + 输出文件  

**修改建议**: 同 `stage3_batch_minimal.py`

---

### 2.3 P1 级别：配置文件（2 个）

不修改会影响批量生产，但不影响单样本测试。

#### 2.3.1 `configs/pipeline.yaml`

**位置**: 行 13-15  
**路径类型**: 数据集路径  
**当前值**:
```yaml
paths:
  raw_root: /mnt/afs/davidwang/workspace/data/sana_wm/raw
  staging: /mnt/afs/davidwang/workspace/data/sana_wm/staging
  out_root: /mnt/afs/davidwang/workspace/data/sana_wm/shards
```

**修改建议**:
```bash
sed -i 's|/mnt/afs/davidwang/workspace/data/sana_wm/|/root/work/filestorage/davidwang/sana_wm/|g' \
  configs/pipeline.yaml
```

---

#### 2.3.2 `configs/sources.yaml`

**位置**: 行 14  
**路径类型**: 数据集路径示例  
**当前值**:
```yaml
local_path_example: /mnt/aos/davidwang/workspace/data/spatialvid_hq/SpatialVID/videos/group_0001
```

**修改建议**: 这是示例路径，实际使用时通过命令行参数传入，可不改。

---

### 2.4 P2 级别：批量生产脚本（CMCC 特有，5 个）

这些脚本是为 CMCC 多机环境编写的，需根据实际路径调整。

| 文件 | 路径类型 | 影响 |
|------|---------|------|
| `experiments/batch_production/config.sh` | 数据集路径 | CMCC 批量生产必改 |
| `experiments/batch_production/launch_all_nodes.sh` | hostfile 路径 | 多机分布式必改 |
| `experiments/batch_production/sync_to_nodes.sh` | 同步路径 | 多机同步必改 |
| `experiments/vipe_comparison/compare.sh` | 测试数据路径 | 可选，调试用 |
| `scripts/run_stage3_cmcc*.py` | CMCC 路径 | 已包含 CMCC 路径，需微调 |

**修改建议**: 这些脚本中的路径已经是 `/root/work/` 开头（CMCC 标准路径），仅需根据实际部署位置微调。

---

### 2.5 P3 级别：测试脚本（11 个，可选）

这些脚本用于本地开发调试，CMCC 生产环境可不复制。

| 文件 | 路径类型 | 是否复制 |
|------|---------|---------|
| `scripts/analyze_smoke_results.py` | 测试数据 | ❌ 跳过 |
| `scripts/compare_scales.py` | 测试数据 | ❌ 跳过 |
| `scripts/diagnose_depth_scale.py` | 测试数据 | ❌ 跳过 |
| `scripts/verify_dover_fix.py` | 测试视频 | ❌ 跳过 |
| `scripts/verify_official_alignment.py` | 测试视频 | ❌ 跳过 |
| `scripts/run_smoke_test_comparison.sh` | 本机路径 | ❌ 跳过 |
| `scripts/smoke_test_*.sh` | 测试数据 | ❌ 跳过 |
| `models/DOVER/test_dover_h100.py` | 测试脚本 | ❌ 跳过 |
| 其他 `test_*.py` | 单元测试 | ⚠️ 可选保留 |

---

### 2.6 third_party 扫描结果

**✅ 无硬编码路径发现**

扫描了 `third_party/vipe/` 的所有 `.py`、`.sh`、`.yaml`、`.json` 文件，**未发现**任何 `/mnt/afs/davidwang` 或 `/home/`、`/root/` 等硬编码路径。

**结论**: VIPE submodule 代码清洁，无需修改路径。

---

### 2.7 批量修改脚本（CMCC 侧执行）

#### 一键修改核心文件
```bash
#!/bin/bash
# 文件名: fix_paths_cmcc.sh
# 在 CMCC 侧执行

set -e

PROJECT_ROOT="/root/work/david_work/sana_wm_pipeline"
cd "$PROJECT_ROOT"

echo "=== 开始修改硬编码路径 ==="

# 备份
echo "1. 备份原文件..."
find src scripts configs -name "*.py" -o -name "*.sh" -o -name "*.yaml" | xargs -I {} cp {} {}.bak

# P0: 核心代码文件
echo "2. 修改核心代码文件..."
sed -i 's|/mnt/afs/davidwang/models/|/root/work/david_work/models/|g' \
  src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py

sed -i 's|/mnt/afs/davidwang/workspace/sana_wm_pipeline|/root/work/david_work/sana_wm_pipeline|g' \
  scripts/stage3_batch_minimal.py \
  scripts/stage3_test_5s.py

sed -i 's|/mnt/afs/davidwang/workspace/data/.*/tmp|/root/work/david_work/tmp|g' \
  scripts/stage3_batch_minimal.py \
  scripts/stage3_test_5s.py

# P1: 配置文件
echo "3. 修改配置文件..."
sed -i 's|/mnt/afs/davidwang/workspace/data/sana_wm/|/root/work/filestorage/davidwang/sana_wm/|g' \
  configs/pipeline.yaml

# 验证
echo "4. 验证修改..."
grep -r "/mnt/afs/davidwang" src/sana_wm_pipeline/sana_wm_data_clean/ scripts/stage3_*.py configs/*.yaml && \
  echo "⚠️ 警告：仍存在未替换的路径" || \
  echo "✓ 核心路径修改完成"

echo "=== 修改完成 ==="
echo "备份文件: *.bak（如需回滚，执行 find . -name '*.bak' -exec bash -c 'mv \"\$0\" \"\${0%.bak}\"' {} \;）"
```

#### 环境变量配置（推荐）
```bash
# 添加到 ~/.bashrc 或启动脚本
cat >> ~/.bashrc << 'EOF'

# SANA-WM Pipeline 环境变量
export SANA_WM_PI3X_WEIGHTS=/root/work/david_work/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/root/work/david_work/models/moge2
export TORCH_HOME=/root/work/david_work/cache/torch
export HF_HOME=/root/work/david_work/cache/huggingface

EOF

source ~/.bashrc
```

---

### 2.8 验证清单

#### CMCC 侧部署后验证
```bash
cd /root/work/david_work/sana_wm_pipeline

# 1. 检查 submodule
python -c "from third_party.vipe.models import Pi3xMogeModel; print('✓ VIPE 可用')"

# 2. 检查模型权重
ls -lh models/DOVER/pretrained_weights/DOVER.pth
ls -lh models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth

# 3. 检查路径配置
python -c "
from src.sana_wm_pipeline.sana_wm_data_clean.pose._real import _PI3X_WEIGHTS, _MOGE2_WEIGHTS
print(f'Pi3x: {_PI3X_WEIGHTS}')
print(f'Moge2: {_MOGE2_WEIGHTS}')
assert '/root/work' in _PI3X_WEIGHTS, '路径未修改'
print('✓ 路径配置正确')
"

# 4. 创建必需目录
mkdir -p /root/work/david_work/tmp
mkdir -p /root/work/filestorage/davidwang/sana_wm/{raw,staging,shards}

# 5. 测试导入
python -c "
import sys
sys.path.insert(0, 'src')
from sana_wm_pipeline.sana_wm_data_clean.pose import stage
print('✓ 模块导入成功')
"
```

---

## 三、常见问题

### Q1: 打包后 submodule 是空目录？

**原因**: 使用了 `git archive`（不含 submodule）。  
**解决**: 用 `tar` 而非 `git archive`，或单独打包 submodule 后合并。

---

### Q2: CMCC 解压后 Python 找不到模块？

**原因**: `PYTHONPATH` 未设置或路径错误。  
**解决**:
```bash
export PYTHONPATH=/root/work/david_work/sana_wm_pipeline/src:$PYTHONPATH
# 或在脚本中
sys.path.insert(0, '/root/work/david_work/sana_wm_pipeline/src')
```

---

### Q3: 环境变量不生效？

**原因**: 未 source `.bashrc` 或在 Python 启动前设置。  
**解决**:
```bash
# 确认环境变量
echo $SANA_WM_PI3X_WEIGHTS

# 重新加载
source ~/.bashrc

# 或在 Python 中显式设置
import os
os.environ['SANA_WM_PI3X_WEIGHTS'] = '/root/work/david_work/models/pi3x'
```

---

### Q4: DOVER 报错找不到 ConvNeXt 权重？

**原因**: DOVER 自动下载权重，离线环境无法下载。  
**解决**: 提前下载并放到 `$TORCH_HOME/hub/checkpoints/`：
```bash
# 本机下载
mkdir -p /tmp/torch_cache/hub/checkpoints
wget -P /tmp/torch_cache/hub/checkpoints \
  https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth

# 打包传输到 CMCC
tar -czf torch_cache.tar.gz -C /tmp torch_cache/

# CMCC 侧解压
mkdir -p /root/work/david_work/cache/torch
tar -xzf torch_cache.tar.gz -C /root/work/david_work/cache/torch --strip-components=1
```

---

## 四、总结

### 关键步骤
1. **本机打包**: 使用方案 A（完整 tar，含 .git）
2. **验证完整性**: 检查 submodule、models、配置文件
3. **传输到 CMCC**: 人工方式 + MD5 校验
4. **CMCC 解压**: 解压到 `/root/work/david_work/`
5. **修改路径**: 执行 `fix_paths_cmcc.sh` + 设置环境变量
6. **功能验证**: 运行验证清单确认无误

### 必改文件（最小集）
1. `src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py`（或用环境变量）
2. `scripts/stage3_batch_minimal.py`
3. `scripts/stage3_test_5s.py`
4. `configs/pipeline.yaml`

### 工作量评估
- 打包 + 传输: 2-4 小时
- 部署 + 配置: 1-2 小时
- 验证测试: 1 小时
- **总计**: 4-7 小时

---

**文档版本**: v2.0  
**最后更新**: 2026-08-16  
**维护者**: Claude (Ponytail Mode)
