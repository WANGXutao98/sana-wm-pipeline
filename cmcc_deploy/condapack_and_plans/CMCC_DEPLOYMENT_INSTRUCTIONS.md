# CMCC 环境部署完整指南

**日期**: 2026-08-23  
**状态**: ⚠️ 部署中断 - Pi3X 推理 Segmentation Fault  
**环境**: NVIDIA H100 80GB, CUDA 12.4, Python 3.10

---

## 🔴 当前状态（2026-08-23 更新）

### 已完成
- ✅ conda 环境打包、传输、解压
- ✅ PYTHONPATH 配置正确
- ✅ 基础模块导入测试通过（torch, vipe, sana_wm_pipeline）
- ✅ 冒烟测试脚本路径修正

### 当前阻塞问题
- ❌ **Pi3X 模型推理时出现 Segmentation Fault**
- 位置：`mode_default.py` 的 `pi3_infer()` 调用
- 阶段：视频归一化后，VIPE SLAM 的 Pi3 深度推理阶段

### 关键发现
1. **本机环境（/mnt/afs/davidwang/workspace）运行正常**
   - 测试脚本：`smoke_spatialvid.sh` 完整流程通过
   - 说明代码逻辑无问题
   
2. **CMCC 环境运行 segfault**
   - 环境预检通过（torch, vipe 导入成功）
   - 视频归一化成功（46 帧）
   - Pi3 推理时崩溃
   
3. **环境差异确认**
   | 环境 | PyTorch | CUDA (编译) | CUDA Driver | cuDNN |
   |------|---------|-------------|-------------|-------|
   | 本机 | 2.12.0+cu130 | 13.0 | 13.0 | 92000 |
   | CMCC | 2.6.0+cu124 | 12.4 | 13.0 | 90100 |
   
4. **Pi3 CUDA 扩展缺失（本机和 CMCC 都缺失）**
   - 两个环境都没有 Pi3 的 CUDA 编译扩展（.so 文件）
   - 都显示警告：`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead`
   - 本机能正常运行，说明缺失扩展不是根因

---

## 🔍 Segmentation Fault 问题分析（2026-08-23）

### 问题现象

```bash
[Stage 2] VIPE SLAM (Pi3X + MoGe-2)...
[mode_default] Phase A: 深度预计算
  读取视频: /root/work/david_work/smoke_pass_results/run_20260823_040216/00094653-a9c6-5558-8e2a-4119e7d64f36/normalized.mp4
  采样帧数: 46
  Pi3推理 (46帧)...
Segmentation fault (core dumped)
```

### 排查过程

#### 1. 验证环境基础功能
- ✅ PyTorch CUDA 可用
- ✅ vipe 模块导入成功
- ✅ Pi3X 模型加载成功
- ❌ Pi3X 推理时 segfault

#### 2. 对比本机与 CMCC
- 本机：`smoke_spatialvid.sh` 完整流程通过
- CMCC：相同代码在 Pi3 推理阶段崩溃
- **结论**：不是代码逻辑问题，是环境差异

#### 3. PyTorch 版本差异
```
本机：PyTorch 2.12.0 + CUDA 13.0
CMCC：PyTorch 2.6.0 + CUDA 12.4
Driver：都是 13.0（向下兼容，理论上无问题）
```

#### 4. Pi3 CUDA 扩展检查
```bash
# 本机和 CMCC 都没有 Pi3 的 .so 扩展文件
find ... -name "*.so" | grep -i pi3
# 结果：空

# 两个环境都显示相同警告
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
```

**关键推论**：
- Pi3 CUDA 扩展缺失不是根因（本机也缺失但能运行）
- PyTorch 版本差异可能导致纯 Python RoPE2D 实现的行为不同
- 或者 PyTorch 2.6 vs 2.12 的 CUDA kernel 层面有差异

---

## 🔧 待验证的解决方案

### 方案 1：conda-unpack 修复（最简单）⭐

**理论**：conda-pack 打包的环境，路径硬编码可能导致库加载失败

```bash
# 在 CMCC 执行
source /root/work/david_work/envs/sana_wm_cmcc/bin/activate
conda-unpack

# 检查是否执行过
ls -la /root/work/david_work/envs/sana_wm_cmcc/conda-meta/state
```

**状态**：待执行

---

### 方案 2：检查 torch 库依赖（排查系统库缺失）

```bash
# 在 CMCC 执行
ldd /root/work/david_work/envs/sana_wm_cmcc/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so
```

查找是否有 `not found` 的依赖库。

**状态**：待执行

---

### 方案 3：检查 PyTorch ABI 兼容性

```bash
# 本机
conda activate sana_wm
python -c "import torch; print(f'ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"

# CMCC
source /root/work/david_work/envs/sana_wm_cmcc/bin/activate
python -c "import torch; print(f'ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
```

如果输出不同 → ABI 不兼容，需要重新编译或重新打包。

**状态**：待执行

---

### 方案 4：在本机重新打包（匹配 CMCC 环境）

**步骤**：

1. 在本机创建新环境（使用更老的 PyTorch 版本）
   ```bash
   cd /mnt/afs/davidwang/workspace
   conda create -n sana_wm_cu124 python=3.10 -y
   conda activate sana_wm_cu124
   
   # 安装和 CMCC 相同的 PyTorch
   pip install torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

2. 安装其他依赖
   ```bash
   conda activate sana_wm
   pip freeze > /tmp/sana_wm_requirements.txt
   
   conda activate sana_wm_cu124
   pip install -r /tmp/sana_wm_requirements.txt
   ```

3. 测试验证
   ```bash
   cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
   conda activate sana_wm_cu124
   bash experiments/data_production_smoke/smoke_spatialvid.sh
   ```

4. 重新打包
   ```bash
   conda pack -n sana_wm_cu124 -o sana_wm_cmcc_cu124.tar.gz --ignore-editable-packages
   ```

5. 传输并重新部署到 CMCC

**状态**：备选方案（如果方案 1-3 无效）

---

### 方案 5：环境变量强制兼容

```bash
# 在 CMCC 脚本中添加
export CUDA_FORCE_PTX_JIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

尝试绕过 CUDA 版本严格检查。

**状态**：备选方案

---

### 方案 6：降低批处理大小（代码修改）

如果是显存分配问题，修改 `mode_default.py` 的 `pi3_infer` 为分批推理：

```python
# 将 46 帧改为每次推理 8 帧
batch_size = 8
for i in range(0, total_frames, batch_size):
    batch = frames[i:i+batch_size]
    result = model(batch)
    results.append(result)
```

**状态**：最后手段（需要修改代码）

---

## 📋 下一步行动指南

### 立即执行（按顺序）

1. **方案 1：conda-unpack**（5 分钟）
   - 在 CMCC 执行 `conda-unpack`
   - 重新运行测试
   
2. **方案 2：检查库依赖**（5 分钟）
   - 在 CMCC 运行 `ldd` 命令
   - 查看是否有缺失的系统库
   
3. **方案 3：ABI 检查**（5 分钟）
   - 本机和 CMCC 对比 ABI 标志
   - 确认是否需要重新编译

### 如果上述都无效

4. **方案 4：重新打包**（1-2 小时）
   - 在本机创建 PyTorch 2.6 环境
   - 完整测试后重新打包
   - 传输到 CMCC 替换

### 紧急绕过

5. **方案 6：修改代码降批处理**（30 分钟）
   - 只在 CMCC 需要快速出结果时使用
   - 性能会下降，但能跑通

---

## 一、部署概述

### 1.1 背景

将 `sana_wm` 和 `sana_qc` 两个 conda 环境打包并部署到 CMCC 离线服务器。由于打包过程中移除了 editable 包（解决 conda-pack 限制），CMCC 侧需要通过 PYTHONPATH 配置来使用项目代码。

### 1.2 实际路径

| 组件 | 打包机器路径 | CMCC 机器路径 |
|------|-------------|--------------|
| 环境包下载目录 | - | `/root/work/david_work/conda_envs_download/conda_envs/` |
| 解压后环境 | - | `/root/work/david_work/envs/sana_wm_cmcc/`<br>`/root/work/david_work/envs/sana_qc_cmcc/` |
| 项目代码 | `/mnt/afs/davidwang/workspace/sana_wm_pipeline/` | `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/` |

---

## 二、部署步骤

### 步骤 1: 解压 sana_wm_cmcc

```bash
mkdir -p /root/work/david_work/envs/sana_wm_cmcc
tar -xzf /root/work/david_work/conda_envs_download/conda_envs/sana_wm_cmcc.tar.gz \
    -C /root/work/david_work/envs/sana_wm_cmcc

cd /root/work/david_work/envs/sana_wm_cmcc
source bin/activate
conda-unpack
python -c "import torch; print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
deactivate
```

**预期输出**: `✓ torch 2.6.0+cu124 (CUDA: True)`

### 步骤 2: 解压 sana_qc_cmcc

```bash
mkdir -p /root/work/david_work/envs/sana_qc_cmcc
tar -xzf /root/work/david_work/conda_envs_download/conda_envs/sana_qc_cmcc.tar.gz \
    -C /root/work/david_work/envs/sana_qc_cmcc

cd /root/work/david_work/envs/sana_qc_cmcc
source bin/activate
conda-unpack
python -c "import torch; print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
deactivate
```

**预期输出**: `✓ torch 2.6.0+cu124 (CUDA: True)`

### 步骤 3: 配置 PYTHONPATH

**关键步骤**：由于环境中移除了 editable 包，必须配置 PYTHONPATH。

```bash
cat >> ~/.bashrc <<'EOF'

# SANA-WM Pipeline
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF

source ~/.bashrc
echo $PYTHONPATH
```

**预期输出**: 包含两个路径
```
/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:...
```

### 步骤 4: 运行冒烟测试

```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash scripts/smoke_test_cmcc_remote.sh
```

**预期输出**: 所有测试通过
```
✓✓✓ sana_wm_cmcc 测试通过 ✓✓✓
✓✓✓ sana_qc_cmcc 测试通过 ✓✓✓
✓✓✓ 所有测试通过，环境部署成功 ✓✓✓
```

---

## 三、关键问题与解决方案

### 3.1 问题：ModuleNotFoundError: No module named 'nvidia_vipe'

**现象**:
```python
from nvidia_vipe.models import Pi3xMogeModel
ModuleNotFoundError: No module named 'nvidia_vipe'
```

**根本原因**:
1. `pip list` 显示 `nvidia-vipe 1.1.0`，但实际无法导入
2. 这是因为 `nvidia-vipe` 在打包时作为 editable 包被移除（解决 conda-pack 限制）
3. 环境中已安装的是占位符，实际模块在源码目录中

**调查过程**:
```bash
# 检查实际模块结构
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe/
# 发现：
# - 有 vipe/ 子目录（真正的模块）
# - 有 vipe_ext.cpython-310-x86_64-linux-gnu.so（预编译扩展）
# - 没有 nvidia_vipe/ 目录

# 正确的导入方式
import vipe  # ✓ 正确
import nvidia_vipe  # ✗ 错误
```

**解决方案**:
```bash
# PYTHONPATH 必须包含 third_party/vipe
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
```

**关键发现**:
- 模块名是 `vipe`，不是 `nvidia_vipe`
- `pip list` 显示的 `nvidia-vipe` 包名不等于 Python 模块名
- 预编译扩展 `vipe_ext.so` 位于 `third_party/vipe/` 根目录
- JIT 编译失败（缺少 cusparse.h）不影响使用，因为会使用预编译扩展

### 3.2 问题：JIT 编译失败

**现象**:
```
fatal error: cusparse.h: No such file or directory
```

**原因**:
- 环境中缺少完整的 CUDA 开发头文件
- `cusparse.h` 位于 `/root/work/david_work/envs/sana_wm_cmcc/lib/python3.10/site-packages/nvidia/cu13/include/`
- 但编译器未正确找到该路径

**为什么不影响使用**:
- 项目源码目录中已有预编译的 `vipe_ext.cpython-310-x86_64-linux-gnu.so`
- Python 会优先使用预编译扩展，而不是尝试 JIT 编译
- 验证：`import vipe` 成功，所有功能正常

**实际验证**:
```bash
# 检查预编译扩展
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe/vipe_ext.cpython-310-x86_64-linux-gnu.so
# 输出：-rwxr-xr-x 1 10508 10508 6682560 May 28 17:53 vipe_ext.cpython-310-x86_64-linux-gnu.so

# 测试导入（成功）
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "
from sana_wm_pipeline.stage02_pose.mode_default import run_default
import torch
print('✓ 所有模块正常')
"
```

### 3.3 问题：PYTHONPATH 配置遗漏 third_party/vipe

**现象**:
```bash
echo $PYTHONPATH
/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:...
# 缺少 third_party/vipe 路径
```

**后果**:
- `import vipe` 失败
- 冒烟测试脚本中的导入检查失败

**解决**:
```bash
# 完整的 PYTHONPATH（必须包含两个路径）
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
```

### 3.4 问题：smoke_test_cmcc_remote.sh 路径错误

**现象**:
脚本中硬编码的路径不匹配实际部署路径。

**修正**:
```bash
# 修正前
PROJ=$BASE/sana_wm_pipeline

# 修正后
PROJ=$BASE/sana_wm_optimized/sana_wm_pipeline
```

**同时修正 PYTHONPATH**:
```bash
# 修正前
export PYTHONPATH=$PROJ/src:${PYTHONPATH:-}

# 修正后
export PYTHONPATH=$PROJ/src:$PROJ/third_party/vipe:${PYTHONPATH:-}
```

### 3.5 问题：smoke_cmcc_test_v1.sh 使用错误的 Python 环境

**现象**:
```bash
Python: 3.13.12 | packaged by Anaconda, Inc.
Conda 环境: base
Traceback: ModuleNotFoundError: No module named 'torch'
```

脚本调用的是系统默认的 `python` 命令，而不是 sana_wm_cmcc 环境的 Python。

**根本原因**:
- 脚本第 29 行注释掉了 `conda activate`
- 但第 45/126/150/185 行仍在使用 `python` 命令（而非绝对路径）
- 当前 shell 在 base 环境，导致调用 base 的 Python（3.13，无 torch）

**修正方案**:
```bash
# 在第 11 行后添加环境路径定义
export ENV_WM="$NEW_BASE/envs/sana_wm_cmcc"
export PYTHON="$ENV_WM/bin/python3"

# 替换所有 python 命令
# 第 45 行：python -c → $PYTHON -c
# 第 126 行：python -c → $PYTHON -c
# 第 150 行：python -c → $PYTHON -c
# 第 185 行：python - → $PYTHON -

# 第 40 行修改显示
echo "Python 环境: $ENV_WM"  # 原来是 Conda 环境: $CONDA_DEFAULT_ENV
```

**完整修改清单**（smoke_cmcc_test_v1.sh）:
1. **第 32 行**：PYTHONPATH 缺少 `/vipe`
   ```bash
   # 错误
   export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party${PYTHONPATH:+:$PYTHONPATH}"
   # 正确
   export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe${PYTHONPATH:+:$PYTHONPATH}"
   ```

2. **第 26 行后添加**：
   ```bash
   export TORCH_CUDA_ARCH_LIST=9.0
   ```

3. **第 11 行后添加**：
   ```bash
   export ENV_WM="$NEW_BASE/envs/sana_wm_cmcc"
   export PYTHON="$ENV_WM/bin/python3"
   ```

4. **所有 python 命令改为 $PYTHON**（第 45/126/150/185 行）

**状态**：已诊断，待用户手动修改

---

## 四、环境验证

### 4.1 快速验证命令

```bash
# 验证 sana_wm_cmcc
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "
from sana_wm_pipeline.stage02_pose.mode_default import run_default
import vipe
import torch
print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✓ vipe {vipe.__version__}')
print('✓ sana_wm_pipeline 模块导入成功')
"

# 验证 sana_qc_cmcc
/root/work/david_work/envs/sana_qc_cmcc/bin/python3 -c "
import torch, torchvision, transformers
print(f'✓ torch {torch.__version__}')
print(f'✓ torchvision {torchvision.__version__}')
print(f'✓ transformers {transformers.__version__}')
"
```

### 4.2 完整冒烟测试

```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash scripts/smoke_test_cmcc_remote.sh
```

**测试覆盖**:
- Python 解释器可执行性
- PyTorch + CUDA 可用性
- JIT 编译能力（C++ 和 CUDA）
- VIPE SLAM 模块导入
- QC 依赖库完整性

---

## 五、使用方式

### 5.1 直接调用（推荐）

```bash
# 使用 sana_wm_cmcc
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 your_script.py

# 使用 sana_qc_cmcc
/root/work/david_work/envs/sana_qc_cmcc/bin/python3 your_script.py
```

**优点**:
- 不依赖 conda 命令
- 环境隔离，路径明确
- 适合脚本自动化

### 5.2 通过 PATH（可选）

```bash
# 临时设置
export PATH=/root/work/david_work/envs/sana_wm_cmcc/bin:$PATH
python3 your_script.py

# 持久化（添加到 ~/.bashrc）
echo 'export PATH=/root/work/david_work/envs/sana_wm_cmcc/bin:$PATH' >> ~/.bashrc
```

---

## 六、故障排查

### 6.1 ModuleNotFoundError: No module named 'sana_wm_pipeline'

**检查 PYTHONPATH**:
```bash
echo $PYTHONPATH
# 应包含：/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src
```

**检查代码是否存在**:
```bash
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src/
```

**临时修复**:
```bash
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
```

### 6.2 ModuleNotFoundError: No module named 'vipe'

**检查 PYTHONPATH 是否包含 third_party/vipe**:
```bash
echo $PYTHONPATH | grep "third_party/vipe"
```

**检查 vipe 目录是否存在**:
```bash
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe/vipe/
```

**修复**:
```bash
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
```

### 6.3 CUDA not available

**检查 GPU**:
```bash
nvidia-smi
```

**检查 PyTorch CUDA**:
```bash
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"
```

---

## 七、技术总结

### 7.1 为什么需要 PYTHONPATH

**背景**:
- conda-pack 无法打包 editable 包（`pip install -e .`）
- 打包过程中移除了 `nvidia-vipe` 和 `sana-wm-pipeline` 的 editable 安装
- 环境中只保留了依赖库，不包含项目代码

**解决方案**:
- 项目代码单独部署
- 通过 PYTHONPATH 指向源码目录
- Python 模块查找顺序：PYTHONPATH（优先）→ site-packages → 标准库

**优点**:
- 完全离线运行（无需 pip install）
- 代码修改立即生效（等效于 editable 安装）
- 环境与代码分离，便于独立更新

### 7.2 vipe 模块的特殊性

**模块结构**:
```
third_party/vipe/
├── vipe/                          # Python 模块（import vipe）
│   ├── __init__.py
│   ├── pipeline/
│   └── ...
└── vipe_ext.cpython-310...so      # 预编译 C++/CUDA 扩展
```

**导入机制**:
- `import vipe` → 从 `third_party/vipe/vipe/` 导入
- `vipe` 模块会自动加载同目录下的 `vipe_ext.so`
- 预编译扩展存在时，不会触发 JIT 编译

**包名 vs 模块名**:
- pip 包名：`nvidia-vipe`（带连字符）
- Python 模块名：`vipe`（不是 `nvidia_vipe`）

### 7.3 部署验证清单

- [ ] 两个环境解压并 unpack 成功
- [ ] PYTHONPATH 包含 `src` 和 `third_party/vipe` 两个路径
- [ ] 可以导入 `sana_wm_pipeline` 模块
- [ ] 可以导入 `vipe` 模块
- [ ] PyTorch CUDA 可用
- [ ] 冒烟测试全部通过

---

## 八、参考信息

### 8.1 相关文档

- `CONDA_ENV_PACKAGING_PLAN.md` - 打包方案设计
- `CONDA_ENV_PACKAGING_EXECUTION_GUIDE.md` - 打包执行指南
- `CONDA_ENV_PACKAGING_BUGFIX_LOG.md` - 打包阶段问题修复日志
- `CMCC_SETUP_GUIDE.md` - 精简操作手册

### 8.2 环境信息

| 项目 | 值 |
|------|-----|
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA H100 80GB HBM3 |
| vipe | 1.1.0 |
| torchvision | (sana_qc_cmcc) |
| transformers | (sana_qc_cmcc) |

### 8.3 实际验证日期

- 打包完成：2026-08-23
- 传输到 CMCC：2026-08-23
- 部署验证：2026-08-23
- 基础导入测试：✅ 通过
- 完整流程测试：⚠️ 阻塞中（Pi3X segfault）

### 8.4 未解决问题跟踪

#### Issue #1: Pi3X 推理 Segmentation Fault
- **优先级**：P0（阻塞）
- **影响范围**：VIPE SLAM 流程无法完成
- **复现条件**：CMCC 环境，46 帧视频，Pi3 推理阶段
- **不复现环境**：本机（PyTorch 2.12）
- **待验证方案**：
  1. conda-unpack 修复
  2. 检查 torch 库依赖
  3. 检查 ABI 兼容性
  4. 重新打包（PyTorch 2.6 环境）
- **临时绕过**：降低批处理大小（修改代码）

---

## 附录：完整 PYTHONPATH 配置

```bash
# 添加到 ~/.bashrc
cat >> ~/.bashrc <<'EOF'

# SANA-WM Pipeline Module Resolution
# 说明：由于环境中移除了 editable 包，需要通过 PYTHONPATH 指向源码
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF

# 立即生效
source ~/.bashrc
```

**验证**:
```bash
echo $PYTHONPATH
# 应输出：
# /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:...
```
