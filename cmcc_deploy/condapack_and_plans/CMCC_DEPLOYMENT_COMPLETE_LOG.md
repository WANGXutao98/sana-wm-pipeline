# CMCC 部署完整日志

**创建日期**: 2026-08-27  
**项目**: SANA-WM Pipeline 数据生产环境部署  
**目标平台**: 中国移动离线训练服务器（NVIDIA H100 80GB）

---

## 📊 部署时间线

| 日期 | 阶段 | 状态 |
|------|------|------|
| 2026-08-20 | 部署规划 | ✅ 完成 |
| 2026-08-21~22 | Conda 环境打包 | ✅ 完成（遇到多个问题，已解决）|
| 2026-08-23 早期 | 初次部署测试 | ⚠️ Segmentation Fault |
| 2026-08-23~27 | 问题诊断与修复 | ✅ 完成 |
| 2026-08-27 当前 | CUDA Kernel 错误 | 🔴 阻塞中 |

---

## 🎯 部署目标

### 核心任务
在 CMCC 离线环境运行 SANA-WM 数据生产管线：
- 输入：Apple Vision Pro 录制的空间视频（`.mov` 格式）
- 处理：7 个阶段（归一化、SLAM、分割、AOT、GeoCalib、聚合、可视化）
- 输出：训练就绪的数据（poses、depths、masks 等）

### 环境约束
- ✅ 无外网访问（所有依赖必须预打包）
- ✅ GPU：NVIDIA H100 80GB × 1
- ✅ CUDA Driver：13.0
- ✅ Python：3.10
- ⚠️ 系统 CUDA Toolkit：12.4（低于本机的 13.0）

---

## 📦 第一阶段：Conda 环境打包（2026-08-21~22）

### 初始方案与失败

**尝试 1：直接打包 `sana_wm` 环境**
```bash
conda pack -n sana_wm -o sana_wm_cmcc.tar.gz --ignore-editable-packages
```

❌ **失败原因**：
- PyTorch 2.12.0 编译时使用 CUDA 13.0
- CMCC 系统只有 CUDA 12.4
- 运行时找不到 `libcudart.so.13.0`

### 关键问题 1：PyTorch CUDA 版本不匹配

**解决方案**：重新打包兼容 CUDA 12.4 的环境

```bash
# 创建新环境
conda create -n sana_wm_cu124 python=3.10 -y
conda activate sana_wm_cu124

# 安装 PyTorch 2.6.0（CUDA 12.4 编译）
pip install torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124
```

✅ **结果**：PyTorch 可以在 CMCC 加载

---

### 关键问题 2：ABI 不兼容

**症状**：
```
OSError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.32` not found
```

**原因分析**：
- PyTorch 2.6.0（CUDA 12.4）使用旧 ABI（`_GLIBCXX_USE_CXX11_ABI=0`）
- 部分依赖包（如 vipe）编译时使用新 ABI
- CMCC 系统 libstdc++.so.6 版本较旧，缺失 GLIBCXX_3.4.32

**解决方案**：使用 PyTorch 2.12.0 + CUDA 13.0（新 ABI）+ 捆绑 CUDA 运行时库

```bash
# 方案：捆绑 CUDA 13.0 运行时库到 conda 环境
conda create -n sana_wm_cu130_bundled python=3.10 -y
conda activate sana_wm_cu130_bundled

# 安装 PyTorch 2.12.0 + CUDA 13.0（新 ABI）
pip install torch==2.12.0+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130

# 关键：安装 cudatoolkit-dev 捆绑运行时库
conda install -c conda-forge cudatoolkit-dev=13.0 -y
```

✅ **结果**：
- PyTorch ABI：`_GLIBCXX_USE_CXX11_ABI=1`（新 ABI）
- CUDA 运行时：捆绑在 conda 环境内（`$ENV/lib/libcudart.so.13.0`）
- 不再依赖系统 CUDA Toolkit

**最终打包命令**：
```bash
conda pack -n sana_wm_cu130_bundled \
  -o /mnt/afs/davidwang/workspace/sana_wm_cmcc_cu130_abi1_bundled.tar.gz \
  --ignore-editable-packages \
  --compress-level 6
```

**包体积**：~8.2 GB

---

### 关键问题 3：Editable 包处理

**症状**：
```
CondaPackException: Cannot pack an environment with editable packages
```

**原因**：
- `sana_wm_pipeline` 以 `pip install -e .` 安装（开发模式）
- `vipe` 也是 editable 安装

**解决方案**：
```bash
# 1. 卸载 editable 安装
pip uninstall sana_wm_pipeline -y
pip uninstall nvidia-vipe -y

# 2. 复制源码到 conda 环境
cp -r /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline \
  $CONDA_PREFIX/lib/python3.10/site-packages/

# 3. vipe 保持在 third_party（通过 PYTHONPATH 引用）
# 不复制到 site-packages，避免路径冲突
```

✅ **结果**：可以成功打包

---

## 🚀 第二阶段：CMCC 部署与测试（2026-08-23）

### 2.1 传输与解压

```bash
# CMCC 侧操作
cd /root/work/david_work/conda_envs_download/conda_envs

# 解压环境
mkdir -p /root/work/david_work/envs/sana_wm_cmcc
tar -xzf sana_wm_cmcc_cu130_abi1_bundled.tar.gz \
  -C /root/work/david_work/envs/sana_wm_cmcc

# 运行 conda-unpack（修复路径硬编码）
cd /root/work/david_work/envs/sana_wm_cmcc
source bin/activate
conda-unpack
```

### 2.2 项目代码部署

```bash
# CMCC 项目路径（注意：比本机多一层 sana_wm_optimized）
/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/
```

**关键配置**：
```bash
export PROJ_DIR="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline"
export ENV_WM="/root/work/david_work/envs/sana_wm_cmcc"
export PYTHON="$ENV_WM/bin/python3"

# PYTHONPATH 必须包含两个路径
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe"

# VIPE 配置
export VIPE_EXT_JIT=0  # 禁用 JIT 编译（离线环境）
export TORCH_CUDA_ARCH_LIST=9.0  # H100 架构
```

---

## 🐛 第三阶段：问题诊断与解决（2026-08-23）

### 问题 3.1：PYTHONPATH 配置错误

**症状**：
```python
ModuleNotFoundError: No module named 'sana_wm_pipeline'
```

**原因**：
```bash
# 错误配置
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party"

# 问题：third_party 下是 vipe/ 目录，不是 vipe 的源码
```

**解决方案**：
```bash
# 正确配置
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe"
```

✅ **验证**：
```bash
$PYTHON -c "import sana_wm_pipeline; print('✓ sana_wm_pipeline')"
$PYTHON -c "import vipe; print('✓ vipe')"
```

---

### 问题 3.2：Segmentation Fault（Pi3X 推理）

**症状**：
```bash
[Stage 2] VIPE SLAM (Pi3X + MoGe-2)...
  Pi3推理 (46帧)...
Segmentation fault (core dumped)
```

**初步诊断**：
- ✅ PyTorch CUDA 可用
- ✅ 模块导入成功
- ✅ 模型加载成功
- ❌ 推理时崩溃

**环境对比**：
| 项目 | 本机 | CMCC | 差异 |
|------|------|------|------|
| PyTorch | 2.12.0+cu130 | 2.12.0+cu130 | ✅ 相同（重新打包后）|
| CUDA 编译 | 13.0 | 13.0 | ✅ 相同 |
| CUDA Driver | 13.0 | 13.0 | ✅ 相同 |
| Pi3 CUDA 扩展 | 无 | 无 | ✅ 相同 |
| ABI | New (1) | New (1) | ✅ 相同（重新打包后）|

**关键线索**：
- 本机相同代码正常运行
- 排除代码逻辑问题
- 怀疑 conda-pack 环境路径问题

**解决方案**：执行 `conda-unpack`

```bash
cd /root/work/david_work/envs/sana_wm_cmcc
source bin/activate
conda-unpack
```

✅ **结果**：Segmentation Fault 问题解决

**根因分析**：
- conda-pack 打包时硬编码了原始路径（`/mnt/afs/davidwang/...`）
- conda-unpack 将这些路径重写为新环境路径（`/root/work/david_work/envs/...`）
- 未执行 conda-unpack 导致某些库加载路径错误，引发 segfault

---

## 🔥 第四阶段：CUDA Kernel Image 错误（2026-08-27 当前）

### 当前状态
- ✅ Segmentation Fault 已解决
- ✅ ABI 不兼容已解决
- 🟡 **疑似问题：CUDA kernel image 不可用**（待验证）

### 错误信息（报告时）

```
CUDA error: no kernel image is available for execution on the device
```

**发生位置**：Pi3X 模型推理时的基本张量运算

---

### 诊断进展（2026-08-27）

#### ✅ 阶段 1：基础诊断（已完成）

**CMCC 环境测试结果**：

```
1. PyTorch 版本信息:
   PyTorch: 2.12.0+cu130 ✓
   CUDA 编译版本: 13.0 ✓
   CUDA 可用: True ✓

2. 设备信息:
   设备名称: NVIDIA H100 80GB HBM3 ✓
   Compute Capability: 9.0 ✓
   编译架构: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120'] ✓
   包含 sm_90: ✓

3. 基本操作: ✓ 矩阵乘法成功
```

**结论**：
- ✅ PyTorch 环境正常
- ✅ CUDA 基础功能正常
- ✅ H100（sm_90）架构支持正常
- ✅ 基本 CUDA kernel 可以执行

**这表明**：
1. 不是 PyTorch 编译架构问题（sm_90 已包含）
2. 不是 CUDA 版本不匹配问题（基本操作成功）
3. 如果错误仍存在，可能是特定模型/算子的问题

---

#### 🟡 阶段 2：完整冒烟测试（待执行）

**下一步需要验证**：错误是否仍然存在

### 诊断步骤（原计划）

#### 步骤 1：检查 PyTorch 支持的 CUDA 架构

```bash
source /root/work/david_work/envs/sana_wm_cmcc/bin/activate

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

# 检查编译的架构
if hasattr(torch.cuda, 'get_arch_list'):
    print(f'Compiled architectures: {torch.cuda.get_arch_list()}')

# 检查当前设备架构
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'Device compute capability: {cap[0]}.{cap[1]}')
"
```

**预期输出**：
- H100 compute capability: `9.0`
- PyTorch 编译架构应包含 `sm_90` 或支持 PTX

#### 步骤 2：测试基本 CUDA 操作

```bash
python -c "
import torch

# 简单张量运算
x = torch.randn(10, 10, device='cuda')
y = x @ x.T
print(f'✓ 基本矩阵乘法成功')

# 测试 nn 模块
import torch.nn as nn
model = nn.Linear(10, 5).cuda()
out = model(x)
print(f'✓ nn.Linear 成功')

# 测试 Conv2d
conv = nn.Conv2d(3, 16, 3).cuda()
img = torch.randn(1, 3, 224, 224, device='cuda')
out = conv(img)
print(f'✓ Conv2d 成功')
"
```

**目的**：确定错误是在所有 CUDA 操作还是特定操作

#### 步骤 3：尝试启用 PTX JIT

**理论**：
- 如果 PyTorch 没有为 sm_90 预编译 kernel
- 可以启用 PTX JIT，让驱动动态编译

```bash
export CUDA_FORCE_PTX_JIT=1

# 重新测试
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh
```

#### 步骤 4：检查 Pi3X 模型特定操作

```bash
python -c "
import torch
from vipe.models.pi3x import Pi3X

# 加载模型
model = Pi3X()
model = model.cuda()
model.eval()

# 测试简单输入
dummy_input = torch.randn(1, 1, 3, 512, 512, device='cuda')

with torch.no_grad():
    try:
        output = model(dummy_input)
        print('✓ Pi3X 推理成功')
    except Exception as e:
        print(f'✗ Pi3X 推理失败: {e}')
"
```

---

### 待执行的修复方案

#### 方案 A：重新编译 PyTorch（根本解决）

**适用场景**：PyTorch wheels 不支持 sm_90

```bash
# 在本机执行
conda create -n pytorch_build python=3.10 -y
conda activate pytorch_build

# 安装编译依赖
conda install cmake ninja mkl mkl-include -y
pip install pyyaml typing_extensions

# 克隆 PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.12.0

# 配置编译（仅 sm_90）
export TORCH_CUDA_ARCH_LIST="9.0"
export USE_CUDNN=1
export USE_MKLDNN=1

# 编译（需 2-4 小时）
python setup.py bdist_wheel

# 打包生成的 wheel
# 传输到 CMCC 并安装
```

#### 方案 B：降级到预编译支持的架构（临时绕过）

**理论**：使用 sm_80（A100）kernel，可能性能下降但能运行

```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 测试
```

#### 方案 C：使用官方 NGC PyTorch 容器

**理论**：NVIDIA NGC 的 PyTorch 预编译了所有架构

```bash
# 在 CMCC 拉取 NGC 容器（如果有 NGC 访问权限）
singularity pull docker://nvcr.io/nvidia/pytorch:24.03-py3

# 或者下载预编译的 wheels
# https://developer.nvidia.com/deep-learning-frameworks
```

---

## 📋 诊断执行清单

### 阶段 1：信息收集（5 分钟）

- [ ] 检查 PyTorch 支持的 CUDA 架构
- [ ] 检查设备 compute capability
- [ ] 测试基本 CUDA 操作是否正常

### 阶段 2：快速尝试（10 分钟）

- [ ] 启用 `CUDA_FORCE_PTX_JIT=1`
- [ ] 设置 `TORCH_CUDA_ARCH_LIST=9.0`
- [ ] 重新测试冒烟流程

### 阶段 3：深度诊断（30 分钟）

- [ ] 隔离 Pi3X 模型测试
- [ ] 识别具体失败的 kernel 操作
- [ ] 检查 RoPE2D 实现（纯 Python vs CUDA）

### 阶段 4：根本修复（根据诊断结果选择）

- [ ] 方案 A：重新编译 PyTorch（2-4 小时）
- [ ] 方案 B：启用架构兼容模式（5 分钟）
- [ ] 方案 C：使用 NGC PyTorch（1 小时）

---

## 📞 相关文档索引

| 文档 | 内容 | 用途 |
|------|------|------|
| `CMCC_DEPLOYMENT_INSTRUCTIONS.md` | 初始部署指南 | 历史参考 |
| `CMCC_SEGFAULT_DEBUG_NEXT_STEPS.md` | Segfault 调试方案 | ✅ 已解决 |
| `CONDA_ENV_PACKAGING_BUGFIX_LOG.md` | 打包阶段问题日志 | ABI 问题参考 |
| `CONDA_ENV_PACKAGING_EXECUTION_GUIDE.md` | 打包执行记录 | 完整打包步骤 |
| **本文档** | 完整部署日志 | **当前主文档** |

---

## 💡 经验总结

### 已验证的关键点

1. **PyTorch CUDA 版本必须与系统兼容**
   - 不要假设 CUDA Driver 向下兼容就万事大吉
   - 捆绑 CUDA 运行时库（cudatoolkit-dev）是更安全的方案

2. **ABI 兼容性至关重要**
   - 混合新旧 ABI 的依赖会导致运行时错误
   - 统一使用新 ABI（`_GLIBCXX_USE_CXX11_ABI=1`）

3. **conda-unpack 不可省略**
   - 即使环境看起来"能用"，路径硬编码会导致隐蔽的崩溃
   - 解压后立即执行 `conda-unpack`

4. **PYTHONPATH 必须精确**
   - `third_party` vs `third_party/vipe` 的区别是模块能否导入
   - 用 `python -c "import ..."` 验证每个模块

5. **H100（sm_90）架构需要特别注意**
   - 不是所有 PyTorch wheels 都预编译了 sm_90
   - 可能需要启用 PTX JIT 或重新编译

### 教训

1. **不要跳过环境检查步骤**
   - conda-unpack 看起来"可选"，实际上是必需的
   
2. **版本差异要逐一确认**
   - PyTorch 版本、CUDA 版本、ABI 版本都要一致

3. **离线环境需要完全自包含**
   - CUDA 运行时库捆绑到 conda 环境
   - 所有模型权重预下载

---

## 🎯 下一步行动

### 立即执行（当前会话）

1. 运行诊断步骤 1-3（信息收集与快速尝试）
2. 根据诊断结果选择修复方案
3. 实施修复并验证
4. 更新本文档的"第四阶段"部分

### 长期优化

1. 在本机建立 CMCC 完全一致的测试环境
2. 自动化打包和部署流程
3. 编写完整的冒烟测试套件
4. 准备多个架构的 PyTorch 备用包

---

**最后更新**: 2026-08-27  
**当前状态**: 🔴 CUDA Kernel Image 错误诊断中  
**责任人**: David Wang  
**下次更新**: 诊断完成后
