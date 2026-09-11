# CMCC Pi3X Segfault 调试指南

**创建日期**: 2026-08-23  
**问题状态**: 🔴 阻塞中  
**问题描述**: CMCC 环境下 Pi3X 推理时出现 Segmentation Fault

---

## 📊 问题总结

### 已知事实

1. **本机环境正常**
   - 路径：`/mnt/afs/davidwang/workspace/sana_wm_pipeline`
   - 测试：`smoke_spatialvid.sh` 完整流程通过
   - PyTorch：2.12.0 + CUDA 13.0

2. **CMCC 环境崩溃**
   - 路径：`/root/work/david_work/sana_wm_optimized/sana_wm_pipeline`
   - 崩溃位置：Pi3 推理阶段（46 帧）
   - PyTorch：2.6.0 + CUDA 12.4

3. **环境对比**
   | 项目 | 本机 | CMCC | 兼容性 |
   |------|------|------|--------|
   | PyTorch | 2.12.0+cu130 | 2.6.0+cu124 | ⚠️ 版本差异大 |
   | CUDA 编译 | 13.0 | 12.4 | ⚠️ 不同 |
   | CUDA Driver | 13.0 | 13.0 | ✅ 相同 |
   | cuDNN | 92000 | 90100 | ⚠️ 不同 |
   | GPU | H100 | H100 | ✅ 相同 |
   | Pi3 CUDA 扩展 | 无 | 无 | ✅ 相同（都缺失）|

4. **关键线索**
   - Pi3 CUDA 扩展缺失不是根因（两边都缺失，但本机能运行）
   - PyTorch 版本差异可能导致纯 Python RoPE2D 实现行为不同
   - 或者 conda-pack 打包的环境有路径硬编码问题

---

## 🔍 诊断方案（按执行顺序）

### 方案 1: conda-unpack 修复 ⭐（最优先）

**时间**: 5 分钟  
**风险**: 低  
**成功率**: 中

```bash
# 在 CMCC 执行
source /root/work/david_work/envs/sana_wm_cmcc/bin/activate
conda-unpack

# 验证是否执行过
ls -la /root/work/david_work/envs/sana_wm_cmcc/conda-meta/state

# 重新测试
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh
```

**如果成功**: 问题解决，继续后续任务  
**如果失败**: 进入方案 2

---

### 方案 2: 检查 torch 库依赖

**时间**: 10 分钟  
**风险**: 低  
**目的**: 排查是否缺失系统库

```bash
# 在 CMCC 执行
source /root/work/david_work/envs/sana_wm_cmcc/bin/activate

# 检查 torch CUDA 库依赖
ldd /root/work/david_work/envs/sana_wm_cmcc/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so | grep "not found"
```

**如果有 not found**:
```bash
# 根据缺失的库名安装
# 例如：
yum install -y libgomp libnuma libstdc++
```

**如果没有 not found**: 进入方案 3

---

### 方案 3: 检查 ABI 兼容性

**时间**: 5 分钟  
**风险**: 低  
**目的**: 确认是否需要重新编译

```bash
# 在本机执行
conda activate sana_wm
python -c "import torch; print(f'ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"

# 在 CMCC 执行
source /root/work/david_work/envs/sana_wm_cmcc/bin/activate
python -c "import torch; print(f'ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
```

**如果输出不同**: ABI 不兼容，需要重新打包（进入方案 4）  
**如果输出相同**: ABI 没问题，继续其他排查

---

### 方案 4: 在本机重新打包（根本解决）

**时间**: 1-2 小时  
**风险**: 低  
**成功率**: 高

#### 步骤 4.1: 创建新环境（PyTorch 2.6）

```bash
cd /mnt/afs/davidwang/workspace

# 创建环境
conda create -n sana_wm_cu124 python=3.10 -y
conda activate sana_wm_cu124

# 安装和 CMCC 相同版本的 PyTorch
pip install torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

#### 步骤 4.2: 安装其他依赖

```bash
# 导出原环境的依赖
conda activate sana_wm
pip freeze > /tmp/sana_wm_requirements.txt

# 在新环境安装（跳过 torch 和 torchvision）
conda activate sana_wm_cu124
grep -v "^torch" /tmp/sana_wm_requirements.txt | \
grep -v "^torchvision" | \
pip install -r /dev/stdin
```

#### 步骤 4.3: 测试新环境

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
conda activate sana_wm_cu124

# 设置环境变量
export PYTHONPATH="$PWD/src:$PWD/third_party/vipe"
export VIPE_EXT_JIT=0
export TORCH_CUDA_ARCH_LIST=9.0

# 运行测试
bash experiments/data_production_smoke/smoke_spatialvid.sh
```

**如果测试失败**: 说明 PyTorch 2.6 本身有问题，需要分析具体错误  
**如果测试成功**: 继续打包

#### 步骤 4.4: 重新打包

```bash
conda activate sana_wm_cu124

# 打包（移除 editable 包）
conda pack -n sana_wm_cu124 \
  -o /mnt/afs/davidwang/workspace/sana_wm_cmcc_cu124_fixed.tar.gz \
  --ignore-editable-packages

# 检查文件大小
ls -lh /mnt/afs/davidwang/workspace/sana_wm_cmcc_cu124_fixed.tar.gz
```

#### 步骤 4.5: 传输到 CMCC

```bash
# 在 CMCC 执行
cd /root/work/david_work/conda_envs_download/conda_envs

# 下载新包（手动传输）
# 假设已传输到当前目录

# 备份旧环境
mv /root/work/david_work/envs/sana_wm_cmcc /root/work/david_work/envs/sana_wm_cmcc.backup

# 解压新环境
mkdir -p /root/work/david_work/envs/sana_wm_cmcc
tar -xzf sana_wm_cmcc_cu124_fixed.tar.gz -C /root/work/david_work/envs/sana_wm_cmcc

# 运行 conda-unpack
cd /root/work/david_work/envs/sana_wm_cmcc
source bin/activate
conda-unpack

# 测试
python -c "import torch; print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
```

#### 步骤 4.6: 重新测试

```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh
```

---

### 方案 5: 添加环境变量（快速尝试）

**时间**: 5 分钟  
**风险**: 低  
**成功率**: 低

在 CMCC 的测试脚本开头添加：

```bash
export CUDA_FORCE_PTX_JIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

---

### 方案 6: 降低批处理大小（代码修改，最后手段）

**时间**: 30 分钟  
**风险**: 中（修改代码逻辑）  
**性能影响**: 会变慢

#### 步骤 6.1: 定位 pi3_infer 实现

```bash
grep -n "def pi3_infer" /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src/sana_wm_pipeline/stage02_pose/mode_default.py
```

#### 步骤 6.2: 修改为分批推理

假设原代码是：
```python
# 一次性推理所有帧
poses, depth = self.pi3_model(frames)  # frames: (1, 46, 3, H, W)
```

改为：
```python
# 分批推理
batch_size = 8
all_poses, all_depth = [], []

for i in range(0, frames.shape[1], batch_size):
    batch = frames[:, i:i+batch_size]
    with torch.no_grad():
        poses_batch, depth_batch = self.pi3_model(batch)
    all_poses.append(poses_batch)
    all_depth.append(depth_batch)

poses = torch.cat(all_poses, dim=1)
depth = torch.cat(all_depth, dim=1)
```

**注意**: 需要先查看实际代码结构，上面只是示例。

---

## 📋 执行检查清单

### 第一轮诊断（快速验证）

- [ ] 方案 1: conda-unpack 执行并测试
- [ ] 方案 2: 检查 torch 库依赖
- [ ] 方案 3: 检查 ABI 兼容性
- [ ] 方案 5: 添加环境变量尝试

**预计时间**: 30 分钟  
**如果全部失败**: 进入第二轮

### 第二轮解决（深度修复）

- [ ] 方案 4: 在本机创建 PyTorch 2.6 环境
- [ ] 在本机测试新环境
- [ ] 重新打包
- [ ] 传输到 CMCC
- [ ] 部署并测试

**预计时间**: 1-2 小时

### 紧急绕过（如需快速出结果）

- [ ] 方案 6: 修改代码降低批处理大小

**预计时间**: 30 分钟  
**代价**: 性能下降，不是长期方案

---

## 🚨 开新对话时需要的信息

### 给新 Claude 的背景信息

1. **问题核心**：
   - CMCC 环境 Pi3X 推理 segfault
   - 本机相同代码正常运行
   - PyTorch 版本差异（2.12 vs 2.6）

2. **已完成的工作**：
   - ✅ conda 环境打包、传输、解压
   - ✅ PYTHONPATH 配置正确
   - ✅ 基础模块导入测试通过
   - ✅ smoke_cmcc_test_v1.sh 脚本修正方案已给出

3. **待验证的方案**（优先级排序）：
   1. conda-unpack 修复
   2. 检查 torch 库依赖
   3. 检查 ABI 兼容性
   4. 重新打包（PyTorch 2.6）
   5. 添加环境变量
   6. 降低批处理大小

4. **关键文档**：
   - `CMCC_DEPLOYMENT_INSTRUCTIONS.md` - 完整部署指南和问题记录
   - `CMCC_SEGFAULT_DEBUG_NEXT_STEPS.md` - 本文件，调试方案
   - `smoke_cmcc_test_v1.sh` - 需要修正的测试脚本

### 关键路径信息

```bash
# CMCC 环境
项目代码: /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
环境路径: /root/work/david_work/envs/sana_wm_cmcc
Python: /root/work/david_work/envs/sana_wm_cmcc/bin/python3

# 本机环境
项目代码: /mnt/afs/davidwang/workspace/sana_wm_pipeline
Conda 环境: sana_wm
测试脚本: experiments/data_production_smoke/smoke_spatialvid.sh (正常运行)
```

### 环境版本信息

```
本机：PyTorch 2.12.0+cu130, CUDA Driver 13.0, cuDNN 92000
CMCC：PyTorch 2.6.0+cu124, CUDA Driver 13.0, cuDNN 90100
```

---

## 💡 建议的调试顺序

1. **先快速尝试**（30 分钟内）
   - conda-unpack → ldd 检查 → ABI 检查 → 环境变量

2. **如果快速尝试无效**
   - 在本机重新打包（PyTorch 2.6 环境）
   - 这是最可靠的解决方案

3. **如果需要紧急出结果**
   - 修改代码降低批处理大小
   - 能跑通但性能下降

4. **长期方案**
   - 统一本机和 CMCC 的 PyTorch 版本
   - 或者升级 CMCC 环境到 PyTorch 2.12

---

## 📞 联系点

- 文档更新：`CMCC_DEPLOYMENT_INSTRUCTIONS.md`
- 脚本修正：`smoke_cmcc_test_v1.sh` 需要手动应用修改
- 测试视频：`/root/work/david_work/smoke_pass_videos/`

**最后更新**: 2026-08-23
