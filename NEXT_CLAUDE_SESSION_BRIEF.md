# 新 Claude 会话快速启动指南

**日期**: 2026-08-23  
**任务**: 解决 CMCC 环境 Pi3X 推理 Segmentation Fault 问题

---

## 🎯 当前任务状态

### 阻塞问题
**Pi3X 模型推理时 Segmentation Fault**
- 位置：`mode_default.py` 的 `pi3_infer()` 调用
- 现象：视频归一化成功（46 帧），Pi3 推理阶段崩溃
- 环境：CMCC 离线服务器

### 关键事实
1. ✅ 本机环境（`/mnt/afs/davidwang/workspace`）**运行正常**
   - 测试脚本：`experiments/data_production_smoke/smoke_spatialvid.sh`
   
2. ❌ CMCC 环境（`/root/work/david_work/sana_wm_optimized/sana_wm_pipeline`）**崩溃**

3. 主要差异：**PyTorch 版本**
   - 本机：PyTorch 2.12.0 + CUDA 13.0
   - CMCC：PyTorch 2.6.0 + CUDA 12.4

---

## 📂 关键文档（按阅读顺序）

### 1. 调试方案（最重要）
**文件**: `CMCC_SEGFAULT_DEBUG_NEXT_STEPS.md`

包含 6 个方案（按优先级排序）：
1. ⭐ conda-unpack 修复（5 分钟，最优先）
2. 检查 torch 库依赖（10 分钟）
3. 检查 ABI 兼容性（5 分钟）
4. 在本机重新打包 PyTorch 2.6 环境（1-2 小时，最可靠）
5. 添加环境变量尝试（5 分钟）
6. 降低批处理大小（30 分钟，最后手段）

### 2. 部署完整记录
**文件**: `CMCC_DEPLOYMENT_INSTRUCTIONS.md`

包含：
- 部署步骤（已完成）
- 已解决的 5 个问题（PYTHONPATH、模块导入等）
- **第 3.5 节**：smoke_cmcc_test_v1.sh 脚本的修正方案
- Segmentation Fault 问题分析

### 3. 其他参考文档
- `CONDA_ENV_PACKAGING_EXECUTION_GUIDE.md` - 打包执行记录
- `CONDA_ENV_PACKAGING_BUGFIX_LOG.md` - 打包阶段问题日志

---

## 🚀 立即执行的命令

### 第一步：快速诊断（在 CMCC 机器上）

```bash
# 1. conda-unpack（最可能的修复）
source /root/work/david_work/envs/sana_wm_cmcc/bin/activate
conda-unpack
ls -la /root/work/david_work/envs/sana_wm_cmcc/conda-meta/state

# 2. 检查 torch 库依赖
ldd /root/work/david_work/envs/sana_wm_cmcc/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so | grep "not found"

# 3. 检查 ABI
python -c "import torch; print(f'ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
```

### 第二步：重新测试

```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh
```

**注意**：`smoke_cmcc_test_v1.sh` 需要先修正（见下文）

---

## 📝 smoke_cmcc_test_v1.sh 修正清单

**文件路径**: `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/experiments/data_production_smoke/smoke_cmcc_test_v1.sh`

### 必须修改的 4 处

1. **第 11 行后添加**：
```bash
export ENV_WM="$NEW_BASE/envs/sana_wm_cmcc"
export PYTHON="$ENV_WM/bin/python3"
```

2. **第 26 行后添加**：
```bash
export TORCH_CUDA_ARCH_LIST=9.0
```

3. **第 32 行修改**：
```bash
# 原来（错误）
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party${PYTHONPATH:+:$PYTHONPATH}"

# 改为（正确）
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party/vipe${PYTHONPATH:+:$PYTHONPATH}"
```

4. **替换所有 python 命令**：
   - 第 45 行：`python -c` → `$PYTHON -c`
   - 第 126 行：`python -c` → `$PYTHON -c`
   - 第 150 行：`python -c` → `$PYTHON -c`
   - 第 185 行：`python -` → `$PYTHON -`

5. **第 40 行修改**：
```bash
# 原来
echo "Conda 环境: $CONDA_DEFAULT_ENV"

# 改为
echo "Python 环境: $ENV_WM"
```

---

## 🔄 如果快速诊断无效

### 方案 4：在本机重新打包（最可靠）

```bash
# 在本机执行
cd /mnt/afs/davidwang/workspace

# 1. 创建 PyTorch 2.6 环境
conda create -n sana_wm_cu124 python=3.10 -y
conda activate sana_wm_cu124
pip install torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. 安装其他依赖
conda activate sana_wm
pip freeze > /tmp/sana_wm_requirements.txt
conda activate sana_wm_cu124
grep -v "^torch" /tmp/sana_wm_requirements.txt | grep -v "^torchvision" | pip install -r /dev/stdin

# 3. 测试新环境
cd sana_wm_pipeline
export PYTHONPATH="$PWD/src:$PWD/third_party/vipe"
bash experiments/data_production_smoke/smoke_spatialvid.sh

# 4. 如果测试通过，重新打包
conda pack -n sana_wm_cu124 -o sana_wm_cmcc_cu124_fixed.tar.gz --ignore-editable-packages
```

详细步骤见 `CMCC_SEGFAULT_DEBUG_NEXT_STEPS.md` 方案 4。

---

## 🗺️ 关键路径速查

```bash
# CMCC 环境
项目: /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
环境: /root/work/david_work/envs/sana_wm_cmcc
Python: /root/work/david_work/envs/sana_wm_cmcc/bin/python3
测试脚本: experiments/data_production_smoke/smoke_cmcc_test_v1.sh

# 本机环境
项目: /mnt/afs/davidwang/workspace/sana_wm_pipeline
Conda: sana_wm
测试脚本: experiments/data_production_smoke/smoke_spatialvid.sh (能正常运行)
```

---

## 💬 如何向新 Claude 提问

### 快速启动模板

```
我正在解决 CMCC 环境的 Pi3X Segmentation Fault 问题。

背景：
- 已阅读 CMCC_SEGFAULT_DEBUG_NEXT_STEPS.md
- 本机环境正常，CMCC 环境崩溃
- PyTorch 版本差异（本机 2.12，CMCC 2.6）

已执行的诊断：
[贴上你执行的命令和输出]

请根据输出判断下一步应该执行哪个方案。
```

### 如果需要实施方案 4

```
快速诊断未解决问题，需要在本机重新打包 PyTorch 2.6 环境。

请帮我：
1. 确认方案 4 的完整步骤
2. 执行创建和测试新环境
3. 如果测试通过，执行打包
```

---

## ✅ 成功标志

当以下命令在 CMCC 不再 segfault 时，问题解决：

```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_test_v1.sh
```

预期输出：
```
✓✓✓ 冒烟测试全部通过 (N/N) ✓✓✓
```

---

## 🎓 经验总结

### 已解决的问题模式

1. **PYTHONPATH 缺失** → 必须包含 `src` 和 `third_party/vipe`
2. **模块名混淆** → `import vipe` 不是 `import nvidia_vipe`
3. **环境路径错误** → 必须用绝对路径调用 Python
4. **项目路径错误** → CMCC 是 `sana_wm_optimized/sana_wm_pipeline`

### 当前问题特征

- ✅ 模块导入成功
- ✅ CUDA 可用
- ❌ 推理时崩溃
- 🔍 可能原因：PyTorch 版本不兼容或打包环境路径问题

---

**祝调试顺利！所有必要信息都在上述文档中。**
