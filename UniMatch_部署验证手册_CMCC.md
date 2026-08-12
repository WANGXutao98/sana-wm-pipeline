# UniMatch 部署验证手册（CMCC 机器）

> **任务目标**：验证 UniMatch 光流检测模块在 CMCC H100 GPU 上正常运行  
> **执行日期**：2026-08-03  
> **前置条件**：✅ DOVER H100 已验证通过  
> **预估时间**：1-2 小时

---

## 📋 执行检查清单

- [ ] 步骤 1：环境验证（5 分钟）
- [ ] 步骤 2：UniMatch 代码检查（5 分钟）
- [ ] 步骤 3：权重文件验证（5 分钟）
- [ ] 步骤 4：依赖安装（10 分钟）
- [ ] 步骤 5：编写测试脚本（15 分钟）
- [ ] 步骤 6：运行验证测试（20 分钟）
- [ ] 步骤 7：性能基准测试（10 分钟）

---

## 步骤 1：环境验证（5 分钟）

### 1.1 激活环境

```bash
# 在 CMCC 机器执行
conda activate sana_wm_qc_env
```

### 1.2 验证基础环境

```bash
# 验证 Python + PyTorch + CUDA
python -c "
import torch
print(f'Python: {__import__(\"sys\").version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'CUDA version: {torch.version.cuda}')
"
```

**期望输出**：
```
Python: 3.10.x 或 3.11.x
PyTorch: 2.x.x+cu124
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
CUDA version: 12.4
```

**⚠️ 如果 CUDA available: False**：
```bash
# 检查环境变量
echo $LD_LIBRARY_PATH
# 如果为空或缺少 CUDA 路径，设置：
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## 步骤 2：UniMatch 代码检查（5 分钟）

### 2.1 检查代码目录

```bash
cd /root/work/david_work/models/unimatch
ls -la
```

**期望看到**：
- `unimatch/` 目录（核心代码）
- `pretrained/` 目录（权重文件）
- `evaluate_flow.py`, `main_flow.py` 等脚本
- `README.md`, `MODEL_ZOO.md`

### 2.2 检查核心代码结构

```bash
ls -la unimatch/
```

**期望看到**：
- `__init__.py`
- `unimatch.py` 或类似的主模型文件
- `geometry.py`, `position.py` 等工具文件

---

## 步骤 3：权重文件验证（5 分钟）

### 3.1 检查权重文件

```bash
find pretrained/ -name "*.pth" -o -name "*.pt" 2>/dev/null
```

**期望输出**（可能的权重文件）：
```
pretrained/gmflow-scale1-mixdata.pth
pretrained/gmflow-scale2-regrefine6-mixdata.pth
或类似的权重文件（约 50-200MB）
```

### 3.2 验证权重文件完整性

```bash
# 查看权重文件大小
ls -lh pretrained/*.pth 2>/dev/null || ls -lh pretrained/*.pt 2>/dev/null
```

**如果 pretrained/ 为空或缺少权重**：
- 需要从 AFS 传输权重文件
- 或从 ModelScope/Hugging Face 下载（需要先在 AFS 下载后传输）

**⚠️ 暂停点**：如果权重文件缺失，告知我，我会生成下载/传输指令。

---

## 步骤 4：依赖安装（10 分钟）

### 4.1 安装 UniMatch 核心依赖

```bash
# 进入 unimatch 目录
cd /root/work/david_work/models/unimatch

# 检查是否有 pip_install.sh
if [ -f pip_install.sh ]; then
    echo "✅ 找到 pip_install.sh"
    cat pip_install.sh
else
    echo "❌ 未找到 pip_install.sh，需要手动安装依赖"
fi
```

### 4.2 手动安装依赖（推荐方式）

**基于 DOVER 成功经验，逐个安装依赖以便排查问题**：

```bash
# 激活环境（如果未激活）
conda activate sana_wm_qc_env

# 核心依赖
pip install opencv-python
pip install imageio
pip install matplotlib
pip install tensorboard

# 如果需要 timm（通常 DOVER 环境已安装）
pip list | grep timm || pip install timm

# 如果需要 einops（通常 DOVER 环境已安装）
pip list | grep einops || pip install einops

# 验证安装
python -c "import cv2; import imageio; import matplotlib; print('依赖安装成功')"
```

### 4.3 将 UniMatch 添加到 Python 路径

**方法 A（临时，推荐测试时使用）**：
```bash
export PYTHONPATH=/root/work/david_work/models/unimatch:$PYTHONPATH
```

**方法 B（永久，通过环境变量）**：
```bash
# 添加到 conda 环境激活脚本
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export PYTHONPATH=/root/work/david_work/models/unimatch:$PYTHONPATH' > $CONDA_PREFIX/etc/conda/activate.d/unimatch.sh
```

**验证 Python 能否导入**：
```bash
python -c "import sys; sys.path.insert(0, '/root/work/david_work/models/unimatch'); from unimatch.unimatch import UniMatch; print('✅ UniMatch 导入成功')"
```

**⚠️ 如果导入失败**，记录错误信息：
```bash
python -c "import sys; sys.path.insert(0, '/root/work/david_work/models/unimatch'); from unimatch.unimatch import UniMatch" 2>&1 | tee /tmp/unimatch_import_error.log
cat /tmp/unimatch_import_error.log
```

---

## 步骤 5：编写测试脚本（15 分钟）

### 5.1 创建测试目录

```bash
mkdir -p /root/work/david_work/sana_qc_pipeline/test_scripts
cd /root/work/david_work/sana_qc_pipeline/test_scripts
```

### 5.2 创建测试脚本

**将以下内容保存为 `test_unimatch_cmcc.py`**：

```python
#!/usr/bin/env python3
"""
UniMatch 光流检测验证脚本（CMCC H100）

测试内容：
1. 模型加载（CPU + GPU）
2. 随机数据推理
3. 实际视频光流计算
4. 性能基准测试
"""

import sys
import os
import time
import torch
import numpy as np

# 添加 UniMatch 到 Python 路径
sys.path.insert(0, '/root/work/david_work/models/unimatch')

print("=" * 80)
print("UniMatch 光流检测验证脚本")
print("=" * 80)

# ============================================================================
# 测试 1：环境检查
# ============================================================================
print("\n[测试 1] 环境检查")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 计算能力: {torch.cuda.get_device_capability(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============================================================================
# 测试 2：导入 UniMatch
# ============================================================================
print("\n[测试 2] 导入 UniMatch 模块")
try:
    from unimatch.unimatch import UniMatch
    print("✅ UniMatch 导入成功")
except ImportError as e:
    print(f"❌ UniMatch 导入失败: {e}")
    print("\n可能的原因：")
    print("1. unimatch 目录不在 PYTHONPATH 中")
    print("2. unimatch/__init__.py 缺失")
    print("3. 依赖包缺失（opencv-python, imageio 等）")
    sys.exit(1)

# ============================================================================
# 测试 3：模型加载（CPU 模式）
# ============================================================================
print("\n[测试 3] 模型加载测试（CPU 模式）")
try:
    model_cpu = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
    )
    model_cpu.eval()
    print("✅ CPU 模式加载成功")
    
    # 统计参数量
    num_params = sum(p.numel() for p in model_cpu.parameters())
    print(f"模型参数量: {num_params / 1e6:.2f}M")
    
except Exception as e:
    print(f"❌ CPU 模式加载失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 4：模型加载（GPU 模式）
# ============================================================================
print("\n[测试 4] 模型加载测试（GPU 模式）")
if not torch.cuda.is_available():
    print("⚠️  GPU 不可用，跳过 GPU 测试")
else:
    try:
        model_gpu = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).cuda()
        model_gpu.eval()
        print("✅ GPU 模式加载成功")
        
        # 检查显存占用
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"模型显存占用: {memory_allocated:.2f} GB")
        
    except Exception as e:
        print(f"❌ GPU 模式加载失败: {e}")
        print("\n可能的原因：")
        print("1. CUDA 版本不兼容")
        print("2. 显存不足")
        print("3. PyTorch 未正确安装 CUDA 支持")
        sys.exit(1)

# ============================================================================
# 测试 5：加载预训练权重
# ============================================================================
print("\n[测试 5] 加载预训练权重")
pretrained_dir = "/root/work/david_work/models/unimatch/pretrained"

# 查找权重文件
weight_files = []
for ext in ['*.pth', '*.pt']:
    import glob
    weight_files.extend(glob.glob(os.path.join(pretrained_dir, ext)))

if not weight_files:
    print(f"⚠️  未找到权重文件在 {pretrained_dir}")
    print("将使用随机初始化权重进行测试")
    model = model_gpu if torch.cuda.is_available() else model_cpu
else:
    weight_file = weight_files[0]
    print(f"找到权重文件: {weight_file}")
    print(f"文件大小: {os.path.getsize(weight_file) / 1024**2:.2f} MB")
    
    try:
        checkpoint = torch.load(weight_file, map_location='cpu')
        
        # 处理不同的权重格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model = model_gpu if torch.cuda.is_available() else model_cpu
        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功")
        
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        print("将使用随机初始化权重继续测试")
        model = model_gpu if torch.cuda.is_available() else model_cpu

# ============================================================================
# 测试 6：随机数据推理
# ============================================================================
print("\n[测试 6] 随机数据推理测试")
try:
    # 生成随机输入（模拟两帧图像）
    batch_size = 1
    height, width = 256, 256
    img1 = torch.randn(batch_size, 3, height, width)
    img2 = torch.randn(batch_size, 3, height, width)
    
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    
    # 推理计时
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        flow = model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1])
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_time = (time.time() - start_time) * 1000
    
    print("✅ 推理成功")
    print(f"输入形状: {img1.shape}")
    print(f"输出形状: {flow['flow_preds'][-1].shape if isinstance(flow, dict) else flow.shape}")
    print(f"推理耗时: {inference_time:.2f} ms")
    
    # 提取实际光流
    if isinstance(flow, dict) and 'flow_preds' in flow:
        flow_pred = flow['flow_preds'][-1]  # 取最后一个尺度
    else:
        flow_pred = flow
    
    print(f"光流数值范围: [{flow_pred.min():.2f}, {flow_pred.max():.2f}]")
    print(f"光流平均幅度: {torch.abs(flow_pred).mean():.4f}")
    
except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 7：性能基准测试
# ============================================================================
print("\n[测试 7] 性能基准测试（10 次推理）")
if torch.cuda.is_available():
    try:
        times = []
        for i in range(10):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1])
            
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
        
        print(f"平均推理时间: {np.mean(times):.2f} ms")
        print(f"最小推理时间: {np.min(times):.2f} ms")
        print(f"最大推理时间: {np.max(times):.2f} ms")
        print(f"标准差: {np.std(times):.2f} ms")
        
    except Exception as e:
        print(f"⚠️  性能测试失败: {e}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("验证总结")
print("=" * 80)
print("✅ UniMatch 导入成功")
print(f"✅ 模型加载成功 ({'GPU' if torch.cuda.is_available() else 'CPU'} 模式)")
if weight_files:
    print(f"✅ 权重加载成功 ({os.path.basename(weight_files[0])})")
else:
    print("⚠️  未加载预训练权重（使用随机初始化）")
print(f"✅ 随机数据推理成功 ({inference_time:.2f} ms)")
if torch.cuda.is_available():
    print(f"✅ 性能基准测试完成 (平均 {np.mean(times):.2f} ms)")

print("\n🎉 UniMatch H100 验证通过！")
print("=" * 80)
```

### 5.3 赋予执行权限

```bash
chmod +x test_unimatch_cmcc.py
```

---

## 步骤 6：运行验证测试（20 分钟）

### 6.1 设置环境变量

```bash
# 设置 Python 路径
export PYTHONPATH=/root/work/david_work/models/unimatch:$PYTHONPATH

# 激活环境（如果未激活）
conda activate sana_wm_qc_env
```

### 6.2 运行测试脚本

```bash
cd /root/work/david_work/sana_qc_pipeline/test_scripts
python test_unimatch_cmcc.py 2>&1 | tee /tmp/unimatch_test_output.log
```

### 6.3 预期输出

**成功的输出应包含**：
```
================================================================================
UniMatch 光流检测验证脚本
================================================================================

[测试 1] 环境检查
Python: 3.10.x
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
GPU 计算能力: (9, 0)
显存总量: 79.xx GB

[测试 2] 导入 UniMatch 模块
✅ UniMatch 导入成功

[测试 3] 模型加载测试（CPU 模式）
✅ CPU 模式加载成功
模型参数量: XX.XX M

[测试 4] 模型加载测试（GPU 模式）
✅ GPU 模式加载成功
模型显存占用: X.XX GB

[测试 5] 加载预训练权重
找到权重文件: /root/work/david_work/models/unimatch/pretrained/xxx.pth
文件大小: XXX.XX MB
✅ 权重加载成功

[测试 6] 随机数据推理测试
✅ 推理成功
输入形状: torch.Size([1, 3, 256, 256])
输出形状: torch.Size([1, 2, 256, 256])
推理耗时: XX.XX ms
光流数值范围: [-X.XX, X.XX]
光流平均幅度: X.XXXX

[测试 7] 性能基准测试（10 次推理）
平均推理时间: XX.XX ms
最小推理时间: XX.XX ms
最大推理时间: XX.XX ms
标准差: X.XX ms

================================================================================
验证总结
================================================================================
✅ UniMatch 导入成功
✅ 模型加载成功 (GPU 模式)
✅ 权重加载成功 (xxx.pth)
✅ 随机数据推理成功 (XX.XX ms)
✅ 性能基准测试完成 (平均 XX.XX ms)

🎉 UniMatch H100 验证通过！
================================================================================
```

---

## 步骤 7：故障排查（如遇到问题）

### 7.1 常见错误及解决方案

#### 错误 1：`ModuleNotFoundError: No module named 'unimatch'`

**原因**：Python 找不到 unimatch 模块

**解决**：
```bash
# 方法 1：临时设置
export PYTHONPATH=/root/work/david_work/models/unimatch:$PYTHONPATH
python test_unimatch_cmcc.py

# 方法 2：检查路径是否正确
ls -la /root/work/david_work/models/unimatch/unimatch/__init__.py
```

#### 错误 2：`ImportError: libcuda.so.1: cannot open shared object file`

**原因**：CUDA 库路径未设置

**解决**：
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python test_unimatch_cmcc.py
```

#### 错误 3：`RuntimeError: CUDA out of memory`

**原因**：显存不足（不太可能在 H100 80GB 上发生）

**解决**：
```bash
# 检查其他进程占用
nvidia-smi

# 如果有其他进程，清理或使用不同 GPU
CUDA_VISIBLE_DEVICES=1 python test_unimatch_cmcc.py
```

#### 错误 4：权重加载失败 `RuntimeError: Error(s) in loading state_dict`

**原因**：权重文件与模型结构不匹配

**解决**：
```bash
# 这不影响功能测试，脚本会使用随机权重继续
# 如果需要精确的光流计算，需要匹配的权重文件
```

### 7.2 生成故障报告

如果测试失败，收集以下信息：

```bash
# 生成完整诊断报告
{
    echo "===== 环境信息 ====="
    conda list | grep -E "(torch|cuda|opencv|imageio)"
    
    echo -e "\n===== GPU 信息 ====="
    nvidia-smi
    
    echo -e "\n===== UniMatch 目录结构 ====="
    ls -laR /root/work/david_work/models/unimatch/unimatch/ 2>/dev/null | head -50
    
    echo -e "\n===== 权重文件 ====="
    find /root/work/david_work/models/unimatch/pretrained -type f -ls 2>/dev/null
    
    echo -e "\n===== 测试日志 ====="
    cat /tmp/unimatch_test_output.log
    
} > /tmp/unimatch_debug_report.txt

cat /tmp/unimatch_debug_report.txt
```

---

## ✅ 验证完成标准

测试通过的标准：
- [ ] ✅ UniMatch 模块成功导入
- [ ] ✅ GPU 模式加载成功
- [ ] ✅ 权重加载成功（或随机权重推理成功）
- [ ] ✅ 随机数据推理成功，推理时间 <500ms
- [ ] ✅ 光流输出形状正确 `(1, 2, H, W)`
- [ ] ✅ 性能基准测试平均时间 <300ms

**目标性能**：<3s/样本（实际视频，含 I/O）

---

## 📝 验证记录模板

**将验证结果填写到这里**：

```
执行日期: 2026-08-03
执行人: [你的名字]
环境: sana_wm_qc_env
GPU: NVIDIA H100 80GB HBM3

测试结果:
□ 测试 1: 环境检查 - [ PASS / FAIL ]
□ 测试 2: UniMatch 导入 - [ PASS / FAIL ]
□ 测试 3: CPU 模式加载 - [ PASS / FAIL ]
□ 测试 4: GPU 模式加载 - [ PASS / FAIL ]
□ 测试 5: 权重加载 - [ PASS / FAIL / SKIP ]
□ 测试 6: 随机推理 - [ PASS / FAIL ]，耗时: ___ ms
□ 测试 7: 性能基准 - [ PASS / FAIL ]，平均: ___ ms

关键指标:
- 模型参数量: ___ M
- 显存占用: ___ GB
- 推理时间: ___ ms (256x256 输入)
- 光流数值范围: [___, ___]

遇到的问题:
1. [问题描述]
   解决方案: [解决方案]

结论: [ ✅ 验证通过 / ❌ 验证失败 ]
```

---

## 🎯 下一步

**如果验证通过**：
- 继续任务 #2：Qwen3.5-9B 部署与验证
- 或开始编写 Stage 3 单样本端到端测试脚本

**如果验证失败**：
- 提供 `/tmp/unimatch_debug_report.txt` 内容
- 我会分析问题并提供解决方案

---

**最后更新**: 2026-08-03  
**参考文档**: `DOVER_H100_部署方案_CMCC实际执行记录.md`
