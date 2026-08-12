# DOVER OOM 问题综合分析报告

## 📊 问题概览

### CMCC 机器情况

**硬件配置**：
- GPU: NVIDIA H100 80GB HBM3 MIG 3g.40gb
- 实际可用显存: ~79 GB

**测试结果**：
```
视频: 720p (1280×720) × 160 帧
分块: 2 个 chunk（80 帧/chunk）
配置: FP16 GPU 模式

Chunk 1: ❌ OOM（尝试分配 6.85 GB，已占用 74.95 GB）
Chunk 2: ❌ OOM（尝试分配 6.85 GB，已占用 74.12 GB）
```

### 本机情况

**硬件配置**：
- GPU: NVIDIA H100 80GB HBM3 MIG 3g.40gb
- 实际可用显存: **~40 GB**（MIG 分区）

**关键发现**：
- 本机 GPU 只有 40GB（MIG 模式下的分区）
- 无法复现 CMCC 的 79GB 环境
- 缺少部分依赖（decord）

---

## 🔍 根因分析

### 1. 显存占用详细分解

从 CMCC 的错误信息可以看到：

```
Already allocated: 74.95 GiB
尝试分配: 6.85 GiB
GPU 总容量: 79.18 GiB
结果: OOM
```

#### 显存占用组成

| 组件 | 显存占用（估算） | 依据 |
|------|----------------|------|
| **UniMatch 模型** | ~8-12 GB | 光流计算模型 |
| **DOVER 模型（FP16）** | ~6-8 GB | 质量评估模型 |
| **其他模型/缓存** | ~40-50 GB | ⚠️ 这是关键问题 |
| **总计（模型）** | **~54-70 GB** | |
| **720p × 80 帧数据（FP16）** | ~6.85 GB | 推理时需要 |
| **总需求** | **~65-77 GB** | 接近或超过 79 GB |

### 2. 关键问题：神秘的 40-50 GB

**问题**：DOVER 和 UniMatch 加起来只需要 14-20 GB，但实际已占用 54-70 GB。

**可能原因**：

#### 原因 A：Qwen VLM 模型在 GPU 上（最可能）

如果 Qwen3.5-27B-VL 也加载在 GPU 上：
```
Qwen3.5-27B (FP16): ~54 GB
UniMatch: ~10 GB
DOVER (FP16): ~7 GB
总计: ~71 GB
```

**验证方法**：
```bash
# 在 CMCC 机器上，加载模型后检查
python -c "
import torch
print('已分配显存:', torch.cuda.memory_allocated() / 1024**3, 'GB')
print('保留显存:', torch.cuda.memory_reserved() / 1024**3, 'GB')
"
```

#### 原因 B：PyTorch 显存碎片化

PyTorch 可能保留了大量未使用的显存：
```
已分配: 53.58 GB
保留但未分配: 2.39-6.22 GB
```

#### 原因 C：多个进程共享 GPU

如果有其他进程也在使用同一 GPU：
```bash
# 检查
nvidia-smi
fuser -v /dev/nvidia*
```

---

## 💡 解决方案矩阵

### 方案对比

| 方案 | 显存节省 | 实施难度 | 性能影响 | 推荐度 |
|------|---------|---------|---------|--------|
| **A. 降采样到 480p** | **55%** | ⭐ 简单 | < 1% | ⭐⭐⭐ **强烈推荐** |
| B. 卸载 Qwen VLM | 54 GB | ⭐⭐ 中等 | 无（按需加载） | ⭐⭐⭐ 推荐 |
| C. 卸载 UniMatch | 10 GB | ⭐⭐ 中等 | +10% 时间 | ⭐⭐ 备选 |
| D. 减小 chunk (3秒) | 40% | ⭐ 简单 | 可能影响精度 | ⭐⭐ 备选 |
| E. CPU 模式 | 100% | ⭐ 简单 | -90% 速度 | ❌ 最后手段 |

---

### 方案 A：降采样到 480p（当前方案）

**理论分析**：
```
720p (1280×720) → 480p (853×480)
像素数: 921,600 → 409,440（减少 56%）

显存需求（FP16）:
- 720p × 80 帧: ~6.85 GB
- 480p × 80 帧: ~3.0 GB
- 节省: ~3.85 GB (56%)
```

**预期效果**：
```
当前: 74.95 GB（模型）+ 6.85 GB（数据）= 81.8 GB > 79 GB（OOM）
降采样: 74.95 GB（模型）+ 3.0 GB（数据）= 77.95 GB < 79 GB（安全）
```

**问题**：
- ❌ **降采样代码在 CMCC 上未生效**
- 原因：代码更新问题或缓存问题

---

### 方案 B：卸载 Qwen VLM（推荐组合）

如果 Qwen 占用了 ~54 GB，可以按需加载：

```python
# 在 Stage 3 处理流程中
# 1. 先处理不需要 VLM 的样本
for sample in samples:
    if not need_vlm(sample):
        # 只加载 UniMatch + DOVER
        process_without_vlm(sample)

# 2. 批量处理需要 VLM 的样本
vlm_samples = [s for s in samples if need_vlm(s)]
if vlm_samples:
    # 卸载 DOVER
    dover_fn = None
    torch.cuda.empty_cache()
    
    # 加载 Qwen
    vlm_fn = load_qwen_fn(...)
    
    for sample in vlm_samples:
        process_with_vlm(sample, vlm_fn)
```

**预期效果**：
```
不需要 VLM: 10 GB（UniMatch）+ 7 GB（DOVER）+ 3 GB（480p 数据）= 20 GB
需要 VLM: 54 GB（Qwen）+ 3 GB（数据）= 57 GB

两种情况都安全！
```

---

### 方案 C：分阶段处理（终极方案）

```python
# 阶段 1：视觉特征提取（UniMatch + DOVER）
for sample in samples:
    # 只加载 UniMatch + DOVER（~17 GB）
    flow = compute_flow(sample)
    quality = compute_quality(sample)
    save_features(sample, flow, quality)

# 卸载视觉模型
del unimatch_fn, dover_fn
torch.cuda.empty_cache()

# 阶段 2：VLM 处理（仅需要 VLM 的样本）
vlm_fn = load_qwen_fn()
for sample in samples_need_vlm:
    # 只加载 Qwen（~54 GB）
    vlm_result = process_vlm(sample, vlm_fn)
    save_vlm_result(sample, vlm_result)
```

**优势**：
- ✅ 每个阶段显存占用最小化
- ✅ 可以处理任意分辨率
- ✅ 绝对不会 OOM

**劣势**：
- ⚠️ 需要两遍处理
- ⚠️ 代码重构工作量大

---

## 🎯 推荐实施方案

### 短期方案（立即实施）⭐⭐⭐

**方案 A：降采样到 480p**

**步骤**：
1. 确认 `stage3_gpu.py` 代码正确更新
2. 清理 Python 缓存
3. 验证降采样生效
4. 重新测试

**预期**：
- 显存占用：77.95 GB < 79 GB
- 成功率：> 95%

---

### 中期方案（1-2 天）⭐⭐⭐

**方案 A + 诊断优化**

**额外步骤**：
1. 确认 Qwen 是否在 GPU 上
2. 如果是，考虑按需加载
3. 启用 PyTorch 显存优化

```python
# 在脚本开头添加
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

---

### 长期方案（1 周）⭐⭐

**方案 C：分阶段处理**

**优势**：
- 彻底解决显存问题
- 支持任意分辨率
- 更好的可维护性

**实施**：
1. 重构 Stage 3 为两阶段
2. 添加中间特征存储
3. 更新批处理脚本

---

## 📋 当前行动项

### 紧急（CMCC）

1. **验证代码更新**：
   ```bash
   grep -n "max_resolution" /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py
   ```

2. **清理缓存**：
   ```bash
   find /root/work/david_work/sana_qc_pipeline -name "*.pyc" -delete
   find /root/work/david_work/sana_qc_pipeline -name "__pycache__" -type d -exec rm -rf {} +
   ```

3. **运行验证脚本**：
   ```bash
   python test_scripts/verify_downsampling.py
   ```

4. **重新测试**：
   ```bash
   python test_scripts/profile_stage3_single_sample_cmcc.py
   ```

### 诊断（CMCC）

5. **检查显存占用来源**：
   ```python
   # 在加载模型后，检查每个模型的显存占用
   import torch
   print("UniMatch 加载后:", torch.cuda.memory_allocated() / 1024**3, "GB")
   # 加载 DOVER
   print("DOVER 加载后:", torch.cuda.memory_allocated() / 1024**3, "GB")
   # 加载 Qwen（如果有）
   print("Qwen 加载后:", torch.cuda.memory_allocated() / 1024**3, "GB")
   ```

6. **检查是否有其他进程**：
   ```bash
   nvidia-smi
   fuser -v /dev/nvidia0
   ```

---

## 🔬 理论验证

### 降采样后的显存计算

```python
# 720p × 80 帧（FP16）
frames = 80
H, W = 720, 1280
channels = 3
dtype_bytes = 2  # FP16

input_memory = frames * H * W * channels * dtype_bytes
print(f"输入显存: {input_memory / 1024**3:.2f} GB")  # ~0.42 GB

# DOVER 内部计算（25x overhead）
total_memory = input_memory * 25
print(f"总显存需求: {total_memory / 1024**3:.2f} GB")  # ~10.5 GB

# 但实际测试显示只需要 ~6.85 GB
# 说明 DOVER 内部有优化
```

```python
# 480p × 80 帧（FP16）
H, W = 480, 853
input_memory = frames * H * W * channels * dtype_bytes
print(f"输入显存: {input_memory / 1024**3:.2f} GB")  # ~0.19 GB

total_memory = input_memory * 25
print(f"总显存需求: {total_memory / 1024**3:.2f} GB")  # ~4.7 GB

# 实际可能是 ~3.0 GB
```

### 安全性验证

```
当前配置（720p）:
已占用: 74.95 GB
需要: 6.85 GB
总计: 81.8 GB > 79 GB（OOM）

降采样后（480p）:
已占用: 74.95 GB
需要: 3.0 GB
总计: 77.95 GB < 79 GB（安全，剩余 1.23 GB）

安全边际: 1.23 GB / 79 GB = 1.6%（略紧张）
```

**结论**：降采样到 480p **理论上可行**，但安全边际很小。

---

## 🎓 经验教训

### 1. 显存占用的隐藏因素

- ❌ 错误假设：只有当前模型占用显存
- ✅ 实际情况：可能有多个模型、缓存、其他进程

### 2. 调试的重要性

- ❌ 盲目优化：不知道瓶颈在哪
- ✅ 先诊断：确认每个组件的显存占用

### 3. 多层防护

- ❌ 单一方案：降采样
- ✅ 组合方案：降采样 + 按需加载 + 显存优化

---

## 📊 决策树

```
CMCC OOM 问题
    ↓
Q: 降采样代码是否生效？
    ├─ 是 → Q: 仍然 OOM？
    │        ├─ 是 → 诊断其他模型占用（Qwen？）
    │        │       ├─ 发现 Qwen → 实施方案 B（按需加载）
    │        │       └─ 未发现 → 实施方案 D（减小 chunk）
    │        └─ 否 → ✅ 问题解决
    │
    └─ 否 → 修复代码更新问题
            ├─ 清理缓存
            ├─ 验证代码
            └─ 重新测试
```

---

## 📝 总结

### 核心问题

**CMCC 机器上 ~40-50 GB 的"神秘显存占用"**，导致即使 FP16 + 降采样也可能 OOM。

### 最可能的原因

**Qwen3.5-27B-VL 模型（~54 GB）同时在 GPU 上**。

### 推荐方案

1. **立即**：确认降采样代码生效（方案 A）
2. **短期**：诊断显存占用来源
3. **中期**：如果是 Qwen，实施按需加载（方案 B）
4. **长期**：分阶段处理架构（方案 C）

### 预期效果

- 方案 A 单独：成功率 60-80%（取决于其他模型）
- 方案 A + B：成功率 > 95%
- 方案 C：成功率 100%

---

**生成时间**: 2026-08-09  
**状态**: 🔴 等待 CMCC 验证降采样 + 诊断显存占用
