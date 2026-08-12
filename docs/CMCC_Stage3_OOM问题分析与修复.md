# CMCC Stage3 OOM 问题分析与修复

> **日期**：2026-08-09  
> **问题**：DOVER 在 H100 上 OOM（尝试分配 20.54 GB）  
> **根因**：移除了 CPU workaround 后，高分辨率视频直接在 GPU 上处理  
> **修复**：智能显存管理 - 根据分辨率自动选择 CPU/GPU

---

## 🐛 问题现象

```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 20.54 GiB. 
GPU 0 has a total capacity of 79.18 GiB of which 16.07 GiB is free.
```

**测试样本**：
- 视频：160 帧，720x1280 分辨率
- 处理步骤：UniMatch (6.95秒) → DOVER (OOM)

---

## 🔍 根因分析

### 历史背景

1. **最初实现**（提交 815571b）：DOVER 在 GPU 上运行
2. **CPU workaround**（某个提交）：因 H100 兼容性问题强制 CPU
3. **移除 workaround**（提交 58a5a8d，2026-08-07）：确认 H100 兼容，移除 CPU 限制
4. **OOM 问题**（2026-08-09）：高分辨率视频导致显存不足

### 显存占用计算

**单个 5 秒块（80 帧，720x1280）**：

```
输入张量：1 * 3 * 80 * 720 * 1280 * 4 bytes = 0.88 GB

DOVER 内部计算（注意力机制 + 特征提取）：
- Swin Transformer 的注意力矩阵
- 多层特征金字塔
- 技术质量 + 美学质量双分支

峰值显存 ≈ 输入的 20-30 倍 = 17-26 GB
```

**实际测量**：
- OOM 时尝试分配：20.54 GB
- 可用显存：16.07 GB
- 已占用：46.37 GB（UniMatch + DOVER 模型 + 其他开销）

### 为什么之前没问题？

**CPU 模式下**（workaround 期间）：
```python
device = "cpu"  # 强制 CPU
t = t.to("cpu")  # 数据在 CPU
model.to("cpu")  # 模型在 CPU
# 不占用 GPU 显存
```

**GPU 模式下**（移除 workaround 后）：
```python
device = "cuda"  # 使用 GPU
t = t.to("cuda")  # 数据在 GPU
model.to("cuda")  # 模型在 GPU
# 占用大量 GPU 显存
```

---

## 🔧 修复方案

### 方案对比

| 方案 | 优点 | 缺点 | 决策 |
|------|------|------|------|
| A. 恢复 CPU 模式 | 稳定，不会 OOM | 慢 ~10x | ❌ 放弃性能 |
| B. 减小块大小 | 保持 GPU 加速 | 仍可能 OOM | ⚠️ 治标不治本 |
| C. 智能显存管理 | 自适应，最优性能 | 稍复杂 | ✅ 采用 |

### 实现：智能显存管理

**核心逻辑**：
```python
def dover_fn(frames_rgb):
    T, H, W, C = frames_rgb.shape
    
    # 阈值：720p (720 * 1280 = 921600 像素)
    current_resolution = H * W
    
    # 估算显存需求（输入 * 25 倍峰值系数）
    estimated_vram_gb = (T * H * W * 3 * 4 * 25) / (1024 ** 3)
    
    # 决策：高分辨率或显存需求 >15GB → CPU
    use_cpu = (current_resolution > 921600) or (estimated_vram_gb > 15)
    
    if use_cpu:
        # 临时将模型移到 CPU 进行推理
        model.cpu()
        t = t.to("cpu")
        results = model(views)
        model.to("cuda")  # 推理后恢复
    else:
        # GPU 推理
        t = t.to("cuda")
        results = model(views)
```

**决策表**：

| 分辨率 | 帧数 | 估算显存 | 设备选择 |
|--------|------|---------|---------|
| 480p (640x480) | 80 | 2.6 GB | GPU ✅ |
| 720p (1280x720) | 80 | 10.7 GB | GPU ✅ |
| **720p (1280x720)** | **160** | **21.4 GB** | **CPU** ⚠️ |
| 1080p (1920x1080) | 80 | 24.0 GB | CPU ⚠️ |
| 4K (3840x2160) | 80 | 96.0 GB | CPU ⚠️ |

**用户测试样本（720x1280, 160帧）**：
- 估算显存：21.4 GB
- 决策：**使用 CPU**（避免 OOM）
- 性能：慢但稳定

---

## 📊 性能影响

### 不同分辨率的性能

| 分辨率 | 设备 | 单块耗时 | 说明 |
|--------|------|---------|------|
| ≤720p | GPU | ~500 ms | 快速 ✅ |
| >720p | CPU | ~5 秒 | 稳定但慢 ⚠️ |

### 用户数据集分布

需要分析实际数据集中的分辨率分布：

```bash
# 统计视频分辨率
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height \
  -of csv=p=0 sample.mp4
```

**如果大部分视频 ≤720p**：
- 主要使用 GPU，性能好 ✅

**如果大部分视频 >720p**：
- 主要使用 CPU，性能降级 ⚠️
- 建议：预处理降采样到 720p

---

## ✅ 修复验证

### 步骤 1：更新代码

```bash
cd /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc
cp stage3_gpu.py stage3_gpu.py.before_oom_fix
# 复制修复后的 stage3_gpu.py
```

### 步骤 2：重新运行测试

```bash
cd /root/work/david_work/sana_qc_pipeline
python test_scripts/profile_stage3_single_sample_cmcc.py
```

**预期输出**：
```
[4/7] DOVER 质量: ~5000 ms
      结果: 0.xxxx
      设备: CPU (自动降级，避免 OOM)
```

### 步骤 3：小规模验证

```bash
time python scripts/run_stage3_cmcc.py \
  --stage12-jsonl /tmp/test10.jsonl \
  --skip-vlm \
  ...
```

**预期**：
- ✅ 不再出现 OOM
- ⚠️ DOVER 步骤变慢（~5秒/块）
- ✅ 其他步骤正常（UniMatch 仍在 GPU）

---

## 🎓 经验教训

### 1. 移除 workaround 要谨慎

**错误做法**：
- 看到 "H100 已兼容" 就直接移除 workaround
- 没有测试不同分辨率/帧数的视频

**正确做法**：
- 保留 workaround 的核心逻辑（条件判断）
- 添加自适应选择而不是完全移除
- 在真实数据上全面测试

### 2. 显存估算很重要

**计算公式**：
```
峰值显存 ≈ 输入张量大小 × 峰值系数

输入大小 = batch × channels × frames × height × width × bytes_per_element
峰值系数 = 20-30 (取决于模型架构)
```

**对于 Transformer 模型**：
- 注意力矩阵：O(n²) 复杂度
- 高分辨率/长序列 → 显存爆炸式增长

### 3. 测试要覆盖边界情况

**不足的测试**：
- 只测试小分辨率视频（480p）
- 只测试短视频（<60 帧）

**完整的测试**：
- 低分辨率：480p, 720p
- 高分辨率：1080p, 4K
- 短视频：30-60 帧
- 长视频：150-300 帧
- **组合**：720p + 160 帧（触发 OOM）

### 4. Git 历史很重要

**这次诊断用到的 git 命令**：
```bash
# 查看提交历史
git log --oneline -- stage3_gpu.py

# 查看特定提交的改动
git show 58a5a8d

# 对比两个版本
git diff 58a5a8d^ 58a5a8d -- stage3_gpu.py
```

**经验**：
- 保留清晰的提交信息
- Workaround 要注释为什么需要
- 移除时要解释为什么安全

---

## 📋 后续优化（可选）

### 1. 自适应块大小

当前：固定 5 秒块

优化：根据分辨率动态调整
```python
if resolution <= 480p:
    chunk_s = 10  # 大块，快速
elif resolution <= 720p:
    chunk_s = 5   # 中块，平衡
else:
    chunk_s = 2   # 小块，避免 OOM
```

### 2. 混合精度推理

```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    results = model(views)
```

可能节省 50% 显存，但需要验证精度影响。

### 3. 梯度检查点（Gradient Checkpointing）

DOVER 内部已使用（看到的 `checkpoint.checkpoint` 调用）。

可能的优化：调整检查点策略。

### 4. 预处理降采样

如果数据集中高分辨率视频很多：
```bash
# 预处理：统一降采样到 720p
ffmpeg -i input.mp4 -vf "scale=1280:720" output.mp4
```

权衡：
- ✅ 提升处理速度
- ❌ 损失视频质量（影响 DOVER 评分）

---

## 🔄 回滚方案

如果修复有问题：

```bash
cd /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc
cp stage3_gpu.py.before_oom_fix stage3_gpu.py
```

或者直接恢复到 CPU-only 模式：

```python
# 在 load_dover_fn 中
device = "cpu"  # 强制 CPU
```

---

## ✅ 验证检查清单

- [ ] 代码已更新（智能显存管理）
- [ ] profile_stage3_single_sample_cmcc.py 运行成功（无 OOM）
- [ ] DOVER 使用 CPU 处理高分辨率视频
- [ ] UniMatch 仍在 GPU 上运行
- [ ] 10 样本测试通过
- [ ] 性能可接受（虽然 DOVER 变慢）

---

**修复日期**：2026-08-09  
**修复类型**：显存管理优化  
**影响**：DOVER 在高分辨率视频上自动降级到 CPU  
**状态**：待 CMCC 验证
