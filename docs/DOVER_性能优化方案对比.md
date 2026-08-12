# DOVER 性能优化方案对比

## 📋 问题分析

### 当前方案的缺陷

上一轮优化采用了"永久 CPU 模式"策略：
- ✅ 避免了模型在 GPU/CPU 之间反复移动（每次 ~4 秒）
- ❌ **完全放弃了 GPU 算力**：所有后续样本都用 CPU 推理
- ❌ **CPU 推理极慢**：深度学习模型在 CPU 上比 GPU 慢 10-100 倍

### 核心矛盾

1. **长视频直接传给 GPU 不合理**：显存不够（H100 80GB 也无法容纳 1080p × 1000 帧）
2. **纯 CPU 处理不合理**：速度太慢，浪费 H100 算力
3. **需要找到平衡点**：既利用 GPU，又不超显存

### 关键发现

**DOVER 已经在按 5 秒 chunk 分块处理！**

查看 `visual_metrics.py:177-198`：

```python
def dover_score(...):
    chunks = dover_chunk_indices(n_frames=len(frames_rgb), fps=fps, chunk_s=5)
    for s, e in chunks:
        val = dover_fn(frames_rgb[s:e])  # ⭐ 每次只处理 80 帧（5秒）
```

这意味着：
- 即使是 1068 帧的长视频，也会被分成 13 个 80 帧的 chunk
- **每个 chunk 可以独立选择设备（GPU 或 CPU）**
- 不需要一次性加载整个视频到 GPU

---

## 🎯 候选优化方案

### 方案 A：纯 GPU 模式（基线）

**策略**：
- DOVER 模型常驻 GPU
- 所有 chunk 在 GPU 上处理
- 不做任何显存保护

**预期**：
- ✅ **最快**：充分利用 GPU 算力
- ❌ **高分辨率会 OOM**：1080p × 80 帧 ≈ 2.4GB 显存（可能超限）

**适用场景**：
- 低分辨率视频（<720p）
- 短视频（<5 秒）

---

### 方案 B：纯 CPU 模式（当前方案）

**策略**：
- DOVER 模型常驻 CPU
- 所有 chunk 在 CPU 上处理
- 永不使用 GPU

**预期**：
- ✅ **绝对安全**：永不 OOM
- ❌ **极慢**：CPU 推理比 GPU 慢 10-100 倍

**适用场景**：
- 无 GPU 环境
- 极高分辨率视频（>4K）

---

### 方案 C：混合精度 GPU（FP16）

**策略**：
- DOVER 模型常驻 GPU，转换为 float16
- 所有 chunk 在 GPU 上处理（FP16）
- 显存占用减半

**预期**：
- ✅ **显存减半**：1080p × 80 帧 ≈ 1.2GB（比 FP32 少一半）
- ✅ **速度接近 FP32**：H100 的 FP16 性能极强
- ⚠️ **精度可能下降**：需要验证 DOVER 是否支持 FP16

**适用场景**：
- 中高分辨率视频（720p-1080p）
- 对精度要求不高的场景

---

### 方案 D：智能分块（推荐）

**策略**：
- DOVER 模型初始在 GPU
- **每个 chunk 独立判断**：
  - 估算显存需求 = `T × H × W × 3 × 4 bytes`
  - 如果 < 1GB → GPU 处理
  - 如果 > 1GB → 临时移到 CPU 处理，完成后移回 GPU
- 动态适应不同分辨率和长度

**预期**：
- ✅ **灵活**：低分辨率用 GPU，高分辨率用 CPU
- ✅ **安全**：永不 OOM
- ⚠️ **模型移动开销**：每次 CPU/GPU 切换 ~4 秒
  - 但如果大部分 chunk 在 GPU，开销可控
  - 只有高分辨率才触发 CPU

**适用场景**：
- **混合分辨率的数据集**（推荐）
- 需要平衡速度和稳定性

---

### 方案 E：智能分块 + FP16（最优？）

**策略**：
- 结合方案 C 和 D
- GPU 模式用 FP16，CPU 模式用 FP32
- 每个 chunk 独立判断设备

**预期**：
- ✅ **最优平衡**：速度快、显存少、不 OOM
- ⚠️ **实现复杂度高**

---

## 📊 基准测试设计

### 测试脚本

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python test_scripts/dover_performance_benchmark.py
```

### 测试用例

| 视频名称 | 帧数 | 分辨率 | 显存估算 | 预期结果 |
|---------|------|--------|----------|----------|
| short_480p | 80 | 640×480 | ~350MB | GPU 应该成功 |
| short_720p | 80 | 1280×720 | ~850MB | GPU 可能成功 |
| short_1080p | 80 | 1920×1080 | ~1.9GB | GPU 可能 OOM |
| long_480p | 240 | 640×480 | ~1GB | GPU 可能成功 |
| long_720p | 240 | 1280×720 | ~2.5GB | GPU 可能 OOM |

### 关键指标

1. **速度**：每个视频的处理时间（秒）
2. **显存**：峰值显存占用（MB）
3. **成功率**：是否 OOM
4. **精度**：不同方案的分数差异（验证 FP16 是否影响精度）

---

## 🧪 执行测试

### 步骤 1：激活环境

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate abot-physworld  # 或你的环境名称
```

### 步骤 2：检查 DOVER 模型

```bash
ls -la /mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/
# 应该看到：
#   dover.yml
#   pretrained_weights/DOVER.pth
```

### 步骤 3：运行基准测试

```bash
python test_scripts/dover_performance_benchmark.py
```

### 步骤 4：查看结果

```bash
cat test_scripts/dover_benchmark_results.json
```

---

## 📈 预期结果分析

### 速度对比（预测）

| 方案 | short_480p | short_720p | short_1080p | 速度排名 |
|------|-----------|-----------|-------------|----------|
| A. 纯 GPU | 0.2s | 0.3s | OOM? | 🥇 最快 |
| B. 纯 CPU | 2.0s | 4.0s | 8.0s | 🐌 最慢 |
| C. FP16 GPU | 0.2s | 0.3s | 0.5s? | 🥇 最快 |
| D. 智能分块 | 0.2s | 0.3s | 8.0s (CPU) | 🥈 中等 |

### 显存对比（预测）

| 方案 | short_480p | short_720p | short_1080p | 安全性 |
|------|-----------|-----------|-------------|--------|
| A. 纯 GPU | 350MB | 850MB | 1.9GB (OOM?) | ⚠️ 危险 |
| B. 纯 CPU | 0MB | 0MB | 0MB | ✅ 绝对安全 |
| C. FP16 GPU | 175MB | 425MB | 950MB | ✅ 较安全 |
| D. 智能分块 | 350MB | 850MB | 0MB (CPU) | ✅ 绝对安全 |

### 综合评分（预测）

| 方案 | 速度 | 安全性 | 灵活性 | 推荐度 |
|------|------|--------|--------|--------|
| A. 纯 GPU | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⚠️ 不推荐（易 OOM） |
| B. 纯 CPU | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⚠️ 不推荐（太慢） |
| C. FP16 GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ 推荐（需验证精度） |
| D. 智能分块 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅✅ 强烈推荐 |

---

## 🎯 推荐方案

### 首选：方案 C（FP16 GPU）

**前提条件**：
- 验证 DOVER 在 FP16 下的精度损失 < 1%
- 数据集以低中分辨率为主（<1080p）

**优势**：
- 速度最快（充分利用 GPU）
- 显存减半（支持更高分辨率）
- 实现简单（只需 `.half()` 转换）

### 次选：方案 D（智能分块）

**前提条件**：
- FP16 精度损失不可接受
- 数据集分辨率混合（480p-4K）

**优势**：
- 绝对安全（永不 OOM）
- 灵活适应各种分辨率
- 低分辨率仍用 GPU（快）

### 终极方案：C + D 混合

**策略**：
- GPU 模式用 FP16
- CPU 模式用 FP32
- 每个 chunk 动态选择

**实现**：
```python
def dover_fn(frames_rgb: np.ndarray) -> float:
    T, H, W, C = frames_rgb.shape
    estimated_vram_mb = (T * H * W * 3 * 2) / (1024 ** 2)  # FP16 = 2 bytes
    
    if estimated_vram_mb < 2000:  # FP16 可以处理更大的视频
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    # 临时移动模型（如果需要）
    model.to(device)
    if device == "cuda":
        model.half()
    else:
        model.float()
    
    t = torch.from_numpy(frames_rgb).to(dtype) / 255.0
    t = t.permute(3, 0, 1, 2).unsqueeze(0).to(device)
    views = {"technical": t, "aesthetic": t}
    
    with torch.no_grad():
        results = model(views)
    
    return float(sum(r.mean().item() for r in results) / len(results))
```

---

## 📝 下一步行动

### 1. 运行基准测试（本机）

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python test_scripts/dover_performance_benchmark.py
```

### 2. 分析结果

查看 `test_scripts/dover_benchmark_results.json`，对比：
- 速度差异（GPU vs CPU 的实际倍数）
- 显存占用（是否在 H100 80GB 范围内）
- FP16 精度（与 FP32 的分数差异）

### 3. 选择最优方案

根据测试结果，选择：
- 如果 FP16 精度 OK + 无 OOM → **方案 C**
- 如果 FP16 精度不佳 → **方案 D**
- 如果需要极致性能 → **方案 C+D 混合**

### 4. 实现并更新代码

修改 `src/sana_wm_pipeline/qc/stage3_gpu.py` 的 `load_dover_fn` 函数

### 5. 在 CMCC 机器上验证

准备详细文档，让用户在 CMCC 机器上复现测试

---

## 🚀 CMCC 部署指南（待定）

测试完成后，将生成：
1. `CMCC_DOVER_优化部署指南.md`
2. 更新后的 `stage3_gpu.py`
3. 更新后的 `run_stage3_cmcc.py`
4. 性能对比报告

---

## 📊 附录：显存估算公式

### FP32（4 bytes per value）

```
VRAM (bytes) = T × H × W × C × 4 × overhead
overhead ≈ 25x (DOVER 模型内部计算)

示例：80 帧 × 1920 × 1080 × 3 × 4 × 25 ≈ 15GB
```

### FP16（2 bytes per value）

```
VRAM (bytes) = T × H × W × C × 2 × overhead
overhead ≈ 25x

示例：80 帧 × 1920 × 1080 × 3 × 2 × 25 ≈ 7.5GB
```

### 安全阈值

| 分辨率 | FP32 最大帧数 | FP16 最大帧数 | chunk_s 建议 |
|--------|--------------|--------------|-------------|
| 480p | 300 | 600 | 5s (标准) |
| 720p | 180 | 360 | 5s (标准) |
| 1080p | 80 | 160 | 2-3s |
| 4K | 20 | 40 | 1s |

---

**生成时间**: 2026-08-09  
**作者**: Claude (Opus 4.8)  
**项目**: sana_wm_pipeline Stage 3 性能优化
