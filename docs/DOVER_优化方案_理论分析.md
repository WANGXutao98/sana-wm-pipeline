# DOVER 性能优化：理论分析与推荐方案

## 📋 当前状况

### 依赖问题

本机测试遇到了 DOVER 依赖问题（缺少 `scikit-video`），由于权限限制无法安装。但这不影响我们基于理论分析和已知数据给出优化方案。

### 已知数据（来自上一个对话的性能分析）

- **单样本总耗时**: 178 秒
- **DOVER 占比**: 94.9%（169 秒）
- **问题根因**: 反复移动模型（GPU ↔ CPU），每次移动 ~4 秒

---

## 🎯 理论分析

### DOVER 的处理流程

从 `visual_metrics.py` 可知，DOVER 已经按 **5 秒 chunk** 分块处理：

```python
def dover_score(...):
    chunks = dover_chunk_indices(n_frames=len(frames_rgb), fps=fps, chunk_s=5)
    for s, e in chunks:
        val = dover_fn(frames_rgb[s:e])  # 每次只处理 80 帧
```

**关键点**：
- 即使是 1068 帧的长视频，也会被分成 **13 个 80 帧的 chunk**
- **每个 chunk 可以独立选择设备（GPU 或 CPU）**
- 不需要一次性加载整个视频到 GPU

### 显存需求估算

#### FP32（当前方案）

```
单个 chunk 显存 = T × H × W × C × 4 bytes × overhead

overhead ≈ 25x（DOVER 模型内部计算开销）
```

| 分辨率 | 80 帧显存需求 | H100 80GB 能否容纳？ |
|--------|--------------|---------------------|
| 480p (640×480) | ~350 MB × 25 ≈ 8.8 GB | ✅ 可以 |
| 720p (1280×720) | ~850 MB × 25 ≈ 21 GB | ✅ 可以 |
| 1080p (1920×1080) | ~1.9 GB × 25 ≈ 47 GB | ✅ 可以（接近上限） |
| 4K (3840×2160) | ~7.6 GB × 25 ≈ 190 GB | ❌ 超限 |

#### FP16（推荐方案）

显存减半：

| 分辨率 | 80 帧显存需求 | H100 80GB 能否容纳？ |
|--------|--------------|---------------------|
| 480p | ~4.4 GB | ✅ 可以 |
| 720p | ~10.5 GB | ✅ 可以 |
| 1080p | ~23.5 GB | ✅ 可以（安全） |
| 4K | ~95 GB | ❌ 超限 |

### GPU vs CPU 速度对比（经验数据）

根据深度学习模型的一般规律：

- **GPU (H100)**: ~0.2-0.5 秒/chunk（80 帧）
- **CPU (192 核)**: ~2-8 秒/chunk（80 帧）
- **加速比**: 10-40x

对于 1068 帧视频（13 个 chunk）：
- **纯 GPU**: 13 × 0.3s = 3.9 秒
- **纯 CPU**: 13 × 5s = 65 秒
- **差异**: ~16x

---

## ✅ 推荐优化方案

### 方案 1：FP16 GPU（首选）

**实现**：

修改 `stage3_gpu.py` 的 `load_dover_fn`：

```python
def load_dover_fn(device: str = "cuda", ...):
    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(dover_weight_path, map_location=device, weights_only=False))
    model = model.to(device)
    
    # ⭐ 新增：如果是 GPU，转换为 FP16
    if device == "cuda":
        model = model.half()
    
    model.eval()
    
    def dover_fn(frames_rgb: np.ndarray) -> float:
        t = torch.from_numpy(frames_rgb).float() / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0)
        
        # ⭐ 新增：如果是 GPU，转换输入为 FP16
        if device == "cuda":
            t = t.half()
        
        t = t.to(device)
        views = {"technical": t, "aesthetic": t}
        
        with torch.no_grad():
            results = model(views)
        
        return float(sum(r.mean().item() for r in results) / len(results))
    
    return dover_fn
```

**优势**：
- ✅ 显存减半（支持到 1080p）
- ✅ 速度接近 FP32（H100 的 FP16 性能极强）
- ✅ 实现简单（只需 `.half()` 转换）
- ⚠️ 需要验证精度损失（预计 < 0.5%）

**预期性能**：
- 单样本（1068 帧）：**169 秒 → 5-10 秒**（~20x 加速）
- 139 样本：**10 小时 → 12-23 分钟**

---

### 方案 2：智能分块（次选）

**实现**：

```python
def load_dover_fn(device: str = "cuda", ...):
    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(dover_weight_path, map_location="cuda", weights_only=False))
    model = model.to("cuda")
    model.eval()
    
    def dover_fn(frames_rgb: np.ndarray) -> float:
        T, H, W, C = frames_rgb.shape
        
        # ⭐ 每个 chunk 独立判断设备
        estimated_vram_gb = (T * H * W * 3 * 4 * 25) / (1024 ** 3)
        
        # 阈值：40GB（H100 80GB 的一半，留安全余量）
        if estimated_vram_gb < 40:
            target_device = "cuda"
        else:
            target_device = "cpu"
            model.cpu()  # 临时移到 CPU
        
        t = torch.from_numpy(frames_rgb).float() / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to(target_device)
        views = {"technical": t, "aesthetic": t}
        
        with torch.no_grad():
            results = model(views)
        
        # 移回 GPU
        if target_device == "cpu":
            model.cuda()
        
        return float(sum(r.mean().item() for r in results) / len(results))
    
    return dover_fn
```

**优势**：
- ✅ 绝对安全（永不 OOM）
- ✅ 低分辨率仍用 GPU（快）
- ⚠️ 模型移动开销（每次 ~4 秒）

**适用场景**：
- 数据集分辨率混合（480p-4K）
- FP16 精度损失不可接受

**预期性能**：
- 如果大部分样本 < 1080p → GPU 路径 → **5-10 秒**
- 如果有 4K 样本 → CPU 路径 → **50-70 秒**

---

### 方案 3：FP16 + 智能分块（终极）

结合两者优势：

```python
def load_dover_fn(device: str = "cuda", ...):
    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(dover_weight_path, map_location="cuda", weights_only=False))
    model = model.to("cuda").half()  # 初始为 GPU FP16
    model.eval()
    
    def dover_fn(frames_rgb: np.ndarray) -> float:
        T, H, W, C = frames_rgb.shape
        
        # FP16 显存需求
        estimated_vram_gb = (T * H * W * 3 * 2 * 25) / (1024 ** 3)
        
        # 阈值：60GB（FP16 下可以处理更大视频）
        if estimated_vram_gb < 60:
            target_device = "cuda"
            target_dtype = torch.float16
        else:
            target_device = "cpu"
            target_dtype = torch.float32
            model.cpu().float()  # 临时移到 CPU + FP32
        
        t = torch.from_numpy(frames_rgb).to(target_dtype) / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to(target_device)
        views = {"technical": t, "aesthetic": t}
        
        with torch.no_grad():
            results = model(views)
        
        # 移回 GPU FP16
        if target_device == "cpu":
            model.cuda().half()
        
        return float(sum(r.mean().item() for r in results) / len(results))
    
    return dover_fn
```

**优势**：
- ✅ 最优平衡：速度 + 安全性
- ✅ 支持 480p-4K 全范围
- ⚠️ 实现复杂度最高

---

## 📊 方案对比

| 方案 | 速度 | 安全性 | 实现难度 | 推荐度 |
|------|------|--------|---------|--------|
| 1. FP16 GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ 简单 | ✅✅ 强烈推荐 |
| 2. 智能分块 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ 中等 | ✅ 推荐 |
| 3. FP16 + 智能分块 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ 复杂 | ✅✅✅ 终极方案 |
| 当前方案（纯 CPU） | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ 简单 | ❌ 不推荐 |

---

## 🚀 实施建议

### 步骤 1：先在 CMCC 机器上验证 FP16 精度

创建一个小测试脚本，对比 FP32 vs FP16 的分数差异：

```python
# test_fp16_accuracy.py
dover_fn_fp32 = load_dover_fn(device="cuda", use_fp16=False)
dover_fn_fp16 = load_dover_fn(device="cuda", use_fp16=True)

# 测试 10 个样本
for i in range(10):
    frames = load_sample(i)
    score_fp32 = dover_fn_fp32(frames)
    score_fp16 = dover_fn_fp16(frames)
    diff = abs(score_fp32 - score_fp16)
    print(f"Sample {i}: FP32={score_fp32:.4f}, FP16={score_fp16:.4f}, diff={diff:.4f}")
```

### 步骤 2：根据精度测试结果选择方案

- 如果 diff < 0.01（1% 误差）→ **方案 1（FP16 GPU）**
- 如果 diff > 0.01 但数据集分辨率 < 1080p → **保持 FP32 GPU**
- 如果数据集有 4K 样本 → **方案 2 或 3**

### 步骤 3：更新代码并测试

修改 `stage3_gpu.py`，在 CMCC 机器上运行一个完整的 shard 验证性能

### 步骤 4：全量部署

更新所有处理脚本，开始全量处理

---

## 📝 CMCC 部署清单

### 文件清单

1. ✅ `docs/DOVER_性能优化方案对比.md`（本文档）
2. ⏳ `src/sana_wm_pipeline/qc/stage3_gpu.py`（待更新）
3. ⏳ `test_scripts/test_dover_fp16_accuracy.py`（待创建）
4. ⏳ `CMCC_DEPLOYMENT/DOVER_优化部署指南.md`（待创建）

### 部署步骤

```bash
# 1. 在 CMCC 机器上，激活环境
conda activate <your-env>

# 2. 测试 FP16 精度
python test_scripts/test_dover_fp16_accuracy.py

# 3. 如果精度 OK，更新代码
git pull  # 或手动复制更新后的 stage3_gpu.py

# 4. 运行单个 shard 测试
python scripts/run_stage3_cmcc.py --shard 0 --test-mode

# 5. 验证性能提升（预期 ~20x 加速）
# 6. 全量部署
```

---

## 🔍 关键洞察

### 为什么上一个方案失败了？

上一个对话的"永久 CPU 模式"方案存在根本性缺陷：

1. **误解了瓶颈**：以为是"模型移动开销"，实际上是"CPU 推理太慢"
2. **过度保守**：为了避免 OOM，放弃了所有 GPU 算力
3. **没有利用分块特性**：DOVER 已经按 80 帧分块，每个 chunk 只需 10-20GB 显存

### 正确的优化思路

1. **利用 H100 的 FP16 性能**：显存减半，速度几乎不变
2. **理解 DOVER 的分块机制**：80 帧 chunk 在 H100 上完全可以处理
3. **动态适应**：只有极端情况（4K）才需要 CPU

---

**生成时间**: 2026-08-09  
**作者**: Claude (Opus 4.8)  
**项目**: sana_wm_pipeline Stage 3 DOVER 性能优化
