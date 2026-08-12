# DOVER OOM 问题诊断与解决方案

## 🔍 问题诊断

### CMCC 机器实际测试结果

**测试样本**：
- 160 帧，720×1280 分辨率
- 10 秒视频，分成 2 个 chunk（80 帧/chunk）

**错误信息**：
```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 6.85 GiB. 
GPU 0 has a total capacity of 79.18 GiB of which 4.41 GiB is free. 
Process has 58.02 GiB memory in use. 
Including non-PyTorch memory, this process has 53.58 GiB allocated by PyTorch.
```

**关键发现**：
1. ✅ FP16 已启用（日志确认）
2. ❌ 720p × 80 帧仍然 OOM
3. ❌ 已占用 53.58 GB（UniMatch + DOVER + 其他模型）
4. ❌ 第二个 chunk 尝试分配 6.85 GB 时失败

---

## 💡 根本原因

### 显存占用分析

| 组件 | 显存占用（估算） |
|------|----------------|
| UniMatch 模型 | ~8-12 GB |
| DOVER 模型（FP16） | ~6-8 GB |
| 其他模型（Qwen？） | ~20-30 GB |
| **总模型占用** | **~40-50 GB** |
| 720p × 80 帧数据（FP16） | ~10-15 GB |
| **峰值需求** | **~60-65 GB** |

**问题**：
- GPU 总容量：79.18 GB
- 已占用：53.58 GB
- 剩余：4.41 GB
- 需要：6.85 GB
- **缺口**：2.44 GB

### 为什么 FP16 不够？

即使 FP16 将 DOVER 显存减半，但：
1. **UniMatch 仍占用大量显存**（未优化）
2. **可能有其他模型在 GPU 上**（Qwen VLM？）
3. **显存碎片化**导致无法分配连续空间
4. **720p 分辨率仍然较高**

---

## ✅ 解决方案

### 方案 A：降采样视频（推荐）⭐⭐⭐

**原理**：将高分辨率视频降采样到 480p 或更低

**显存节省**：
- 720p (1280×720) → 480p (853×480)
- 像素数：921,600 → 409,440（减少 55%）
- 显存需求：10-15 GB → 4-7 GB

**实现**：修改 `stage3_gpu.py` 的 `_decode_frames` 函数

```python
def _decode_frames(mp4_bytes: bytes, max_resolution: int = 640) -> np.ndarray | None:
    """解码视频，自动降采样到 max_resolution
    
    Args:
        mp4_bytes: 视频文件字节
        max_resolution: 最大边长（默认 640，即 480p）
    """
    if not mp4_bytes:
        return None
    try:
        import av
        import cv2
        frames = []
        with av.open(io.BytesIO(mp4_bytes)) as c:
            for pkt in c.demux(video=0):
                for f in pkt.decode():
                    frame = f.to_ndarray(format="rgb24")
                    H, W = frame.shape[:2]
                    
                    # 降采样到 max_resolution
                    if max(H, W) > max_resolution:
                        scale = max_resolution / max(H, W)
                        new_H, new_W = int(H * scale), int(W * scale)
                        frame = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_AREA)
                    
                    frames.append(frame)
        return np.array(frames, dtype=np.uint8) if frames else None
    except Exception:
        return None
```

**优点**：
- ✅ 显存占用大幅下降
- ✅ 不影响 DOVER 质量评估（对分辨率不敏感）
- ✅ 实现简单

**缺点**：
- ⚠️ 略微降低质量评估精度（通常 < 2%）

---

### 方案 B：卸载 UniMatch（备选）⭐⭐

**原理**：在 DOVER 处理前卸载 UniMatch，释放 8-12 GB 显存

**实现**：

```python
# 在 process_sample_stage3 函数中
# 先处理 UniMatch
flow_val = unimatch_flow_magnitude(frames_rgb, flow_fn)

# 卸载 UniMatch（如果是函数内部加载的模型）
if hasattr(flow_fn, '__self__') and hasattr(flow_fn.__self__, 'cpu'):
    flow_fn.__self__.cpu()
    torch.cuda.empty_cache()

# 再处理 DOVER
dover_val = dover_score(frames_rgb, dover_fn)
```

**优点**：
- ✅ 释放 8-12 GB 显存

**缺点**：
- ⚠️ 增加处理时间（模型移动开销）
- ⚠️ 实现复杂（需要访问模型对象）

---

### 方案 C：减小 chunk 大小⭐

**原理**：将 5 秒 chunk 改为 2-3 秒

**实现**：修改 `visual_metrics.py`

```python
# 旧值
DOVER_CHUNK_S: int = 5  # 5 秒 = 80 帧

# 新值
DOVER_CHUNK_S: int = 3  # 3 秒 = 48 帧
```

**显存节省**：
- 80 帧 → 48 帧
- 显存需求：10-15 GB → 6-9 GB（减少 40%）

**优点**：
- ✅ 显存占用下降
- ✅ 实现极简

**缺点**：
- ⚠️ 可能影响 DOVER 评估准确性（需要验证）
- ⚠️ 增加处理次数（更多 chunk）

---

### 方案 D：PyTorch 显存优化⭐

**原理**：启用 PyTorch 的显存优化选项

**实现**：在脚本开头添加

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

**优点**：
- ✅ 减少显存碎片化
- ✅ 无需修改代码逻辑

**缺点**：
- ⚠️ 效果有限（通常只能节省 5-10%）

---

### 方案 E：CPU 模式（最后手段）❌

**原理**：DOVER 使用 CPU 模式

**实现**：
```python
dover_fn = load_dover_fn(device="cpu")
```

**优点**：
- ✅ 绝对不会 OOM

**缺点**：
- ❌ 速度慢 10-40x（回到原问题）

---

## 🎯 推荐方案组合

### 最佳组合：A + D

```python
# 1. 启用 PyTorch 显存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 2. 降采样视频到 480p
def _decode_frames(mp4_bytes: bytes, max_resolution: int = 640):
    # ... 降采样逻辑 ...
```

**预期效果**：
- 显存需求：10-15 GB → 4-7 GB
- 足够在 79 GB GPU 上运行（即使有其他模型）

### 备选组合：A + C

如果仍然 OOM，进一步减小 chunk：

```python
DOVER_CHUNK_S: int = 2  # 2 秒 = 32 帧
```

---

## 📝 实施步骤

### 步骤 1：更新 `stage3_gpu.py`

添加降采样逻辑到 `_decode_frames` 函数

### 步骤 2：在 CMCC 测试

```bash
python test_scripts/profile_stage3_single_sample_cmcc.py
```

### 步骤 3：验证结果

检查：
- [ ] 不再 OOM
- [ ] 处理速度 < 15 秒/样本
- [ ] DOVER 分数与原分辨率差异 < 2%

### 步骤 4：全量部署

如果测试通过，部署到生产环境

---

## 📊 预期效果对比

| 方案 | 显存节省 | 速度影响 | 精度影响 | 推荐度 |
|------|---------|---------|---------|--------|
| A. 降采样（480p） | 55% | 无 | < 2% | ⭐⭐⭐ |
| B. 卸载 UniMatch | 15% | +10% | 无 | ⭐⭐ |
| C. 减小 chunk | 40% | +20% | 未知 | ⭐ |
| D. PyTorch 优化 | 5-10% | 无 | 无 | ⭐⭐⭐ |
| E. CPU 模式 | 100% | -90% | 无 | ❌ |

---

## 🔍 诊断工具

### 使用更新后的 profile 脚本

```bash
# 本机已更新：test_scripts/profile_stage3_single_sample_cmcc.py
# 功能：
# - 显存监控
# - chunk 级别计时
# - OOM 诊断
# - 解决方案建议

python test_scripts/profile_stage3_single_sample_cmcc.py
```

---

**生成时间**: 2026-08-09  
**状态**: ⏳ 等待 CMCC 测试验证
