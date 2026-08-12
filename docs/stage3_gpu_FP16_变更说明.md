# stage3_gpu.py FP16 优化变更说明

## 📋 变更概述

**日期**: 2026-08-09  
**文件**: `src/sana_wm_pipeline/qc/stage3_gpu.py`  
**函数**: `load_dover_fn` (第 297-379 行)  
**备份**: `stage3_gpu.py.backup_20260809_before_fp16`

---

## 🔄 主要变更

### 1. 函数签名变更

**旧版本**：
```python
def load_dover_fn(device: str = "cuda", dover_config_path: str = None, dover_weight_path: str = None):
```

**新版本**：
```python
def load_dover_fn(device: str = "cuda", dover_config_path: str = None, dover_weight_path: str = None, use_fp16: bool = True):
```

**新增参数**：
- `use_fp16: bool = True` - 是否在 GPU 上使用 FP16（默认启用）

---

### 2. 核心逻辑变更

#### 移除的代码（旧逻辑）

```python
# 移除：动态 OOM 检测和 CPU 切换逻辑（~50 行）
_cpu_mode = [False]
_checked = [False]

def dover_fn(frames_rgb):
    T, H, W, C = frames_rgb.shape
    
    # 首次调用检测分辨率
    if not _checked[0]:
        _checked[0] = True
        if current_resolution > RESOLUTION_THRESHOLD or estimated_vram_gb > 15:
            _cpu_mode[0] = True
            model.cpu()  # ❌ 永久切换到 CPU
    
    # 选择设备
    if _cpu_mode[0]:
        target_device = "cpu"
    else:
        target_device = device
```

**问题**：
- 一旦触发 CPU 模式，所有后续样本都用 CPU（包括低分辨率样本）
- CPU 推理比 GPU 慢 10-40x
- 浪费了 H100 80GB 的强大算力

#### 新增的代码（FP16 逻辑）

```python
# 新增：GPU FP16 模式（第 341-348 行）
if device == "cuda" and use_fp16:
    model = model.half()
    print(f"[DOVER] GPU FP16 模式已启用（显存减半，支持到 1080p）", flush=True)
elif device == "cuda":
    print(f"[DOVER] GPU FP32 模式（use_fp16=False）", flush=True)
else:
    print(f"[DOVER] CPU 模式（性能较慢，建议使用 GPU）", flush=True)

# 新增：推理时转换输入为 FP16（第 361-363 行）
def dover_fn(frames_rgb):
    t = torch.from_numpy(frames_rgb).float() / 255.0
    t = t.permute(3, 0, 1, 2).unsqueeze(0)
    
    # GPU FP16 模式：转换输入为 FP16
    if device == "cuda" and use_fp16:
        t = t.half()
    
    t = t.to(device)
```

**优势**：
- 模型常驻 GPU（不再反复移动）
- FP16 显存减半（支持到 1080p）
- 速度几乎不变（H100 的 FP16 性能极强）
- 代码更简洁（-50 行）

---

## 📊 预期性能提升

| 指标 | 旧方案（CPU） | 新方案（FP16 GPU） | 加速比 |
|------|--------------|-------------------|--------|
| 单 chunk (80 帧) | 5-8 秒 | 0.3-0.5 秒 | **10-27x** |
| 单样本 (1068 帧) | 169 秒 | 5-10 秒 | **17-34x** |
| 单 shard (139 样本) | ~6.5 小时 | ~12-23 分钟 | **17-32x** |

---

## 🔍 代码对比

### 旧版本（CPU 模式，第 340-377 行）

```python
# Track if we've switched to CPU mode permanently
_cpu_mode = [False]
_checked = [False]

def dover_fn(frames_rgb: np.ndarray) -> float:
    T, H, W, C = frames_rgb.shape
    
    # One-time check: should we use CPU mode?
    if not _checked[0]:
        _checked[0] = True
        RESOLUTION_THRESHOLD = 720 * 1280
        current_resolution = H * W
        estimated_vram_gb = (T * H * W * 3 * 4 * 25) / (1024 ** 3)
        
        if device == "cuda" and ((current_resolution > RESOLUTION_THRESHOLD) or (estimated_vram_gb > 15)):
            _cpu_mode[0] = True
            warnings.warn(...)
            model.cpu()  # ❌ 永久移到 CPU
    
    # 选择设备
    if _cpu_mode[0]:
        target_device = "cpu"
    else:
        target_device = device
    
    t = torch.from_numpy(frames_rgb).float() / 255.0
    t = t.permute(3, 0, 1, 2).unsqueeze(0).to(target_device)
    views = {"technical": t, "aesthetic": t}
    
    with torch.no_grad():
        results = model(views)
    
    return float(sum(r.mean().item() for r in results) / len(results))
```

### 新版本（FP16 GPU 模式，第 341-379 行）

```python
# GPU 模式：启用 FP16（显存减半）
if device == "cuda" and use_fp16:
    model = model.half()
    print(f"[DOVER] GPU FP16 模式已启用（显存减半，支持到 1080p）", flush=True)
elif device == "cuda":
    print(f"[DOVER] GPU FP32 模式（use_fp16=False）", flush=True)
else:
    print(f"[DOVER] CPU 模式（性能较慢，建议使用 GPU）", flush=True)

model.eval()

def dover_fn(frames_rgb: np.ndarray) -> float:
    # Convert to (1, 3, T, H, W) float normalized
    t = torch.from_numpy(frames_rgb).float() / 255.0
    t = t.permute(3, 0, 1, 2).unsqueeze(0)
    
    # GPU FP16 模式：转换输入为 FP16
    if device == "cuda" and use_fp16:
        t = t.half()
    
    t = t.to(device)
    views = {"technical": t, "aesthetic": t}
    
    with torch.no_grad():
        results = model(views)
    
    return float(sum(r.mean().item() for r in results) / len(results))
```

**关键差异**：
- ✅ 移除了 `_cpu_mode` 和 `_checked` 状态跟踪
- ✅ 移除了 OOM 检测和动态设备切换
- ✅ 添加了 `model.half()` 转换（FP16）
- ✅ 添加了 `t.half()` 输入转换（FP16）
- ✅ 添加了清晰的日志输出

---

## 📝 CMCC 部署步骤

### 1. 文件传输

将更新后的文件复制到 CMCC 机器：

```bash
# 在本机
scp src/sana_wm_pipeline/qc/stage3_gpu.py <user>@<cmcc-host>:/path/to/sana_wm_pipeline/src/sana_wm_pipeline/qc/

# 在 CMCC 机器
cd /path/to/sana_wm_pipeline
cp src/sana_wm_pipeline/qc/stage3_gpu.py src/sana_wm_pipeline/qc/stage3_gpu.py.backup_before_fp16
```

### 2. 验证语法

```bash
python -m py_compile src/sana_wm_pipeline/qc/stage3_gpu.py
# 如果没有输出，说明语法正确
```

### 3. 测试单 shard

```bash
python scripts/run_stage3_cmcc.py \
    --input-jsonl stage1_results/group_X/stage1_shard-000003-000001.jsonl \
    --output-jsonl stage3_results/group_X/stage3_shard-000003-000001.jsonl \
    --caption-overrides-jsonl stage3_results/group_X/caption_overrides_shard-000003-000001.jsonl \
    --group-name group_X
```

**检查日志**：
```bash
# 应该看到这行（说明 FP16 已启用）
[DOVER] GPU FP16 模式已启用（显存减半，支持到 1080p）
```

### 4. 监控 GPU 使用

```bash
# 另开一个终端
nvidia-smi -l 1
```

**预期**：
- GPU 使用率 > 90%
- 显存占用 10-20GB（每个 chunk）
- 功率接近上限（说明满载）

### 5. 性能对比

| 检查项 | 旧方案（CPU） | 新方案（FP16 GPU） | 状态 |
|--------|--------------|-------------------|------|
| 单样本耗时 | 169 秒 | < 15 秒 | ⬜ 待验证 |
| GPU 使用率 | ~0% | > 90% | ⬜ 待验证 |
| 日志提示 | 无 | `[DOVER] GPU FP16 模式已启用` | ⬜ 待验证 |

---

## 🔧 故障排查

### 问题 1：OOM（显存溢出）

**症状**：
```
RuntimeError: CUDA out of memory
```

**原因**：视频分辨率 > 1080p

**解决方案**：

方案 A：禁用 FP16，回退到 FP32
```python
# 在调用 load_dover_fn 时
dover_fn = load_dover_fn(device="cuda", use_fp16=False)
```

方案 B：降采样高分辨率视频
```python
# 在 _decode_frames 函数中添加
def _decode_frames(mp4_bytes: bytes, max_resolution: int = 1080) -> np.ndarray | None:
    # ... 解码 ...
    for f in frames:
        H, W = f.shape[:2]
        if max(H, W) > max_resolution:
            scale = max_resolution / max(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
            f = cv2.resize(f, (new_W, new_H))
```

方案 C：临时切换到 CPU 模式
```python
dover_fn = load_dover_fn(device="cpu")  # 仅用于 4K 视频
```

---

### 问题 2：精度损失

**症状**：FP16 vs FP32 分数差异 > 1%

**验证方法**：
```python
# 测试脚本
dover_fn_fp32 = load_dover_fn(device="cuda", use_fp16=False)
dover_fn_fp16 = load_dover_fn(device="cuda", use_fp16=True)

for i in range(10):
    frames = load_sample(i)
    score_fp32 = dover_fn_fp32(frames)
    score_fp16 = dover_fn_fp16(frames)
    diff = abs(score_fp32 - score_fp16)
    print(f"Sample {i}: FP32={score_fp32:.6f}, FP16={score_fp16:.6f}, diff={diff:.6f}")
```

**解决方案**：如果 diff > 1%，禁用 FP16
```python
dover_fn = load_dover_fn(device="cuda", use_fp16=False)
```

---

### 问题 3：速度没提升

**可能原因**：
1. 意外使用了 CPU 模式
2. GPU 驱动问题
3. DOVER 模型加载失败

**排查步骤**：

1. 检查日志是否有 FP16 启用提示
```bash
grep "DOVER.*FP16" <log_file>
# 应该看到：[DOVER] GPU FP16 模式已启用（显存减半，支持到 1080p）
```

2. 检查 GPU 使用率
```bash
nvidia-smi -l 1
# GPU-Util 应该 > 90%
```

3. 检查代码中的 device 参数
```python
# 确保是 "cuda"，不是 "cpu"
dover_fn = load_dover_fn(device="cuda")
```

---

## ✅ 验收清单

部署成功的标志：

- [ ] 文件已传输到 CMCC 机器
- [ ] 语法检查通过（`python -m py_compile`）
- [ ] 日志显示 `[DOVER] GPU FP16 模式已启用`
- [ ] GPU 使用率 > 90%
- [ ] 单样本处理时间 < 15 秒（vs 旧方案 169 秒）
- [ ] 单 shard 处理时间 < 30 分钟（vs 旧方案 6.5 小时）
- [ ] 显存占用 10-20GB（正常范围）

---

## 📚 相关文档

- [DOVER 优化任务总结](./DOVER_优化_任务总结.md)
- [CMCC DOVER 优化部署指南](./CMCC_DOVER_优化部署指南.md)
- [DOVER 性能优化完整分析报告](./DOVER_性能优化_完整分析报告.md)

---

**变更日期**: 2026-08-09  
**变更人**: Claude (Opus 4.8)  
**状态**: ✅ 代码已更新，等待 CMCC 验证
