# DOVER 性能优化部署指南（CMCC 机器）

## 📋 概述

### 问题

当前 Stage 3 处理速度慢的**真正原因**：

- ❌ **错误诊断**：上一个方案认为是"模型移动开销"
- ✅ **真实原因**：采用了"永久 CPU 模式"，完全放弃 GPU 算力
- 📊 **数据支持**：CPU 推理比 GPU 慢 10-40x

### 解决方案

**方案 1：FP16 GPU（推荐）**

- 显存减半（FP32 → FP16）
- 速度几乎不变（H100 的 FP16 性能极强）
- 支持到 1080p 分辨率

**预期效果**：
- 单样本：169 秒 → 5-10 秒（**~20x 加速**）
- 139 样本：10 小时 → 12-23 分钟

---

## 🔧 部署步骤

### 步骤 1：理解当前代码的问题

查看 `src/sana_wm_pipeline/qc/stage3_gpu.py` 第 297-394 行的 `load_dover_fn` 函数。

**当前逻辑**（错误）：
```python
# 首次调用时检测分辨率
if current_resolution > RESOLUTION_THRESHOLD:
    _cpu_mode[0] = True
    model.cpu()  # ❌ 永久移到 CPU

# 后续所有 chunk 都用 CPU
if _cpu_mode[0]:
    target_device = "cpu"  # ❌ 太慢！
```

**问题**：
1. 一旦触发 CPU 模式，所有后续样本都用 CPU（包括低分辨率样本）
2. CPU 推理比 GPU 慢 10-40x
3. 浪费了 H100 80GB 的强大算力

---

### 步骤 2：备份当前代码

```bash
cd /path/to/sana_wm_pipeline
cp src/sana_wm_pipeline/qc/stage3_gpu.py src/sana_wm_pipeline/qc/stage3_gpu.py.backup_20260809
```

---

### 步骤 3：更新 `load_dover_fn` 函数

用以下代码替换 `stage3_gpu.py` 中的 `load_dover_fn` 函数（第 297-394 行）：

```python
def load_dover_fn(device: str = "cuda", dover_config_path: str = None, dover_weight_path: str = None, use_fp16: bool = True):
    """Load DOVER and return dover_fn(frames_rgb: (T,H,W,3) uint8) -> float.

    Args:
        device: torch device (e.g., 'cuda' or 'cpu')
        dover_config_path: path to dover.yml (default: auto-detect from DOVER package)
        dover_weight_path: path to DOVER.pth (default: auto-detect from DOVER package)
        use_fp16: use FP16 on GPU for 2x memory reduction (default: True)

    Note: 2026-08-09 FP16 优化
          - GPU 模式默认使用 FP16（显存减半，支持 1080p）
          - CPU 模式强制使用 FP32（精度优先）
          - 移除旧的 OOM 检测和模型移动逻辑（不再需要）
    """
    from dover import DOVER  # type: ignore
    import torch
    import yaml
    from pathlib import Path

    # Auto-detect DOVER paths if not provided
    if dover_config_path is None or dover_weight_path is None:
        try:
            import dover
            dover_pkg_dir = Path(dover.__file__).parent.parent
            if dover_config_path is None:
                dover_config_path = str(dover_pkg_dir / "dover.yml")
            if dover_weight_path is None:
                dover_weight_path = str(dover_pkg_dir / "pretrained_weights" / "DOVER.pth")
        except Exception:
            raise RuntimeError(
                "Could not auto-detect DOVER paths. Please provide dover_config_path and dover_weight_path explicitly."
            )

    # Load config and initialize model
    with open(dover_config_path, "r") as f:
        dover_opt = yaml.safe_load(f)

    # Initialize model on specified device
    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(dover_weight_path, map_location=device, weights_only=False))
    model = model.to(device)

    # ⭐ 新增：GPU 模式自动使用 FP16
    if device == "cuda" and use_fp16:
        model = model.half()
        print(f"[DOVER] GPU FP16 模式已启用（显存减半，速度不变）", flush=True)

    model.eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        import torch
        # DOVER expects a dict with 'technical' and 'aesthetic' views
        # frames_rgb: (T, H, W, 3) uint8

        # Convert to (1, 3, T, H, W) float normalized
        t = torch.from_numpy(frames_rgb).float() / 255.0  # (T, H, W, 3)
        t = t.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)

        # ⭐ 新增：GPU FP16 模式转换输入
        if device == "cuda" and use_fp16:
            t = t.half()

        t = t.to(device)

        views = {
            "technical": t,
            "aesthetic": t,
        }

        with torch.no_grad():
            results = model(views)

        # results is a list of [technical_score, aesthetic_score]
        # Return the mean of both
        return float(sum(r.mean().item() for r in results) / len(results))

    return dover_fn
```

**关键变更**：

1. ✅ **移除了旧的 OOM 检测逻辑**（第 341-372 行）
2. ✅ **添加了 FP16 支持**（`model.half()` + `t.half()`）
3. ✅ **模型常驻 GPU**（不再反复移动）
4. ✅ **默认启用 FP16**（`use_fp16=True`）

---

### 步骤 4：验证代码变更

```bash
# 检查语法错误
python -m py_compile src/sana_wm_pipeline/qc/stage3_gpu.py

# 如果没有输出，说明语法正确
```

---

### 步骤 5：测试 FP16 精度（可选但推荐）

创建测试脚本 `test_scripts/test_dover_fp16_accuracy.py`：

```python
#!/usr/bin/env python3
"""测试 FP16 vs FP32 的精度差异"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.qc.stage3_gpu import load_dover_fn

# 生成测试数据
frames = np.random.randint(0, 256, (80, 720, 1280, 3), dtype=np.uint8)

# 加载两个版本
dover_fn_fp32 = load_dover_fn(device="cuda", use_fp16=False)
dover_fn_fp16 = load_dover_fn(device="cuda", use_fp16=True)

# 对比
score_fp32 = dover_fn_fp32(frames)
score_fp16 = dover_fn_fp16(frames)
diff = abs(score_fp32 - score_fp16)
diff_pct = (diff / score_fp32) * 100

print(f"FP32 score: {score_fp32:.6f}")
print(f"FP16 score: {score_fp16:.6f}")
print(f"Absolute diff: {diff:.6f}")
print(f"Relative diff: {diff_pct:.2f}%")

if diff_pct < 1.0:
    print("✅ FP16 精度损失可接受（< 1%）")
else:
    print("⚠️ FP16 精度损失较大（> 1%），建议使用 FP32")
```

运行测试：

```bash
python test_scripts/test_dover_fp16_accuracy.py
```

**预期输出**：
```
[DOVER] GPU FP16 模式已启用（显存减半，速度不变）
[DOVER] GPU FP16 模式已启用（显存减半，速度不变）
FP32 score: 0.654321
FP16 score: 0.654198
Absolute diff: 0.000123
Relative diff: 0.02%
✅ FP16 精度损失可接受（< 1%）
```

---

### 步骤 6：运行单 shard 性能测试

```bash
# 测试单个 shard（假设是 shard-000003-000001.tar）
python scripts/run_stage3_cmcc.py \
    --input-jsonl stage1_results/group_X/stage1_shard-000003-000001.jsonl \
    --output-jsonl stage3_results/group_X/stage3_shard-000003-000001.jsonl \
    --caption-overrides-jsonl stage3_results/group_X/caption_overrides_shard-000003-000001.jsonl \
    --group-name group_X
```

**监控日志**：

```bash
# 应该看到这行
[DOVER] GPU FP16 模式已启用（显存减半，速度不变）

# 处理速度应该明显加快
[stage3] 100 samples processed  # 应该在几分钟内完成，而不是几小时
```

---

### 步骤 7：性能对比

| 指标 | 旧方案（CPU） | 新方案（FP16 GPU） | 加速比 |
|------|--------------|-------------------|--------|
| 单样本耗时 | 169 秒 | 5-10 秒 | **17-34x** |
| 单 shard (139 样本) | ~6.5 小时 | ~12-23 分钟 | **17-32x** |
| 全部 7 个 shard | ~45 小时 | ~1.4-2.7 小时 | **17-32x** |

---

### 步骤 8：全量部署

如果性能测试通过，更新所有处理脚本：

```bash
# 批量处理所有 shard
bash scripts/batch_run_stage3_all_shards.sh
```

---

## 🔍 故障排查

### 问题 1：OOM（显存溢出）

**症状**：
```
RuntimeError: CUDA out of memory
```

**原因**：视频分辨率过高（> 1080p）

**解决方案**：

方案 A：切换到 FP32 并调低 batch size（如果 DOVER 支持）

方案 B：对高分辨率视频降采样：

```python
# 在 _decode_frames 函数中添加降采样
def _decode_frames(mp4_bytes: bytes, max_resolution: int = 1080) -> np.ndarray | None:
    # ... 现有代码 ...
    for pkt in c.demux(video=0):
        for f in pkt.decode():
            frame = f.to_ndarray(format="rgb24")
            H, W = frame.shape[:2]
            if max(H, W) > max_resolution:
                scale = max_resolution / max(H, W)
                new_H, new_W = int(H * scale), int(W * scale)
                frame = cv2.resize(frame, (new_W, new_H))
            frames.append(frame)
```

方案 C：使用方案 2（智能分块）代替方案 1

---

### 问题 2：精度损失过大

**症状**：FP16 vs FP32 差异 > 1%

**解决方案**：

禁用 FP16，保持 FP32：

```python
dover_fn = load_dover_fn(device="cuda", use_fp16=False)  # 禁用 FP16
```

---

### 问题 3：速度没有提升

**症状**：处理速度仍然很慢（每样本 > 60 秒）

**排查步骤**：

1. 检查是否真的使用了 GPU：
```bash
nvidia-smi -l 1  # 实时监控 GPU 使用率，应该接近 100%
```

2. 检查日志中是否有 FP16 启用提示：
```
[DOVER] GPU FP16 模式已启用（显存减半，速度不变）
```

3. 检查是否意外使用了 CPU 模式：
```python
# 在 run_stage3_cmcc.py 中检查
dover_fn = load_dover_fn(device="cuda")  # 确保是 "cuda"，不是 "cpu"
```

---

## 📊 性能监控

### 实时监控 GPU 使用

```bash
# 终端 1：运行处理脚本
python scripts/run_stage3_cmcc.py ...

# 终端 2：监控 GPU
watch -n 1 nvidia-smi
```

**预期输出**：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.XX.XX    Driver Version: 535.XX.XX    CUDA Version: 12.X   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  H100 80GB HBM3     Off  | 00000000:XX:XX.X Off |                    0 |
| N/A   45C    P0    250W / 700W |  20000MiB / 81559MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

**关键指标**：
- **GPU-Util**: 应该接近 95-100%（说明 GPU 被充分利用）
- **Memory-Usage**: 每个 chunk ~10-20GB（FP16 模式）
- **Pwr:Usage**: 接近功率上限（说明满载运行）

---

## 📝 代码变更总结

### 变更文件

1. `src/sana_wm_pipeline/qc/stage3_gpu.py`

### 变更内容

| 函数 | 变更类型 | 详情 |
|------|---------|------|
| `load_dover_fn` | 重构 | 移除 OOM 检测，添加 FP16 支持 |
| `dover_fn` (闭包) | 简化 | 移除动态设备切换，固定使用 GPU |

### 代码行数变化

- 删除：~30 行（OOM 检测逻辑）
- 新增：~10 行（FP16 转换）
- 净变化：-20 行（更简洁）

---

## ✅ 验收标准

部署成功的标志：

1. ✅ 日志中看到 `[DOVER] GPU FP16 模式已启用`
2. ✅ `nvidia-smi` 显示 GPU 使用率 > 90%
3. ✅ 单样本处理时间 < 15 秒（相比之前 169 秒）
4. ✅ 单 shard (139 样本) 处理时间 < 30 分钟（相比之前 6.5 小时）
5. ✅ FP16 vs FP32 精度差异 < 1%（如果运行了精度测试）

---

## 🔗 相关文档

- [DOVER 性能优化方案对比](./DOVER_性能优化方案对比.md)
- [DOVER 优化方案理论分析](./DOVER_优化方案_理论分析.md)
- [Stage 3 性能分析完整诊断](./CMCC_Stage3_最终性能分析_完整诊断.md)

---

**生成时间**: 2026-08-09  
**作者**: Claude (Opus 4.8)  
**版本**: v1.0（FP16 GPU 优化）
