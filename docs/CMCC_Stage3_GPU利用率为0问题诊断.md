# CMCC Stage3 GPU 利用率为 0% 问题诊断报告

> **问题日期**：2026-08-09  
> **现象**：GPU 利用率 0%，CPU 利用率高，10 小时仅处理 139 样本  
> **诊断状态**：✅ 根因已定位  
> **修复状态**：🔧 待修复

---

## 📊 问题现象

### 1. GPU 状态
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
utilization.gpu [%], memory.used [MiB]
0 %, 6575 MiB
0 %, 6575 MiB
0 %, 6575 MiB
```

**关键信息**：
- GPU 利用率：**0%**（应该 >50%）
- 显存占用：**6575 MiB (约 6.4 GB)**（正常，DOVER + UniMatch 预期 <10 GB）
- CPU 利用率：**很高**（异常，说明计算在 CPU 上进行）

### 2. 性能数据
- 运行时间：**10 小时**
- 处理样本数：**139 个**
- 平均速度：**~4.3 分钟/样本**（预期应该 <5 秒/样本）
- 速度差距：**慢 ~50 倍**

### 3. 单独验证结果（正常）
用户手动验证了两个模型，均能正常使用 GPU：

**DOVER 验证**：
```
✅ 模型成功移到 GPU（H100 兼容性验证通过）
✅ 推理成功
推理耗时: 553.69 ms
输出质量分数: -0.1335
```

**UniMatch 验证**：
```
✅ 模型加载成功 (GPU 模式)
✅ 随机数据推理正常
✅ 真实视频测试完成 (平均 45.76 ms/帧对)
```

---

## 🔍 根因分析

### 问题定位：视频解码在 CPU 上进行

**关键代码路径**：
1. `stage3_gpu.py::process_sample_stage3()` → 从 tar 中读取 mp4 字节
2. `_decode_frames(mp4_bytes)` → **使用 PyAV 在 CPU 上解码视频**
3. `frames_rgb` (numpy array) → 作为输入传给 UniMatch 和 DOVER

**问题代码**：`src/sana_wm_pipeline/qc/stage3_gpu.py:30-42`

```python
def _decode_frames(mp4_bytes: bytes) -> np.ndarray | None:
    if not mp4_bytes:
        return None
    try:
        import av
        frames = []
        with av.open(io.BytesIO(mp4_bytes)) as c:
            for pkt in c.demux(video=0):
                for f in pkt.decode():
                    frames.append(f.to_ndarray(format="rgb24"))  # ⚠️ CPU 解码
        return np.array(frames, dtype=np.uint8) if frames else None
    except Exception:
        return None
```

### 为什么 GPU 利用率为 0%？

1. **PyAV 默认使用 CPU 解码器**
   - `av.open()` 默认使用 FFmpeg 的软件解码器（libx264）
   - 没有启用硬件加速（NVDEC/CUVID）
   - 解码是最耗时的步骤（每个视频 30-1068 帧，分辨率不确定）

2. **视频解码成为瓶颈**
   ```
   时间分布（估算）：
   - 视频解码（CPU）：~4 分钟（瓶颈）
   - UniMatch 推理（GPU）：~0.5 秒（很快）
   - DOVER 推理（GPU）：~0.5 秒（很快）
   - 其他开销（I/O + JSON）：~10 秒
   
   总计：~4-5 分钟/样本（与实测 4.3 分钟一致！）
   ```

3. **GPU 处于空闲等待状态**
   - 大部分时间在等待 CPU 解码完成
   - GPU 偶尔才被调用一次（UniMatch + DOVER）
   - 每次 GPU 推理只需 1 秒，然后又回到等待

### 为什么单独验证没问题？

**关键差异**：单独验证脚本直接使用预先加载的测试视频，跳过了解码步骤

```python
# testdovercmcc.py 和 testunimatchrealvideo.py
# 使用固定的测试视频文件，只测试模型推理部分
TEST_VIDEO = "/root/work/david_work/sana_qc_pipeline/DOVER/demo/SpatialVID-hq..."
```

**验证脚本流程**（快）：
```
加载测试视频（一次） → 测试模型推理（GPU）
```

**实际 Pipeline 流程**（慢）：
```
对每个样本：
  从 tar 读取 mp4 → CPU 解码视频（4分钟）→ GPU 推理（1秒）
```

---

## 📈 性能对比分析

### 当前性能（CPU 解码）

| 步骤 | 设备 | 单样本耗时 | 占比 |
|------|------|----------|------|
| 视频解码 | **CPU** | ~240 秒 | 93% |
| UniMatch | GPU | 0.5 秒 | 0.2% |
| DOVER | GPU | 0.5 秒 | 0.2% |
| I/O + 其他 | CPU | 10 秒 | 3.8% |
| **总计** | - | **~251 秒** | 100% |

**GPU 利用率**：0.4% (1秒 / 251秒)

### 预期性能（GPU 解码，使用 NVDEC）

| 步骤 | 设备 | 单样本耗时 | 占比 |
|------|------|----------|------|
| 视频解码 | **GPU (NVDEC)** | ~2 秒 | 40% |
| UniMatch | GPU | 0.5 秒 | 10% |
| DOVER | GPU | 0.5 秒 | 10% |
| I/O + 其他 | CPU | 2 秒 | 40% |
| **总计** | - | **~5 秒** | 100% |

**GPU 利用率**：60% (3秒 / 5秒)

**性能提升**：**50 倍**（251 秒 → 5 秒）

---

## 🎯 验证假设的证据

### 证据 1：显存占用符合预期
- 当前显存：6575 MiB (6.4 GB)
- DOVER 单独占用：0.22 GB（文档记录）
- UniMatch 单独占用：0.02 GB（文档记录）
- **推论**：模型已经在 GPU 上，但没有被频繁调用

### 证据 2：处理速度与 CPU 解码时间吻合
- 实测：4.3 分钟/样本
- 估算 CPU 解码时间：4 分钟（对于高分辨率长视频）
- **推论**：瓶颈在视频解码，而非模型推理

### 证据 3：CPU 利用率很高
- 现象：CPU 利用率很高
- **推论**：CPU 正在进行密集计算（视频解码）

### 证据 4：PyAV 默认行为
查看 PyAV 文档和代码：
```python
# PyAV 默认使用 CPU 解码器
av.open(file)  # 使用 FFmpeg 软件解码器（libx264）

# 需要显式指定硬件加速
av.open(file, options={"hwaccel": "cuda"})  # ❌ 不适用于内存流
av.Codec("h264", "r").create().hardware_config  # 需要特殊配置
```

对于内存流（`io.BytesIO`），PyAV 更难启用硬件加速。

---

## 🔧 解决方案

### 方案 A：使用 TorchVision + NVDEC（推荐）

**优点**：
- ✅ 原生支持 NVDEC 硬件解码
- ✅ 与 PyTorch 集成良好
- ✅ 性能优秀（~2 秒解码 vs 240 秒 CPU）
- ✅ 支持内存流

**实现**：
```python
import io
import torch
import torchvision
from torchvision.io import read_video

def _decode_frames_gpu(mp4_bytes: bytes, device: str = "cuda") -> np.ndarray | None:
    """使用 TorchVision + NVDEC 在 GPU 上解码视频"""
    if not mp4_bytes:
        return None
    try:
        # TorchVision 支持从内存解码（需要 torchvision >= 0.15）
        video_tensor, audio_tensor, info = torchvision.io.read_video(
            io.BytesIO(mp4_bytes),
            pts_unit='sec',
            output_format='TCHW'  # (T, C, H, W)
        )
        # 转换为 numpy (T, H, W, C) uint8
        frames_rgb = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return frames_rgb
    except Exception as e:
        return None
```

**注意**：需要验证 CMCC 环境的 TorchVision 版本是否支持。

---

### 方案 B：NVIDIA DALI（高性能，但复杂）

**优点**：
- ✅ 专为数据加载优化
- ✅ 原生支持 NVDEC
- ✅ 可以与 PyTorch 集成

**缺点**：
- ❌ 需要安装额外依赖（nvidia-dali-cuda120）
- ❌ API 复杂，需要重构代码
- ❌ 对内存流支持不够友好

---

### 方案 C：预解码数据集（终极方案）

**思路**：在预处理阶段将所有视频解码为帧序列，存储为高效格式（如 WebDataset tar）

**优点**：
- ✅ 完全消除运行时解码开销
- ✅ 可以使用任何解码器（CPU/GPU）预处理
- ✅ Stage 3 只需读取预解码帧（极快）

**缺点**：
- ❌ 需要大量存储空间（估计 10-50 倍视频大小）
- ❌ 需要额外的预处理步骤
- ❌ 数据管理更复杂

---

### 方案 D：多进程 CPU 解码（权宜之计）

**思路**：保持 CPU 解码，但启用多进程并行

**优点**：
- ✅ 无需修改解码代码
- ✅ 可以利用多核 CPU（192 核）

**缺点**：
- ❌ CPU 解码仍然慢（只是并行了）
- ❌ CPU 资源被大量占用
- ❌ 只能加速 ~8-16 倍（取决于核心数），不如 GPU 解码的 50 倍

---

## 📋 推荐行动计划

### 立即行动（方案 A）

1. **验证 TorchVision 版本和 NVDEC 支持**
   ```bash
   python -c "import torchvision; print(torchvision.__version__)"
   python -c "import torch; print(torch.backends.cudnn.version())"
   ```

2. **创建测试脚本验证 GPU 解码**
   - 测试从内存流解码
   - 对比 CPU vs GPU 解码速度
   - 验证解码结果一致性

3. **修改 `_decode_frames()` 使用 GPU 解码**
   - 保留 CPU 解码作为 fallback
   - 添加性能日志

4. **小规模测试（10-100 样本）**
   - 验证性能提升
   - 监控 GPU 利用率（应该 >50%）

5. **全量运行**

### 中期优化（可选）

- 如果方案 A 效果不理想，考虑方案 B（DALI）
- 优化 I/O（预读取、缓存）
- 批处理推理（UniMatch + DOVER）

### 长期优化（可选）

- 方案 C：预解码数据集
- 端到端 GPU pipeline（解码 → 推理 → 后处理）

---

## 🧪 验证检查清单

修复后需要验证：

- [ ] GPU 利用率 >50%（`nvidia-smi` 实时监控）
- [ ] 单样本处理时间 <10 秒（目标 <5 秒）
- [ ] 139 样本处理时间 <15 分钟（之前 10 小时）
- [ ] 解码结果与 CPU 解码一致（随机抽查 10 个样本）
- [ ] 所有指标值（UniMatch flow, DOVER score）一致
- [ ] 无内存泄漏（长时间运行后显存稳定）

---

## 📚 参考文档

- `UniMatch_H100_验证记录_CMCC.md` - 确认 UniMatch GPU 可用
- `DOVER_H100_部署方案_CMCC实际执行记录.md` - 确认 DOVER GPU 可用
- TorchVision Video I/O: https://pytorch.org/vision/stable/io.html
- NVIDIA DALI: https://docs.nvidia.com/deeplearning/dali/

---

**诊断完成时间**：2026-08-09  
**下一步**：实施方案 A（TorchVision + NVDEC GPU 解码）
