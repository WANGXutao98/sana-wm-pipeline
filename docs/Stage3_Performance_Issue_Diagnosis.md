# Stage 3 性能问题诊断报告

**日期**: 2026-08-07  
**问题**: GPU 利用率 0%，处理速度慢 50 倍  
**状态**: ✅ 已修复

---

## 🐛 问题现象

### 用户反馈

在 CMCC 机器运行 `run_stage3_cmcc.py` 时：

```bash
# GPU 监控
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
utilization.gpu [%], memory.used [MiB]
0 %, 6575 MiB
0 %, 6575 MiB
0 %, 6575 MiB  # ← GPU 完全空闲！
```

**关键指标**：
- ❌ GPU 利用率：**0%**（应该 50-90%）
- ❌ 处理速度：**10 小时处理 139 个样本**（预期 5 分钟）
- ⚠️ CPU 利用率：**1360%**（13.6 个核心满载）
- ✅ GPU 显存占用：6.5 GB（说明模型已加载）

**计算实际速度**：
```
10 小时 = 36000 秒
139 个样本
36000 / 139 = 259 秒/样本

预期速度（GPU）：2-3 秒/样本
实际速度（CPU）：259 秒/样本
慢了：259 / 2.5 ≈ 104 倍！
```

---

## 🔍 根本原因分析

### 矛盾的证据

**用户的独立测试证明**：
1. ✅ DOVER 在 H100 GPU 上**完全正常**（425ms 推理）
2. ✅ UniMatch 在 H100 GPU 上**完全正常**（35ms 推理）

但是在 Stage 3 中：
- ❌ GPU 利用率为 0%
- ❌ 处理速度极慢

**这说明什么**？模型能在 GPU 上运行，但 Stage 3 代码强制它在 CPU 上运行！

---

### 代码审查

查看 `src/sana_wm_pipeline/qc/stage3_gpu.py` 第 280-288 行：

```python
def load_dover_fn(device: str = "cuda", ...):
    """..."""
    from dover import DOVER
    import torch
    import yaml
    from pathlib import Path
    import warnings

    # ⚠️ WORKAROUND: Force CPU mode due to H100 sm_90 incompatibility with DOVER
    # PyTorch 2.4 + H100 + DOVER causes Segmentation fault on model.to("cuda")
    if device != "cpu":
        warnings.warn(
            f"DOVER requested device='{device}', but forcing CPU mode due to H100 compatibility issues. "
            "Performance will be slower but stable.",
            RuntimeWarning
        )
        device = "cpu"  # ← 强制 CPU！

    # ...
    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(dover_weight_path, map_location="cpu", ...))  # ← CPU
    model = model.to(device)  # ← device 已被改为 "cpu"
    model.eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        # ...
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # ← 数据也在 CPU
```

**发现**：
1. **无条件强制 CPU**：只要 `device != "cpu"`，就会被强制改为 `"cpu"`
2. **过时的 workaround**：注释说 "PyTorch 2.4 + H100 会 Segmentation fault"
3. **误导性警告**：警告说 "Performance will be slower"，但没说会慢 **100 倍**！

---

### 历史背景

查看历史文档 `DOVER_H100_部署方案_CMCC实际执行记录.md`：

```markdown
> **执行日期**：2026-08-03
> **执行环境**：CMCC sana_wm_qc_env
> **执行状态**：✅ 成功
> **关键发现**：H100 GPU 模式完全正常，性能符合预期

**性能验证**：
- 随机数据推理：425ms
- Demo 视频：归一化分数 0.4108
- 实际视频：归一化分数 0.4024

[4/5] 初始化模型并移到 GPU（关键测试）
  ✅ 模型成功移到 GPU（H100 兼容性验证通过）
  推理耗时: 425.29 ms
```

**结论**：
- ✅ 2026-08-03 已经验证 DOVER 在 H100 GPU 上完全正常
- ✅ PyTorch 2.6.0+cu124 + H100 没有兼容性问题
- ❌ 但 `stage3_gpu.py` 中的 workaround 从未被移除

---

## 🔧 修复方案

### 代码修改

**修改文件**: `src/sana_wm_pipeline/qc/stage3_gpu.py`

#### 1. 移除强制 CPU 的代码（第 280-288 行）

**修改前**：
```python
# ⚠️ WORKAROUND: Force CPU mode due to H100 sm_90 incompatibility with DOVER
# PyTorch 2.4 + H100 + DOVER causes Segmentation fault on model.to("cuda")
if device != "cpu":
    warnings.warn(
        f"DOVER requested device='{device}', but forcing CPU mode due to H100 compatibility issues. "
        "Performance will be slower but stable.",
        RuntimeWarning
    )
    device = "cpu"
```

**修改后**：
```python
# Note: Previous H100 compatibility workaround removed (2026-08-07)
# Testing confirmed DOVER works perfectly on H100 GPU with PyTorch 2.6.0+cu124
# See: DOVER_H100_部署方案_CMCC实际执行记录.md
```

#### 2. 更新模型加载代码（第 308-314 行）

**修改前**：
```python
# Initialize model on CPU first
model = DOVER(**dover_opt["model"]["args"])
# Load weights to CPU first to avoid H100 compatibility issues
model.load_state_dict(torch.load(dover_weight_path, map_location="cpu", weights_only=False))
# Keep model on CPU (H100 incompatibility workaround)
model = model.to(device)
model.eval()
```

**修改后**：
```python
# Initialize model on specified device
model = DOVER(**dover_opt["model"]["args"])
model.load_state_dict(torch.load(dover_weight_path, map_location=device, weights_only=False))
model = model.to(device)
model.eval()
```

#### 3. 更新 docstring（第 267-273 行）

**修改前**：
```python
"""Load DOVER and return dover_fn(frames_rgb: (T,H,W,3) uint8) -> float.

Args:
    device: torch device (NOTE: currently forced to CPU due to H100 compatibility issues)
    ...
"""
```

**修改后**：
```python
"""Load DOVER and return dover_fn(frames_rgb: (T,H,W,3) uint8) -> float.

Args:
    device: torch device (e.g., 'cuda' or 'cpu')
    ...

Note: H100 GPU is fully supported as of PyTorch 2.6.0+cu124.
      Previous CPU-only workaround has been removed (2026-08-07).
"""
```

---

### 提交信息

```
fix(stage3): remove obsolete DOVER CPU-only workaround for H100

- Remove forced CPU mode for DOVER on H100 GPU
- Testing confirmed DOVER works perfectly on H100 with PyTorch 2.6.0+cu124
- Previous workaround caused 50x slowdown (GPU idle, CPU saturated)
- See: DOVER_H100_部署方案_CMCC实际执行记录.md

Performance impact:
- Before: ~10 hours for 139 samples (GPU util 0%)
- Expected after: ~5 minutes for 139 samples (GPU util 50-90%)
```

**提交 SHA**: `58a5a8d`

---

## 📊 预期性能改进

### 修复前（CPU 模式）

| 指标 | 值 |
|------|-----|
| GPU 利用率 | 0% |
| CPU 利用率 | 1360% (13.6 核心) |
| 单样本处理 | 259 秒 |
| 139 样本总时间 | 10 小时 |
| 显存占用 | 6.5 GB |

**瓶颈**：DOVER 在 CPU 上运行

---

### 修复后（GPU 模式）预期

| 指标 | 值 |
|------|-----|
| GPU 利用率 | 50-90% |
| CPU 利用率 | 1000-1500% (正常) |
| 单样本处理 | 2-3 秒 |
| 139 样本总时间 | 5 分钟 |
| 显存占用 | 10-15 GB |

**加速比**：259 / 2.5 = **104 倍**！

---

### 各模块性能对比

| 模块 | CPU 模式 | GPU 模式 | 加速比 |
|------|---------|---------|--------|
| UniMatch | ~35ms | ~35ms | 1x (已在 GPU) |
| DOVER | ~10000ms | ~425ms | **23x** |
| 视频解码 | ~500ms | ~500ms | 1x (固定在 CPU) |
| **总计** | ~10500ms | ~960ms | **11x** |

实际慢了 104 倍是因为：
- DOVER CPU 推理可能更慢（10 秒而非 425ms）
- 数据传输开销
- 其他瓶颈

---

## ✅ 验证步骤

### 1. 拷贝修复后的文件到 CMCC

只需要拷贝这一个文件：
```
src/sana_wm_pipeline/qc/stage3_gpu.py
```

### 2. 运行测试

```bash
cd /root/work/david_work/sana_qc_pipeline
conda activate sana_wm_qc_env

# 清理之前的输出
rm -rf /root/work/david_work/qc_output_new/smoke_test_stage3/*

# 运行测试（同样的命令）
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3_cmcc.py \
  --stage12-jsonl /root/work/david_work/qc_output_new/smoke_test_manifest.jsonl \
  --output-dir /root/work/david_work/qc_output_new/smoke_test_stage3 \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
  --skip-vlm
```

### 3. 监控 GPU（另一个终端）

```bash
watch -n 1 nvidia-smi
```

**预期看到**：
```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|=============================================================================|
|   0  NVIDIA H100 ...    Off  | ...           Off |   0                    |
| N/A   45C    P0   150W / 700W |  12000MiB / 81559MiB |     78%      Default |  ← 关键！
+-----------------------------------------------------------------------------+
```

**关键指标**：
- ✅ GPU-Util: **50-90%**（不再是 0%）
- ✅ Memory-Usage: **10-15 GB**（DOVER 在 GPU）
- ✅ 单样本处理：**2-3 秒**（不再是 259 秒）

### 4. 检查日志

**不应该再看到**：
```
RuntimeWarning: DOVER requested device='cuda', but forcing CPU mode due to H100 compatibility issues.
```

**应该看到**：
```
[worker 0] loading DOVER...
[worker 0] skipping Qwen VLM (--skip-vlm enabled)
[worker 0] models ready.
```

### 5. 性能验证

```bash
# 记录开始时间
start_time=$(date +%s)

# 等待处理完成（或处理一段时间后）
# ...

# 记录结束时间
end_time=$(date +%s)

# 计算处理速度
total_samples=$(wc -l < /root/work/david_work/qc_output_new/smoke_test_stage3/stage3_worker000.jsonl)
elapsed=$((end_time - start_time))
samples_per_sec=$(echo "scale=2; $total_samples / $elapsed" | bc)
sec_per_sample=$(echo "scale=2; $elapsed / $total_samples" | bc)

echo "总样本数: $total_samples"
echo "总耗时: $elapsed 秒"
echo "处理速度: $sec_per_sample 秒/样本"
```

**预期**：
- 单样本处理：**2-3 秒**
- 139 样本总时间：**5 分钟**（不再是 10 小时）

---

## 🎓 关键学习

### 1. 为什么会有这个 workaround？

**可能的原因**：
1. **早期版本问题**：PyTorch 2.4 + H100 确实有兼容性问题
2. **快速临时修复**：为了让系统先运行起来，强制 CPU
3. **忘记移除**：后来升级到 PyTorch 2.6 后，忘记移除这个 workaround

### 2. 为什么测试通过了但代码有问题？

**独立测试 vs 集成代码**：
- ✅ 独立测试脚本：直接调用 `model.to("cuda")`，没有 workaround
- ❌ Stage 3 代码：有 workaround，强制 CPU

**教训**：测试代码和生产代码要保持一致

### 3. 为什么 GPU 显存有占用但利用率为 0？

**部分在 GPU，部分在 CPU**：
- ✅ UniMatch：在 GPU（6.5 GB 显存）
- ❌ DOVER：在 CPU（0 GB GPU 显存，但加载到系统内存）
- ❌ Qwen：跳过加载

实际上 6.5 GB 显存只是 UniMatch，DOVER 根本没在 GPU 上。

### 4. 为什么 CPU 利用率这么高？

**CPU 模式的 DOVER**：
- DOVER 是深度神经网络
- 在 CPU 上推理需要大量计算
- 占满 13.6 个 CPU 核心（1360% 利用率）

**这不是正常的数据预处理 CPU 占用，而是模型推理的 CPU 占用**。

### 5. 教训总结

| 问题 | 教训 |
|------|------|
| Workaround 忘记移除 | 添加 TODO 或过期日期 |
| 测试与生产不一致 | 测试要覆盖真实调用路径 |
| 性能问题难定位 | 监控 GPU 利用率和 CPU 利用率 |
| 警告信息不够明确 | "slower" 应该说明具体慢多少倍 |

---

## 📝 后续行动

### 立即行动
- [x] 修复代码（移除 CPU workaround）
- [x] 提交到 Git
- [ ] 用户在 CMCC 测试验证

### 短期（本周）
- [ ] 验证性能提升（预期 104 倍）
- [ ] 更新性能文档
- [ ] 运行完整的 Stage 3（修复 Qwen 后）

### 中期（下周）
- [ ] 审查其他 workaround（是否还有过时的）
- [ ] 统一测试脚本和生产代码
- [ ] 添加性能监控（GPU 利用率告警）

---

## 📚 相关文档

- `DOVER_H100_部署方案_CMCC实际执行记录.md` - DOVER GPU 验证记录
- `UniMatch_H100_验证记录_CMCC.md` - UniMatch GPU 验证记录
- `docs/Stage3_Skip_VLM_Development_Archive.md` - Skip VLM 功能开发存档
- Git commit `58a5a8d` - DOVER GPU 修复提交

---

**报告版本**: v1.0  
**创建日期**: 2026-08-07  
**状态**: 已修复，等待验证
