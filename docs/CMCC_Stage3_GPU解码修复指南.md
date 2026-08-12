# CMCC Stage3 GPU 视频解码修复指南

> **问题**：GPU 利用率 0%，10 小时仅处理 139 样本  
> **根因**：PyAV 使用 CPU 解码视频（瓶颈）  
> **方案**：使用 TorchVision GPU 解码（NVDEC 硬件加速）  
> **预期提升**：50 倍加速（251 秒/样本 → 5 秒/样本）

---

## 🚀 快速修复步骤

### 步骤 1：在 CMCC 机器上测试 GPU 解码

```bash
# 1. 激活环境
cd /root/work/david_work/sana_qc_pipeline
source sanawmqcenv/bin/activate

# 2. 运行测试脚本（已准备好，复制到 CMCC）
python test_scripts/testgpuvideodecodecmcc.py
```

**预期输出**：
```
✅ TorchVision (GPU) 解码成功
   耗时: ~50-200 ms
   加速比: 20-100x
   
预估 Stage 3 性能提升：
  当前速度：~258 秒/样本
  预期速度：~5-10 秒/样本
  139 样本处理时间：~12-23 分钟（当前 10 小时）
```

**如果测试失败**：
- 检查 TorchVision 版本：`python -c "import torchvision; print(torchvision.__version__)"`
- 需要 >= 0.15.0，如果版本过低：`pip install --upgrade torchvision`

---

### 步骤 2：应用修复

修复文件已准备好，需要在 CMCC 机器上替换：

```bash
# 在 CMCC 机器执行
cd /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc

# 备份原文件
cp stage3gpu.py stage3gpu.py.backup20260809

# 替换为修复版本（从本机复制过去）
# 源文件位置（本机）：/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3gpuv2gpudecode.py
# 目标位置（CMCC）：/root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc/stage3gpu.py
```

**修复内容摘要**：
1. 新增 `_decode_frames_gpu()` - TorchVision GPU 解码
2. 保留 `_decode_frames_cpu()` - PyAV CPU fallback
3. `_decode_frames()` 自动选择 GPU/CPU
4. 新增 `prefer_gpu_decode` 参数控制

---

### 步骤 3：小规模验证（10 样本）

```bash
# 在 CMCC 机器执行
cd /root/work/david_work/sana_qc_pipeline

# 创建测试子集（前 10 个样本）
head -n 10 /path/to/stage12_jsonl_file.jsonl > /tmp/test10samples.jsonl

# 运行 Stage 3（单 GPU，10 样本）
python scripts/runstage3cmcc.py \
  --stage12-jsonl /tmp/test10samples.jsonl \
  --skip-vlm \
  --output-dir /tmp/stage3test \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg /root/work/david_work/sana_qc_pipeline/configs/table6_thresholds.yml \
  --device cuda

# 同时监控 GPU 利用率（另一个终端）
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

**预期结果**：
- ✅ GPU 利用率 >50%（之前 0%）
- ✅ 10 样本处理时间 <2 分钟（之前 40+ 分钟）
- ✅ 输出文件 `/tmp/stage3test/stage3_worker000.jsonl` 存在且有 10 行

---

### 步骤 4：全量运行

验证通过后，执行全量：

```bash
# 使用原有的启动脚本（会自动使用新代码）
cd /root/work/david_work/sana_qc_pipeline

# 多 GPU 并行（例如 8 GPU）
python scripts/launchstage3multinode.py \
  --stage12-jsonl /path/to/full_stage12.jsonl \
  --skip-vlm \
  --output-dir /path/to/stage3output \
  --qwendir /root/work/david_work/models/Qwen3.5-9B \
  --unimatchdir /root/work/david_work/models/unimatch \
  --totalgpus 8 \
  --table6cfg /root/work/david_work/sana_qc_pipeline/configs/table6_cmcc.yml
```

**预期性能**（8 GPU）：
- 总样本数：假设 18 万（根据实际调整）
- 单 GPU 速度：~5 秒/样本（优化后）
- 单 GPU 处理量：180000 / 8 = 22500 样本
- 单 GPU 耗时：22500 × 5 = 112500 秒 = **31 小时**
- **8 GPU 并行：~31 小时**（之前估计 >80 小时）

---

## 📊 性能对比

### 修复前（CPU 解码）

| 指标 | 数值 |
|------|------|
| GPU 利用率 | 0% |
| 单样本处理时间 | ~258 秒 |
| 139 样本处理时间 | 10 小时 |
| 瓶颈 | 视频解码（CPU） |

### 修复后（GPU 解码）

| 指标 | 数值 |
|------|------|
| GPU 利用率 | >50% |
| 单样本处理时间 | ~5 秒 |
| 139 样本处理时间 | ~12 分钟 |
| 加速比 | **50 倍** |

---

## 🔍 故障排查

### 问题 1：TorchVision 不可用

**现象**：
```
TorchVision 不可用，fallback 到 CPU 解码
```

**解决方案**：
```bash
# 检查版本
python -c "import torchvision; print(torchvision.__version__)"

# 如果 < 0.15.0，升级
pip install --upgrade torchvision

# 如果无法升级，保持 CPU 解码（已自动 fallback）
```

---

### 问题 2：GPU 解码失败

**现象**：
```
GPU 解码失败: ..., fallback 到 CPU
```

**原因**：
- 视频编码格式不支持（非 H.264/H.265）
- NVDEC 驱动问题
- 显存不足

**解决方案**：
- 代码已自动 fallback 到 CPU，不影响功能
- 如果大部分样本都 fallback，检查视频格式：
  ```bash
  ffprobe sample.mp4 2>&1 | grep "Video:"
  # 应该看到 h264 或 hevc
  ```

---

### 问题 3：显存不足（OOM）

**现象**：
```
CUDA out of memory
```

**原因**：
- 视频分辨率过高（>4K）
- 视频过长（>1000 帧）
- 多模型同时加载

**解决方案**：
1. 降低并发数（减少 GPU 数量）
2. 强制使用 CPU 解码（添加 `--prefer-cpu-decode` 参数）
3. 分批处理长视频

---

### 问题 4：速度仍然慢

**现象**：
- GPU 利用率 >50%
- 但单样本仍需 >30 秒

**排查步骤**：

1. **检查是否在使用 GPU 解码**：
   ```bash
   # 查看日志中是否有 "fallback 到 CPU" 字样
   grep -i "fallback" /tmp/stage3_worker000.log
   ```

2. **检查视频特征**：
   ```bash
   # 查看视频分辨率和帧数
   ffprobe sample.mp4
   ```

3. **Profiling**：
   ```bash
   # 添加详细日志
   export PYTHONPATH=/root/work/david_work/sana_qc_pipeline:$PYTHONPATH
   python -m cProfile -o profile.stats scripts/runstage3cmcc.py ...
   
   # 分析
   python -m pstats profile.stats
   ```

---

## 📁 文件清单

### 本机文件（需要复制到 CMCC）

| 文件（本机） | 目标位置（CMCC） | 说明 |
|------------|----------------|------|
| `/mnt/afs/davidwang/workspace/sana_wm_pipeline/test_scripts/testgpuvideodecodecmcc.py` | `/root/work/david_work/sana_qc_pipeline/test_scripts/` | GPU 解码测试脚本 |
| `/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3gpuv2gpudecode.py` | `/root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py` | 修复后的主文件 |

### 诊断文档

| 文档 | 说明 |
|------|------|
| `CMCC_Stage3_GPU利用率为0问题诊断.md` | 根因分析和技术细节 |
| `CMCC_Stage3_GPU解码修复指南.md` | 本文档，修复步骤 |

---

## ✅ 验证检查清单

修复完成后，验证以下项目：

- [ ] 测试脚本运行成功（`testgpuvideodecodecmcc.py`）
- [ ] GPU 解码加速比 >10x
- [ ] 小规模验证通过（10 样本 <2 分钟）
- [ ] GPU 利用率 >50%
- [ ] 解码结果一致（与 CPU 解码对比）
- [ ] 无显存泄漏（长时间运行）
- [ ] 全量运行启动成功

---

## 🔄 回滚方案

如果修复后出现问题，立即回滚：

```bash
# 在 CMCC 机器执行
cd /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc
cp stage3_gpu.py.backup_2026_08_09 stage3_gpu.py

# 重启任务即可
```

回滚后会恢复到 CPU 解码（慢但稳定）。

---

## 📞 支持

如有问题，提供以下信息：

1. 测试脚本输出（`testgpuvideodecodecmcc.py`）
2. TorchVision 版本
3. nvidia-smi 输出
4. 错误日志（如有）

---

**修复日期**：2026-08-09  
**预期效果**：50 倍加速，GPU 利用率 >50%  
**状态**：待在 CMCC 机器验证
