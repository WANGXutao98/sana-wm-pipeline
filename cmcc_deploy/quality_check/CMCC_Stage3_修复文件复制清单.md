# CMCC 文件复制清单 - Stage3 GPU 解码修复

> **任务**：修复 GPU 利用率为 0% 的问题（视频解码瓶颈）  
> **预期效果**：50 倍加速（10 小时 → 12 分钟处理 139 样本）

---

## 📋 需要复制的文件

### 1. 测试脚本

**源文件（本机）**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/test_scripts/test_gpu_video_decode_cmcc.py
```

**目标位置（CMCC）**：
```
/root/work/david_work/sana_qc_pipeline/test_scripts/testgpuvideodecodecmcc.py
```

**用途**：验证 GPU 解码性能

---

### 2. 修复后的主代码

**源文件（本机）**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu_v2_gpu_decode.py
```

**目标位置（CMCC）**：
```
/root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py
```

**⚠️ 重要**：先备份原文件！
```bash
cp /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py \
   /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py.backup_2026_08_09
```

---

### 3. 文档（可选，供参考）

**诊断报告（本机）**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/CMCC_Stage3_GPU利用率为0问题诊断.md
```

**修复指南（本机）**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/CMCC_Stage3_GPU解码修复指南.md
```

**目标位置（CMCC）**：
```
/root/work/david_work/sana_qc_pipeline/docs/
```

---

## 🚀 执行步骤（在 CMCC 机器）

### 步骤 1：复制文件

```bash
# 假设你已经通过 scp/rsync 将文件传输到 CMCC 机器的临时目录
# 例如：/tmp/stage3_fix/

# 1. 复制测试脚本
mkdir -p /root/work/david_work/sana_qc_pipeline/test_scripts
cp /tmp/stage3_fix/test_gpu_video_decode_cmcc.py \
   /root/work/david_work/sana_qc_pipeline/test_scripts/testgpuvideodecodecmcc.py
chmod +x /root/work/david_work/sana_qc_pipeline/test_scripts/testgpuvideodecodecmcc.py

# 2. 备份并替换主代码
cd /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc
cp stage3_gpu.py stage3_gpu.py.backup_2026_08_09
cp /tmp/stage3_fix/stage3_gpu_v2_gpu_decode.py stage3_gpu.py

# 3. 复制文档（可选）
mkdir -p /root/work/david_work/sana_qc_pipeline/docs
cp /tmp/stage3_fix/*.md /root/work/david_work/sana_qc_pipeline/docs/
```

---

### 步骤 2：运行测试

```bash
# 激活环境
cd /root/work/david_work/sana_qc_pipeline
source sanawmqcenv/bin/activate

# 运行 GPU 解码测试
python test_scripts/testgpuvideodecodecmcc.py
```

**预期输出关键信息**：
```
✅ TorchVision (GPU) 解码成功
   耗时: ~50-200 ms
   加速比: 20-100x

推荐方案：
  ✅ TorchVision (GPU) - 速度快，与 PyTorch 集成好

预估 Stage 3 性能提升：
  当前速度：~258 秒/样本
  预期速度：~5 秒/样本
  加速比：50.0x
  139 样本处理时间：11.6 分钟（当前 10 小时）
```

---

### 步骤 3：小规模验证（10 样本）

```bash
# 创建 10 样本测试集
head -n 10 /path/to/your/stage12.jsonl > /tmp/test10.jsonl

# 运行 Stage 3
python scripts/runstage3cmcc.py \
  --stage12-jsonl /tmp/test10.jsonl \
  --skip-vlm \
  --output-dir /tmp/stage3test \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg /root/work/david_work/sana_qc_pipeline/configs/table6_cmcc.yml \
  --device cuda

# 同时在另一个终端监控 GPU
watch -n 1 nvidia-smi
```

**验证指标**：
- ✅ GPU 利用率 >50%（之前 0%）
- ✅ 10 样本处理时间 <2 分钟（之前 40+ 分钟）
- ✅ 输出文件存在且正确

---

### 步骤 4：全量运行（验证通过后）

使用你原有的启动命令即可，代码已自动使用 GPU 解码。

---

## 🔄 如果需要回滚

```bash
cd /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc
cp stage3_gpu.py.backup_2026_08_09 stage3_gpu.py
```

---

## 📊 预期性能对比

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| GPU 利用率 | 0% | >50% | ∞ |
| 单样本耗时 | ~258 秒 | ~5 秒 | 50x |
| 139 样本耗时 | 10 小时 | 12 分钟 | 50x |
| 瓶颈 | CPU 解码 | I/O + 推理 | - |

---

## ✅ 快速检查清单

- [ ] 文件已复制到 CMCC
- [ ] 原文件已备份（`stage3_gpu.py.backup_2026_08_09`）
- [ ] 测试脚本运行成功
- [ ] GPU 解码加速比 >10x
- [ ] 小规模验证通过（10 样本）
- [ ] GPU 利用率 >50%
- [ ] 准备启动全量运行

---

**准备日期**：2026-08-09  
**状态**：文件已在本机准备好，等待复制到 CMCC  
**下一步**：在 CMCC 机器执行上述步骤
