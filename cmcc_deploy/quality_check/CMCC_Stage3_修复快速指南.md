# CMCC Stage3 修复 - 快速执行指南

> **根因**：代码从 tar 重复读取，而不是使用已解压的目录  
> **修复**：1 个文件修改，预期 10-100x I/O 加速

---

## 🚀 快速修复步骤

### 1. 复制文件到 CMCC（本机执行）

**需要复制的文件**：
```
源文件（本机）：
  /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py
  /mnt/afs/davidwang/workspace/sana_wm_pipeline/test_scripts/test_tar_vs_directory_io_cmcc.py

目标位置（CMCC）：
  /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py
  /root/work/david_work/sana_qc_pipeline/test_scripts/test_tar_vs_directory_io_cmcc.py
```

---

### 2. 在 CMCC 机器上验证

#### 步骤 2.1：I/O 性能测试

```bash
cd /root/work/david_work/sana_qc_pipeline/test_scripts

# 运行测试（预期：从目录读取快 10-50x）
python test_tar_vs_directory_io_cmcc.py
```

**预期输出**：
```
从 tar 读取:      100-500 ms
从目录读取:      5-20 ms  (10-50x 加速)
```

---

#### 步骤 2.2：备份并替换代码

```bash
cd /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc

# 备份原文件
cp stage3_gpu.py stage3_gpu.py.backup_2026_08_09

# 替换为修复版本（已从本机复制过来）
# 验证新代码
grep -A 5 "优先从解压目录读取" stage3_gpu.py
```

---

#### 步骤 2.3：小规模测试（10 样本）

```bash
cd /root/work/david_work/sana_qc_pipeline

# 创建测试集
head -n 10 /root/work/david_work/qc_output_new/smoke_test_manifest.jsonl > /tmp/test10.jsonl

# 运行测试（计时）
time python scripts/runstage3cmcc.py \
  --stage12-jsonl /tmp/test10.jsonl \
  --skip-vlm \
  --output-dir /tmp/stage3test \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04filter/table6_thresholds.yaml \
  --device cuda
```

**预期结果**：
- ✅ 10 样本处理时间：<5 分钟（之前 ~40 分钟）
- ✅ 输出文件：`/tmp/stage3test/stage3_worker000.jsonl` 存在且有 10 行

---

### 3. 如果验证通过，启动全量运行

使用你原有的启动命令即可。

---

## 🔄 回滚方案

如果出现问题：

```bash
cd /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc
cp stage3_gpu.py.backup_2026_08_09 stage3_gpu.py
```

---

## 📊 修复内容摘要

**修改文件**：`src/sana_wm_pipeline/qc/stage3_gpu.py`

**修改函数**：`process_sample_stage3()`

**核心逻辑**：
```python
# 优先从解压目录读取
extracted_dir = tar_path.with_suffix('')  # .tar -> /
if extracted_dir.exists():
    mp4_bytes = (extracted_dir / f"{sample_id}.mp4").read_bytes()
    cap_bytes = (extracted_dir / f"{sample_id}.caption.txt").read_bytes()
else:
    # Fallback 到 tar 读取
    with tarfile.open(tar_path) as tf:
        mp4_bytes = tf.extractfile(...).read()
```

**效果**：
- 从 tar 读取：100-500 ms/样本
- 从目录读取：5-20 ms/样本
- **加速比：10-50x I/O 性能提升**

---

## ✅ 验证检查清单

- [ ] I/O 测试脚本运行成功
- [ ] I/O 加速比 >10x
- [ ] 代码已备份并替换
- [ ] 10 样本测试 <5 分钟
- [ ] 输出文件正确
- [ ] 准备全量运行

---

**修复日期**：2026-08-09  
**修复类型**：I/O 优化（使用解压目录）  
**预期提升**：10-100 倍
