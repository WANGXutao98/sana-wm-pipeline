# Stage 3 CMCC 执行快速指南（最终版 2026-08-07）

> **基于你的实际需求定制**：从 Stage 1+2 JSONL 加载样本 → 从解压目录读取文件 → 执行 Stage 3

---

## 🎯 执行目标

- **输入**：Stage 1+2 结果 JSONL（包含通过的样本）
- **数据源**：解压后的目录（`/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output`）
- **输出**：Stage 3 结果 JSONL（每个 worker 一个文件）

---

## ✅ 代码已就绪

### 修改的脚本

**`scripts/run_stage3_cmcc.py`** - 已更新为 v2.0
- ✅ 新增 `--data-root` 参数（自动索引解压数据）
- ✅ 保持与你的原命令格式兼容
- ✅ 智能回退机制（索引 → tar_path）

### 核心代码

**`src/sana_wm_pipeline/qc/stage3_gpu.py`**
- ✅ 第 70-79 行已支持解压目录优先读取
- ✅ Qwen 思维链已修复（第 357 行）
- ✅ 无需修改

---

## 🚀 快速开始（3 步）

### 步骤 1：单样本冒烟测试（5 分钟）

在 CMCC 机器上执行：

```bash
# 激活环境
conda activate sana_wm_qc_env

# 设置环境变量
export TORCH_HOME=/root/work/david_work/cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 运行冒烟测试（单个 worker，处理前 10 个样本）
python scripts/run_stage3_cmcc.py \
  --stage1-jsonl /root/work/david_work/qc_output_new/smoke_test_manifest.jsonl \
  --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output-dir /tmp/stage3_smoke_test \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml
```

**预期输出**：
```
[index] 扫描数据目录: /root/work/filestorage/...
[index] 索引完成，共 333163 个样本
[worker 0] loading UniMatch...
[worker 0] loading DOVER...
[worker 0] loading Qwen3.5-9B...
[worker 0] models ready.
[worker 0] 已处理 100 个样本
[worker 0] 完成！
  已处理: XXX
  已跳过: XXX
  输出: /tmp/stage3_smoke_test/stage3_worker000.jsonl
```

**验证结果**：
```bash
# 查看输出
cat /tmp/stage3_smoke_test/stage3_worker000.jsonl | jq . | head -50

# 检查通过率
total=$(wc -l < /tmp/stage3_smoke_test/stage3_worker000.jsonl)
passed=$(grep -c '"table6_accepted": true' /tmp/stage3_smoke_test/stage3_worker000.jsonl)
echo "处理: $total, 通过: $passed, 通过率: $(echo "scale=2; $passed * 100 / $total" | bc)%"
```

---

### 步骤 2：小批量测试（10 分钟）

测试 2 个 worker 并行处理：

```bash
# 准备测试数据（前 100 行）
head -100 /root/work/david_work/qc_output_new/full_manifest.jsonl > /tmp/test_100.jsonl

# 启动 2 个 worker
for worker_id in 0 1; do
    CUDA_VISIBLE_DEVICES=$worker_id nohup python scripts/run_stage3_cmcc.py \
      --stage1-jsonl /tmp/test_100.jsonl \
      --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
      --output-dir /tmp/stage3_batch_test \
      --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
      --unimatch-dir /root/work/david_work/models/unimatch \
      --worker-id $worker_id \
      --total-workers 2 \
      --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
      > /tmp/stage3_worker${worker_id}.log 2>&1 &
done

echo "Worker PIDs: $!"
```

**监控进度**：
```bash
# 查看实时日志
tail -f /tmp/stage3_worker0.log

# 检查完成情况
ls -lh /tmp/stage3_batch_test/

# 统计已处理样本
cat /tmp/stage3_batch_test/stage3_worker*.jsonl | wc -l
```

---

### 步骤 3：全量执行（7-8 小时）

确认小批量测试无误后，启动全量执行：

```bash
# 完整的 Stage 1+2 结果 JSONL 路径
STAGE12_JSONL="/root/work/david_work/qc_output_new/full_manifest.jsonl"

# 输出目录
OUTPUT_DIR="/root/work/david_work/qc_output_stage3_final"
mkdir -p $OUTPUT_DIR

# 启动 48 个 worker（每个 GPU 一个）
for worker_id in {0..47}; do
    CUDA_VISIBLE_DEVICES=$worker_id nohup python scripts/run_stage3_cmcc.py \
      --stage1-jsonl $STAGE12_JSONL \
      --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
      --output-dir $OUTPUT_DIR \
      --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
      --unimatch-dir /root/work/david_work/models/unimatch \
      --worker-id $worker_id \
      --total-workers 48 \
      --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
      > $OUTPUT_DIR/worker${worker_id}.log 2>&1 &
done

echo "已启动 48 个 worker"
echo "日志目录: $OUTPUT_DIR/"
```

**监控执行**：
```bash
# 实时监控进度
watch -n 30 'echo "已完成样本数:"; cat /root/work/david_work/qc_output_stage3_final/stage3_worker*.jsonl 2>/dev/null | wc -l; echo ""; echo "Worker 状态:"; ls -lh /root/work/david_work/qc_output_stage3_final/*.done 2>/dev/null | wc -l; echo "/ 48"'

# 查看某个 worker 的日志
tail -f $OUTPUT_DIR/worker00.log

# 检查 GPU 使用率
watch -n 10 nvidia-smi
```

---

## 📊 结果汇总

全量执行完成后：

```bash
cd /root/work/david_work/qc_output_stage3_final

# 1. 合并所有 worker 输出
cat stage3_worker*.jsonl > stage3_all_merged.jsonl

# 2. 统计通过率
total=$(wc -l < stage3_all_merged.jsonl)
passed=$(grep -c '"table6_accepted": true' stage3_all_merged.jsonl)
echo "总样本数: $total"
echo "通过样本数: $passed"
echo "通过率: $(echo "scale=2; $passed * 100 / $total" | bc)%"

# 3. 提取通过的样本 ID
grep '"table6_accepted": true' stage3_all_merged.jsonl | \
  jq -r '.sample_id' > stage3_pass_list.txt

echo "通过样本列表: stage3_pass_list.txt ($(wc -l < stage3_pass_list.txt) 个)"

# 4. 合并 caption 改写
cat caption_overrides_worker*.jsonl > caption_overrides_all.jsonl
echo "Caption 改写: $(wc -l < caption_overrides_all.jsonl) 个"
```

---

## 🔧 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--stage1-jsonl` | Stage 1+2 结果 JSONL | `/root/work/.../manifest.jsonl` |
| `--data-root` | 解压数据根目录 | `/root/work/.../jdvbbfb_output` |
| `--output-dir` | 输出目录 | `/root/work/.../stage3_output` |
| `--qwen-dir` | Qwen 模型路径 | `/root/work/.../Qwen3.5-9B` |
| `--unimatch-dir` | UniMatch 模型路径 | `/root/work/.../unimatch` |
| `--worker-id` | Worker ID（0-indexed）| `0` |
| `--total-workers` | 总 Worker 数 | `48` |
| `--table6-cfg` | Table 6 配置文件 | `src/.../table6_thresholds.yaml` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | CUDA 设备 | `cuda` |
| `--skip-index` | 跳过索引构建（使用 tar_path）| `False` |

---

## 📈 性能预估

### 索引构建（首次）

| 样本数 | 时间 |
|--------|------|
| 333,163 | ~60-120 秒 |

### 模型加载（每个 worker）

| 模块 | 时间 |
|------|------|
| UniMatch | ~2.5 秒 |
| DOVER | ~1.8 秒 |
| Qwen3.5-9B | ~8.2 秒 |
| **总计** | ~12 秒 |

### 单样本处理

| 操作 | 时间 |
|------|------|
| 文件读取 | ~0.05 秒 |
| UniMatch | ~0.8 秒 |
| DOVER | ~0.4 秒 |
| Qwen | ~0.6 秒 |
| **总计** | ~2 秒 |

### 全量执行（假设 10 万样本通过 Stage 1+2）

| 指标 | 数值 |
|------|------|
| 总样本数 | 100,000 |
| 每 GPU 样本数 | 100,000 / 48 ≈ 2,083 |
| 单 GPU 耗时 | 2,083 × 2 秒 ≈ 69 分钟 |
| **总耗时**（含启动） | **~90 分钟** |

---

## 🐛 故障排查

### 问题 1：索引构建失败

**症状**：`[index] 索引完成，共 0 个样本`

**原因**：数据根目录路径错误

**解决**：
```bash
# 检查路径
ls -lh /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output

# 确认有 final_wds-* 目录
ls /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/final_wds-*

# 如果路径错误，修正 --data-root 参数
```

---

### 问题 2：模型加载失败

**症状**：`ModuleNotFoundError: No module named 'unimatch'`

**原因**：环境变量未设置

**解决**：
```bash
# 设置 PYTHONPATH
export PYTHONPATH=/root/work/david_work/models/unimatch:$PYTHONPATH

# 或在脚本开头添加（已包含在代码中）
sys.path.insert(0, '/root/work/david_work/models/unimatch')
```

---

### 问题 3：Qwen 推理慢（10 秒+）

**症状**：单样本处理超过 10 秒

**原因**：思维链未禁用

**解决**：
```bash
# 检查代码修复
grep -n "enable_thinking" src/sana_wm_pipeline/qc/stage3_gpu.py

# 应该看到第 357 行有：enable_thinking=False
# 如果没有，参考 QWEN_THINKING_FIX.md 应用修复
```

---

### 问题 4：某些样本找不到

**症状**：日志中有 "样本不在索引中，回退到 tar_path"

**原因**：
- 样本文件确实缺失
- 或 sample_id 格式不匹配

**解决**：
```bash
# 手动查找样本
find /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  -name "样本ID.mp4"

# 如果找不到，说明文件确实缺失
# 检查 JSONL 中的 tar_path 是否有效
```

---

## 📋 执行检查清单

### 执行前 ✓

- [ ] Conda 环境已激活（`sana_wm_qc_env`）
- [ ] 环境变量已设置（`TORCH_HOME`, `TRANSFORMERS_OFFLINE` 等）
- [ ] Stage 1+2 结果 JSONL 存在且有效
- [ ] 数据根目录存在（`/root/work/.../jdvbbfb_output`）
- [ ] 输出目录有足够空间（~50 GB）
- [ ] GPU 空闲（`nvidia-smi` 检查）

### 执行中 ⚠️

- [ ] 索引构建成功（显示样本数 > 0）
- [ ] 所有 worker 都在运行（`ps aux | grep run_stage3_cmcc`）
- [ ] GPU 利用率正常（50-80%）
- [ ] 输出文件在增长（`ls -lh *.jsonl`）
- [ ] 无大量错误日志

### 执行后 ✅

- [ ] 所有 worker 完成（48 个 `.done` 文件）
- [ ] 输出文件数量正确（48 个 `stage3_worker*.jsonl`）
- [ ] 总样本数匹配（合并后的行数 ≈ Stage 1+2 Pass 样本数）
- [ ] 通过率合理（预期 85-90%）
- [ ] 结果格式正确（`jq . stage3_all_merged.jsonl | head`）

---

## 🎯 成功标准

| 指标 | 标准 |
|------|------|
| 索引构建 | < 2 分钟，样本数 > 300,000 |
| 模型加载 | < 15 秒/worker |
| 单样本处理 | ~2 秒 |
| 错误率 | < 1% |
| 通过率 | 85-90%（取决于数据质量）|
| 全量执行 | < 2 小时（10 万样本）|

---

## 📞 获取帮助

### 相关文档

1. **本文档** - 执行快速指南
2. `run_stage3_cmcc更新说明_2026-08-07.md` - 代码变更详解
3. `SESSION_SUMMARY_2026-08-04.md` - 上一版本总结
4. `QWEN_THINKING_FIX.md` - Qwen 修复指南

### 日志位置

- **Worker 日志**：`$OUTPUT_DIR/worker*.log`
- **主输出**：`$OUTPUT_DIR/stage3_worker*.jsonl`
- **Caption 改写**：`$OUTPUT_DIR/caption_overrides_worker*.jsonl`
- **完成标记**：`$OUTPUT_DIR/*.done`

---

**版本**：v1.0  
**日期**：2026-08-07  
**状态**：✅ 就绪，可立即执行  
**预计执行时间**：90 分钟（假设 10 万样本）
