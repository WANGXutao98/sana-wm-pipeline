# CMCC 快速部署检查清单

## ✅ 部署前检查

### 1. 文件准备
- [ ] 本机已备份：`stage3_gpu.py.backup_20260809_before_fp16`
- [ ] 本机已更新：`stage3_gpu.py`（新增 FP16 支持）
- [ ] 语法检查通过：`python -m py_compile src/sana_wm_pipeline/qc/stage3_gpu.py`

### 2. 文档准备
- [ ] 已阅读：`DOVER_优化_任务总结.md`
- [ ] 已阅读：`CMCC_DOVER_优化部署指南.md`
- [ ] 已阅读：`stage3_gpu_FP16_变更说明.md`

---

## 📦 传输到 CMCC

### 方式 1：直接 scp（如果有网络连接）
```bash
scp src/sana_wm_pipeline/qc/stage3_gpu.py <user>@<cmcc-host>:/path/to/sana_wm_pipeline/src/sana_wm_pipeline/qc/
```

### 方式 2：手动复制（推荐）
1. 在本机查看文件：
   ```bash
   cat src/sana_wm_pipeline/qc/stage3_gpu.py
   ```
2. 在 CMCC 机器上备份：
   ```bash
   cd /path/to/sana_wm_pipeline
   cp src/sana_wm_pipeline/qc/stage3_gpu.py src/sana_wm_pipeline/qc/stage3_gpu.py.backup_before_fp16
   ```
3. 在 CMCC 机器上编辑：
   ```bash
   vim src/sana_wm_pipeline/qc/stage3_gpu.py
   ```
4. 替换 `load_dover_fn` 函数（第 297-379 行）

---

## 🔍 CMCC 机器上验证

### 1. 语法检查
```bash
cd /path/to/sana_wm_pipeline
python -m py_compile src/sana_wm_pipeline/qc/stage3_gpu.py
# ✅ 无输出 = 语法正确
# ❌ 有报错 = 检查代码
```

### 2. 激活环境
```bash
conda activate <your-env-name>
```

### 3. 测试单个样本（可选，快速验证）
```bash
python -c "
from sana_wm_pipeline.qc.stage3_gpu import load_dover_fn
import numpy as np

# 加载 DOVER（应该看到 FP16 启用提示）
dover_fn = load_dover_fn(device='cuda')

# 测试推理
frames = np.random.randint(0, 256, (80, 720, 1280, 3), dtype=np.uint8)
score = dover_fn(frames)
print(f'DOVER score: {score:.4f}')
"
```

**预期输出**：
```
[DOVER] GPU FP16 模式已启用（显存减半，支持到 1080p）
DOVER score: 0.6543
```

### 4. 运行单 shard 测试
```bash
python scripts/run_stage3_cmcc.py \
    --input-jsonl stage1_results/group_X/stage1_shard-000003-000001.jsonl \
    --output-jsonl stage3_results/group_X/stage3_shard-000003-000001.jsonl \
    --caption-overrides-jsonl stage3_results/group_X/caption_overrides_shard-000003-000001.jsonl \
    --group-name group_X
```

### 5. 监控 GPU（另开终端）
```bash
nvidia-smi -l 1
```

---

## 📊 验收标准

### 关键指标对比

| 检查项 | 旧方案（CPU） | 新方案（FP16 GPU） | 实际值 | 状态 |
|--------|--------------|-------------------|--------|------|
| **日志提示** | 无 | `[DOVER] GPU FP16 模式已启用` | _______ | ⬜ |
| **GPU 使用率** | ~0% | > 90% | _______% | ⬜ |
| **显存占用** | ~0 GB | 10-20 GB | _______GB | ⬜ |
| **单样本耗时** | ~169 秒 | < 15 秒 | _______秒 | ⬜ |
| **单 shard (139 样本)** | ~6.5 小时 | < 30 分钟 | _______分钟 | ⬜ |

### 必须通过的检查

- [ ] ✅ 日志显示：`[DOVER] GPU FP16 模式已启用`
- [ ] ✅ GPU 使用率 > 90%
- [ ] ✅ 单样本耗时 < 15 秒（至少比旧方案快 10x）
- [ ] ✅ 显存占用在 10-20GB 范围（不 OOM）

---

## 🚨 常见问题快速处理

### 问题 1：看不到 FP16 启用提示

**检查**：
```bash
# 查看日志
tail -f <log_file> | grep DOVER
```

**原因**：
- 代码未更新成功
- 使用了 CPU 模式

**解决**：
```bash
# 确认函数签名
grep "def load_dover_fn" src/sana_wm_pipeline/qc/stage3_gpu.py
# 应该看到：def load_dover_fn(device: str = "cuda", ..., use_fp16: bool = True)
```

---

### 问题 2：GPU 使用率低（< 50%）

**检查**：
```bash
nvidia-smi
```

**可能原因**：
- 正在处理 I/O（读取视频文件）
- DOVER 模型未加载到 GPU
- 意外使用了 CPU 模式

**解决**：
```bash
# 检查进程
ps aux | grep python

# 检查日志
grep "DOVER.*模式" <log_file>
```

---

### 问题 3：OOM（显存溢出）

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**临时解决**：
```python
# 在 run_stage3_cmcc.py 中禁用 FP16
dover_fn = load_dover_fn(device="cuda", use_fp16=False)
```

**长期解决**：
- 降采样高分辨率视频到 1080p
- 或对 4K 视频使用 CPU 模式

---

### 问题 4：速度没提升

**检查步骤**：

1. 确认 GPU 模式：
```bash
grep "DOVER.*GPU" <log_file>
# 应该看到：[DOVER] GPU FP16 模式已启用
```

2. 确认 GPU 使用率：
```bash
nvidia-smi
# GPU-Util 应该 > 90%
```

3. 确认处理速度：
```bash
# 查看日志，计算每个样本的处理时间
# 应该 < 15 秒/样本
```

---

## 📈 性能记录表（填写实际值）

### 测试配置
- **GPU**: ______________________
- **CUDA 版本**: ______________________
- **测试日期**: ______________________
- **测试 shard**: ______________________
- **样本数量**: ______________________

### 性能数据

| 指标 | 旧方案（CPU） | 新方案（FP16 GPU） | 加速比 |
|------|--------------|-------------------|--------|
| 单样本平均耗时 | 169 秒 | _______秒 | _______x |
| GPU 使用率 | ~0% | _______%  | - |
| 显存占用 | ~0 GB | _______GB | - |
| 单 shard 总耗时 | ~6.5 小时 | _______分钟 | _______x |

### 问题记录
- [ ] 无问题，一切正常
- [ ] 遇到问题：_________________________________
  - 解决方案：_________________________________

---

## ✅ 最终确认

- [ ] 所有验收标准通过
- [ ] GPU 使用率 > 90%
- [ ] 加速比 > 10x
- [ ] 无 OOM 错误
- [ ] 准备全量部署

---

## 📞 支持

如遇到问题，请提供：
1. 完整的错误日志
2. `nvidia-smi` 输出
3. 处理速度数据（秒/样本）
4. GPU 型号和显存大小

---

**文档版本**: v1.0  
**生成日期**: 2026-08-09  
**适用范围**: CMCC Stage 3 DOVER FP16 优化部署
