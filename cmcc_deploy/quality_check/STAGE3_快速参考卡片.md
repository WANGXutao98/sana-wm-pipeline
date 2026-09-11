# Stage 3 快速参考卡片（CMCC）

> **一页纸速查**：所有关键信息汇总

---

## ✅ 当前状态（2026-08-04）

| 模块 | 状态 | 性能 | 显存 |
|------|------|------|------|
| UniMatch | ✅ 验证通过 | 28 ms/帧对 | 0.02 GB |
| DOVER | ✅ 验证通过 | 425 ms/样本 | 0.22 GB |
| Qwen3.5-9B | ✅ 验证通过 | 597 ms/样本 | 16.68 GB |
| **单样本总计** | ✅ 就绪 | **~2 秒** | **~17 GB** |

**全量预估**：18 万样本 / 48 GPU = **7-8 小时**

---

## 🚀 启动命令速查

### 环境激活
```bash
# 激活环境
conda activate sana_wm_qc_env

# 设置环境变量（必需！）
export TORCH_HOME=/root/work/david_work/cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 单样本测试
```bash
cd /root/work/david_work/sana_qc_pipeline/scripts

python run_stage3_single_sample.py \
  --sample-id "your_sample_id" \
  --video-path "/path/to/video.mp4" \
  --caption-path "/path/to/caption.txt" \
  --output /tmp/test_result.jsonl
```

### 小批量测试（100 样本）
```bash
python run_stage3_cmcc.py \
  --stage2-results /path/to/stage2_results.jsonl \
  --video-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output-dir /tmp/stage3_test \
  --num-gpus 2 \
  --max-samples 100
```

### 全量执行（48 GPU）
```bash
nohup python run_stage3_cmcc.py \
  --stage2-results /path/to/stage2_results.jsonl \
  --video-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output-dir /root/work/david_work/qc_output/stage3 \
  --num-gpus 48 \
  > /root/work/david_work/qc_output/stage3/run.log 2>&1 &

echo $! > /tmp/stage3_pid.txt
```

### 监控进度
```bash
# 查看已完成样本数
find /root/work/david_work/qc_output/stage3 -name "worker_*.jsonl" \
  -exec wc -l {} \; | awk '{s+=$1} END {print s}'

# GPU 利用率
watch -n 5 nvidia-smi

# 查看日志
tail -f /root/work/david_work/qc_output/stage3/run.log
```

---

## 🔧 关键配置

### 模型路径
```bash
UniMatch:    /root/work/david_work/models/unimatch
DOVER:       /root/work/david_work/sana_qc_pipeline/DOVER
Qwen3.5-9B:  /root/work/david_work/models/Qwen3.5-9B
```

### 质量阈值（可调整）
```python
THRESHOLDS = {
    'unimatch_flow_max': 50.0,    # 光流最大值（像素）
    'dover_min': 0.3,              # DOVER 最低分
}
```

### Qwen 关键修复（已应用）
```python
# src/sana_wm_pipeline/qc/stage3_gpu.py:357
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 🔑 必需！
)
```

---

## 🐛 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| `CUDA out of memory` | 检查 GPU 是否被占用：`nvidia-smi` |
| `ModuleNotFoundError: unimatch` | 设置 `export PYTHONPATH=/root/work/david_work/models/unimatch:$PYTHONPATH` |
| `No module named 'skvideo'` | `pip install scikit-video` |
| `HTTPSConnectionPool timeout` | 确认已设置 `TRANSFORMERS_OFFLINE=1` |
| 下载 convnext 权重 | 设置 `export TORCH_HOME=/root/work/david_work/cache/torch` |
| Qwen 推理慢（10秒+）| 确认 `enable_thinking=False` 已应用 |

---

## 📊 预期输出

### 单样本 JSON 格式
```json
{
  "sample_id": "wds-DL3DV-ALL-2K/w000/000012345",
  "stage3": {
    "unimatch_flow": 12.5,
    "dover": 0.42,
    "vlm_entity_count": 2,
    "vlm_quality": 0.75,
    "caption_revised": "A person walking in the park.",
    "table6_accepted": true,
    "reasons": []
  },
  "stage3_pass": true
}
```

### 输出目录结构
```
/root/work/david_work/qc_output/stage3/
├── worker_00.jsonl ... worker_47.jsonl  (48 个)
├── stage3_results_merged.jsonl          (合并后)
├── logs/
│   └── worker_*.log
└── run.log
```

---

## 📞 需要的信息（待提供）

在开始全量执行前，需要确认：

1. **数据目录结构**：
   ```bash
   cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output
   ls -lh
   find . -maxdepth 2 -type d | head -20
   ```

2. **Stage 1+2 结果位置**：
   ```bash
   find /root/work -name "*stage*.jsonl"
   ```

3. **Stage 2 通过样本数**：
   ```bash
   grep -c '"stage2_pass": true' /path/to/stage2_results.jsonl
   ```

4. **一个完整样本示例**：
   - 样本 ID
   - 视频路径
   - Caption 路径

---

## 📚 文档索引

| 文档 | 用途 |
|------|------|
| `STAGE3_完整打通方案_2026-08-04.md` | **主文档**：完整执行方案 |
| `STAGE3_技术总结与发现.md` | 技术细节和性能数据 |
| `QWEN_THINKING_FIX.md` | Qwen 思维链修复指南 |
| `UniMatch_H100_验证记录_CMCC.md` | UniMatch 验证详情 |
| `DOVER_H100_部署方案_CMCC实际执行记录.md` | DOVER 部署详情 |
| `Qwen3.5-9B_部署验证记录_CMCC.md` | Qwen 部署详情 |

---

## ⚡ 性能优化记录

| 优化项 | 提升倍数 | 关键技术 |
|--------|---------|---------|
| Qwen 思维链禁用 | **17.9x** | `enable_thinking=False` |
| UniMatch H100 | 10.7x | H100 GPU + 参数配置 |
| DOVER H100 | 2-5x | GPU 模式 + 离线配置 |
| **整体** | **5-7.5x** | 组合优化 |

---

## 🎯 下一步行动

### 立即执行（需要你）
1. 在 CMCC 机器提供数据信息（见上方"需要的信息"）
2. 验证单样本测试脚本可用

### 准备阶段（2-3 小时）
3. 根据数据结构编写完整执行脚本
4. 小批量测试（100 样本）

### 全量执行（7-8 小时）
5. 启动 48 GPU workers
6. 监控进度
7. 结果汇总

---

**最后更新**：2026-08-04  
**版本**：v1.0  
**状态**：⏳ 等待数据信息
