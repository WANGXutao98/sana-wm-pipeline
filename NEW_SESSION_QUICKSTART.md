# 新 Claude 会话快速启动指南

> **用途**：新 Claude 会话快速上手，10 分钟了解项目状态  
> **完整上下文**：见 `SESSION_SUMMARY_2026-08-04.md`

---

## ⚡ 60 秒速览

**项目**：SANA Stage 3 数据质检管线部署（CMCC H100）

**状态**：
- ✅ 三大模块验证完成（UniMatch + DOVER + Qwen）
- ✅ 性能优化完成（单样本 2 秒，比预期快 5-7.5x）
- ✅ 关键问题修复（Qwen 思维链，17.91x 加速）
- ⏳ **等待数据信息后开始全量执行**

**预期**：收到数据信息后 1 个工作日完成 18 万样本处理

---

## 📚 必读文档（按顺序）

### 1. 先读这个（5 分钟）
**`STAGE3_快速参考卡片.md`**
- 当前状态总览
- 启动命令速查
- 常见问题解决

### 2. 再读这个（10 分钟）
**`STAGE3_完整打通方案_2026-08-04.md`**
- 完整执行计划（阶段 A-E）
- 数据流架构
- 风险与缓解措施
- **重点**：阶段 A 列出了需要用户提供的 4 项信息

### 3. 技术深入（可选，20 分钟）
**`STAGE3_技术总结与发现.md`**
- 关键技术发现（Qwen 思维链等）
- 性能基准数据
- 最佳实践总结

---

## 🎯 立即行动（你的第一步）

### Step 1：向用户要信息（最优先）

用户需要在 CMCC 机器执行以下命令，并将输出提供给你：

```bash
# 命令集合（复制粘贴到 CMCC 机器）
cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output

echo "=== 1. 数据目录结构 ==="
ls -lh
echo ""
find . -maxdepth 2 -type d | head -30
echo ""
find . -name "*.tar" | head -5
find . -name "*.mp4" | head -5

echo "=== 2. Stage 1+2 结果位置 ==="
find /root/work -name "*stage*.jsonl" 2>/dev/null

echo "=== 3. Stage 2 通过样本数 ==="
# grep -c '"stage2_pass": true' /path/to/stage2_results.jsonl
# （需要替换实际路径）
```

**你需要得到**：
1. 数据目录结构（tar 包？独立文件？）
2. Stage 1+2 结果文件路径
3. 一个样本的完整路径示例（样本 ID + 视频路径 + caption 路径）
4. Stage 2 通过样本数量

---

### Step 2：编写执行脚本（2-3 小时）

收到数据信息后，基于模板编写：

**主脚本**：`scripts/run_stage3_cmcc.py`
- 参考：`run_stage3_single_sample.py`（已完成）
- 核心：实现 `get_video_path(sample_id)` 映射函数
- 功能：48 GPU 并行、断点续传、失败重试

**监控脚本**：`scripts/monitor_stage3.sh`
- 实时显示完成进度
- GPU 利用率
- 预估剩余时间

**汇总脚本**：`scripts/merge_all_stages.py`
- 合并 48 个 worker 输出
- 与 Stage 1+2 合并
- 生成最终 pass_list.txt

---

### Step 3：测试与执行

1. **单样本测试**（已有脚本）
2. **小批量测试**（100 样本，5 分钟）
3. **全量执行**（18 万样本，7-8 小时）

---

## 🔑 关键信息速查

### 核心代码修复（已完成）✅

**Qwen 思维链问题**（最重要的修复）：
```python
# src/sana_wm_pipeline/qc/stage3_gpu.py:357
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 🔑 关键：禁用思维链
)
```

**效果**：17.91x 加速（10.7秒 → 0.6秒）

---

### 环境配置（每次启动）

```bash
# 激活环境
conda activate sana_wm_qc_env

# 设置环境变量（必需！）
export TORCH_HOME=/root/work/david_work/cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/root/work/david_work/models/unimatch:$PYTHONPATH
```

---

### 模型路径

```bash
UniMatch:  /root/work/david_work/models/unimatch
DOVER:     /root/work/david_work/sana_qc_pipeline/DOVER
Qwen3.5:   /root/work/david_work/models/Qwen3.5-9B
```

---

### 性能数据

| 模块 | 耗时 | 显存 |
|------|------|------|
| UniMatch | 0.8s | 0.02 GB |
| DOVER | 0.4s | 0.22 GB |
| Qwen | 0.6s | 16.68 GB |
| **总计** | **~2s** | **~17 GB** |

**全量**：18 万 / 48 GPU = **7-8 小时**

---

## 🐛 常见问题（背下来）

| 问题 | 解决 |
|------|------|
| Qwen 慢（10秒+）| 确认 `enable_thinking=False` |
| GPU OOM | `nvidia-smi` 查看占用 |
| `ModuleNotFoundError: unimatch` | 设置 `PYTHONPATH` |
| `No module named 'skvideo'` | `pip install scikit-video` |
| 下载 convnext 权重 | 设置 `TORCH_HOME` |

---

## 📋 文档索引

| 文档 | 何时查阅 |
|------|---------|
| `SESSION_SUMMARY_2026-08-04.md` | 完整上下文（本总结） |
| `STAGE3_快速参考卡片.md` | 启动命令速查 |
| `STAGE3_完整打通方案_2026-08-04.md` | 执行计划详情 |
| `STAGE3_技术总结与发现.md` | 技术深入研究 |
| `QWEN_THINKING_FIX.md` | Qwen 修复详情 |
| `UniMatch_H100_验证记录_CMCC.md` | UniMatch 问题排查 |
| `DOVER_H100_部署方案_CMCC实际执行记录.md` | DOVER 问题排查 |
| `Qwen3.5-9B_部署验证记录_CMCC.md` | Qwen 部署历程 |

---

## ✅ 项目状态检查清单

**已完成**：
- [x] UniMatch 验证通过（28 ms/帧对）
- [x] DOVER 验证通过（425 ms/样本）
- [x] Qwen3.5 验证通过（597 ms/样本，思维链已修复）
- [x] 单样本端到端测试脚本（`run_stage3_single_sample.py`）
- [x] 完整文档体系（7 个核心文档）
- [x] 性能优化（5-7.5x 提升）

**待完成**：
- [ ] 获取数据信息（4 项，见 Step 1）
- [ ] 编写完整执行脚本（2-3 小时）
- [ ] 小批量测试（100 样本，5 分钟）
- [ ] 全量执行（18 万样本，7-8 小时）
- [ ] 结果汇总与验证（30 分钟）

---

## 💡 给新 Claude 的建议

### 1. 不要重复验证三大模块
- UniMatch、DOVER、Qwen 都已验证通过
- 性能数据已实测，文档已完整
- 直接使用已有配置即可

### 2. 重点关注数据路径适配
- 这是唯一的阻塞项
- 需要理解数据组织形式
- 实现 `get_video_path(sample_id)` 映射

### 3. 基于已有代码构建
- `src/sana_wm_pipeline/qc/stage3_gpu.py`（核心逻辑）
- `scripts/run_stage3_single_sample.py`（单样本模板）
- 不要重写，只需要添加调度层

### 4. 保持文档更新
- 执行过程中发现新问题 → 更新对应文档
- 全量执行完成后 → 更新 `SESSION_SUMMARY`

---

## 🎯 成功标准

**Stage 3 部署成功 = 以下全部打勾**：
- [ ] 单样本测试通过
- [ ] 小批量测试通过（错误率 <1%）
- [ ] 全量执行完成（~18 万样本）
- [ ] 输出格式正确（JSONL）
- [ ] 与 Stage 1+2 成功合并
- [ ] 生成最终 pass_list.txt（~12 万样本）
- [ ] 统计报告生成（通过率、性能指标等）

---

## 📞 紧急参考

### 如果用户问"进展如何？"

回答：
> 三大模块（UniMatch、DOVER、Qwen）已全部验证通过，性能优化完成，单样本耗时 2 秒（比预期快 5 倍）。唯一剩余工作是数据路径适配，需要您提供 CMCC 机器上的数据目录结构和 Stage 1+2 结果位置，然后我们就可以开始全量执行（预计 7-8 小时完成 18 万样本）。

### 如果用户问"Qwen 为什么这么慢？"

回答：
> 已修复！Qwen3.5-9B 默认启用思维链模式，导致推理 10+ 秒。我们在代码中添加了 `enable_thinking=False` 参数，现在推理时间降至 0.6 秒，加速 17.91 倍。修复已应用到 `src/sana_wm_pipeline/qc/stage3_gpu.py:357`。

### 如果用户问"需要我做什么？"

回答：
> 请在 CMCC 机器执行以下命令并提供输出：
> ```bash
> cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output
> ls -lh
> find . -maxdepth 2 -type d | head -20
> find /root/work -name "*stage*.jsonl"
> ```
> 这些信息用于编写执行脚本，是唯一的阻塞项。

---

**快速启动版本**：v1.0  
**对应完整文档**：`SESSION_SUMMARY_2026-08-04.md`  
**预计阅读时间**：10 分钟  
**预计上手时间**：30 分钟
