# Stage 3 完整打通项目 - 对话总结（2026-08-04）

> **用途**：为新的 Claude 会话提供完整上下文，快速接手项目  
> **项目**：SANA 数据质检管线 Stage 3 部署（CMCC H100 环境）  
> **状态**：✅ 三大模块验证完成，⏳ 等待数据信息后开始全量执行

---

## 🎯 项目目标

在 CMCC 机器上部署并运行 Stage 3 数据质检管线，对 18 万视频样本进行：
1. **UniMatch 光流检测** - 检测运动连续性
2. **DOVER 质量评分** - 检测模糊/抖动
3. **Qwen Caption 改写** - 移除相机运动词汇

**预期产出**：
- ~12 万高质量训练样本（Pass 率 ~65%）
- 完整的质检报告和统计数据
- 可复用的部署文档和脚本

---

## ✅ 已完成工作（本次对话）

### 1. 三大模块验证与优化 ⭐⭐⭐

#### UniMatch（光流检测）✅
- **状态**：完全验证通过
- **性能**：28 ms/帧对（256×256）
- **显存**：0.02 GB
- **文档**：`UniMatch_H100_验证记录_CMCC.md`
- **关键发现**：H100 性能超预期（比预估快 10 倍）

#### DOVER（质量评分）✅
- **状态**：H100 GPU 模式验证通过
- **性能**：425 ms/样本
- **显存**：0.22 GB
- **关键配置**：`export TORCH_HOME=/root/work/david_work/cache/torch`
- **文档**：`DOVER_H100_部署方案_CMCC实际执行记录.md`

#### Qwen3.5-9B（Caption 改写）✅ ⭐⭐⭐
- **状态**：验证通过，**关键问题已修复**
- **性能**：597 ms/样本（修复后）
- **显存**：16.68 GB
- **加速比**：**17.91x**（10,695 ms → 597 ms）
- **文档**：`Qwen3.5-9B_部署验证记录_CMCC.md`

**关键修复**：Qwen3.5 思维链问题
```python
# 问题：模型默认输出 "Thinking Process:"，耗时 10+ 秒
# 解决：src/sana_wm_pipeline/qc/stage3_gpu.py:357
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 🔑 关键修复
)
```

**单样本总耗时**：~2 秒（优化前预估 10-15 秒，提升 5-7.5x）

---

### 2. 完整文档体系建立 📚

创建了 7 个核心文档：

| 文档名称 | 类型 | 关键内容 |
|---------|------|---------|
| `STAGE3_完整打通方案_2026-08-04.md` | **主方案** | 完整执行计划、风险缓解、实施步骤（A-E 阶段） |
| `STAGE3_技术总结与发现.md` | 技术总结 | 关键技术发现、性能基准、最佳实践 |
| `STAGE3_快速参考卡片.md` | 速查手册 | 一页纸速查、启动命令、常见问题 |
| `QWEN_THINKING_FIX.md` | 修复指南 | Qwen 思维链问题完整解决方案（3 个方案） |
| `UniMatch_H100_验证记录_CMCC.md` | 验证记录 | UniMatch 完整验证过程和性能数据 |
| `DOVER_H100_部署方案_CMCC实际执行记录.md` | 执行记录 | DOVER 部署步骤和故障排查 |
| `Qwen3.5-9B_部署验证记录_CMCC.md` | 验证记录 | Qwen 部署历程和思维链修复 |

**文档使用指南**：
- 新手入门：`STAGE3_快速参考卡片.md`
- 执行计划：`STAGE3_完整打通方案_2026-08-04.md`
- 技术深入：`STAGE3_技术总结与发现.md`
- 排错参考：各模块验证记录

---

### 3. 测试脚本创建 🔧

**已创建**：
- `test_scripts/test_qwen_disable_thinking.py` - Qwen 思维链对比测试
- `test_scripts/test_qwen_postprocess.py` - 后处理备选方案
- `test_scripts/verify_fix.sh` - 一键验证修复脚本
- `scripts/run_stage3_single_sample.py` - 单样本端到端测试

**待创建**（需数据信息）：
- `scripts/run_stage3_cmcc.py` - 48 GPU 完整执行脚本
- `scripts/monitor_stage3.sh` - 实时进度监控
- `scripts/merge_all_stages.py` - 三阶段结果合并

---

## 🔑 关键技术发现

### 发现 1：Qwen3.5 思维链问题（最重要）⭐⭐⭐

**问题描述**：
- Qwen3.5-9B 是类 OpenAI o1 的推理模型
- 默认启用 `<think>` 标签，强制输出 256 tokens 思维过程
- 导致推理时间 10+ 秒，生产环境完全不可用

**根本原因**：
```jinja
# models/Qwen3.5-9B/chat_template.jinja (第 149-153 行)
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}  # 👈 默认启用
{%- endif %}
```

**解决方案**：
```python
# 在 apply_chat_template 调用时传入 enable_thinking=False
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 禁用思维链
)
```

**验证结果**（test_qwen_disable_thinking.py）：
- 启用思维链：10,695 ms，输出 256 tokens，包含 "Thinking Process:"
- 禁用思维链：597 ms，输出 10-15 tokens，纯净结果
- **加速比：17.91x**
- Caption 质量：4/4 样本相机词汇 100% 移除

**重要性**：
- 这是本次部署最关键的优化
- 通用性强，适用于所有 Qwen3.5 系列
- 文档完整，可避免其他项目踩坑

---

### 发现 2：UniMatch 推理参数陷阱

**问题**：参数列表长度必须 = `num_scales`（本模型为 2）

**错误示例**：
```python
# ❌ 长度不匹配
attn_splits_list=[2]  # 长度 = 1，但 num_scales = 2
```

**正确配置**：
```python
INFERENCE_PARAMS = {
    'attn_type': 'swin',
    'attn_splits_list': [2, 8],     # 长度 = 2
    'corr_radius_list': [-1, 4],    # 长度 = 2
    'prop_radius_list': [-1, 1],    # 长度 = 2
}
```

---

### 发现 3：DOVER 离线模式配置

**问题**：DOVER 依赖 timm 库，会自动下载 convnext 权重

**解决**：
```bash
# 设置 PyTorch 缓存路径（必需）
export TORCH_HOME=/root/work/david_work/cache/torch

# 安装必需依赖
pip install scikit-video  # 容易遗漏但必需
```

CMCC 机器已有权重：`/root/work/david_work/cache/torch/hub/checkpoints/convnext_tiny_1k_224_ema.pth`

---

### 发现 4：H100 性能超预期

**观察**：
- UniMatch 在 H100 上比预期快 10 倍（28ms vs 300ms 预估）
- 真实视频测试优于随机数据（28ms vs 35ms）

**启示**：
- 高端 GPU 上性能预估应基于实测
- 真实数据测试比合成数据更重要

---

## 📊 性能基准数据

### 单样本性能

| 模块 | 耗时 | 显存占用 |
|------|------|---------|
| UniMatch | 0.8 秒（30 帧对） | 0.02 GB |
| DOVER | 0.4 秒 | 0.22 GB |
| Qwen Caption | 0.6 秒 | 16.68 GB |
| I/O + 其他 | 0.2 秒 | - |
| **总计** | **~2 秒** | **~17 GB** |

### 全量处理预估

- 18 万样本 / 48 GPU = 3,750 样本/GPU
- 3,750 × 2 秒 = 7,500 秒 = 2.1 小时/GPU
- 考虑 I/O 和调度开销：**实际 7-8 小时**
- 比初始预估（10-14 小时）快 **3-6 小时**

---

## 🏗️ 系统架构设计

### 数据流

```
原始数据（18 万）
  ↓
Stage 1（基础过滤）✅ Pass ~77%
  ↓
Stage 2（场景检测）✅ Pass ~95%
  ↓ ~14 万样本
Stage 3（GPU 密集）⏳ 本次实施
  ├─ UniMatch（光流）
  ├─ DOVER（质量）
  └─ Qwen（Caption）
  ↓ ~12 万样本
最终训练数据集
```

### Stage 3 核心模块

**已实现**：`src/sana_wm_pipeline/qc/stage3_gpu.py`

**关键函数**：
- `load_unimatch_fn()` - 加载 UniMatch 模型
- `load_dover_fn()` - 加载 DOVER 模型
- `load_qwen_fn()` - 加载 Qwen 模型（**已修复思维链问题**）
- `process_sample_stage3()` - 单样本处理主函数

**配置路径**：
```python
MODEL_PATHS = {
    'unimatch': '/root/work/david_work/models/unimatch',
    'dover': '/root/work/david_work/sana_qc_pipeline/DOVER',
    'qwen': '/root/work/david_work/models/Qwen3.5-9B',
}
```

---

## 🚧 当前卡点与待办

### 阻塞项（需要用户提供）⏳

为了编写完整的执行脚本，需要以下信息：

#### 1. 数据目录结构
```bash
# 在 CMCC 机器执行
cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output
ls -lh
find . -maxdepth 2 -type d | head -30
find . -name "*.tar" | head -5
find . -name "*.mp4" | head -5
```

**需要知道**：
- 数据组织形式（tar 包？独立文件？）
- 子目录结构
- 单个样本包含哪些文件

#### 2. Stage 1+2 结果位置
```bash
find /root/work -name "*stage*.jsonl" 2>/dev/null
```

**需要知道**：
- Stage 1 结果文件路径
- Stage 2 结果文件路径
- 结果文件格式（单文件还是分 group）

#### 3. 样本 ID 映射规则

**需要提供**：
- 一个完整样本示例：
  - 样本 ID（如：`wds-DL3DV-ALL-2K/w000/000012345`）
  - 对应的视频文件路径
  - 对应的 caption 文件路径
  - 该样本在 Stage 1+2 的结果 JSON

#### 4. Stage 2 通过样本数量
```bash
grep -c '"stage2_pass": true' /path/to/stage2_results.jsonl
```

---

### 待编写脚本（收到信息后 2-3 小时）

#### A. 数据加载模块
```python
class Stage3DataLoader:
    """
    根据实际数据结构实现：
    - load_stage2_pass_samples() - 加载 Stage 2 通过的样本
    - get_video_path(sample_id) - 样本 ID → 视频路径
    - get_caption_path(sample_id) - 样本 ID → Caption 路径
    """
```

#### B. 完整执行脚本
```python
# scripts/run_stage3_cmcc.py
# 功能：
# - 分配 18 万样本到 48 个 GPU
# - 启动 48 个独立 worker 进程
# - 实现断点续传（避免重复处理）
# - 失败重试机制
# - 进度实时监控
```

#### C. 监控脚本
```bash
# scripts/monitor_stage3.sh
# 功能：
# - 实时显示完成进度
# - GPU 利用率监控
# - 预估剩余时间
# - 错误率统计
```

#### D. 结果汇总脚本
```python
# scripts/merge_all_stages.py
# 功能：
# - 合并 48 个 worker 输出
# - 与 Stage 1+2 结果合并
# - 生成最终 pass_list.txt
# - 生成统计报告
```

---

## 📋 实施步骤（待执行）

### 阶段 A：信息收集 ⏳
- [ ] 提供数据目录结构
- [ ] 提供 Stage 1+2 结果路径
- [ ] 提供样本映射示例
- [ ] 确认 Stage 2 通过样本数

### 阶段 B：代码适配（2-3 小时）
- [ ] 实现 `get_video_path()` 映射函数
- [ ] 编写 `run_stage3_cmcc.py` 主脚本
- [ ] 编写监控和汇总脚本
- [ ] 单样本端到端测试

### 阶段 C：小批量测试（5 分钟）
- [ ] 随机选择 100 个样本
- [ ] 启动 2 个 GPU worker
- [ ] 验证性能和错误率
- [ ] 检查输出格式

### 阶段 D：全量执行（7-8 小时）
- [ ] 启动 48 GPU workers
- [ ] 实时监控进度
- [ ] 处理失败样本
- [ ] 汇总结果

### 阶段 E：结果验证（30 分钟）
- [ ] 合并所有 worker 输出
- [ ] 统计 Stage 3 通过率
- [ ] 生成质检报告
- [ ] 与 Stage 1+2 合并

---

## 🔧 环境配置检查清单

### 必需环境变量（每次启动前设置）
```bash
export TORCH_HOME=/root/work/david_work/cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/root/work/david_work/models/unimatch:$PYTHONPATH
```

### 依赖检查
```bash
# 激活环境
conda activate sana_wm_qc_env

# 验证核心库
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import unimatch; print('UniMatch OK')"
python -c "import dover; print('DOVER OK')"
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"
python -c "import skvideo; print('scikit-video OK')"  # 容易遗漏
```

### 模型权重检查
```bash
# UniMatch
ls /root/work/david_work/models/unimatch/pretrained/*.pth

# DOVER
ls /root/work/david_work/sana_qc_pipeline/DOVER/pretrained_weights/DOVER.pth

# Qwen
ls /root/work/david_work/models/Qwen3.5-9B/*.safetensors

# Convnext（DOVER 依赖）
ls /root/work/david_work/cache/torch/hub/checkpoints/convnext_tiny_1k_224_ema.pth
```

---

## 🐛 常见问题速查

| 问题症状 | 可能原因 | 解决方案 |
|---------|---------|---------|
| Qwen 推理慢（10秒+） | 思维链未禁用 | 确认 `enable_thinking=False` 已应用 |
| `CUDA out of memory` | GPU 被占用 | `nvidia-smi` 查看，释放显存 |
| `ModuleNotFoundError: unimatch` | PYTHONPATH 未设置 | 设置 `export PYTHONPATH=...` |
| `No module named 'skvideo'` | 依赖缺失 | `pip install scikit-video` |
| 下载 convnext 权重 | TORCH_HOME 未设置 | `export TORCH_HOME=/root/work/...` |
| `HTTPSConnectionPool timeout` | 离线模式未开启 | 设置 `TRANSFORMERS_OFFLINE=1` |

---

## 📊 预期输出格式

### 单样本 JSON
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
├── worker_00.jsonl ... worker_47.jsonl  # 48 个 worker 输出
├── stage3_results_merged.jsonl          # 合并后结果
├── stage3_report.html                   # 可视化报告
├── failed_samples.txt                   # 失败样本列表
├── logs/
│   └── worker_*.log                     # 各 worker 日志
└── run.log                              # 主进程日志
```

### 统计报告
```json
{
  "total_samples": 180000,
  "stage1_pass": 138600,
  "stage2_pass": 131670,
  "stage3_pass": 118503,
  "final_pass_rate": 0.658,
  "stage3_metrics": {
    "unimatch_flow": {"mean": 15.3, "std": 8.2},
    "dover": {"mean": 0.45, "std": 0.12},
    "caption_rewrite": {"success_rate": 0.98}
  },
  "execution_time": "7.5 hours"
}
```

---

## 🎯 关键决策点

### 决策 1：分辨率选择
- **推荐**：256×256（最快，先跑通）
- 备选：480×480（平衡质量与速度）
- 如质量不满意，可用高分辨率重跑

### 决策 2：Caption 改写策略
- **推荐**：关键词匹配预筛选（减少 VLM 调用）
- 备选：全部过 VLM（准确但慢 2x）

### 决策 3：失败重试次数
- **推荐**：3 次重试，延迟 [5s, 30s, 300s]
- GPU OOM：清空缓存后重试
- 文件缺失：不重试，记录到失败列表

---

## 💡 成功经验总结

### 经验 1：模型部署前必做预热测试
- 单样本加载 → 单样本推理 → 小批量 → 全量
- Qwen3.5 如果没有预热测试，48 GPU 跑完才发现慢，损失巨大

### 经验 2：离线环境配置检查清单
- 环境变量必须每次设置（不持久化）
- 权重文件存在性检查
- 依赖完整性验证

### 经验 3：性能优化优先级
1. 算法层（如禁用思维链）- 10-20x
2. 硬件层（如 H100）- 2-10x
3. 参数层（如分辨率）- 1.5-3x
4. 代码层（如批处理）- 1.2-2x

---

## 📞 快速联系方式

### 关键文件位置（AFS）
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/
├── STAGE3_完整打通方案_2026-08-04.md     # 主文档
├── STAGE3_快速参考卡片.md                 # 速查
├── STAGE3_技术总结与发现.md               # 技术深入
├── QWEN_THINKING_FIX.md                   # Qwen 修复
├── src/sana_wm_pipeline/qc/stage3_gpu.py # 核心代码（已修复）
├── scripts/run_stage3_single_sample.py    # 单样本测试
└── test_scripts/test_qwen_disable_thinking.py # 验证脚本
```

### CMCC 关键路径
```
模型：
- UniMatch: /root/work/david_work/models/unimatch
- DOVER: /root/work/david_work/sana_qc_pipeline/DOVER
- Qwen: /root/work/david_work/models/Qwen3.5-9B

数据：
- 原始数据: /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output
- Stage 1+2 结果: 待确认
- Stage 3 输出: /root/work/david_work/qc_output/stage3（建议）
```

---

## 🚀 立即行动项

### 新 Claude 会话应该做什么？

1. **首先阅读**：
   - `STAGE3_快速参考卡片.md`（快速了解）
   - `STAGE3_完整打通方案_2026-08-04.md`（详细计划）

2. **获取用户提供的数据信息**（4 项，见"阻塞项"部分）

3. **编写执行脚本**（基于数据信息）：
   - `run_stage3_cmcc.py` - 主执行脚本
   - `monitor_stage3.sh` - 监控脚本
   - `merge_all_stages.py` - 汇总脚本

4. **协助测试与执行**：
   - 单样本测试 → 小批量测试 → 全量执行
   - 实时监控和问题解决

---

## 📈 项目时间线

| 阶段 | 预估耗时 | 状态 |
|------|---------|------|
| 三大模块验证 | 3 天 | ✅ 已完成（2026-08-03~04） |
| 信息收集 | 30 分钟 | ⏳ 等待用户 |
| 代码适配 | 2-3 小时 | ⏳ 待开始 |
| 小批量测试 | 5 分钟 | ⏳ 待开始 |
| 全量执行 | 7-8 小时 | ⏳ 待开始 |
| 结果汇总 | 30 分钟 | ⏳ 待开始 |
| **总计** | **~1 个工作日** | 50% 完成 |

---

## 🎖️ 核心成就

1. ✅ **Qwen3.5 思维链问题修复**（17.91x 加速）
2. ✅ **三大模块完整验证**（性能超预期）
3. ✅ **完整文档体系建立**（7 个核心文档）
4. ✅ **单样本耗时优化**（10-15s → 2s，提升 5-7.5x）
5. ⏳ **等待数据信息**（唯一剩余阻塞项）

---

**文档版本**：v1.0  
**创建时间**：2026-08-04  
**对话 ID**：当前会话  
**下一步**：提供数据信息 → 编写执行脚本 → 小批量测试 → 全量执行  
**预计完成**：收到数据信息后 1 个工作日
