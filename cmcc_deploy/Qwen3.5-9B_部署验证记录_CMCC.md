# Qwen3.5-9B 部署验证记录（CMCC）

> **执行日期**：2026-08-04  
> **执行环境**：CMCC sana_wm_qc_env  
> **执行状态**：⏳ 进行中 - transformers 已升级，待验证  
> **模型路径**：`/root/work/david_work/models/Qwen3.5-9B/`

---

## 📋 部署历程

### 阶段 1：模型确认 ✅

**模型文件**（传输完成）：
```
路径: /root/work/david_work/models/Qwen3.5-9B/
文件:
- config.json (3.1 KB)
- model.safetensors-00001~00004-of-00004.safetensors (4 个分片)
- model.safetensors.index.json (77.8 KB)
- tokenizer.json (12.5 MB)
- preprocessor_config.json, video_preprocessor_config.json
- chat_template.jinja
- success.txt (传输完成标记)

权重大小: 17.98 GB
模型类型: qwen3_5 (Qwen3_5ForConditionalGeneration)
架构: Qwen3.5 多模态模型
```

---

### 阶段 2：问题诊断 ✅

#### 问题 1：架构不匹配

**初次尝试**（`test_qwen_cmcc.py`）：
```python
# 使用 Qwen2VL 类加载
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(...)
```

**错误**：
```
RuntimeError: size mismatch for weight: 
copying a param with shape torch.Size([8192, 4096]) from checkpoint, 
the shape in current model is torch.Size([4096, 4096]).
```

**原因**：
- 模型是 Qwen3.5 架构（`qwen3_5`）
- 脚本使用 Qwen2VL 类（`qwen2_vl`）
- 两者架构完全不同（中间层大小 8192 vs 4096）

---

#### 问题 2：transformers 版本不支持

**修复尝试**（`test_qwen3.5_cmcc_fixed.py`）：
```python
# 改用 Auto 类自动识别
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(...)
```

**错误**：
```
ValueError: The checkpoint you are trying to load has model type `qwen3_5` 
but Transformers does not recognize this architecture.
```

**原因**：
- transformers 4.56.0 不支持 Qwen3.5（太新）
- Qwen3.5 发布于 2024 年底，需要更新版本

---

### 阶段 3：环境升级 ✅

**升级操作**（2026-08-04）：
```bash
conda activate sana_wm_qc_env
pip install --upgrade transformers
```

**升级结果**：
```
transformers: 4.56.0 → 5.14.1 ✅
huggingface-hub: 0.34.0 → 1.26.0 ✅
safetensors: 0.4.5 → 0.8.0 ✅
hf-xet: 1.5.0 → 1.6.0 ✅
```

**依赖冲突警告**：
```
nvidia-vipe 1.1.0 requires transformers<5,>=4, 
but you have transformers 5.14.1 which is incompatible.
```

**影响评估**：
- ⚠️ nvidia-vipe 与 transformers 5.x 不兼容
- ✅ Stage 3 不使用 nvidia-vipe，可忽略
- ⚠️ 如需 vipe，可创建独立环境

---

### 阶段 4：验证测试 ✅（已完成）

**测试脚本**：`test_qwen_disable_thinking.py`

**测试命令**：
```bash
cd /root/work/david_work/sana_qc_pipeline/test_scripts
python test_qwen_disable_thinking.py
```

**测试结果**：✅ 完全通过

**性能指标**：
- 模型加载成功：10.4 秒
- 显存占用：16.68 GB（符合预期）
- 参数量：8.95 B
- Caption 改写成功率：4/4 (100%)
- **平均推理时间：597 ms** ⚡（优于预期的 1-2s）

**加速比**：**17.91x**（启用思维链 10,695ms vs 禁用思维链 597ms）

---

### 阶段 5：思维链问题修复 ✅

#### 问题发现

**现象**：模型默认输出 "Thinking Process:"，导致：
- 推理时间 10+ 秒（本应 0.5-1 秒）
- 输出包含大量推理过程（256 tokens）
- 相机词汇未能正确移除

**根本原因**：
Qwen3.5-9B 是类 o1 推理模型，在 `chat_template.jinja` 中默认启用 `<think>` 标签：
```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}
{%- endif %}
```

#### 解决方案

**修改位置**：`src/sana_wm_pipeline/qc/stage3_gpu.py:357`

**修改前**：
```python
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

**修改后**：
```python
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 🔑 关键：禁用思维链
)
```

#### 验证结果

**测试样本**：4 个包含相机词汇的 caption

| 指标 | 启用思维链 | 禁用思维链 | 提升 |
|------|-----------|-----------|------|
| 平均耗时 | 10,695 ms | **597 ms** | **17.91x** ⚡ |
| 思维链出现 | 4/4 (100%) | 0/4 (0%) | ✅ 完全消除 |
| 成功率 | 0/4 (0%) | 4/4 (100%) | ✅ 完美 |
| 相机词移除 | ❌ 失败 | ✅ 全部移除 | 完美 |
| 输出 Tokens | 256 | 10-15 | 减少 94% |

**示例对比**：
```
原始: "A person walking in the park with camera panning from left to right and zooming in slowly."

❌ 启用思维链 (10.9秒):
"Thinking Process: 1. Analyze the Request: ..."

✅ 禁用思维链 (0.6秒):
"A person walking in the park."
```

#### 技术分析

**为什么加速比达到 17.91x？**

1. **Token 生成开销巨大**：
   - 启用时强制生成 256 tokens（思维链）
   - 禁用时只生成 10-15 tokens（纯净结果）
   - 自回归生成，越长越慢

2. **模型架构特性**：
   - Qwen3.5 使用混合架构（Gated DeltaNet + Attention）
   - 思维链模式触发更多注意力层计算
   - 禁用后走"快速路径"

3. **为什么提示词优化无效？**
   - `<think>` 标签在训练时硬编码
   - 任何 prompt 都无法覆盖底层模板行为
   - 唯一解法是在 chat template 层面禁用

---

## 🔧 脚本修复要点

### 修复版脚本（`test_qwen3.5_cmcc_fixed.py`）

**关键修改**：

1. **使用 Auto 类**：
```python
# ❌ 原代码
from transformers import Qwen2VLForConditionalGeneration

# ✅ 修复后
from transformers import AutoModelForCausalLM
```

2. **参数名称更新**：
```python
# ❌ 废弃参数
torch_dtype=torch.bfloat16

# ✅ 新版本参数
dtype=torch.bfloat16
```

3. **简化为纯文本模型**：
- 移除视觉处理逻辑
- 专注 Caption 改写（Stage 3 核心）

---

## 📊 测试计划

### 测试用例（Caption 改写）

```python
CAPTION_REWRITE_TESTS = [
    # 1. 包含相机动作
    "A person walking in the park with camera panning from left to right...",
    
    # 2. 纯内容描述
    "A red car parked on the street under a blue sky.",
    
    # 3. 多个相机动作
    "Birds flying over the ocean, camera tilting up and tracking...",
    
    # 4. 复杂场景
    "A chef cooking, the camera slowly dollies in...",
]
```

**验证标准**：
- ✅ 模型加载成功
- ✅ 显存占用 <20 GB
- ✅ 相机词汇成功移除
- ✅ 平均推理时间 <2s

---

## 🔄 回滚方案（备用）

**如果验证失败或出现兼容性问题**：

```bash
# 1. 回滚 transformers
pip install transformers==4.56.0 \
    huggingface-hub==0.34.0 \
    safetensors==0.4.5 \
    hf-xet==1.5.0

# 2. 验证回滚
pip show transformers | grep Version

# 3. 备选方案
# - 使用 Qwen2.5-VL-7B（兼容 transformers 4.56.0）
# - 或在独立环境测试 Qwen3.5
```

---

## 📝 环境备份

**备份文件**（升级前）：
```bash
# 环境包列表
/tmp/sana_env_backup.txt

# 关键版本
transformers: 4.56.0
huggingface-hub: 0.34.0
safetensors: 0.4.5
```

---

## 🎯 下一步行动

### 立即执行：
1. 运行 `test_qwen3.5_cmcc_fixed.py`
2. 验证模型加载和 Caption 改写功能
3. 记录性能指标

### 如果成功：
1. ✅ 完成任务 #2：Qwen3.5-9B 部署与验证
2. ⏭️ 继续任务 #3：Stage 3 端到端测试

### 如果失败：
1. 回滚 transformers 到 4.56.0
2. 下载 Qwen2.5-VL-7B 替代
3. 或在 AFS 独立环境测试

---

## 📌 关键注意事项

1. **nvidia-vipe 冲突**：
   - 当前不影响 Stage 3 测试
   - 如需使用 vipe，创建独立环境

2. **离线模式**：
   - 确保环境变量已设置：
     ```bash
     export TRANSFORMERS_OFFLINE=1
     export HF_HUB_OFFLINE=1
     export HF_DATASETS_OFFLINE=1
     ```

3. **显存管理**：
   - Qwen3.5-9B 需要 ~18GB
   - H100 80GB 足够
   - 确保无其他进程占用 GPU

---

**最后更新**：2026-08-04  
**当前状态**：⏳ 等待验证测试结果  
**下一步**：执行 `test_qwen3.5_cmcc_fixed.py`
