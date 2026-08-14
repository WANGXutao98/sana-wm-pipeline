# Qwen3.5 思维链问题修复指南

## 问题分析

Qwen3.5-9B/27B 模型在训练时被设计为**默认输出思维链**（类似 OpenAI o1），会在生成前插入 `<think>` 标签并输出 "Thinking Process:"。

### 根本原因

查看 `/mnt/afs/davidwang/workspace/models/Qwen3.5-9B/chat_template.jinja` 第 149-153 行：

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}
{%- endif %}
```

**默认行为**：模型会自动进入思维链模式，输出推理过程。

## 解决方案总结

| 方案 | 难度 | 效果 | 推荐度 |
|------|------|------|--------|
| 1. 禁用思维链（`enable_thinking=False`） | ⭐ | ⭐⭐⭐⭐⭐ | 🏆 最优 |
| 2. 后处理提取结果 | ⭐⭐ | ⭐⭐⭐ | 备选 |
| 3. 切换到 Qwen2.5-VL | ⭐⭐⭐ | ⭐⭐⭐⭐ | 非快速方案 |

---

## 方案 1：禁用思维链（推荐）⭐

### 修改位置

**文件**: `src/sana_wm_pipeline/qc/stage3_gpu.py`  
**函数**: `load_qwen_fn` (第 335-363 行)  
**修改行**: 第 357 行

### 修改内容

```python
# 修改前（第 357 行）
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 修改后
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True,
    enable_thinking=False  # 🔑 关键：禁用思维链
)
```

### 完整修复代码

```python
def load_qwen_fn(model_dir: str, device: str = "cuda"):
    """Load Qwen3.5-27B-VL and return vlm_call(prompt, keyframes) -> str."""
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch
    from PIL import Image

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    def vlm_call(prompt: str, keyframes: list) -> str:
        pil_imgs = [Image.fromarray(f) for f in keyframes]
        content = [{"type": "text", "text": prompt}]
        for img in pil_imgs:
            content.insert(-1, {"type": "image", "image": img})
        messages = [{"role": "user", "content": content}]
        
        # 🔑 关键修改：添加 enable_thinking=False
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # 禁用思维链
        )
        
        inputs = processor(text=[text], images=pil_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        return processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return vlm_call
```

### 预期效果

- ✅ **推理时间**: 从 10+ 秒降至 1-3 秒
- ✅ **输出格式**: 直接返回 JSON，无 "Thinking Process:"
- ✅ **相机词移除**: 更准确（无思维链干扰）

---

## 方案 2：后处理提取（备选）

如果 `enable_thinking=False` 无效（可能某些版本不支持），使用后处理：

### 修改 `vlm_call` 函数

```python
import re

def vlm_call(prompt: str, keyframes: list) -> str:
    pil_imgs = [Image.fromarray(f) for f in keyframes]
    content = [{"type": "text", "text": prompt}]
    for img in pil_imgs:
        content.insert(-1, {"type": "image", "image": img})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_imgs, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512)  # 增加 tokens
    
    raw_output = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 🔑 后处理：提取 </think> 之后的内容
    if "</think>" in raw_output:
        return raw_output.split("</think>", 1)[1].strip()
    
    # 或提取最后一个 JSON 块
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_output, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return raw_output
```

---

## 测试脚本

### 1. 验证 `enable_thinking=False` 是否有效

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/test_scripts
python test_qwen_disable_thinking.py
```

**预期输出**：
- 启用思维链：~10 秒，包含 "Thinking Process:"
- 禁用思维链：~1-3 秒，直接输出结果

### 2. 测试后处理方案

```bash
python test_qwen_postprocess.py
```

---

## 应用到生产环境

### 步骤 1：备份原文件

```bash
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py \
   /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py.backup
```

### 步骤 2：应用修改

手动编辑或使用 Edit 工具修改第 357 行。

### 步骤 3：验证修改

```bash
# 查看修改
git diff /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py

# 或直接查看第 357 行附近
sed -n '355,360p' /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py
```

### 步骤 4：在 CMCC 机器上测试

```bash
# 激活环境
conda activate sana_wm_qc_env

# 运行小批量测试
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python scripts/run_stage3_cmcc.py \
  --stage1-jsonl /path/to/test_stage1.jsonl \
  --output-jsonl /tmp/test_stage3_fixed.jsonl \
  --qwen-dir /root/work/david_work/models/Qwen3.5-27B \
  --worker-id 0 \
  --num-workers 1
```

---

## 性能对比预期

| 指标 | 修改前 | 修改后 | 提升 |
|------|--------|--------|------|
| 单样本推理 | 10-12 秒 | 1-3 秒 | **5-10x** |
| 输出格式 | 包含思维链 | 纯 JSON | ✅ |
| Caption 质量 | 可能含相机词 | 更准确 | ✅ |

---

## 常见问题

### Q1: `enable_thinking` 参数不存在？

**A**: 检查 transformers 版本：

```bash
pip show transformers
# 需要 >= 4.57.0（根据 config.json）
```

如果版本过低，使用**方案 2（后处理）**。

### Q2: 修改后仍然输出思维链？

**A**: 可能原因：
1. 代码未正确加载（重启 Python 进程）
2. 模型缓存问题（清除 `~/.cache/huggingface`）
3. transformers 版本不支持该参数（降级到方案 2）

### Q3: Qwen3.5-27B 和 9B 都需要修改吗？

**A**: 是的，所有 Qwen3.5 系列都有这个问题。修改同一个函数即可（它们共用 `load_qwen_fn`）。

---

## 相关文件

- 修复的主文件: `src/sana_wm_pipeline/qc/stage3_gpu.py:357`
- 测试脚本 1: `test_scripts/test_qwen_disable_thinking.py`
- 测试脚本 2: `test_scripts/test_qwen_postprocess.py`
- 原始测试: `test_scripts/test_qwen_prompt_optimization.py`

---

## 总结

**推荐执行顺序**：
1. ✅ 先运行 `test_qwen_disable_thinking.py` 验证方案 1 可行
2. ✅ 如果可行，修改 `stage3_gpu.py:357` 行
3. ✅ 在测试数据上验证性能提升
4. ✅ 部署到生产环境

**预期收益**：
- 推理速度提升 5-10 倍
- 输出格式更规范
- Caption 质量提升
